"""
Entry-point for training the ECI CausalModel with k-fold cross-validation.

Usage::

    python main.py --dataset data/ESC.pkl --dataset_name ESC --num_folds 5
"""

import argparse
import datetime
import logging
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import GraphTextCollate, load_and_preprocess_data
from src.models import CausalModel
from src.trainer import ModelTrainer
from src.utils import EarlyStopping, record_best_scores, setup_seed

logger = logging.getLogger(__name__)

# Special event marker tokens shared across tokenizer and model
SPECIAL_TOKENS: List[str] = ["<e1>", "</e1>", "<e2>", "</e2>"]

# Columns to drop after tokenization (raw text fields)
_COLS_TO_REMOVE = [
    "sentence",
    "event_tagged_sentence",
    "event_masked_sentence",
    "e1",
    "e2",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for binary classification."""
    count_0 = labels.count(0)
    count_1 = labels.count(1)
    total = len(labels)
    weight_0 = total / (2.0 * count_0)
    weight_1 = total / (2.0 * count_1)
    return torch.tensor([weight_0, weight_1], dtype=torch.float)


def create_dataloaders(
    dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Tokenize a dataset into masked and tagged variants and wrap them in
    DataLoaders.

    Returns:
        ``(mask_dataloader, tag_dataloader)``
    """

    def tokenize_col(text_column: str):
        return lambda x: tokenizer(x[text_column], truncation=True)

    masked = (
        dataset
        .map(tokenize_col("event_masked_sentence"), batched=True, batch_size=32)
        .remove_columns(_COLS_TO_REMOVE)
    )
    tagged = (
        dataset
        .map(tokenize_col("event_tagged_sentence"), batched=True, batch_size=32)
        .remove_columns(_COLS_TO_REMOVE)
    )

    collator = GraphTextCollate(tokenizer=tokenizer)

    dl_mask = DataLoader(
        masked,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_tag = DataLoader(
        tagged,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl_mask, dl_tag


def build_optimizer(
    model: CausalModel,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW with proper weight-decay grouping.

    Best practice for fine-tuning transformers: apply weight decay to Linear
    weights only, and exclude all bias terms and LayerNorm parameters.
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"}
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(param_groups, lr=learning_rate)


def train_fold(
    fold_idx: int,
    model: CausalModel,
    optimizer: torch.optim.Optimizer,
    trainer: ModelTrainer,
    train_loaders: Tuple[DataLoader, DataLoader],
    test_loaders: Tuple[DataLoader, DataLoader],
    args: argparse.Namespace,
    checkpoint_path: str,
) -> None:
    """Run the full training + evaluation loop for a single fold."""
    dl_mask_train, dl_tag_train = train_loaders
    dl_mask_test, dl_tag_test = test_loaders

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.num_epochs):
        ep_info = f"[Ep {epoch + 1}/{args.num_epochs}]"

        train_p, train_r, train_f1, train_loss = trainer.train_epoch(
            dl_mask_train, dl_tag_train, ep_info,
        )
        test_p, test_r, test_f1, test_loss = trainer.evaluate(
            dl_mask_test, dl_tag_test, ep_info,
        )

        logger.info("[Epoch %d] loss: %.6f", epoch + 1, train_loss)
        logger.info(
            "Validation — P: %.4f  R: %.4f  F1: %.4f", test_p, test_r, test_f1,
        )

        is_new_best = early_stopping(test_f1 * 100)

        if is_new_best:
            ckpt_file = os.path.join(checkpoint_path, f"best_model_fold{fold_idx}.pt")
            torch.save(model.state_dict(), ckpt_file)
            logger.info("-> NEW BEST F1: %.2f%%. Checkpoint saved!", test_f1 * 100)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record_best_scores(
                current_time,
                test_p,
                test_r,
                test_f1,
                os.path.join(checkpoint_path, f"best_scores_fold{fold_idx}.txt"),
            )

        if early_stopping.early_stop:
            logger.info("[!] EARLY STOPPING TRIGGERED | FOLD %d", fold_idx)
            break

    logger.info("=== END OF TRAINING | FOLD %d ===\n", fold_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Orchestrate k-fold cross-validation training."""
    setup_seed(args.SEED)
    torch.backends.cudnn.benchmark = True

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Dataset
    total_dataset = load_and_preprocess_data(args.dataset)
    if "ESC" not in args.dataset_name or args.shuffle:
        total_dataset = total_dataset.shuffle(seed=args.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    fold_size = len(total_dataset) // args.num_folds
    checkpoint_path = f'checkpoints/{args.dataset_name}_{args.bert_path.split("/")[-1]}'
    os.makedirs(checkpoint_path, exist_ok=True)
    logger.info("Checkpoint directory: %s", checkpoint_path)

    # ----- K-fold loop -----
    for i in range(args.num_folds):
        fold_idx = i + 1
        logger.info("\n=== Training | Fold %d/%d ===", fold_idx, args.num_folds)

        # Split
        test_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = list(set(range(len(total_dataset))) - set(test_indices))

        train_fold_ds = total_dataset.select(train_indices).shuffle(seed=args.SEED)
        test_fold_ds = total_dataset.select(test_indices)

        # Class weights
        class_weights = compute_class_weights(train_fold_ds["labels"])
        logger.info(
            "[*] Class weights fold %d: Normal=%.4f  Causal=%.4f",
            fold_idx,
            class_weights[0].item(),
            class_weights[1].item(),
        )

        # DataLoaders
        train_loaders = create_dataloaders(
            train_fold_ds, tokenizer, args.train_batchsize,
        )
        test_loaders = create_dataloaders(
            test_fold_ds, tokenizer, args.test_batchsize,
        )

        # Model, optimizer, scheduler, trainer
        model = CausalModel(
            bert_path=args.bert_path,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            device=device,
            special_tokens=SPECIAL_TOKENS,
            visualize=args.visualize,
        ).to(device)

        optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)

        # LR scheduler: linear warmup then linear decay to 0
        steps_per_epoch = len(train_loaders[0])  # batches in mask dataloader
        total_steps = steps_per_epoch * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        logger.info(
            "[*] Scheduler: %d total steps, %d warmup steps",
            total_steps, warmup_steps,
        )

        trainer = ModelTrainer(
            model, optimizer, device, class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            scheduler=scheduler,
        )

        # Train
        train_fold(
            fold_idx,
            model,
            optimizer,
            trainer,
            train_loaders,
            test_loaders,
            args,
            checkpoint_path,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the ECI CausalModel with k-fold cross-validation.",
    )
    parser.add_argument("--dataset", type=str, default="data/ESC_dataset")
    parser.add_argument("--dataset_name", type=str, default="ESC")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_batchsize", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--test_batchsize", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for Linear weights (bias/LayerNorm excluded)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--bert_path", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--SEED", type=int, default=3407)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--focal_gamma", type=float, default=1.0, help="Gamma for Focal Loss (lower = less sensitive to outliers)",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing factor (0.0 = off, 0.1 = recommended)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(parse_args())
