"""
Training and evaluation logic for the CausalModel.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.loss import FocalLoss
from src.utils import compute_metrics

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles the training loop, mixed-precision scaling, and evaluation.

    Args:
        model: The :class:`~src.models.CausalModel` instance.
        optimizer: PyTorch optimizer (e.g. ``AdamW``).
        device: Target device string (``'cuda'`` or ``'cpu'``).
        class_weights: Optional class-balance weights for the loss function.
        gamma: Focal-loss gamma parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device

        if class_weights is not None:
            class_weights = class_weights.to(device)

        self.criterion = FocalLoss(weight=class_weights, gamma=gamma)
        self.scaler = GradScaler()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_graph_data(
        mask_data: Dict, tag_data: Dict
    ) -> Optional[Dict]:
        """Pop graph structures out of the batch dicts so the remaining
        keys are plain tensors suitable for ``model(**batch)``."""
        graph_data = mask_data.pop("graph_data", None)
        tag_data.pop("graph_data", None)  # discard duplicate copy
        return graph_data

    @staticmethod
    def _move_to_device(
        batch: Dict[str, torch.Tensor], device: str, *, exclude: str = "labels"
    ) -> Dict[str, torch.Tensor]:
        """Move all tensor values in *batch* to *device*, skipping *exclude*."""
        return {k: v.to(device) for k, v in batch.items() if k != exclude}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        dataloader_mask: DataLoader,
        dataloader_tag: DataLoader,
        epoch_info: str,
    ) -> Tuple[float, float, float, float]:
        """Train for one epoch over paired mask/tag dataloaders.

        Returns:
            ``(precision, recall, f1, mean_loss)``
        """
        self.model.train()
        mean_loss = torch.zeros(1, device=self.device)
        predicted_all: List[int] = []
        gold_all: List[int] = []

        pbar = tqdm(
            zip(dataloader_mask, dataloader_tag),
            total=len(dataloader_mask),
            desc=f"Train {epoch_info}",
            dynamic_ncols=True,
            leave=False,
        )

        for iteration, (mask_data, tag_data) in enumerate(pbar):
            graph_data = self._extract_graph_data(mask_data, tag_data)

            mask_data = self._move_to_device(mask_data, self.device)
            labels = tag_data["labels"].to(self.device)
            tag_data = self._move_to_device(tag_data, self.device)

            with autocast("cuda"):
                outputs = self.model(mask_data, tag_data, graph_data).squeeze(1)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)
            predicted_all += list(torch.argmax(outputs, dim=-1).cpu().numpy())
            gold_all += list(labels.cpu().numpy())

        p, r, f1 = compute_metrics(gold_all, predicted_all)
        return p, r, f1, mean_loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        dataloader_mask_test: DataLoader,
        dataloader_tag_test: DataLoader,
        epoch_info: str,
    ) -> Tuple[float, float, float, float]:
        """Evaluate the model on the test fold.

        Returns:
            ``(precision, recall, f1, mean_loss)``
        """
        self.model.eval()
        mean_loss_test: float = 0.0
        predicted_all_test: List[int] = []
        gold_all_test: List[int] = []

        pbar = tqdm(
            zip(dataloader_mask_test, dataloader_tag_test),
            total=len(dataloader_mask_test),
            desc=f"Eval  {epoch_info}",
            dynamic_ncols=True,
            leave=False,
        )

        with torch.no_grad():
            for iteration, (mask_data, tag_data) in enumerate(pbar):
                graph_data = self._extract_graph_data(mask_data, tag_data)

                labels = tag_data["labels"].to(self.device)
                mask_data = self._move_to_device(mask_data, self.device)
                tag_data = self._move_to_device(tag_data, self.device)

                outputs = self.model(mask_data, tag_data, graph_data).squeeze(1)
                loss = self.criterion(outputs, labels)
                mean_loss_test = (mean_loss_test * iteration + loss.detach()) / (iteration + 1)

                predicted = torch.argmax(outputs, dim=-1)
                predicted_all_test += list(predicted.cpu().numpy())
                gold_all_test += list(labels.cpu().numpy())

        p, r, f1 = compute_metrics(gold_all_test, predicted_all_test)
        return p, r, f1, mean_loss_test.item()