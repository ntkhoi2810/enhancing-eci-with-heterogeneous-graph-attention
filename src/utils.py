"""
Utility helpers: seeding, metrics, early stopping, and score recording.
"""

import logging
import random
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def setup_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility across all RNGs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def negative_sampling(example: dict) -> bool:
    """Randomly drop 70 % of negative (non-causal) samples.

    Useful for balancing heavily skewed datasets like CTB.
    """
    if example["labels"] == 0:
        return random.random() > 0.7
    return True


def compute_metrics(
    gold: List[int], predicted: List[int]
) -> Tuple[float, float, float]:
    """Compute binary precision, recall, and F1.

    Returns:
        ``(precision, recall, f1)`` — each in ``[0, 1]``.
    """
    p = precision_score(gold, predicted, zero_division=0)
    r = recall_score(gold, predicted, zero_division=0)
    f = f1_score(gold, predicted, zero_division=0)
    return p, r, f


def record_best_scores(
    timestamp: str,
    precision: float,
    recall: float,
    f1: float,
    filename: str,
) -> None:
    """Append a tab-separated score line to *filename*."""
    with open(filename, "a") as fh:
        fh.write(f"{timestamp}\t{precision * 100:.2f}\t{recall * 100:.2f}\t{f1 * 100:.2f}\n")


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience: How many epochs without improvement to wait.
        verbose: Log a message on each non-improving epoch.
    """

    def __init__(self, patience: int = 10, verbose: bool = False) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter: int = 0
        self.best_score: float | None = None
        self.early_stop: bool = False

    def __call__(self, current_score: float) -> bool:
        """Update state and return ``True`` if this is a new best score."""
        if self.best_score is None:
            self.best_score = current_score
            return True

        if current_score <= self.best_score:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "Early stopping patience: %d/%d", self.counter, self.patience
                )
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        self.best_score = current_score
        self.counter = 0
        return True