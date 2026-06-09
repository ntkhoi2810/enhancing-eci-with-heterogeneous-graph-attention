"""
Focal Loss implementation for class-imbalanced classification.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.

    Down-weights well-classified examples so the model focuses on hard
    negatives.  When ``gamma = 0`` this reduces to standard cross-entropy.

    Args:
        weight: Per-class weights tensor of shape ``(num_classes,)``.
        gamma: Focusing parameter (≥ 0). Higher values suppress easy examples
            more aggressively.
        label_smoothing: Smoothing factor in ``[0, 1)``. Softens targets from
            hard ``{0, 1}`` to ``{ε/C, 1-ε+ε/C}`` to prevent overconfident
            logits and stabilise validation loss.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Raw logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss (or per-sample if ``reduction='none'``).
        """
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss