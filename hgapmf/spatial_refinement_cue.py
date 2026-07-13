from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpatialRefinementCuePredictor(nn.Module):
    """
    Predicts R_l before the current-stage segmentation head.

    It only consumes current CT/MR features and the previous-stage semantic
    prior. Training-time disagreement targets are computed outside this
    module and used only as supervision.
    """

    def __init__(self, in_channels: int, semantic_channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(8, in_channels // 2))
        self.semantic_channels = int(semantic_channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels * 2 + self.semantic_channels, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_ct: torch.Tensor, f_mr: torch.Tensor, semantic_prior: torch.Tensor) -> torch.Tensor:
        if semantic_prior.shape[2:] != f_ct.shape[2:]:
            semantic_prior = F.interpolate(
                semantic_prior,
                size=f_ct.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if semantic_prior.shape[1] != self.semantic_channels:
            raise ValueError(
                f"semantic_prior has {semantic_prior.shape[1]} channels, expected {self.semantic_channels}."
            )
        return self.net(torch.cat([f_ct, f_mr, semantic_prior], dim=1))
