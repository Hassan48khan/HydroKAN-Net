"""
loss.py — Water-Aware Frequency Loss (WAFL) for HydroKAN-Net.

WAFL combines three terms for UAV flood segmentation:
    L = w_dice * Dice + w_bce * BCE + w_freq * FreqConsistency

  - Dice            : region-overlap consistency (handles class imbalance).
  - BCE             : pixel-wise discrimination.
  - FreqConsistency : L1 distance between the 2D-FFT magnitude spectra of the
                      predicted and ground-truth masks. Because standing water
                      is dominated by a smooth, low-frequency response, matching
                      the spectra discourages fragmented / speckled predictions
                      and promotes spatially continuous water regions.

The loss supports multi-scale deep supervision: the same WAFL objective is
applied to the main prediction and to auxiliary decoder heads, with
decaying weights for coarser heads.

Reference: HydroKAN-Net (Spectral-Gated KAN with Boundary-Aware Cross-Scale
Aggregation for Flood Segmentation in UAV Imagery).

Usage
-----
>>> criterion = WaterAwareFrequencyLoss()
>>> # main only:
>>> loss = criterion(logits, target)                 # logits, target: [B,1,H,W]
>>> # with deep supervision (model returns (main, [ds1, ds2, ds3])):
>>> loss = criterion((main, ds_list), target)
"""

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Outputs = Union[torch.Tensor, Tuple[torch.Tensor, Sequence[torch.Tensor]]]


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation. Expects raw logits."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(2, 3)) + self.eps
        den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.eps
        return (1.0 - num / den).mean()


class FrequencyConsistencyLoss(nn.Module):
    """
    L1 distance between the 2D rFFT magnitude spectra of prediction and target.

    Operates on probabilities (sigmoid of logits) so that the comparison is in
    [0, 1] and differentiable end-to-end.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        # rfft2 -> half-spectrum; magnitude is what we match.
        pf = torch.fft.rfft2(prob, norm="ortho").abs()
        tf = torch.fft.rfft2(target, norm="ortho").abs()
        return F.l1_loss(pf, tf)


class WaterAwareFrequencyLoss(nn.Module):
    """
    Water-Aware Frequency Loss (WAFL) with optional multi-scale deep supervision.

    Parameters
    ----------
    w_dice, w_bce, w_freq : float
        Weights for the Dice, BCE, and frequency-consistency terms.
        Defaults (0.5, 0.3, 0.2) match the HydroKAN-Net paper.
    ds_weights : sequence of float
        Weights applied to auxiliary deep-supervision heads, ordered from the
        highest-resolution head to the lowest. Defaults (0.5, 0.3, 0.15).
    pos_weight : float, optional
        Positive-class weight for BCE to counter flood/non-flood imbalance.
        If None, standard BCE is used.
    """

    def __init__(
        self,
        w_dice: float = 0.5,
        w_bce: float = 0.3,
        w_freq: float = 0.2,
        ds_weights: Sequence[float] = (0.5, 0.3, 0.15),
        pos_weight: float = None,
    ):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.w_freq = w_freq
        self.ds_weights = tuple(ds_weights)

        self.dice = DiceLoss()
        self.freq = FrequencyConsistencyLoss()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        else:
            self.pos_weight = None

    def _bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )

    def single(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """WAFL on a single prediction/target pair."""
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        return (
            self.w_dice * self.dice(logits, target)
            + self.w_bce * self._bce(logits, target)
            + self.w_freq * self.freq(logits, target)
        )

    def forward(self, outputs: Outputs, target: torch.Tensor) -> torch.Tensor:
        """
        outputs : either a single logit tensor [B,1,H,W], or a tuple
                  (main_logits, [ds1, ds2, ...]) for deep supervision.
        target  : ground-truth mask [B,1,H,W] with values in {0,1} (float).
        """
        if isinstance(outputs, (tuple, list)):
            main, ds_list = outputs[0], list(outputs[1])
        else:
            main, ds_list = outputs, []

        target = target.float()
        loss = self.single(main, target)
        for w, ds in zip(self.ds_weights, ds_list):
            loss = loss + w * self.single(ds, target)
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 256, 256
    main = torch.randn(B, 1, H, W, requires_grad=True)
    ds = [torch.randn(B, 1, H, W, requires_grad=True) for _ in range(3)]
    target = (torch.rand(B, 1, H, W) > 0.5).float()

    crit = WaterAwareFrequencyLoss()

    l_main = crit(main, target)
    l_ds = crit((main, ds), target)
    print(f"main-only WAFL : {l_main.item():.4f}")
    print(f"with deep sup. : {l_ds.item():.4f}")

    l_ds.backward()
    print("backward OK, grad on main:", main.grad is not None)
