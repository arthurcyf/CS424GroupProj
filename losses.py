"""
losses.py — LSGanLoss, CycleLoss, IdentityLoss, IdRetentionLoss

All L1-based losses are masked to exclude the black circular background
(pixels at exactly -1.0 across all channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Valid-pixel Mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_valid_mask(image: Tensor, threshold: float = -0.999) -> Tensor:
    """
    Compute a binary float mask that is 1 where any channel is above `threshold`.

    Pixels in the pure-black circular background are ≈ -1.0 across all channels;
    this mask suppresses them so masked L1 losses only train on face content.

    Args:
        image     : (B, 3, H, W)  image normalised to [-1, 1]
        threshold : pixels with max-channel value ≤ threshold are background
    Returns:
        mask      : (B, 1, H, W)  float mask, 1 = face region, 0 = background
    """
    return (image.max(dim=1, keepdim=True).values > threshold).float()


# ─────────────────────────────────────────────────────────────────────────────
# Least-Squares GAN Loss
# ─────────────────────────────────────────────────────────────────────────────

class LSGanLoss(nn.Module):
    """
    Least-Squares GAN loss (Mao et al., 2017).

    Generator target : E[(D(fake) - 1)²]
    Discriminator    : 0.5 * ( E[(D(real) - 1)²] + E[(D(fake))²] )
    """

    def generator_loss(self, fake_logits: Tensor) -> Tensor:
        """
        Args:
            fake_logits : (B, 1, H', W')  discriminator output on generated images
        Returns:
            loss        : scalar
        """
        return torch.mean((fake_logits - 1.0) ** 2)

    def discriminator_loss(
        self,
        real_logits: Tensor,
        fake_logits: Tensor,
    ) -> Tensor:
        """
        Args:
            real_logits : (B, 1, H', W')  discriminator output on real images
            fake_logits : (B, 1, H', W')  discriminator output on generated images
                          (should be detached from the generator graph)
        Returns:
            loss        : scalar
        """
        real_loss = torch.mean((real_logits - 1.0) ** 2)
        fake_loss = torch.mean(fake_logits ** 2)
        return 0.5 * (real_loss + fake_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Masked Cycle-Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

class CycleLoss(nn.Module):
    """
    Masked L1 cycle-consistency loss.

    L_cycle = lambda_cycle * mean( valid_mask * |rec - real| )

    Args:
        lambda_cycle : loss weight (default 10.0)
    """

    def __init__(self, lambda_cycle: float = 10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle

    def forward(self, rec: Tensor, real: Tensor) -> Tensor:
        """
        Args:
            rec  : (B, 3, H, W)  reconstructed image in [-1, 1]
            real : (B, 3, H, W)  original image in [-1, 1]
        Returns:
            loss : scalar, weighted by lambda_cycle
        """
        valid_mask = compute_valid_mask(real)
        diff = valid_mask * torch.abs(rec - real)
        return self.lambda_cycle * diff.sum() / (valid_mask.sum().clamp(min=1) * 3)


# ─────────────────────────────────────────────────────────────────────────────
# Masked Identity Loss
# ─────────────────────────────────────────────────────────────────────────────

class IdentityLoss(nn.Module):
    """
    Masked L1 identity-preservation loss.

    Encourages G(x, id_x) ≈ x, preventing unnecessary colour/structure shifts.

    L_identity = lambda_identity * mean( valid_mask * |idt - real| )

    Args:
        lambda_identity : loss weight (default 5.0 = 0.5 × lambda_cycle)
    """

    def __init__(self, lambda_identity: float = 5.0):
        super().__init__()
        self.lambda_identity = lambda_identity

    def forward(self, idt: Tensor, real: Tensor) -> Tensor:
        """
        Args:
            idt  : (B, 3, H, W)  identity-mapped image G(real, id_real), in [-1, 1]
            real : (B, 3, H, W)  original image in [-1, 1]
        Returns:
            loss : scalar, weighted by lambda_identity
        """
        valid_mask = compute_valid_mask(real)
        diff = valid_mask * torch.abs(idt - real)
        return self.lambda_identity * diff.sum() / (valid_mask.sum().clamp(min=1) * 3)


# ─────────────────────────────────────────────────────────────────────────────
# Identity Retention Loss
# ─────────────────────────────────────────────────────────────────────────────

class IdRetentionLoss(nn.Module):
    """
    Cosine-distance identity retention loss.

    Penalises deviation between the ArcFace embedding of the generated image
    and the target identity vector.

    L_id = lambda_id * mean( 1 - CosSim(ArcFace(fake), id_tgt) )

    Args:
        lambda_id : loss weight (default 2.0)
    """

    def __init__(self, lambda_id: float = 2.0):
        super().__init__()
        self.lambda_id = lambda_id

    def forward(self, fake_emb: Tensor, id_tgt: Tensor) -> Tensor:
        """
        Args:
            fake_emb : (B, 512)  ArcFace embedding of the generated image
                                 (must remain attached to the generator graph)
            id_tgt   : (B, 512)  target identity embedding (L2-normalised)
        Returns:
            loss     : scalar, weighted by lambda_id
        """
        cos_sim = F.cosine_similarity(fake_emb, id_tgt.detach(), dim=1)  # (B,)
        return self.lambda_id * torch.mean(1.0 - cos_sim)
