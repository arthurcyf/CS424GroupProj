# =============================================================================
# losses.py — LSGanLoss, CycleLoss, IdentityLoss, ClassificationLoss,
#             PerceptualLoss
#
# All loss functions used in the conditional CycleGAN training pipeline.
# LSGanLoss accepts both single tensors and lists (from MultiScaleDiscriminator)
# so the training loop remains the same whether or not multi-scale is used.
# =============================================================================

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tvm


# =============================================================================
# Section 1: Least-Squares GAN Loss (LSGAN)
# =============================================================================

class LSGanLoss(nn.Module):
    """
    Least-Squares GAN adversarial loss (Mao et al., 2017).

    Accepts either a single patch-logit tensor or a list of tensors (from a
    MultiScaleDiscriminator).  When a list is passed, the per-scale losses are
    averaged so the total magnitude stays comparable regardless of num_scales.

    Generator loss:     E[(D(G(x,c), c) - 1)^2]              (target = 1)
    Discriminator loss: 0.5*(E[(D(real,c)-1)^2] + E[D(fake,c)^2])
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def _g_loss_single(self, pred: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, torch.ones_like(pred))

    def _d_loss_single(
        self, pred_real: torch.Tensor, pred_fake: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.mse(pred_real, torch.ones_like(pred_real))
            + self.mse(pred_fake, torch.zeros_like(pred_fake))
        ) * 0.5

    def generator_loss(
        self,
        pred_fake: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Generator adversarial loss.

        Args:
            pred_fake : (B, 1, H', W') or List[(B, 1, H'_i, W'_i)]
                        discriminator patch logits for generated images
        Returns:
            loss      : scalar tensor
        """
        if isinstance(pred_fake, list):
            return torch.stack(
                [self._g_loss_single(p) for p in pred_fake]
            ).mean()
        return self._g_loss_single(pred_fake)

    def discriminator_loss(
        self,
        pred_real: Union[torch.Tensor, List[torch.Tensor]],
        pred_fake: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Discriminator loss (real→1, fake→0), averaged over scales if lists.

        Args:
            pred_real : single tensor or list of tensors for real images
            pred_fake : single tensor or list of tensors for generated images
        Returns:
            loss      : scalar tensor
        """
        if isinstance(pred_real, list):
            return torch.stack([
                self._d_loss_single(r, f)
                for r, f in zip(pred_real, pred_fake)
            ]).mean()
        return self._d_loss_single(pred_real, pred_fake)


# =============================================================================
# Section 2: Cycle-Consistency Loss
# =============================================================================

class CycleLoss(nn.Module):
    """
    L1 label-aware cycle-consistency loss.

    L_cycle = lambda_cycle * ||G(G(x, c_tgt), c_src) - x||_1

    Args:
        lambda_cycle : loss weight (default 10.0)
    """

    def __init__(self, lambda_cycle: float = 10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.l1           = nn.L1Loss()

    def forward(self, rec: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rec  : (B, 3, 256, 256) reconstructed image
            real : (B, 3, 256, 256) original source image
        Returns:
            loss : scalar tensor
        """
        return self.lambda_cycle * self.l1(rec, real)


# =============================================================================
# Section 3: Identity Loss
# =============================================================================

class IdentityLoss(nn.Module):
    """
    L1 identity-preservation loss.

    L_identity = lambda_identity * ||G(x, c_src) - x||_1

    Args:
        lambda_identity : loss weight (default 5.0)
    """

    def __init__(self, lambda_identity: float = 5.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.l1              = nn.L1Loss()

    def forward(self, idt: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idt  : (B, 3, 256, 256) identity-mapped image G(x, c_src)
            real : (B, 3, 256, 256) original source image
        Returns:
            loss : scalar tensor
        """
        return self.lambda_identity * self.l1(idt, real)


# =============================================================================
# Section 4: Auxiliary Classification Loss
# =============================================================================

class ClassificationLoss(nn.Module):
    """
    Cross-entropy classification loss for the auxiliary jersey classifier.

    L_cls = lambda_cls * CrossEntropy(C(fake_B), c_tgt)

    Args:
        lambda_cls : loss weight (default 2.0)
    """

    def __init__(self, lambda_cls: float = 2.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_teams) raw classifier logits
            targets : (B,)           long tensor of target team indices
        Returns:
            loss    : scalar tensor
        """
        return self.lambda_cls * self.ce(logits, targets)


# =============================================================================
# Section 5: Perceptual Loss (VGG16, frozen)
# =============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual feature-matching loss using a pretrained, frozen VGG16 network.

    Computes L1 distance between VGG16 intermediate feature maps of the
    generated image and the real target image at three depths:
        relu1_2  (features[:4])   — low-level edges / colours
        relu2_2  (features[4:9])  — mid-level textures
        relu3_3  (features[9:16]) — higher-level patterns

    VGG16 parameters are permanently frozen (requires_grad=False).

    Input normalisation:
        GAN images are in [-1, 1] (mean=0.5, std=0.5).
        VGG16 expects ImageNet normalisation (mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]).  Conversion is applied internally.

    Args:
        lambda_perceptual : loss weight (default 1.0)
    """

    def __init__(self, lambda_perceptual: float = 1.0):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual

        vgg_feats   = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg_feats.children())[:4])   # → relu1_2
        self.slice2 = nn.Sequential(*list(vgg_feats.children())[4:9])  # → relu2_2
        self.slice3 = nn.Sequential(*list(vgg_feats.children())[9:16]) # → relu3_3

        # Permanently frozen — VGG provides a fixed perceptual metric
        for param in self.parameters():
            param.requires_grad_(False)

        # ImageNet normalisation constants as buffers (device-portable)
        self.register_buffer(
            'mean_vgg', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std_vgg',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _to_vgg_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert GAN [-1,1] image tensor to VGG ImageNet-normalised space."""
        x = x * 0.5 + 0.5                           # [-1,1] → [0,1]
        return (x - self.mean_vgg) / self.std_vgg    # VGG normalised

    def _extract(self, x: torch.Tensor):
        """
        Extract features at relu1_2, relu2_2, relu3_3 sequentially.

        Args:
            x : (B, 3, H, W) already in VGG normalised space
        Returns:
            Tuple of three feature tensors
        """
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return f1, f2, f3

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Compute mean L1 perceptual distance between fake_B and real target.

        Args:
            fake : (B, 3, 256, 256) generated image in [-1, 1]
            real : (B, 3, 256, 256) real target-domain image in [-1, 1]
        Returns:
            loss : scalar tensor = lambda_perceptual * mean L1 over three layers
        """
        fake_vgg = self._to_vgg_space(fake)
        real_vgg = self._to_vgg_space(real)

        # Gradients flow through fake features only; real is a fixed reference
        fake_feats = self._extract(fake_vgg)
        with torch.no_grad():
            real_feats = self._extract(real_vgg)

        loss = sum(F.l1_loss(f, r) for f, r in zip(fake_feats, real_feats)) / 3.0
        return self.lambda_perceptual * loss
