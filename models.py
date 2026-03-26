# =============================================================================
# models.py — Generator, MultiScaleDiscriminator, Classifier, LabelEmbedding
#
# Upgraded architecture for ~10,000 images per domain:
#   Generator    : Pretrained ResNet-50 encoder (up to layer2) + 12 AdaIN
#                  residual blocks + bilinear-upsample decoder
#   Discriminator: MultiScaleDiscriminator wrapping 2 PatchGAN networks at
#                  full and half resolution
#   Classifier   : Lightweight 3-layer CNN auxiliary classifier
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import torchvision.models as tvm


# =============================================================================
# Section 1: AdaIN Helper Function
# =============================================================================

def adain(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Instance Normalisation.

    Args:
        x     : (B, C, H, W) feature map
        gamma : (B, C)       scale parameters from label embedding
        beta  : (B, C)       shift parameters from label embedding
    Returns:
        out   : (B, C, H, W) normalised and re-scaled feature map
    """
    mean = x.mean(dim=[2, 3], keepdim=True)
    std  = x.std(dim=[2, 3], keepdim=True) + 1e-8
    return gamma[:, :, None, None] * ((x - mean) / std) + beta[:, :, None, None]


# =============================================================================
# Section 2: Label Embedding Module
# =============================================================================

class LabelEmbedding(nn.Module):
    """
    Shared label embedding producing per-block AdaIN (gamma, beta) pairs.

    One gamma head and one beta head per residual block, both projecting from
    embed_dim to the bottleneck channel count.

    Initialisation for identity start (critical for stable early training):
        Gamma heads: weight = 0, bias = 1.0
        Beta  heads: weight = 0, bias = 0.0

    Args:
        num_teams  : number of team domains
        embed_dim  : lookup embedding dimension (default 512)
        num_blocks : number of residual blocks to service (default 12)
        out_dim    : output channel count matching the bottleneck (default 256)
    """

    def __init__(
        self,
        num_teams:  int,
        embed_dim:  int = 512,
        num_blocks: int = 12,
        out_dim:    int = 256,
    ):
        super().__init__()
        self.embedding   = nn.Embedding(num_teams, embed_dim)
        self.gamma_heads = nn.ModuleList(
            [nn.Linear(embed_dim, out_dim) for _ in range(num_blocks)]
        )
        self.beta_heads = nn.ModuleList(
            [nn.Linear(embed_dim, out_dim) for _ in range(num_blocks)]
        )
        # Identity initialisation
        for head in self.gamma_heads:
            nn.init.zeros_(head.weight)
            nn.init.ones_(head.bias)
        for head in self.beta_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            label : (B,) long tensor of team label indices
        Returns:
            List of (gamma, beta) tuples — one per residual block.
            gamma : (B, out_dim), beta : (B, out_dim)
        """
        emb = self.embedding(label)  # (B, embed_dim)
        return [
            (self.gamma_heads[i](emb), self.beta_heads[i](emb))
            for i in range(len(self.gamma_heads))
        ]


# =============================================================================
# Section 3: AdaIN Residual Block
# =============================================================================

class AdaINResBlock(nn.Module):
    """
    Residual block with AdaIN conditioning replacing InstanceNorm2d.

    The same (gamma, beta) pair is applied after each of the two conv layers.

    Args:
        channels : feature channel count (256 throughout the bottleneck)
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode='reflect'
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode='reflect'
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x:     torch.Tensor,
        gamma: torch.Tensor,
        beta:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x     : (B, 256, H, W) input feature map
            gamma : (B, 256)       AdaIN scale
            beta  : (B, 256)       AdaIN shift
        Returns:
            out   : (B, 256, H, W) residual output
        """
        residual = x
        out = self.conv1(x)
        out = adain(out, gamma, beta)
        out = self.relu(out)
        out = self.conv2(out)
        out = adain(out, gamma, beta)
        return out + residual


# =============================================================================
# Section 4: Pretrained ResNet-50 Encoder
# =============================================================================

class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet-50 feature encoder for strong initialisation.

    Slices the backbone at layer2, producing spatially downsampled features
    at 1/8th the input resolution with rich ImageNet-pretrained representations.

    Spatial flow for 256×256 input:
        conv1 (stride 2) → 128×128×64
        maxpool (stride 2) → 64×64×64
        layer1 (stride 1) → 64×64×256
        layer2 (stride 2) → 32×32×512
        projection 1×1   → 32×32×256  (to match bottleneck width)

    The encoder's BatchNorm layers are kept in eval mode during training
    (via the train() override below) so ImageNet-calibrated running statistics
    are used without being corrupted by small GAN batch sizes.

    Args:
        out_channels : projected output channel count (default 256)
        pretrained   : load ImageNet weights (default True)
    """

    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet50(weights=weights)

        # Slice backbone: conv1 → bn1 → relu → maxpool → layer1 → layer2
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )  # output: (B, 512, H/8, W/8)

        # 1×1 projection to bottleneck width + InstanceNorm (replaces BN at boundary)
        self.proj = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def train(self, mode: bool = True):
        """Keep backbone BatchNorm in eval mode regardless of training state."""
        super().train(mode)
        self.features.eval()   # freeze BN running stats
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, 3, 256, 256) image normalised to [-1, 1]
        Returns:
            out : (B, out_channels, 32, 32) encoded features
        """
        return self.proj(self.features(x))


# =============================================================================
# Section 5: Generator
# =============================================================================

class Generator(nn.Module):
    """
    Conditional CycleGAN Generator with pretrained ResNet-50 encoder and
    AdaIN label conditioning in the residual bottleneck.

    Architecture:
        Encoder  : ResNet-50 backbone (pretrained, up to layer2)
                   → (B, 256, 32, 32)
        ResBlocks: num_blocks AdaINResBlocks at 256 channels
        Decoder  : 3× bilinear-upsample + conv stages
                   32×32 → 64×64 → 128×128 → 256×256

    Args:
        num_teams  : total number of team domains
        embed_dim  : label embedding dimension (default 512)
        num_blocks : number of AdaIN residual blocks (default 12)
        pretrained : initialise encoder with ImageNet weights (default True)
    """

    BOTTLENECK_CH = 256

    def __init__(
        self,
        num_teams:  int,
        embed_dim:  int  = 512,
        num_blocks: int  = 12,
        pretrained: bool = True,
    ):
        super().__init__()
        C = self.BOTTLENECK_CH

        self.encoder         = ResNetEncoder(out_channels=C, pretrained=pretrained)
        self.label_embedding = LabelEmbedding(num_teams, embed_dim, num_blocks, out_dim=C)
        self.res_blocks      = nn.ModuleList([AdaINResBlock(C) for _ in range(num_blocks)])
        self.num_blocks      = num_blocks

        # Bilinear-upsample decoder: 32→64→128→256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, 3, 256, 256) source image normalised to [-1, 1]
            label : (B,)             long tensor of target team label indices
        Returns:
            out   : (B, 3, 256, 256) translated image normalised to [-1, 1]
        """
        adain_params = self.label_embedding(label)  # List[(gamma, beta)]

        feat = self.encoder(x)                      # (B, 256, 32, 32)

        for i, block in enumerate(self.res_blocks):
            gamma, beta = adain_params[i]
            feat = block(feat, gamma, beta)

        feat = self.up1(feat)   # (B, 128, 64, 64)
        feat = self.up2(feat)   # (B, 64, 128, 128)
        out  = self.up3(feat)   # (B, 3, 256, 256)
        return out


# =============================================================================
# Section 6: Single-Scale PatchGAN Discriminator
# =============================================================================

class SingleScaleDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator at one spatial scale.

    The team label is injected as a learned spatial embedding of shape
    (B, 1, img_size, img_size) concatenated to the RGB input (4 channels total).

    Args:
        num_teams : number of team domains
        img_size  : spatial resolution this discriminator operates at
    """

    def __init__(self, num_teams: int, img_size: int = 256):
        super().__init__()
        self.img_size    = img_size
        self.label_embed = nn.Embedding(num_teams, img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # raw logits
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, 3, img_size, img_size) image in [-1, 1]
            label : (B,)                        long tensor of team label indices
        Returns:
            out   : (B, 1, H', W') patch logits (no activation)
        """
        B         = x.size(0)
        label_map = self.label_embed(label).view(B, 1, self.img_size, self.img_size)
        return self.net(torch.cat([x, label_map], dim=1))


# =============================================================================
# Section 7: Multi-Scale Discriminator
# =============================================================================

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator operating at multiple spatial resolutions.

    Applies a SingleScaleDiscriminator at each scale. Between scales, the input
    image is average-pooled by 2× so coarser and finer detail are both judged.
    Returns a list of patch-logit tensors — one per scale.

    Typically used with num_scales=2:
        Scale 0: 256×256 (full resolution)
        Scale 1: 128×128 (avg-pooled 2×)

    Args:
        num_teams  : number of team domains
        img_size   : input spatial resolution (default 256)
        num_scales : number of discriminator scales (default 2)
    """

    def __init__(self, num_teams: int, img_size: int = 256, num_scales: int = 2):
        super().__init__()
        self.num_scales      = num_scales
        self.discriminators  = nn.ModuleList([
            SingleScaleDiscriminator(num_teams, img_size=img_size // (2 ** i))
            for i in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    def forward(
        self,
        x:     torch.Tensor,
        label: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            x     : (B, 3, 256, 256) image in [-1, 1]
            label : (B,)             long tensor of team label indices
        Returns:
            outputs : List[Tensor] of length num_scales,
                      each (B, 1, H'_i, W'_i) patch logits
        """
        outputs  = []
        x_scaled = x
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x_scaled, label))
            if i < self.num_scales - 1:
                x_scaled = self.downsample(x_scaled)
        return outputs


# Alias kept for backwards-compatible checkpoint loading in infer.py
Discriminator = MultiScaleDiscriminator


# =============================================================================
# Section 8: Auxiliary Classifier
# =============================================================================

class Classifier(nn.Module):
    """
    Lightweight auxiliary jersey classifier for the classification guidance loss.

    3-layer CNN → Global Average Pool → Linear → num_teams raw logits.
    Trained on real images; used to penalise G if translated jerseys are
    not classifiable as the target team.

    Args:
        num_teams : number of team domains / output classes
    """

    def __init__(self, num_teams: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 128×128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32×32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_teams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, 3, 256, 256) image normalised to [-1, 1]
        Returns:
            out : (B, num_teams)   raw logits
        """
        feat = self.features(x).view(x.size(0), -1)  # (B, 128)
        return self.classifier(feat)
