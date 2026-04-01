"""
models.py — Generator, Discriminator, IdentityExtractor (ArcFace), ParsingNet (BiSeNet)

Architecture overview:
  Generator : encoder (14ch) → ResBlocks with AdaIN → decoder (3ch)
  Discriminator : PatchGAN, 526-channel input (3 RGB + 11 mask + 512 id)
  IdentityExtractor : wraps facenet_pytorch InceptionResnetV1 (VGGFace2)
  ParsingNet : wraps BiSeNet, outputs 11-channel one-hot mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# AdaIN
# ─────────────────────────────────────────────────────────────────────────────

def adain(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    """
    Adaptive Instance Normalisation.

    Args:
        x     : (B, C, H, W)  feature map
        gamma : (B, C)        per-channel scale (predicted from identity vector)
        beta  : (B, C)        per-channel shift (predicted from identity vector)
    Returns:
        out   : (B, C, H, W)  normalised and scaled/shifted feature map
    """
    mean = x.mean(dim=[2, 3], keepdim=True)
    std  = x.std(dim=[2, 3],  keepdim=True) + 1e-8
    normalised = (x - mean) / std
    return gamma[:, :, None, None] * normalised + beta[:, :, None, None]


# ─────────────────────────────────────────────────────────────────────────────
# AdaIN Projection Head (one per AdaIN application)
# ─────────────────────────────────────────────────────────────────────────────

class AdaINProjection(nn.Module):
    """
    Two linear heads (gamma, beta) that project a 512-D identity vector
    to per-channel scale and shift for a single AdaIN operation.

    Initialised so AdaIN is identity at the start of training:
      gamma weights = 0, gamma bias = 1
      beta  weights = 0, beta  bias = 0

    Args:
        id_dim     : identity vector dimensionality (default 512)
        n_channels : number of feature channels to modulate
    """

    def __init__(self, id_dim: int = 512, n_channels: int = 256):
        super().__init__()
        self.gamma = nn.Linear(id_dim, n_channels)
        self.beta  = nn.Linear(id_dim, n_channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_ (self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, id_vec: Tensor):
        """
        Args:
            id_vec : (B, id_dim)
        Returns:
            gamma  : (B, n_channels)
            beta   : (B, n_channels)
        """
        return self.gamma(id_vec), self.beta(id_vec)


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block with AdaIN
# ─────────────────────────────────────────────────────────────────────────────

class ResBlockAdaIN(nn.Module):
    """
    Residual block where both InstanceNorm layers are replaced by AdaIN.
    Each of the two conv positions has its own AdaINProjection.

    Args:
        channels : number of feature channels (= base_channels * 4)
        id_dim   : identity vector dimensionality (default 512)
    """

    def __init__(self, channels: int, id_dim: int = 512):
        super().__init__()
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect')
        self.proj1  = AdaINProjection(id_dim, channels)
        self.proj2  = AdaINProjection(id_dim, channels)

    def forward(self, x: Tensor, id_vec: Tensor) -> Tensor:
        """
        Args:
            x      : (B, C, H, W)  input feature map
            id_vec : (B, 512)      identity embedding
        Returns:
            out    : (B, C, H, W)  residual-connected output
        """
        gamma1, beta1 = self.proj1(id_vec)
        out = self.conv1(x)
        out = adain(out, gamma1, beta1)
        out = F.relu(out, inplace=True)

        gamma2, beta2 = self.proj2(id_vec)
        out = self.conv2(out)
        out = adain(out, gamma2, beta2)
        return out + x


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional face-swap generator.

    Structure: Encoder → ResBlocks (AdaIN) → Decoder.
    Input is the concatenation of RGB image (3 ch) and parsing mask (11 ch) = 14 ch.
    The identity vector conditions all residual blocks via AdaIN.

    Args:
        base_channels : base channel width (default 64)
        n_res_blocks  : number of AdaIN residual blocks (default 9)
        id_dim        : identity vector dimensionality (default 512)
    """

    def __init__(
        self,
        base_channels: int = 64,
        n_res_blocks:  int = 9,
        id_dim:        int = 512,
    ):
        super().__init__()
        ch    = base_channels
        inner = ch * 4

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc = nn.Sequential(
            # 14 → ch, 256×256
            nn.Conv2d(14, ch, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            # ch → ch*2, 128×128
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            # ch*2 → ch*4, 64×64
            nn.Conv2d(ch * 2, inner, 3, stride=2, padding=1),
            nn.InstanceNorm2d(inner),
            nn.ReLU(inplace=True),
        )

        # ── AdaIN Residual Blocks ─────────────────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlockAdaIN(inner, id_dim) for _ in range(n_res_blocks)]
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec = nn.Sequential(
            # ch*4 → ch*2, 128×128
            nn.ConvTranspose2d(inner, ch * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            # ch*2 → ch, 256×256
            nn.ConvTranspose2d(ch * 2, ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            # ch → 3, 256×256
            nn.Conv2d(ch, 3, 7, padding=3, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, mask: Tensor, id_vec: Tensor) -> Tensor:
        """
        Args:
            x      : (B,  3, 256, 256)  input image normalised to [-1, 1]
            mask   : (B, 11, 256, 256)  one-hot parsing mask
            id_vec : (B, 512)           target identity embedding
        Returns:
            out    : (B,  3, 256, 256)  translated image normalised to [-1, 1]
        """
        inp  = torch.cat([x, mask], dim=1)   # (B, 14, 256, 256)
        feat = self.enc(inp)                  # (B, inner, 64, 64)
        for blk in self.res_blocks:
            feat = blk(feat, id_vec)
        return self.dec(feat)                 # (B, 3, 256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator (PatchGAN, 526-channel input)
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator conditioned on spatial mask and identity.

    Input: 3 (RGB) + 11 (parsing mask) + 512 (identity, spatially tiled) = 526 channels.
    Returns a patch map of real/fake logits (no sigmoid — raw scores for LS-GAN).

    Args:
        None (fixed architecture per spec)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 526 → 64, stride-2  (no norm on first layer)
            nn.Conv2d(526, 64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 128, stride-2
            nn.Conv2d(64,  128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 → 256, stride-2
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 → 512, stride-1
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 → 1, stride-1 (raw logits)
            nn.Conv2d(512, 1,   4, stride=1, padding=1),
        )

    def forward(self, x: Tensor, mask: Tensor, id_vec: Tensor) -> Tensor:
        """
        Args:
            x      : (B,   3, H, W)  image normalised to [-1, 1]
            mask   : (B,  11, H, W)  one-hot parsing mask
            id_vec : (B, 512)        identity embedding
        Returns:
            out    : (B, 1, H', W')  raw PatchGAN logits
        """
        B, _, H, W = x.shape
        id_spatial  = id_vec[:, :, None, None].expand(B, 512, H, W)
        inp = torch.cat([x, mask, id_spatial], dim=1)  # (B, 526, H, W)
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# Identity Extractor  (ArcFace via facenet_pytorch)
# ─────────────────────────────────────────────────────────────────────────────

class IdentityExtractor(nn.Module):
    """
    Wraps facenet_pytorch InceptionResnetV1 pre-trained on VGGFace2.

    Extracts L2-normalised 512-D identity embeddings.
    Model parameters are frozen (requires_grad=False) but the forward
    pass remains in the autograd graph so that identity-retention loss
    can back-propagate through generated images to the generator.

    Requires: pip install facenet-pytorch

    Args:
        device : torch device for the model
    """

    def __init__(self, device: torch.device):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x   : (B, 3, 256, 256)  image normalised to [-1, 1]
        Returns:
            emb : (B, 512)          L2-normalised identity embedding
        """
        # Resize to 160×160 expected by InceptionResnetV1
        x160 = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        # Convert from [-1,1] to VGGFace2 normalisation: (pixel_[0,255] - 127.5) / 128
        x255 = (x160 + 1.0) / 2.0 * 255.0
        x_std = (x255 - 127.5) / 128.0
        emb = self.model(x_std)
        return F.normalize(emb, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# BiSeNet Face Parsing Network
# Adapted from face-parsing.PyTorch (MIT Licence, zllrunning)
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU building block."""

    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        ks:      int = 3,
        stride:  int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _SpatialPath(nn.Module):
    """Three strided convolutions to capture spatial detail."""

    def __init__(self):
        super().__init__()
        self.conv1   = _ConvBnRelu(3,  64, ks=7, stride=2, padding=3)
        self.conv2   = _ConvBnRelu(64, 64, ks=3, stride=2, padding=1)
        self.conv3   = _ConvBnRelu(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = _ConvBnRelu(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x   : (B, 3,   H,   W)
        Returns:
            out : (B, 128, H/8, W/8)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv_out(x)


class _AttentionRefinement(nn.Module):
    """Attention Refinement Module (ARM)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvBnRelu(in_ch, out_ch)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv(x)
        return feat * self.attn(feat)


class _ContextPath(nn.Module):
    """ResNet-18 backbone with two ARM modules for context."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=None)
        self.conv1    = resnet.conv1
        self.bn1      = resnet.bn1
        self.relu     = resnet.relu
        self.maxpool  = resnet.maxpool
        self.layer1   = resnet.layer1   # 64 ch
        self.layer2   = resnet.layer2   # 128 ch  → stride-8
        self.layer3   = resnet.layer3   # 256 ch  → stride-16
        self.layer4   = resnet.layer4   # 512 ch  → stride-32
        self.arm16    = _AttentionRefinement(256, 128)
        self.arm32    = _AttentionRefinement(512, 128)
        self.gc       = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBnRelu(512, 128, ks=1, stride=1, padding=0),
        )
        self.up32     = _ConvBnRelu(128, 128, ks=1, stride=1, padding=0)
        self.up16     = _ConvBnRelu(128, 128, ks=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x   : (B, 3, H, W)
        Returns:
            out : (B, 128, H/8, W/8)
        """
        x       = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x       = self.layer1(x)
        feat8   = self.layer2(x)
        feat16  = self.layer3(feat8)
        feat32  = self.layer4(feat16)

        gc = self.gc(feat32)
        a32 = self.arm32(feat32) + gc
        a32 = F.interpolate(a32, size=feat16.shape[2:], mode='bilinear', align_corners=False)
        a32 = self.up32(a32)

        a16 = self.arm16(feat16) + a32
        a16 = F.interpolate(a16, size=feat8.shape[2:],  mode='bilinear', align_corners=False)
        a16 = self.up16(a16)
        return a16


class _FeatureFusion(nn.Module):
    """Feature Fusion Module (FFM)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _ConvBnRelu(in_ch, out_ch, ks=1, stride=1, padding=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, sp: Tensor, cp: Tensor) -> Tensor:
        x    = torch.cat([sp, cp], dim=1)
        feat = self.conv(x)
        return feat + feat * self.attn(feat)


class BiSeNet(nn.Module):
    """
    BiSeNet face-parsing network (19 semantic classes).

    Args:
        n_classes : number of output semantic classes (default 19)
    """

    def __init__(self, n_classes: int = 19):
        super().__init__()
        self.spatial_path = _SpatialPath()
        self.context_path = _ContextPath()
        self.ffm  = _FeatureFusion(256, 256)
        self.head = nn.Sequential(
            _ConvBnRelu(256, 64),
            nn.Dropout(0.1),
            nn.Conv2d(64, n_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x   : (B, 3, H, W)  image normalised to [0, 1]
        Returns:
            out : (B, n_classes, H, W)  class logits (upsampled to input resolution)
        """
        sp  = self.spatial_path(x)
        cp  = self.context_path(x)
        sp  = F.interpolate(sp, size=cp.shape[2:], mode='bilinear', align_corners=False)
        out = self.ffm(sp, cp)
        out = self.head(out)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# ParsingNet Wrapper  (11-channel one-hot mask)
# ─────────────────────────────────────────────────────────────────────────────

# CelebAMask-HQ 19-class → 11-channel grouping
_PARSE_MAP: dict[int, int] = {
    0:  0,   # background
    1:  1,   # skin
    2:  2,   # l_brow  → brows
    3:  2,   # r_brow  → brows
    4:  3,   # l_eye   → eyes
    5:  3,   # r_eye   → eyes
    6:  4,   # glasses
    7:  5,   # l_ear   → ears
    8:  5,   # r_ear   → ears
    9:  5,   # ear_r   → ears (earrings)
    10: 6,   # nose
    11: 7,   # mouth   → mouth/lips
    12: 7,   # u_lip
    13: 7,   # l_lip
    14: 8,   # neck
    15: 8,   # neck_l
    16: 10,  # cloth   → other
    17: 9,   # hair
    18: 9,   # hat     → hair/head-covering
}
_NUM_MASK_CHANNELS = 11


class ParsingNet(nn.Module):
    """
    Wraps BiSeNet to produce an 11-channel one-hot parsing mask.

    Model parameters are frozen. Intended to run in eval mode.

    Args:
        weights_path : path to BiSeNet .pth checkpoint (79999_iter.pth)
        device       : torch device
    """

    def __init__(self, weights_path: str, device: torch.device):
        super().__init__()
        self.bisenet = BiSeNet(n_classes=19).to(device)
        state = torch.load(weights_path, map_location=device)
        self.bisenet.load_state_dict(state, strict=False)
        self.bisenet.eval()
        for p in self.bisenet.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x    : (B, 3, 256, 256)  image normalised to [-1, 1]
        Returns:
            mask : (B, 11, 256, 256) one-hot float mask in [0, 1]
        """
        x01    = (x.float() + 1.0) / 2.0           # → [0, 1]
        logits = self.bisenet(x01)                   # (B, 19, H, W)
        labels = logits.argmax(dim=1)                # (B, H, W)

        B, H, W = labels.shape
        mask = torch.zeros(B, _NUM_MASK_CHANNELS, H, W,
                           device=x.device, dtype=torch.float32)
        for src_cls, dst_ch in _PARSE_MAP.items():
            mask[:, dst_ch] += (labels == src_cls).float()
        return mask.clamp(0.0, 1.0)
