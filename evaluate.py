"""
evaluate.py — Quantitative evaluation for Conditional CycleGAN face-swap

Metrics
-------
ID-Sim      Primary metric. Mean ArcFace cosine similarity between each generated
            image and the target reference identity embedding. Range [−1, 1];
            higher is better. Measures whether the swap successfully transfers
            the target identity.

FID         Fréchet Inception Distance between the swapped images and real Domain B
            images. Lower is better. Measures perceptual realism.
            Requires: pip install torchmetrics[image]

Cycle SSIM  Structural Similarity Index between cycle-reconstructed images
            (G_BA(G_AB(source))) and the original source images. Range [0, 1];
            higher is better. Measures pose/structure preservation.

Usage
-----
    python evaluate.py \\
        --checkpoint checkpoints/epoch_500.pt \\
        --source_dir faces/ \\
        --target_dir bradpitt_circle_256/ \\
        --reference  bradpitt_circle_256/001_c04300ef.png

    # Evaluate specific checkpoint against multiple reference images:
    python evaluate.py \\
        --checkpoint checkpoints/epoch_500.pt \\
        --source_dir faces/ \\
        --target_dir bradpitt_circle_256/ \\
        --reference  bradpitt_circle_256/001_c04300ef.png \\
        --batch_size 8
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models import Generator, IdentityExtractor, ParsingNet
from utils import download_bisenet_weights


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers  (shared with infer.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

_TO_TENSOR = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _load_image(path: str, device: torch.device) -> Tensor:
    img = Image.open(path).convert('RGB')
    return _TO_TENSOR(img).unsqueeze(0).to(device)


def _denorm(t: Tensor) -> Tensor:
    """[-1, 1] → [0, 1]"""
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


class _FolderDataset(Dataset):
    def __init__(self, folder: str):
        self.paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not self.paths:
            raise ValueError(f"No images found in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[Tensor, str]:
        img = Image.open(self.paths[i]).convert('RGB')
        return _TO_TENSOR(img), self.paths[i]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(
    checkpoint_path: str,
    bisenet_weights:  str,
    device:           torch.device,
) -> Tuple[Generator, Generator, IdentityExtractor, ParsingNet]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_channels = ckpt.get('base_channels', 64)
    n_res_blocks  = ckpt.get('n_res_blocks',  9)

    G_AB = Generator(base_channels=base_channels, n_res_blocks=n_res_blocks).to(device)
    G_BA = Generator(base_channels=base_channels, n_res_blocks=n_res_blocks).to(device)
    G_AB.load_state_dict(ckpt['G_AB'])
    G_BA.load_state_dict(ckpt['G_BA'])
    G_AB.eval()
    G_BA.eval()

    id_extractor = IdentityExtractor(device)
    parsing_net  = ParsingNet(bisenet_weights, device)

    print(
        f"[eval] Loaded checkpoint epoch {ckpt.get('epoch', '?')}  "
        f"(base_channels={base_channels}, n_res_blocks={n_res_blocks})"
    )
    return G_AB, G_BA, id_extractor, parsing_net


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_torchmetrics():
    """
    Import torchmetrics components used for FID and SSIM.
    Raises a clear RuntimeError if torchmetrics is not installed.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        return FrechetInceptionDistance, StructuralSimilarityIndexMeasure
    except ImportError:
        raise RuntimeError(
            "torchmetrics is required for FID and SSIM evaluation.\n"
            "Install it with:  pip install torchmetrics[image]"
        )


def _to_uint8(imgs_01: Tensor) -> Tensor:
    """Convert float [0, 1] tensor to uint8 [0, 255] for FID."""
    return (imgs_01 * 255).clamp(0, 255).to(torch.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[eval] Device: {device}")

    bisenet_weights = args.bisenet_weights
    if not bisenet_weights or not Path(bisenet_weights).exists():
        bisenet_weights = download_bisenet_weights()

    G_AB, G_BA, id_extractor, parsing_net = load_models(
        args.checkpoint, bisenet_weights, device
    )

    FrechetInceptionDistance, StructuralSimilarityIndexMeasure = _try_import_torchmetrics()

    # ── Pre-extract fixed reference identity ──────────────────────────────────
    ref_img  = _load_image(args.reference, device)   # (1, 3, 256, 256)
    id_ref   = id_extractor(ref_img)                 # (1, 512) — L2-normalised
    print(f"[eval] Reference identity extracted from: {args.reference}")

    # ── Prepare data loaders ─────────────────────────────────────────────────
    source_dataset = _FolderDataset(args.source_dir)
    target_dataset = _FolderDataset(args.target_dir)

    source_loader = DataLoader(source_dataset, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                               num_workers=0, pin_memory=True)

    n_source = len(source_dataset)
    n_target = len(target_dataset)
    print(f"[eval] Source images : {n_source}  ({args.source_dir})")
    print(f"[eval] Target images : {n_target}  ({args.target_dir})")

    # ── Initialise metric accumulators ────────────────────────────────────────
    # FID needs feature vectors gathered across the full dataset before scoring
    fid_metric  = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    id_sim_scores: List[float] = []
    ssim_scores:   List[float] = []

    print(f"\n[eval] Running inference on {n_source} source images …")

    for batch_imgs, _ in source_loader:
        batch_imgs = batch_imgs.to(device)          # (B, 3, 256, 256) in [-1, 1]
        B = batch_imgs.shape[0]
        id_batch = id_ref.expand(B, -1)             # (B, 512)

        # Generate swapped image
        mask_src  = parsing_net(batch_imgs)         # (B, 11, 256, 256)
        fake_B    = G_AB(batch_imgs, mask_src, id_batch)   # (B, 3, 256, 256)

        # ── 1. ID-Sim: ArcFace similarity to target reference ─────────────
        fake_B_emb = id_extractor(fake_B.float())   # (B, 512)
        cos_sim = F.cosine_similarity(fake_B_emb, id_batch, dim=1)  # (B,)
        id_sim_scores.extend(cos_sim.cpu().tolist())

        # ── 2. Cycle SSIM: source → fake_B → rec_A ───────────────────────
        id_src    = id_extractor(batch_imgs.float())        # (B, 512)
        mask_fB   = parsing_net(fake_B.float())             # (B, 11, 256, 256)
        rec_A     = G_BA(fake_B, mask_fB, id_src)           # (B, 3, 256, 256)

        src_01  = _denorm(batch_imgs)   # [0, 1]
        rec_01  = _denorm(rec_A)        # [0, 1]
        ssim_val = ssim_metric(rec_01, src_01).item()
        ssim_scores.extend([ssim_val] * B)

        # ── 3. FID — accumulate generated image features ─────────────────
        fake_B_01 = _denorm(fake_B)
        fid_metric.update(_to_uint8(fake_B_01), real=False)

    # Feed real Domain B images into FID
    print(f"[eval] Feeding {n_target} real Domain B images into FID …")
    for batch_imgs, _ in target_loader:
        batch_imgs = batch_imgs.to(device)
        real_01 = _denorm(batch_imgs)
        fid_metric.update(_to_uint8(real_01), real=True)

    # ── Compute final scores ─────────────────────────────────────────────────
    mean_id_sim  = sum(id_sim_scores) / len(id_sim_scores)
    mean_ssim    = sum(ssim_scores)   / len(ssim_scores)
    fid_score    = fid_metric.compute().item()

    print()
    print("=" * 48)
    print("  Evaluation Results")
    print("=" * 48)
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Source images : {n_source}")
    print(f"  Target images : {n_target}")
    print("-" * 48)
    print(f"  ID-Sim  (↑)   : {mean_id_sim:.4f}   [primary metric]")
    print(f"  FID     (↓)   : {fid_score:.2f}")
    print(f"  Cycle SSIM (↑): {mean_ssim:.4f}")
    print("=" * 48)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Conditional CycleGAN face-swap evaluator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint',      required=True,
                   help='Path to .pt training checkpoint')
    p.add_argument('--source_dir',      required=True,
                   help='Folder of source (Domain A) test images')
    p.add_argument('--target_dir',      required=True,
                   help='Folder of real Domain B images (used for FID)')
    p.add_argument('--reference',       required=True,
                   help='Single reference image providing the target identity for ID-Sim')
    p.add_argument('--bisenet_weights', default=None,
                   help='Path to BiSeNet 79999_iter.pth (auto-downloaded if absent)')
    p.add_argument('--batch_size',      type=int, default=8,
                   help='Batch size for inference')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
