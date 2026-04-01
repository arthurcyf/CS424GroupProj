"""
infer.py — Inference pipeline for Conditional CycleGAN face swapping

Single-image mode:
    python infer.py \\
        --checkpoint  checkpoints/epoch_200.pt \\
        --target_image    path/to/target.png \\
        --reference_image path/to/reference.png \\
        --output          output.png

Batch mode (folder of targets, single reference):
    python infer.py \\
        --checkpoint      checkpoints/epoch_200.pt \\
        --target_folder   faces/ \\
        --reference_image bradpitt_circle_256/001_c04300ef.png \\
        --output_folder   swapped_output/
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator, IdentityExtractor, ParsingNet
from utils import download_bisenet_weights


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

_TO_TENSOR = transforms.Compose([
    transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _load_image(path: str, device: torch.device) -> Tensor:
    """
    Load a single image from disk, resize to 256×256 and normalise to [-1, 1].

    Args:
        path   : path to image file
        device : torch device
    Returns:
        img    : (1, 3, 256, 256) float tensor on `device`, values in [-1, 1]
    """
    img = Image.open(path).convert('RGB')
    return _TO_TENSOR(img).unsqueeze(0).to(device)


def _denorm(t: Tensor) -> Tensor:
    """
    Denormalise from [-1, 1] to [0, 1].

    Args:
        t   : (*, 3, H, W) tensor in [-1, 1]
    Returns:
        out : (*, 3, H, W) tensor in [0, 1]
    """
    return (t.clamp(-1.0, 1.0) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(
    checkpoint_path: str,
    bisenet_weights: str,
    device:          torch.device,
) -> Tuple[Generator, IdentityExtractor, ParsingNet]:
    """
    Load the trained generator and pretrained auxiliary models in eval mode.

    Args:
        checkpoint_path : path to .pt training checkpoint (must contain 'G_AB')
        bisenet_weights : path to BiSeNet 79999_iter.pth
        device          : torch device
    Returns:
        G_AB         : Generator A→B in eval mode
        id_extractor : IdentityExtractor in eval mode
        parsing_net  : ParsingNet in eval mode
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_channels = ckpt.get('base_channels', 64)
    n_res_blocks  = ckpt.get('n_res_blocks',  9)

    G_AB = Generator(base_channels=base_channels, n_res_blocks=n_res_blocks).to(device)
    G_AB.load_state_dict(ckpt['G_AB'])
    G_AB.eval()

    id_extractor = IdentityExtractor(device)
    parsing_net  = ParsingNet(bisenet_weights, device)

    print(
        f"[infer] Loaded checkpoint epoch {ckpt.get('epoch', '?')}  "
        f"(base_channels={base_channels}, n_res_blocks={n_res_blocks})"
    )
    return G_AB, id_extractor, parsing_net


# ─────────────────────────────────────────────────────────────────────────────
# Single-image inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    G_AB:           torch.nn.Module,
    id_extractor:   torch.nn.Module,
    parsing_net:    torch.nn.Module,
    target_path:    str,
    reference_path: str,
    output_path:    str,
    device:         torch.device,
) -> None:
    """
    Translate a single target image using a reference identity.

    Steps:
      1. Load and normalise target and reference images.
      2. Extract 512-D identity vector from the reference image.
      3. Extract 11-channel parsing mask from the target image.
      4. Forward through G_AB: fake = G(target_with_mask, id_ref).
      5. Reapply the target's circular background mask to the output.
      6. Denormalise and save as PNG.

    Args:
        G_AB           : trained Generator (A→B)
        id_extractor   : ArcFace identity extractor
        parsing_net    : face parsing network
        target_path    : path to target image (provides pose/expression)
        reference_path : path to reference image (provides identity)
        output_path    : destination PNG path
        device         : torch device
    """
    target    = _load_image(target_path,    device)   # (1, 3, 256, 256)
    reference = _load_image(reference_path, device)   # (1, 3, 256, 256)

    mask_target = parsing_net(target)                 # (1, 11, 256, 256)
    id_ref      = id_extractor(reference)             # (1, 512)

    fake = G_AB(target, mask_target, id_ref)          # (1, 3, 256, 256)

    # Restore the circular background mask from the target image
    bg_mask = (target.max(dim=1, keepdim=True).values > -0.99).float()
    fake    = fake * bg_mask + (-1.0) * (1.0 - bg_mask)

    out = _denorm(fake)                               # (1, 3, 256, 256) in [0, 1]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(out, output_path)
    print(f"[infer] Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch inference
# ─────────────────────────────────────────────────────────────────────────────

class _FolderDataset(Dataset):
    """Minimal dataset that loads images from a flat folder."""

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


@torch.no_grad()
def batch_inference(
    G_AB:           torch.nn.Module,
    id_extractor:   torch.nn.Module,
    parsing_net:    torch.nn.Module,
    target_folder:  str,
    reference_path: str,
    output_folder:  str,
    device:         torch.device,
    batch_size:     int = 8,
) -> None:
    """
    Run inference on every image in `target_folder` using a single reference identity.

    Results are saved to `output_folder` with the same filenames as the inputs.
    Per-image inference time and throughput are printed on completion.

    Args:
        G_AB           : trained Generator (A→B)
        id_extractor   : ArcFace identity extractor
        parsing_net    : face parsing network
        target_folder  : folder of target images (all are processed)
        reference_path : path to the single reference image (identity source)
        output_folder  : folder where translated images are written
        device         : torch device
        batch_size     : number of images processed per forward pass
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    dataset = _FolderDataset(target_folder)
    loader  = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # Pre-extract reference identity once
    ref    = _load_image(reference_path, device)   # (1, 3, 256, 256)
    id_ref = id_extractor(ref)                     # (1, 512)

    total_images   = 0
    per_img_times  = []
    wall_start     = time.perf_counter()

    for imgs, paths in loader:
        imgs    = imgs.to(device)                  # (B, 3, 256, 256)
        B       = imgs.shape[0]
        id_batch = id_ref.expand(B, -1)            # (B, 512)

        t0 = time.perf_counter()

        masks = parsing_net(imgs)                  # (B, 11, 256, 256)
        fakes = G_AB(imgs, masks, id_batch)        # (B, 3,  256, 256)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_img_times.append(elapsed / B)

        # Reapply circular background mask from each target
        bg_mask = (imgs.max(dim=1, keepdim=True).values > -0.99).float()
        fakes   = fakes * bg_mask + (-1.0) * (1.0 - bg_mask)
        fakes   = _denorm(fakes)                   # (B, 3, 256, 256) in [0, 1]

        for i, path in enumerate(paths):
            out_path = Path(output_folder) / Path(path).name
            save_image(fakes[i], str(out_path))

        total_images += B

    wall_elapsed  = time.perf_counter() - wall_start
    avg_ms        = sum(per_img_times) / len(per_img_times) * 1000
    throughput    = total_images / wall_elapsed

    print(f"[batch_infer] Processed    : {total_images} images")
    print(f"[batch_infer] Avg time/img : {avg_ms:.1f} ms")
    print(f"[batch_infer] Throughput   : {throughput:.1f} images/sec")
    print(f"[batch_infer] Output saved : {output_folder}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Conditional CycleGAN face-swap inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint',       required=True,
                   help='Path to .pt training checkpoint')
    p.add_argument('--bisenet_weights',  default=None,
                   help='Path to BiSeNet 79999_iter.pth (auto-downloaded if absent)')
    p.add_argument('--reference_image',  required=True,
                   help='Reference image providing the target identity')
    # Single-image mode
    p.add_argument('--target_image',     default=None,
                   help='Target image (pose/expression canvas) for single-image mode')
    p.add_argument('--output',           default='output.png',
                   help='Output path for single-image mode')
    # Batch mode
    p.add_argument('--target_folder',    default=None,
                   help='Folder of target images for batch mode')
    p.add_argument('--output_folder',    default='output_batch',
                   help='Output folder for batch mode')
    p.add_argument('--batch_size',       type=int, default=8,
                   help='Batch size for batch mode')
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[infer] Device: {device}")

    bisenet_weights = args.bisenet_weights
    if not bisenet_weights or not Path(bisenet_weights).exists():
        bisenet_weights = download_bisenet_weights()

    G_AB, id_extractor, parsing_net = load_models(
        args.checkpoint, bisenet_weights, device
    )

    if args.target_folder:
        batch_inference(
            G_AB, id_extractor, parsing_net,
            args.target_folder,
            args.reference_image,
            args.output_folder,
            device,
            batch_size=args.batch_size,
        )
    elif args.target_image:
        run_inference(
            G_AB, id_extractor, parsing_net,
            args.target_image,
            args.reference_image,
            args.output,
            device,
        )
    else:
        print("[infer] Error: provide --target_image (single) or --target_folder (batch).")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
