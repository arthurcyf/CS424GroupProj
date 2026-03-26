# =============================================================================
# infer.py — Inference Pipeline for Conditional CycleGAN
#
# Single-image inference:
#   python infer.py --checkpoint checkpoints/epoch_200.pt \
#                   --input      test_samples/player_01.png \
#                   --target_team 2 \
#                   --num_teams  4 \
#                   --output     results/player_01_teamC.png
#
# Batch inference:
#   python infer.py --checkpoint checkpoints/epoch_200.pt \
#                   --input_dir  test_samples/ \
#                   --target_team 2 \
#                   --num_teams  4 \
#                   --output_dir results/
# =============================================================================

import argparse
import os
import sys
import time

# Ensure project directory is first on sys.path so local modules take
# priority over any same-named packages in the Python environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models import Generator


# =============================================================================
# Section 1: Image Pre/Post-Processing Utilities
# =============================================================================

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_image(path: str) -> torch.Tensor:
    """
    Load and normalise a single PNG image to [-1, 1].

    Args:
        path : path to a 256×256 PNG image
    Returns:
        img  : (1, 3, 256, 256) float tensor normalised to [-1, 1]
    """
    img = Image.open(path).convert('RGB')
    return INFER_TRANSFORM(img).unsqueeze(0)  # (1, 3, 256, 256)


def denormalise_and_save(tensor: torch.Tensor, output_path: str) -> None:
    """
    Denormalise a [-1, 1] tensor to [0, 255] and save as PNG.

    Args:
        tensor      : (1, 3, 256, 256) float tensor in [-1, 1]
        output_path : destination file path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_image(tensor.clamp(-1, 1), output_path, normalize=True, value_range=(-1, 1))


# =============================================================================
# Section 2: Model Loading
# =============================================================================

def load_generator(
    checkpoint_path: str,
    num_teams:       int,
    device:          torch.device,
) -> Generator:
    """
    Load a trained Generator from a checkpoint file.

    Args:
        checkpoint_path : path to a .pt checkpoint saved by train.py
        num_teams       : total number of teams the model was trained on
        device          : torch device to load onto
    Returns:
        G : Generator in eval mode, weights restored from checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Recover embed_dim and num_blocks from checkpoint config if available
    cfg        = ckpt.get('config', {})
    embed_dim  = cfg.get('embed_dim',      256)
    num_blocks = cfg.get('num_res_blocks', 9)

    G = Generator(num_teams, embed_dim=embed_dim, num_blocks=num_blocks).to(device)
    G.load_state_dict(ckpt['G_state'])
    G.eval()
    print(f"[infer] Loaded generator from {checkpoint_path}  "
          f"(embed_dim={embed_dim}, num_blocks={num_blocks}, num_teams={num_teams})")
    return G


# =============================================================================
# Section 3: Single-Image Inference
# =============================================================================

def infer_single(
    G:           Generator,
    input_path:  str,
    target_team: int,
    output_path: str,
    device:      torch.device,
) -> None:
    """
    Translate a single jersey image to the target team domain and save.

    Args:
        G           : loaded Generator in eval mode
        input_path  : path to a preprocessed 256×256 PNG jersey image
        target_team : integer label index of the target team (e.g. 0, 1, 2)
        output_path : file path for the translated output PNG
        device      : torch device
    """
    img   = load_image(input_path).to(device)                        # (1, 3, 256, 256)
    label = torch.tensor([target_team], dtype=torch.long, device=device)

    with torch.no_grad():
        fake = G(img, label)                                          # (1, 3, 256, 256)

    denormalise_and_save(fake.cpu(), output_path)
    print(f"[infer] {input_path}  →  {output_path}  (team {target_team})")


# =============================================================================
# Section 4: Batch Inference
# =============================================================================

IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def infer_batch(
    G:           Generator,
    input_dir:   str,
    target_team: int,
    output_dir:  str,
    device:      torch.device,
) -> None:
    """
    Run inference on every image in input_dir and save results to output_dir.

    Reports per-image inference time and average throughput.

    Args:
        G           : loaded Generator in eval mode
        input_dir   : folder containing preprocessed 256×256 PNG images
        target_team : integer label index of the target team
        output_dir  : folder to write translated images (created if absent)
        device      : torch device
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])

    if not image_paths:
        print(f"[infer] No images found in {input_dir}")
        return

    label      = torch.tensor([target_team], dtype=torch.long, device=device)
    times      = []

    print(f"[infer] Batch: {len(image_paths)} images  →  team {target_team}")
    print(f"[infer] Output: {output_dir}")

    for img_path in image_paths:
        img = load_image(str(img_path)).to(device)

        # Synchronise GPU before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            fake = G(img, label)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        out_path = os.path.join(output_dir, img_path.name)
        denormalise_and_save(fake.cpu(), out_path)
        print(f"  {img_path.name}  {elapsed * 1000:.1f} ms  →  {out_path}")

    avg_time       = sum(times) / len(times)
    throughput     = 1.0 / avg_time
    print(f"\n[infer] Processed {len(image_paths)} images")
    print(f"[infer] Average time : {avg_time * 1000:.1f} ms/image")
    print(f"[infer] Throughput   : {throughput:.2f} images/sec")


# =============================================================================
# Section 5: CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single and batch inference."""
    p = argparse.ArgumentParser(
        description='Inference for Conditional CycleGAN Jersey Translator'
    )
    p.add_argument('--checkpoint',   type=str, required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--num_teams',    type=int, required=True,
                   help='Total number of teams the model was trained on')
    p.add_argument('--target_team',  type=int, required=True,
                   help='Target team label index (0-based)')

    # Single-image mode
    p.add_argument('--input',        type=str, default=None,
                   help='Path to a single 256×256 input image')
    p.add_argument('--output',       type=str, default=None,
                   help='Output file path for single-image inference')

    # Batch mode
    p.add_argument('--input_dir',    type=str, default=None,
                   help='Folder of input images for batch inference')
    p.add_argument('--output_dir',   type=str, default=None,
                   help='Output folder for batch inference results')

    p.add_argument('--cpu',          action='store_true',
                   help='Force CPU inference even if CUDA is available')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"[infer] Device: {device}")

    # Validate target team index
    if not (0 <= args.target_team < args.num_teams):
        raise ValueError(
            f"--target_team {args.target_team} is out of range "
            f"[0, {args.num_teams - 1}]"
        )

    G = load_generator(args.checkpoint, args.num_teams, device)

    if args.input is not None:
        # ── Single-image mode ──
        if args.output is None:
            raise ValueError("--output must be specified with --input")
        infer_single(G, args.input, args.target_team, args.output, device)

    elif args.input_dir is not None:
        # ── Batch mode ──
        if args.output_dir is None:
            raise ValueError("--output_dir must be specified with --input_dir")
        infer_batch(G, args.input_dir, args.target_team, args.output_dir, device)

    else:
        raise ValueError("Provide either --input (single) or --input_dir (batch)")
