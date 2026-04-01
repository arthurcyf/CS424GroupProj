"""
utils.py — count_params, save_sample_grid, load_checkpoint, BiSeNet weight downloader
"""

import sys
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import save_image


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Counter
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model : any nn.Module
    Returns:
        count : int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Sample Grid
# ─────────────────────────────────────────────────────────────────────────────

def save_sample_grid(
    real_A: Tensor,
    fake_B: Tensor,
    rec_A:  Tensor,
    path:   str,
    nrow:   int = 4,
) -> None:
    """
    Save a 3-row visual sample grid to disk.

      Row 1 : real_A — original Domain A images
      Row 2 : fake_B — translated to Domain B by G_AB
      Row 3 : rec_A  — reconstructed from fake_B by G_BA

    Args:
        real_A : (B, 3, H, W)  in [-1, 1]
        fake_B : (B, 3, H, W)  in [-1, 1]
        rec_A  : (B, 3, H, W)  in [-1, 1]
        path   : output PNG file path
        nrow   : images per row (should equal batch size, default 4)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs = torch.cat(
        [real_A.cpu(), fake_B.cpu(), rec_A.cpu()],
        dim=0,
    )                                          # (3B, 3, H, W)
    save_image(imgs, path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    path:   str,
    G_AB:   nn.Module,
    G_BA:   nn.Module,
    D_A:    nn.Module,
    D_B:    nn.Module,
    opt_G:  torch.optim.Optimizer,
    opt_DA: torch.optim.Optimizer,
    opt_DB: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """
    Load a full training checkpoint into models and optimisers.

    Args:
        path   : path to .pt checkpoint file
        G_AB   : Generator A→B
        G_BA   : Generator B→A
        D_A    : Discriminator A
        D_B    : Discriminator B
        opt_G  : Generator optimiser
        opt_DA : Discriminator A optimiser
        opt_DB : Discriminator B optimiser
        device : device to map tensors onto
    Returns:
        epoch  : the epoch number stored in the checkpoint (int)
    """
    ckpt = torch.load(path, map_location=device)
    G_AB.load_state_dict(ckpt['G_AB'])
    G_BA.load_state_dict(ckpt['G_BA'])
    D_A.load_state_dict(ckpt['D_A'])
    D_B.load_state_dict(ckpt['D_B'])
    opt_G.load_state_dict(ckpt['opt_G'])
    opt_DA.load_state_dict(ckpt['opt_DA'])
    opt_DB.load_state_dict(ckpt['opt_DB'])
    return int(ckpt['epoch'])


# ─────────────────────────────────────────────────────────────────────────────
# BiSeNet Weight Downloader
# ─────────────────────────────────────────────────────────────────────────────

# Primary and fallback URLs for BiSeNet face-parsing weights (79999_iter.pth, ~53 MB)
_BISENET_URLS = [
    # HuggingFace mirrors — direct HTTP downloads, no auth required
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth",
    "https://huggingface.co/vivym/face-parsing-bisenet/resolve/768606b84908769d31ddd78b2e1105319839edfa/79999_iter.pth",
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/79999_iter.pth",
]
_BISENET_DEST = "pretrained/bisenet_79999.pth"


def download_bisenet_weights(dest: str = _BISENET_DEST) -> str:
    """
    Download pre-trained BiSeNet face-parsing weights (79999_iter.pth) if absent.

    Tries multiple HuggingFace mirrors in order; raises RuntimeError if all fail.

    Args:
        dest : local destination path (default 'pretrained/bisenet_79999.pth')
    Returns:
        dest : path to the (now-present) weights file
    """
    if Path(dest).exists():
        return dest
    Path(dest).parent.mkdir(parents=True, exist_ok=True)

    last_exc: Optional[Exception] = None
    for url in _BISENET_URLS:
        print(f"[utils] Downloading BiSeNet weights → {dest}")
        print(f"[utils] Source: {url}")
        try:
            urllib.request.urlretrieve(url, dest, _progress_hook)
            print()   # newline after progress bar
            return dest
        except Exception as exc:
            print(f"\n[utils] Failed ({exc}), trying next mirror…")
            last_exc = exc
            # Remove partial download before retrying
            if Path(dest).exists():
                Path(dest).unlink()

    raise RuntimeError(
        f"\nAll download mirrors failed for BiSeNet weights.\n"
        f"Please download 79999_iter.pth manually from any of:\n"
        + "\n".join(f"  {u}" for u in _BISENET_URLS)
        + f"\nand place the file at:  {dest}\n"
        f"Then pass --bisenet_weights {dest} to the script.\n"
        f"Last error: {last_exc}"
    ) from last_exc


def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    """urllib.request.urlretrieve progress callback."""
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        sys.stdout.write(f'\r  {pct:3d}%')
        sys.stdout.flush()
