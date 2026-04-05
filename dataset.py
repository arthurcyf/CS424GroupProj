"""
dataset.py — FaceDataset, ImageBuffer, dataloader factory

Assumes pre-processed images: circle-cropped, black background, 256×256 PNG.
"""

import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Image Transforms
# ─────────────────────────────────────────────────────────────────────────────

def _make_train_transform(image_size: int = 256) -> transforms.Compose:
    """CycleGAN-style training augmentation: resize, flip, normalise to [-1,1]."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _make_eval_transform(image_size: int = 256) -> transforms.Compose:
    """Evaluation / inference transform: resize and normalise only."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _list_images(folder: str) -> List[str]:
    """Return sorted list of all image file paths in `folder`."""
    p = Path(folder)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)


# ─────────────────────────────────────────────────────────────────────────────
# Unpaired Face Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Unpaired dataset for CycleGAN-style training.

    Domain A and Domain B are loaded independently.  If Domain B is smaller
    than Domain A, its images are cycled (wrapped) to match the length of A.

    Args:
        domain_a_dir : path to Domain A image folder (e.g. trainA/)
        domain_b_dir : path to Domain B image folder (e.g. trainB/)
        image_size   : spatial resolution after resize (default 256)
    """

    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str,
        image_size:   int = 256,
    ):
        self.files_a   = _list_images(domain_a_dir)
        self.files_b   = _list_images(domain_b_dir)
        if not self.files_a:
            raise ValueError(f"No images found in {domain_a_dir}")
        if not self.files_b:
            raise ValueError(f"No images found in {domain_b_dir}")
        self.transform = _make_train_transform(image_size)
        self.len_b     = len(self.files_b)

    def __len__(self) -> int:
        return len(self.files_a)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Args:
            idx : dataset index (0-indexed)
        Returns:
            dict:
                'real_A' : (3, H, W) tensor in [-1, 1]
                'real_B' : (3, H, W) tensor in [-1, 1]
                'path_A' : str absolute path to Domain A image
                'path_B' : str absolute path to Domain B image
        """
        path_a = self.files_a[idx]
        path_b = self.files_b[idx % self.len_b]          # cycle B
        img_a  = Image.open(path_a).convert('RGB')
        img_b  = Image.open(path_b).convert('RGB')
        return {
            'real_A': self.transform(img_a),
            'real_B': self.transform(img_b),
            'path_A': path_a,
            'path_B': path_b,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ImageBuffer:
    """
    Discriminator replay buffer for training stabilisation (Shrivastava et al., 2017).

    Stores up to `max_size` previously generated (image, mask) pairs so that
    the discriminator always receives a mask that matches the image it sees.

    Args:
        max_size : maximum buffer capacity (default 50)
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer: List[tuple] = []  # list of (image, mask) tuples

    def push_and_pop(self, images: Tensor, masks: Tensor) -> tuple:
        """
        Args:
            images : (B, 3,  H, W)  batch of generated images
            masks  : (B, 11, H, W)  corresponding parsing masks
        Returns:
            out_images : (B, 3,  H, W)  mixed batch (buffered + current)
            out_masks  : (B, 11, H, W)  matching masks for each returned image
        """
        result_imgs: List[Tensor] = []
        result_msks: List[Tensor] = []
        for img, msk in zip(images.detach(), masks.detach()):
            if len(self.buffer) < self.max_size:
                self.buffer.append((img.clone(), msk.clone()))
                result_imgs.append(img)
                result_msks.append(msk)
            elif random.random() < 0.5:
                idx = random.randrange(len(self.buffer))
                old_img, old_msk = self.buffer[idx]
                self.buffer[idx] = (img.clone(), msk.clone())
                result_imgs.append(old_img)
                result_msks.append(old_msk)
            else:
                result_imgs.append(img)
                result_msks.append(msk)
        return torch.stack(result_imgs, dim=0), torch.stack(result_msks, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataloader Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloader(
    domain_a_dir: str,
    domain_b_dir: str,
    batch_size:   int  = 4,
    num_workers:  int  = 4,
    image_size:   int  = 256,
    shuffle:      bool = True,
) -> DataLoader:
    """
    Build a DataLoader for unpaired face-swap training.

    Args:
        domain_a_dir : path to Domain A image folder
        domain_b_dir : path to Domain B image folder
        batch_size   : samples per batch
        num_workers  : DataLoader worker processes (set 0 on Windows if issues)
        image_size   : target spatial resolution
        shuffle      : whether to shuffle each epoch
    Returns:
        loader       : DataLoader yielding dicts with keys 'real_A', 'real_B'
    """
    dataset = FaceDataset(domain_a_dir, domain_b_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
