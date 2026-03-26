# =============================================================================
# dataset.py — JerseyDataset, ImageBuffer, dataloader factory
#
# Handles multi-domain jersey image loading, augmentation, paired batch
# construction, and the discriminator image replay buffer.
# =============================================================================

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# =============================================================================
# Section 1: JerseyDataset
# =============================================================================

class JerseyDataset(Dataset):
    """
    Multi-domain jersey image dataset.

    Accepts a list of domain folder paths, assigns integer label indices
    (0, 1, 2, …) to each folder, and samples a random image from a random
    domain on every __getitem__ call.

    Augmentations (training):
        - Random horizontal flip
        - Colour jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        - Random crop to 256×256 with padding 10
        - Normalise to [-1, 1] (mean=0.5, std=0.5 per channel)

    Args:
        domain_folders : list of folder paths, e.g. ["trainA/", "trainB/", "trainC/"]
        augment        : apply training augmentations when True (default True)
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    def __init__(self, domain_folders: List[str], augment: bool = True):
        super().__init__()
        self.domain_folders = [Path(f) for f in domain_folders]
        self.num_domains    = len(domain_folders)

        # Build per-domain sorted image path lists
        self.domain_images: List[List[Path]] = []
        for folder in self.domain_folders:
            if not folder.exists():
                raise FileNotFoundError(f"Domain folder not found: {folder}")
            imgs = sorted([
                p for p in folder.iterdir()
                if p.suffix.lower() in self.IMG_EXTENSIONS
            ])
            if not imgs:
                raise ValueError(f"No images found in domain folder: {folder}")
            self.domain_images.append(imgs)

        self.total_images = sum(len(imgs) for imgs in self.domain_images)

        # ── Transform pipeline ──
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
                transforms.RandomCrop(256, padding=10, pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        """Returns total number of images across all domains."""
        return self.total_images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a random image from a random domain (idx is not used directly).

        Args:
            idx : dataset index (ignored — random sampling used instead)
        Returns:
            image : (3, 256, 256) float tensor normalised to [-1, 1]
            label : scalar long tensor (domain index 0 … num_domains-1)
        """
        domain_idx = random.randint(0, self.num_domains - 1)
        img_path   = random.choice(self.domain_images[domain_idx])

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(domain_idx, dtype=torch.long)

        return image, label

    def sample_from_domain(
        self,
        domain_idx: int,
        n: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n images from a specific domain (used for discriminator real images).

        Args:
            domain_idx : integer domain index (0 … num_domains-1)
            n          : number of images to sample
        Returns:
            images : (n, 3, 256, 256) float tensor normalised to [-1, 1]
            labels : (n,) long tensor all equal to domain_idx
        """
        imgs = []
        for _ in range(n):
            img_path = random.choice(self.domain_images[domain_idx])
            img      = Image.open(img_path).convert('RGB')
            imgs.append(self.transform(img))
        images = torch.stack(imgs, dim=0)
        labels = torch.full((n,), domain_idx, dtype=torch.long)
        return images, labels


# =============================================================================
# Section 2: Image Replay Buffer
# =============================================================================

class ImageBuffer:
    """
    Discriminator image replay buffer for training stabilisation.

    Maintains a pool of previously generated images.  With probability 0.5,
    returns a randomly selected buffered image instead of the current fake,
    and updates the buffer slot with the current fake.

    Args:
        max_size : maximum number of images to store (default 50)
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer: List[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Push new images into the buffer and return a mixed batch.

        Args:
            images : (B, 3, H, W) batch of generated images (detached)
        Returns:
            out    : (B, 3, H, W) batch mixing buffered and new images
        """
        return_images = []
        for img in images.detach():
            img = img.unsqueeze(0)  # (1, C, H, W)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img.clone())
                return_images.append(img)
            else:
                if random.random() < 0.5:
                    # Replace a random buffer slot and return the old image
                    idx      = random.randint(0, self.max_size - 1)
                    buffered = self.buffer[idx].clone()
                    self.buffer[idx] = img.clone()
                    return_images.append(buffered)
                else:
                    return_images.append(img)
        return torch.cat(return_images, dim=0)


# =============================================================================
# Section 3: Dataloader Factory
# =============================================================================

def make_paired_dataloaders(
    domain_folders: List[str],
    batch_size:     int  = 4,
    num_workers:    int  = 4,
    augment:        bool = True,
) -> Tuple[DataLoader, JerseyDataset]:
    """
    Create a DataLoader for multi-domain training and return the underlying dataset.

    The DataLoader yields (image, label) pairs sampled randomly across all
    domains.  The caller samples a target label from the dataset's
    sample_from_domain() method for discriminator real-image batches.

    Args:
        domain_folders : list of folder paths (one per team domain)
        batch_size     : training batch size (default 4)
        num_workers    : DataLoader worker processes (default 4)
        augment        : apply training augmentations (default True)
    Returns:
        loader  : DataLoader yielding (image_tensor, label) per batch
        dataset : underlying JerseyDataset (use for sample_from_domain)
    """
    dataset = JerseyDataset(domain_folders, augment=augment)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset


def sample_target_labels(
    source_labels: torch.Tensor,
    num_domains:   int,
) -> torch.Tensor:
    """
    Sample one target label per image, uniformly from all domains OTHER
    than the image's source domain.

    Args:
        source_labels : (B,) long tensor of source domain indices
        num_domains   : total number of domains
    Returns:
        target_labels : (B,) long tensor of target domain indices
    """
    target_labels = []
    for src in source_labels.tolist():
        others = [d for d in range(num_domains) if d != src]
        target_labels.append(random.choice(others))
    return torch.tensor(target_labels, dtype=torch.long)
