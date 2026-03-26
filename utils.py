# =============================================================================
# utils.py — count_params, save_sample_grid, checkpoint I/O, loss logger
#
# Shared utilities used by train.py and infer.py.
# =============================================================================

import os
import csv
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.utils import save_image


# =============================================================================
# Section 1: Parameter Count Verification
# =============================================================================

def count_params(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a module.

    Args:
        model : any nn.Module
    Returns:
        count : total number of parameters where requires_grad=True
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_param_count(
    G:   nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    C:   nn.Module,
) -> int:
    """
    Print per-model and total parameter counts, then assert total ≤ 60M.

    Args:
        G   : Generator
        D_A : Discriminator A
        D_B : Discriminator B
        C   : Auxiliary Classifier
    Returns:
        total : combined trainable parameter count
    Raises:
        AssertionError : if total exceeds 60,000,000
    """
    g_params   = count_params(G)
    d_a_params = count_params(D_A)
    d_b_params = count_params(D_B)
    c_params   = count_params(C)
    total      = g_params + d_a_params + d_b_params + c_params

    print(f"G params:     {g_params:,}")
    print(f"D_A params:   {d_a_params:,}")
    print(f"D_B params:   {d_b_params:,}")
    print(f"C params:     {c_params:,}")
    print(f"Total params: {total:,}")
    # No hard cap — report only (pretrained encoder removes the earlier 60M constraint)
    return total


# =============================================================================
# Section 2: Sample Grid Saving
# =============================================================================

def save_sample_grid(
    real_A:    torch.Tensor,
    fake_B:    torch.Tensor,
    rec_A:     torch.Tensor,
    save_path: str,
    nrow:      int = 4,
) -> None:
    """
    Save a 3-row visual grid to disk using torchvision.

        Row 1: real_A  — original source images
        Row 2: fake_B  — G(real_A, c_tgt) translated images
        Row 3: rec_A   — G(fake_B, c_src) reconstructed images

    Args:
        real_A    : (B, 3, 256, 256) real source images in [-1, 1]
        fake_B    : (B, 3, 256, 256) translated images in [-1, 1]
        rec_A     : (B, 3, 256, 256) reconstructed images in [-1, 1]
        save_path : output file path (PNG)
        nrow      : images per row in the grid (default 4)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    grid = torch.cat([
        real_A.float().clamp(-1, 1),
        fake_B.float().clamp(-1, 1),
        rec_A.float().clamp(-1, 1),
    ], dim=0)
    save_image(grid, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))


# =============================================================================
# Section 3: Checkpoint Save / Load
# =============================================================================

def save_checkpoint(
    path:      str,
    epoch:     int,
    G:         nn.Module,
    D_A:       nn.Module,
    D_B:       nn.Module,
    C:         nn.Module,
    opt_G:     torch.optim.Optimizer,
    opt_D_A:   torch.optim.Optimizer,
    opt_D_B:   torch.optim.Optimizer,
    opt_C:     torch.optim.Optimizer,
    scaler_G:  torch.amp.GradScaler,
    scaler_D:  torch.amp.GradScaler,
    scaler_C:  torch.amp.GradScaler,
    config:    Dict[str, Any],
) -> None:
    """
    Save a full training checkpoint to disk.

    Stores model weights, optimiser states, GradScaler states, epoch index,
    and the training config so a run can be resumed exactly.

    Args:
        path     : output .pt file path
        epoch    : current epoch number (saved so resume starts at epoch+1)
        G        : Generator
        D_A      : Discriminator A
        D_B      : Discriminator B
        C        : Auxiliary Classifier
        opt_G    : Generator optimiser state
        opt_D_A  : Discriminator A optimiser state
        opt_D_B  : Discriminator B optimiser state
        opt_C    : Classifier optimiser state
        scaler_G : GradScaler for generator
        scaler_D : GradScaler for discriminators
        scaler_C : GradScaler for classifier
        config   : training configuration dict
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'epoch':          epoch,
        'G_state':        G.state_dict(),
        'D_A_state':      D_A.state_dict(),
        'D_B_state':      D_B.state_dict(),
        'C_state':        C.state_dict(),
        'opt_G_state':    opt_G.state_dict(),
        'opt_D_A_state':  opt_D_A.state_dict(),
        'opt_D_B_state':  opt_D_B.state_dict(),
        'opt_C_state':    opt_C.state_dict(),
        'scaler_G_state': scaler_G.state_dict(),
        'scaler_D_state': scaler_D.state_dict(),
        'scaler_C_state': scaler_C.state_dict(),
        'config':         config,
    }, path)


def load_checkpoint(
    path:     str,
    G:        nn.Module,
    D_A:      nn.Module,
    D_B:      nn.Module,
    C:        nn.Module,
    opt_G:    torch.optim.Optimizer,
    opt_D_A:  torch.optim.Optimizer,
    opt_D_B:  torch.optim.Optimizer,
    opt_C:    torch.optim.Optimizer,
    scaler_G: torch.amp.GradScaler,
    scaler_D: torch.amp.GradScaler,
    scaler_C: torch.amp.GradScaler,
    device:   torch.device,
) -> int:
    """
    Load a full training checkpoint and restore all states in-place.

    Args:
        path     : .pt checkpoint file path
        G        : Generator        (mutated in-place)
        D_A      : Discriminator A  (mutated in-place)
        D_B      : Discriminator B  (mutated in-place)
        C        : Classifier       (mutated in-place)
        opt_G    : Generator optimiser       (mutated in-place)
        opt_D_A  : Discriminator A optimiser (mutated in-place)
        opt_D_B  : Discriminator B optimiser (mutated in-place)
        opt_C    : Classifier optimiser      (mutated in-place)
        scaler_G : GradScaler for G  (mutated in-place)
        scaler_D : GradScaler for D  (mutated in-place)
        scaler_C : GradScaler for C  (mutated in-place)
        device   : device to map checkpoint tensors to
    Returns:
        start_epoch : epoch to resume from (= saved_epoch + 1)
    """
    ckpt = torch.load(path, map_location=device)

    G.load_state_dict(ckpt['G_state'])
    D_A.load_state_dict(ckpt['D_A_state'])
    D_B.load_state_dict(ckpt['D_B_state'])
    C.load_state_dict(ckpt['C_state'])

    opt_G.load_state_dict(ckpt['opt_G_state'])
    opt_D_A.load_state_dict(ckpt['opt_D_A_state'])
    opt_D_B.load_state_dict(ckpt['opt_D_B_state'])
    opt_C.load_state_dict(ckpt['opt_C_state'])

    scaler_G.load_state_dict(ckpt['scaler_G_state'])
    scaler_D.load_state_dict(ckpt['scaler_D_state'])
    scaler_C.load_state_dict(ckpt['scaler_C_state'])

    start_epoch = ckpt['epoch'] + 1
    print(f"[checkpoint] Resumed from {path}  (epoch {ckpt['epoch']} → resuming at {start_epoch})")
    return start_epoch


# =============================================================================
# Section 4: CSV Loss Logger
# =============================================================================

class LossLogger:
    """
    Append-mode CSV logger for per-iteration training losses.

    Columns: epoch, iteration, loss_G, loss_D_A, loss_D_B,
             loss_cycle, loss_identity, loss_cls

    The CSV header is written once on first creation; subsequent runs that
    resume from a checkpoint append to the same file.

    Args:
        log_path : path to the output CSV file (created if absent)
    """

    FIELDS = [
        'epoch', 'iteration',
        'loss_G', 'loss_D_A', 'loss_D_B',
        'loss_cycle', 'loss_identity', 'loss_cls', 'loss_perceptual',
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(
        self,
        epoch:            int,
        iteration:        int,
        loss_G:           float,
        loss_D_A:         float,
        loss_D_B:         float,
        loss_cycle:       float,
        loss_identity:    float,
        loss_cls:         float,
        loss_perceptual:  float,
    ) -> None:
        """
        Append one row to the CSV.

        Args:
            epoch            : current epoch number
            iteration        : current iteration within the epoch
            loss_G           : total generator loss (weighted sum)
            loss_D_A         : discriminator A loss
            loss_D_B         : discriminator B loss
            loss_cycle       : cycle-consistency loss (weighted)
            loss_identity    : identity loss (weighted)
            loss_cls         : classification loss (weighted)
            loss_perceptual  : perceptual (VGG16) loss (weighted)
        """
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({
                'epoch':            epoch,
                'iteration':        iteration,
                'loss_G':           f"{loss_G:.6f}",
                'loss_D_A':         f"{loss_D_A:.6f}",
                'loss_D_B':         f"{loss_D_B:.6f}",
                'loss_cycle':       f"{loss_cycle:.6f}",
                'loss_identity':    f"{loss_identity:.6f}",
                'loss_cls':         f"{loss_cls:.6f}",
                'loss_perceptual':  f"{loss_perceptual:.6f}",
            })
