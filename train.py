# =============================================================================
# train.py — Conditional CycleGAN Training Loop (upgraded architecture)
#
# Supports:
#   --config  <json>   load hyperparameters from a JSON file
#   --resume  <ckpt>   resume training from a saved checkpoint
#   --log_file <path>  redirect stdout/stderr to a log file
#
# Usage:
#   python train.py --config configs/adain_4team.json
#   python train.py --config configs/adain_4team.json --resume checkpoints/epoch_090.pt
# =============================================================================

import os
import sys
import json
import random
import argparse
from typing import Any, Dict

# Ensure project directory is first on sys.path so local modules take
# priority over any same-named packages in the Python environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from models  import Generator, MultiScaleDiscriminator, Classifier
from dataset import JerseyDataset, ImageBuffer, make_paired_dataloaders
from losses  import (
    LSGanLoss, CycleLoss, IdentityLoss,
    ClassificationLoss, PerceptualLoss,
)
from utils   import (
    verify_param_count, save_sample_grid,
    save_checkpoint, load_checkpoint, LossLogger,
)


# =============================================================================
# Section 1: Default Configuration
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "num_teams":         4,
    "epochs":            200,
    "batch_size":        4,
    "lr":                2e-4,
    "lambda_cycle":      10.0,
    "lambda_identity":   5.0,
    "lambda_cls":        2.0,
    "lambda_perceptual": 1.0,
    "embed_dim":         512,
    "num_res_blocks":    12,
    "num_scales":        2,
    "checkpoint_dir":    "checkpoints",
    "sample_dir":        "samples",
    "log_dir":           "logs",
    "data_root":         ".",
    "num_workers":       4,
    "buffer_size":       50,
    "warmup_epochs":     5,
    "pretrained_enc":    True,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a JSON config file and merge with defaults.

    Args:
        config_path : path to a JSON configuration file
    Returns:
        config      : merged configuration dict
    """
    cfg = DEFAULT_CONFIG.copy()
    with open(config_path, 'r') as f:
        cfg.update(json.load(f))
    return cfg


# =============================================================================
# Section 2: Learning Rate Schedule
# =============================================================================

def make_lr_lambda(warmup_epochs: int, total_epochs: int, decay_start: int):
    """
    Build a LambdaLR schedule function:
        Epochs 1 … warmup_epochs   : linear ramp up
        Epochs warmup_epochs+1 … decay_start : constant = 1.0
        Epochs decay_start+1 … total_epochs  : linear decay to 0

    Args:
        warmup_epochs : warm-up length in epochs
        total_epochs  : total training epochs
        decay_start   : epoch at which linear decay begins
    Returns:
        lr_lambda : callable for LambdaLR
    """
    def lr_lambda(epoch: int) -> float:
        e = epoch + 1  # convert 0-based LambdaLR index to 1-based epoch
        if e <= warmup_epochs:
            return max(1e-6, e / warmup_epochs)
        elif e <= decay_start:
            return 1.0
        else:
            remaining = total_epochs - decay_start
            elapsed   = e - decay_start
            return max(0.0, 1.0 - elapsed / remaining)

    return lr_lambda


# =============================================================================
# Section 3: Main Training Loop
# =============================================================================

def train(config: Dict[str, Any], resume_path: str = None) -> None:
    """
    Full training loop for the conditional CycleGAN.

    Each iteration:
        1. Sample source batch (real_img, src_label) from the loader.
        2. Choose one target domain c_tgt; sample real_tgt for D_B.
        3. Update Generator G (adversarial + cycle + identity + cls + perceptual).
        4. Update Classifier C on real images.
        5. Update Discriminator D_A on (real_img, rec_A) pairs.
        6. Update Discriminator D_B on (real_tgt, fake_B) pairs.

    Args:
        config      : training configuration dict
        resume_path : optional checkpoint path to resume from
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    # ── Domain folders ──
    data_root      = config['data_root']
    num_teams      = config['num_teams']
    domain_folders = [
        os.path.join(data_root, f'train{chr(65 + i)}')
        for i in range(num_teams)
    ]
    print(f"[train] Domains: {domain_folders}")

    for d in [config['checkpoint_dir'], config['sample_dir'], config['log_dir']]:
        os.makedirs(d, exist_ok=True)

    # ── Data ──
    loader, dataset = make_paired_dataloaders(
        domain_folders,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment=True,
    )

    # ── Models ──
    G   = Generator(
            num_teams,
            embed_dim=config['embed_dim'],
            num_blocks=config['num_res_blocks'],
            pretrained=config.get('pretrained_enc', True),
          ).to(device)
    D_A = MultiScaleDiscriminator(
            num_teams, img_size=256, num_scales=config['num_scales']
          ).to(device)
    D_B = MultiScaleDiscriminator(
            num_teams, img_size=256, num_scales=config['num_scales']
          ).to(device)
    C   = Classifier(num_teams).to(device)

    verify_param_count(G, D_A, D_B, C)

    # ── Loss functions ──
    gan_loss  = LSGanLoss()
    cyc_loss  = CycleLoss(lambda_cycle=config['lambda_cycle'])
    idt_loss  = IdentityLoss(lambda_identity=config['lambda_identity'])
    cls_loss  = ClassificationLoss(lambda_cls=config['lambda_cls'])
    perc_loss = PerceptualLoss(
                    lambda_perceptual=config['lambda_perceptual']
                ).to(device)
    ce_loss   = nn.CrossEntropyLoss()

    # ── Optimisers ──
    # Generator: two parameter groups — encoder at lower lr, rest at full lr
    lr = config['lr']
    g_param_groups = [
        {'params': G.encoder.parameters(),         'lr': lr * 0.1},
        {'params': G.proj.parameters(),            'lr': lr},
        {'params': G.label_embedding.parameters(), 'lr': lr},
        {'params': G.res_blocks.parameters(),      'lr': lr},
        {'params': G.up1.parameters(),             'lr': lr},
        {'params': G.up2.parameters(),             'lr': lr},
        {'params': G.up3.parameters(),             'lr': lr},
    ]
    opt_G   = Adam(g_param_groups,       betas=(0.5, 0.999))
    opt_D_A = Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D_B = Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_C   = Adam(C.parameters(),   lr=lr, betas=(0.5, 0.999))

    # ── Mixed-precision GradScalers ──
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')
    scaler_C = torch.amp.GradScaler('cuda')

    # ── LR Schedulers ──
    total_epochs  = config['epochs']
    warmup_epochs = config['warmup_epochs']
    decay_start   = total_epochs // 2

    lr_fn     = make_lr_lambda(warmup_epochs, total_epochs, decay_start)
    sched_G   = LambdaLR(opt_G,   lr_lambda=lr_fn)
    sched_D_A = LambdaLR(opt_D_A, lr_lambda=lr_fn)
    sched_D_B = LambdaLR(opt_D_B, lr_lambda=lr_fn)
    sched_C   = LambdaLR(opt_C,   lr_lambda=lr_fn)

    # ── Image buffers ──
    buf_A = ImageBuffer(max_size=config['buffer_size'])
    buf_B = ImageBuffer(max_size=config['buffer_size'])

    # ── Logger ──
    log_path = os.path.join(config['log_dir'], 'loss_log.csv')
    logger   = LossLogger(log_path)

    # ── Optionally resume ──
    start_epoch = 1
    if resume_path:
        start_epoch = load_checkpoint(
            resume_path,
            G, D_A, D_B, C,
            opt_G, opt_D_A, opt_D_B, opt_C,
            scaler_G, scaler_D, scaler_C,
            device,
        )
        for _ in range(start_epoch - 1):
            sched_G.step(); sched_D_A.step(); sched_D_B.step(); sched_C.step()
        print(f"[train] Schedulers fast-forwarded to epoch {start_epoch}")

    print(f"[train] Starting from epoch {start_epoch}/{total_epochs}")

    # ── Training loop ──
    for epoch in range(start_epoch, total_epochs + 1):
        G.train(); D_A.train(); D_B.train(); C.train()

        for iteration, (real_img, src_label) in enumerate(loader, start=1):
            real_img  = real_img.to(device)
            src_label = src_label.to(device)
            B         = real_img.size(0)

            # ── Choose target domain (scalar for whole batch) ──
            src0    = int(src_label[0].item())
            others  = [d for d in range(num_teams) if d != src0]
            c_tgt   = random.choice(others)
            tgt_label = torch.full((B,), c_tgt, dtype=torch.long, device=device)

            # ── Sample real target images for D_B ──
            real_tgt, _ = dataset.sample_from_domain(c_tgt, B)
            real_tgt    = real_tgt.to(device)

            # =================================================================
            # Step 1 — Update Generator G
            # =================================================================
            opt_G.zero_grad(set_to_none=True)

            # Freeze C so its weights aren't updated by the G cls loss path
            for p in C.parameters():
                p.requires_grad_(False)

            with torch.autocast('cuda', dtype=torch.float16):
                fake_B = G(real_img, tgt_label)    # source → target
                rec_A  = G(fake_B,   src_label)    # target → source (cycle)
                idt_A  = G(real_img, src_label)    # source → source (identity)

                # Adversarial: D_B judges fake_B, D_A judges rec_A
                l_gan_B = gan_loss.generator_loss(D_B(fake_B, tgt_label))
                l_gan_A = gan_loss.generator_loss(D_A(rec_A,  src_label))
                l_gan   = l_gan_B + l_gan_A

                l_cyc = cyc_loss(rec_A, real_img)
                l_idt = idt_loss(idt_A, real_img)
                l_cls = cls_loss(C(fake_B), tgt_label)

            # Perceptual loss runs in float32 for numerical stability
            with torch.autocast('cuda', enabled=False):
                l_perc = perc_loss(fake_B.float(), real_tgt.float())

            loss_G = l_gan + l_cyc + l_idt + l_cls + l_perc

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            for p in C.parameters():
                p.requires_grad_(True)

            # =================================================================
            # Step 2 — Update Classifier C on real images
            # =================================================================
            opt_C.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss_C = ce_loss(C(real_img), src_label)
            scaler_C.scale(loss_C).backward()
            scaler_C.step(opt_C)
            scaler_C.update()

            # =================================================================
            # Step 3 — Update Discriminator D_A (source domain)
            # Real: (real_img, src_label)   Fake: (rec_A buffered, src_label)
            # =================================================================
            opt_D_A.zero_grad(set_to_none=True)
            rec_A_buf = buf_A.push_and_pop(rec_A.detach())
            with torch.autocast('cuda', dtype=torch.float16):
                loss_D_A = gan_loss.discriminator_loss(
                    D_A(real_img,  src_label),
                    D_A(rec_A_buf, src_label),
                )
            scaler_D.scale(loss_D_A).backward()
            scaler_D.step(opt_D_A)
            scaler_D.update()

            # =================================================================
            # Step 4 — Update Discriminator D_B (target domain)
            # Real: (real_tgt, tgt_label)   Fake: (fake_B buffered, tgt_label)
            # =================================================================
            opt_D_B.zero_grad(set_to_none=True)
            fake_B_buf = buf_B.push_and_pop(fake_B.detach())
            with torch.autocast('cuda', dtype=torch.float16):
                loss_D_B = gan_loss.discriminator_loss(
                    D_B(real_tgt,   tgt_label),
                    D_B(fake_B_buf, tgt_label),
                )
            scaler_D.scale(loss_D_B).backward()
            scaler_D.step(opt_D_B)
            scaler_D.update()

            # =================================================================
            # Logging
            # =================================================================
            logger.log(
                epoch=epoch,
                iteration=iteration,
                loss_G=loss_G.item(),
                loss_D_A=loss_D_A.item(),
                loss_D_B=loss_D_B.item(),
                loss_cycle=l_cyc.item(),
                loss_identity=l_idt.item(),
                loss_cls=l_cls.item(),
                loss_perceptual=l_perc.item(),
            )

            if iteration % 50 == 0:
                lr_cur = opt_G.param_groups[1]['lr']  # group 1 = main lr
                print(
                    f"Epoch [{epoch:3d}/{total_epochs}] Iter [{iteration:4d}]  "
                    f"G: {loss_G.item():.4f}  D_A: {loss_D_A.item():.4f}  "
                    f"D_B: {loss_D_B.item():.4f}  Cyc: {l_cyc.item():.4f}  "
                    f"Idt: {l_idt.item():.4f}  Cls: {l_cls.item():.4f}  "
                    f"Perc: {l_perc.item():.4f}  lr: {lr_cur:.2e}"
                )

        # ── LR step ──
        sched_G.step(); sched_D_A.step(); sched_D_B.step(); sched_C.step()

        # ── Periodic checkpoint + sample grid ──
        if epoch % 10 == 0:
            ckpt_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch:03d}.pt')
            save_checkpoint(
                ckpt_path, epoch,
                G, D_A, D_B, C,
                opt_G, opt_D_A, opt_D_B, opt_C,
                scaler_G, scaler_D, scaler_C,
                config,
            )
            print(f"[checkpoint] Saved: {ckpt_path}")

            G.eval()
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    vis_fake = G(real_img[:4], tgt_label[:4])
                    vis_rec  = G(vis_fake,     src_label[:4])

            grid_path = os.path.join(config['sample_dir'], f'epoch_{epoch:03d}.png')
            save_sample_grid(
                real_img[:4].cpu().float(),
                vis_fake.cpu().float(),
                vis_rec.cpu().float(),
                grid_path,
            )
            print(f"[samples]    Saved: {grid_path}")
            G.train()

    print("[train] Training complete.")


# =============================================================================
# Section 4: CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train Conditional CycleGAN')
    p.add_argument('--config',    type=str, default=None,
                   help='Path to JSON configuration file')
    p.add_argument('--resume',    type=str, default=None,
                   help='Checkpoint to resume from')
    p.add_argument('--log_file',  type=str, default=None,
                   help='Redirect stdout/stderr to this file')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_fh     = open(args.log_file, 'a', buffering=1)
        sys.stdout = log_fh
        sys.stderr = log_fh

    config = load_config(args.config) if args.config else DEFAULT_CONFIG.copy()
    print(f"[config] {json.dumps(config, indent=2)}")

    train(config, resume_path=args.resume)
