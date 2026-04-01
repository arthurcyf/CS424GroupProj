"""
train.py — Conditional CycleGAN training loop

Usage:
    python train.py --train_a faces --train_b bradpitt_circle_256

Key schedule:
    Epochs 1-5    : linear LR warmup (epoch/5 × base_lr)
    Epochs 6-100  : constant LR = 2e-4
    Epochs 101-200: linear decay to 0

Checkpoints saved every 10 epochs to:  checkpoints/epoch_NNN.pt
Sample grids saved every 10 epochs to: samples/epoch_NNN.png
Loss log written every iteration to:   logs/loss_log.csv
"""

import argparse
import csv
import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models import Generator, Discriminator, IdentityExtractor, ParsingNet
from dataset import make_dataloader, ImageBuffer
from losses import LSGanLoss, CycleLoss, IdentityLoss, IdRetentionLoss
from utils import count_params, save_sample_grid, load_checkpoint, download_bisenet_weights


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_lambda(
    total_epochs:  int = 200,
    warmup_epochs: int = 5,
    decay_start:   int = 100,
):
    """
    Build the lambda function used with torch.optim.lr_scheduler.LambdaLR.

    Schedule (1-indexed epochs):
        1 … warmup_epochs  : linear warmup, lr × epoch/warmup_epochs
        warmup+1 … 100     : constant lr × 1.0
        101 … total_epochs : linear decay, lr × (1 - progress)

    Args:
        total_epochs  : total number of training epochs
        warmup_epochs : number of warmup epochs at the start
        decay_start   : epoch (1-indexed) at which linear decay begins
    Returns:
        fn : callable(epoch_0indexed) → float multiplier
    """
    def fn(epoch_0idx: int) -> float:
        e = epoch_0idx + 1                             # convert to 1-indexed
        if e <= warmup_epochs:
            return e / warmup_epochs
        elif e <= decay_start:
            return 1.0
        else:
            progress = (e - decay_start) / max(1, total_epochs - decay_start)
            return max(0.0, 1.0 - progress)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# CSV Loss Logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    """
    Appends one row of scalar loss values to a CSV file each iteration.

    Args:
        path : destination CSV file path (created with header if absent)
    """

    _FIELDS = [
        'epoch', 'iteration',
        'loss_G', 'loss_D_A', 'loss_D_B',
        'loss_cycle', 'loss_identity', 'loss_id',
    ]

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists():
            with open(path, 'w', newline='') as fh:
                csv.DictWriter(fh, fieldnames=self._FIELDS).writeheader()

    def log(self, **kwargs) -> None:
        """
        Write one row.  Keyword argument names must match _FIELDS.

        Args:
            epoch     : int
            iteration : int
            loss_G, loss_D_A, loss_D_B, loss_cycle, loss_identity, loss_id : float
        """
        row = {
            k: (f'{v:.6f}' if isinstance(v, float) else str(v))
            for k, v in kwargs.items()
        }
        with open(self.path, 'a', newline='') as fh:
            csv.DictWriter(fh, fieldnames=self._FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    Main training entry point.

    Args:
        args : parsed CLI arguments (see parse_args())
    """
    # ── Device ──────────────────────────────────────────────────────────────
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp     = device.type == 'cuda'
    amp_dtype   = torch.float16 if use_amp else torch.float32
    print(f"[train] Device: {device}  |  AMP: {use_amp}")

    # ── Output directories ───────────────────────────────────────────────────
    for d in ('checkpoints', 'samples', 'logs', 'pretrained'):
        Path(d).mkdir(exist_ok=True)

    # ── BiSeNet weights ──────────────────────────────────────────────────────
    bisenet_weights = args.bisenet_weights
    if not bisenet_weights or not Path(bisenet_weights).exists():
        bisenet_weights = download_bisenet_weights()

    # ── Pretrained auxiliary models (frozen) ─────────────────────────────────
    id_extractor = IdentityExtractor(device)
    parsing_net  = ParsingNet(bisenet_weights, device)
    print("[train] Loaded IdentityExtractor and ParsingNet.")

    # ── Trainable models ─────────────────────────────────────────────────────
    G_AB = Generator(args.base_channels, args.n_res_blocks).to(device)
    G_BA = Generator(args.base_channels, args.n_res_blocks).to(device)
    D_A  = Discriminator().to(device)
    D_B  = Discriminator().to(device)
    print(
        f"[train] G_AB params: {count_params(G_AB):,}  "
        f"D_A params: {count_params(D_A):,}"
    )

    # ── Optimisers ───────────────────────────────────────────────────────────
    opt_G  = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_DA = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_DB = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ── LR Schedulers ────────────────────────────────────────────────────────
    lr_fn    = make_lr_lambda(args.epochs, warmup_epochs=5, decay_start=100)
    sched_G  = torch.optim.lr_scheduler.LambdaLR(opt_G,  lr_lambda=lr_fn)
    sched_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=lr_fn)
    sched_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=lr_fn)

    # ── Mixed-precision scaler ───────────────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(
            args.resume,
            G_AB, G_BA, D_A, D_B,
            opt_G, opt_DA, opt_DB,
            device,
        ) + 1
        # Step schedulers so they resume at the correct LR
        for _ in range(start_epoch - 1):
            sched_G.step(); sched_DA.step(); sched_DB.step()
        print(f"[train] Resumed from {args.resume}, starting at epoch {start_epoch}.")

    # ── Loss functions ───────────────────────────────────────────────────────
    gan_loss = LSGanLoss()
    cyc_loss = CycleLoss(lambda_cycle=args.lambda_cycle)
    idt_loss = IdentityLoss(lambda_identity=args.lambda_cycle * 0.5)
    id_ret   = IdRetentionLoss(lambda_id=args.lambda_id)

    # ── Data & buffers ───────────────────────────────────────────────────────
    loader = make_dataloader(
        args.train_a, args.train_b,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    buf_A  = ImageBuffer(max_size=50)
    buf_B  = ImageBuffer(max_size=50)
    logger = CSVLogger('logs/loss_log.csv')

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        for it, batch in enumerate(pbar, start=1):
            real_A = batch['real_A'].to(device)   # (B, 3, 256, 256)
            real_B = batch['real_B'].to(device)   # (B, 3, 256, 256)

            # ── Auxiliary features (no grad) ──────────────────────────────
            with torch.no_grad():
                mask_A = parsing_net(real_A)       # (B, 11, 256, 256)
                mask_B = parsing_net(real_B)
                id_A   = id_extractor(real_A)      # (B, 512)
                id_B   = id_extractor(real_B)

            # ══════════════════════════════════════════════════════════════
            # Generator Update
            # ══════════════════════════════════════════════════════════════
            opt_G.zero_grad(set_to_none=True)

            # Step 1: Generate fake images
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                fake_B = G_AB(real_A, mask_A, id_B)   # A → B
                fake_A = G_BA(real_B, mask_B, id_A)   # B → A

            # Step 2: Extract masks for fake images (no grad, uses detached fakes)
            with torch.no_grad():
                mask_fB = parsing_net(fake_B.detach().float())   # (B, 11, 256, 256)
                mask_fA = parsing_net(fake_A.detach().float())

            # Step 3: Identity embeddings of fakes
            #   Note: id_extractor has frozen weights but NO @no_grad decorator,
            #   so gradients flow through it from fake_B/A back to G_AB/BA.
            fake_B_emb = id_extractor(fake_B.float())   # (B, 512)
            fake_A_emb = id_extractor(fake_A.float())

            # Step 4: Compute all generator losses inside autocast
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Cycle reconstruction
                rec_A = G_BA(fake_B, mask_fB, id_A)    # fake_B → rec_A
                rec_B = G_AB(fake_A, mask_fA, id_B)    # fake_A → rec_B

                # Identity mapping (G applied to own domain)
                idt_A = G_BA(real_A, mask_A, id_A)
                idt_B = G_AB(real_B, mask_B, id_B)

                # Adversarial losses (generator side)
                l_gan_AB = gan_loss.generator_loss(D_B(fake_B, mask_fB, id_B))
                l_gan_BA = gan_loss.generator_loss(D_A(fake_A, mask_fA, id_A))
                l_gan    = l_gan_AB + l_gan_BA

                # Masked cycle-consistency losses
                l_cyc = cyc_loss(rec_A, real_A) + cyc_loss(rec_B, real_B)

                # Masked identity losses
                l_idt = idt_loss(idt_A, real_A) + idt_loss(idt_B, real_B)

            # Identity retention loss (float32 for numerical stability)
            l_id = (
                id_ret(fake_B_emb.float(), id_B.float())
                + id_ret(fake_A_emb.float(), id_A.float())
            )

            loss_G = l_gan + l_cyc + l_idt + l_id
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ══════════════════════════════════════════════════════════════
            # Discriminator D_B Update
            # ══════════════════════════════════════════════════════════════
            opt_DB.zero_grad(set_to_none=True)
            fake_B_buf = buf_B.push_and_pop(fake_B.detach()).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                l_DB = gan_loss.discriminator_loss(
                    D_B(real_B,       mask_B,  id_B),
                    D_B(fake_B_buf,   mask_fB, id_B),
                )
            scaler.scale(l_DB).backward()
            scaler.step(opt_DB)

            # ══════════════════════════════════════════════════════════════
            # Discriminator D_A Update
            # ══════════════════════════════════════════════════════════════
            opt_DA.zero_grad(set_to_none=True)
            fake_A_buf = buf_A.push_and_pop(fake_A.detach()).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                l_DA = gan_loss.discriminator_loss(
                    D_A(real_A,       mask_A,  id_A),
                    D_A(fake_A_buf,   mask_fA, id_A),
                )
            scaler.scale(l_DA).backward()
            scaler.step(opt_DA)

            # ── Scaler update (after all three optimizer steps) ───────────
            scaler.update()

            # ── Progress bar & logging ────────────────────────────────────
            pbar.set_postfix({
                'G':  f'{loss_G.item():.3f}',
                'DA': f'{l_DA.item():.3f}',
                'DB': f'{l_DB.item():.3f}',
            })
            logger.log(
                epoch=epoch,
                iteration=it,
                loss_G=float(loss_G),
                loss_D_A=float(l_DA),
                loss_D_B=float(l_DB),
                loss_cycle=float(l_cyc),
                loss_identity=float(l_idt),
                loss_id=float(l_id),
            )

        # ── End of epoch ─────────────────────────────────────────────────────
        sched_G.step()
        sched_DA.step()
        sched_DB.step()

        current_lr = sched_G.get_last_lr()[0]
        print(f"[train] Epoch {epoch:03d} done  |  lr={current_lr:.2e}  "
              f"G={loss_G.item():.4f}  DA={l_DA.item():.4f}  DB={l_DB.item():.4f}")

        # ── Checkpoint + sample grid every 10 epochs ─────────────────────────
        if epoch % 10 == 0:
            ckpt_path = f'checkpoints/epoch_{epoch:03d}.pt'
            torch.save({
                'epoch':         epoch,
                'base_channels': args.base_channels,
                'n_res_blocks':  args.n_res_blocks,
                'G_AB':          G_AB.state_dict(),
                'G_BA':          G_BA.state_dict(),
                'D_A':           D_A.state_dict(),
                'D_B':           D_B.state_dict(),
                'opt_G':         opt_G.state_dict(),
                'opt_DA':        opt_DA.state_dict(),
                'opt_DB':        opt_DB.state_dict(),
            }, ckpt_path)
            print(f"[train] Checkpoint saved: {ckpt_path}")

            # Visualisation grid (last batch of the epoch, eval mode)
            G_AB.eval(); G_BA.eval()
            with torch.no_grad():
                vis_A    = real_A[:4]
                vis_mA   = mask_A[:4]
                vis_idB  = id_B[:4]
                vis_idA  = id_A[:4]
                fake_B_v = G_AB(vis_A, vis_mA, vis_idB)
                mask_fBv = parsing_net(fake_B_v.float())
                rec_A_v  = G_BA(fake_B_v, mask_fBv, vis_idA)
            save_sample_grid(
                vis_A, fake_B_v, rec_A_v,
                path=f'samples/epoch_{epoch:03d}.png',
            )
            print(f"[train] Sample grid saved: samples/epoch_{epoch:03d}.png")
            G_AB.train(); G_BA.train()

    print("[train] Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Conditional CycleGAN face-swap trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument('--train_a',         default='trainA',
                   help='Domain A image folder (source faces)')
    p.add_argument('--train_b',         default='trainB',
                   help='Domain B image folder (target identity faces)')
    # Architecture
    p.add_argument('--base_channels',   type=int,   default=64,
                   help='Base channel width for generators')
    p.add_argument('--n_res_blocks',    type=int,   default=9,
                   help='Number of AdaIN residual blocks in each generator')
    # Training hyperparameters
    p.add_argument('--epochs',          type=int,   default=200)
    p.add_argument('--batch_size',      type=int,   default=4)
    p.add_argument('--lr',              type=float, default=2e-4)
    p.add_argument('--lambda_cycle',    type=float, default=10.0,
                   help='Weight for cycle-consistency loss')
    p.add_argument('--lambda_id',       type=float, default=2.0,
                   help='Weight for identity-retention loss')
    # Infrastructure
    p.add_argument('--num_workers',     type=int,   default=0,
                   help='DataLoader workers (use 0 on Windows to avoid issues)')
    p.add_argument('--bisenet_weights', default=None,
                   help='Path to BiSeNet 79999_iter.pth (auto-downloaded if absent)')
    p.add_argument('--resume',          default=None,
                   help='Path to checkpoint .pt file to resume training from')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
