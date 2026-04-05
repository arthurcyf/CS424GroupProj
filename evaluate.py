# =============================================================================
# evaluate.py — Quantitative evaluation for Conditional CycleGAN checkpoints
#
# Metrics (unpaired-friendly):
#   - Cycle consistency:   L1 / PSNR / SSIM for A->B->A and B->A->B
#   - Identity mapping:    L1 / PSNR / SSIM for A->A and B->B
#   - Optional cls score:  translated-domain accuracy via checkpoint classifier C
#
# Example:
#   python evaluate.py --checkpoint checkpoints/epoch_040.pt --max_images 200
# =============================================================================

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project directory is first on sys.path so local modules take
# priority over any same-named packages in the Python environment.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import Generator, Classifier


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def list_images(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXTENSIONS]
    if not files:
        raise ValueError(f"No images found in folder: {folder}")
    return files


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = EVAL_TRANSFORM(img).unsqueeze(0)
    return t.to(device)


def mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(x, y)
    if mse.item() <= 1e-12:
        return torch.tensor(99.0, device=x.device)
    return 20.0 * torch.log10(torch.tensor(data_range, device=x.device)) - 10.0 * torch.log10(mse)


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0, window_size: int = 11) -> torch.Tensor:
    # x,y shape: (B, C, H, W), values in [-1, 1]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=window_size, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, kernel_size=window_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, kernel_size=window_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=window_size, stride=1, padding=pad) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return {"mean": m, "std": math.sqrt(v)}


def resolve_domains(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[str, str]:
    if args.domain_a_dir and args.domain_b_dir:
        return args.domain_a_dir, args.domain_b_dir

    data_root = cfg.get("data_root", ".")
    domain_a = cfg.get("domain_a_dir", "outputs_faces_ronaldo_filtered/faces")
    domain_b = cfg.get("domain_b_dir", "bradpitt_circle_bounded")
    return (
        os.path.normpath(os.path.join(data_root, domain_a)),
        os.path.normpath(os.path.join(data_root, domain_b)),
    )


def evaluate(
    checkpoint_path: str,
    domain_a_dir: str,
    domain_b_dir: str,
    max_images: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    embed_dim = int(cfg.get("embed_dim", 512))
    num_blocks = int(cfg.get("num_res_blocks", 12))
    num_teams = int(cfg.get("num_teams", 2))

    G = Generator(num_teams, embed_dim=embed_dim, num_blocks=num_blocks).to(device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()

    C = None
    has_classifier = "C_state" in ckpt
    if has_classifier:
        C = Classifier(num_teams).to(device)
        C.load_state_dict(ckpt["C_state"])
        C.eval()

    a_images = list_images(domain_a_dir)
    b_images = list_images(domain_b_dir)

    random.seed(seed)
    if max_images > 0:
        a_images = random.sample(a_images, k=min(max_images, len(a_images)))
        b_images = random.sample(b_images, k=min(max_images, len(b_images)))

    print(f"[eval] Device: {device}")
    print(f"[eval] Checkpoint: {checkpoint_path}")
    print(f"[eval] Domain A: {domain_a_dir} ({len(a_images)} images)")
    print(f"[eval] Domain B: {domain_b_dir} ({len(b_images)} images)")
    print(f"[eval] num_teams={num_teams}, embed_dim={embed_dim}, num_blocks={num_blocks}")

    metrics = {
        "cycle_A_l1": [], "cycle_A_psnr": [], "cycle_A_ssim": [],
        "identity_A_l1": [], "identity_A_psnr": [], "identity_A_ssim": [],
        "cycle_B_l1": [], "cycle_B_psnr": [], "cycle_B_ssim": [],
        "identity_B_l1": [], "identity_B_psnr": [], "identity_B_ssim": [],
    }
    cls_correct = 0
    cls_total = 0

    with torch.no_grad():
        # A -> B -> A and A -> A
        for p in a_images:
            x = load_image(p, device)
            lbl_a = torch.tensor([0], dtype=torch.long, device=device)
            lbl_b = torch.tensor([1], dtype=torch.long, device=device)

            fake_b = G(x, lbl_b)
            rec_a = G(fake_b, lbl_a)
            idt_a = G(x, lbl_a)

            if not torch.isfinite(fake_b).all() or not torch.isfinite(rec_a).all() or not torch.isfinite(idt_a).all():
                continue

            metrics["cycle_A_l1"].append(float(mae(rec_a, x).item()))
            metrics["cycle_A_psnr"].append(float(psnr(rec_a, x).item()))
            metrics["cycle_A_ssim"].append(float(ssim(rec_a, x).item()))

            metrics["identity_A_l1"].append(float(mae(idt_a, x).item()))
            metrics["identity_A_psnr"].append(float(psnr(idt_a, x).item()))
            metrics["identity_A_ssim"].append(float(ssim(idt_a, x).item()))

            if C is not None:
                pred = C(fake_b).argmax(dim=1)
                cls_correct += int((pred == lbl_b).sum().item())
                cls_total += 1

        # B -> A -> B and B -> B
        for p in b_images:
            x = load_image(p, device)
            lbl_a = torch.tensor([0], dtype=torch.long, device=device)
            lbl_b = torch.tensor([1], dtype=torch.long, device=device)

            fake_a = G(x, lbl_a)
            rec_b = G(fake_a, lbl_b)
            idt_b = G(x, lbl_b)

            if not torch.isfinite(fake_a).all() or not torch.isfinite(rec_b).all() or not torch.isfinite(idt_b).all():
                continue

            metrics["cycle_B_l1"].append(float(mae(rec_b, x).item()))
            metrics["cycle_B_psnr"].append(float(psnr(rec_b, x).item()))
            metrics["cycle_B_ssim"].append(float(ssim(rec_b, x).item()))

            metrics["identity_B_l1"].append(float(mae(idt_b, x).item()))
            metrics["identity_B_psnr"].append(float(psnr(idt_b, x).item()))
            metrics["identity_B_ssim"].append(float(ssim(idt_b, x).item()))

            if C is not None:
                pred = C(fake_a).argmax(dim=1)
                cls_correct += int((pred == lbl_a).sum().item())
                cls_total += 1

    summary = {
        "checkpoint": checkpoint_path,
        "domain_a_dir": domain_a_dir,
        "domain_b_dir": domain_b_dir,
        "num_eval_A": len(metrics["cycle_A_l1"]),
        "num_eval_B": len(metrics["cycle_B_l1"]),
        "cycle_A": {
            "l1": mean_std(metrics["cycle_A_l1"]),
            "psnr": mean_std(metrics["cycle_A_psnr"]),
            "ssim": mean_std(metrics["cycle_A_ssim"]),
        },
        "identity_A": {
            "l1": mean_std(metrics["identity_A_l1"]),
            "psnr": mean_std(metrics["identity_A_psnr"]),
            "ssim": mean_std(metrics["identity_A_ssim"]),
        },
        "cycle_B": {
            "l1": mean_std(metrics["cycle_B_l1"]),
            "psnr": mean_std(metrics["cycle_B_psnr"]),
            "ssim": mean_std(metrics["cycle_B_ssim"]),
        },
        "identity_B": {
            "l1": mean_std(metrics["identity_B_l1"]),
            "psnr": mean_std(metrics["identity_B_psnr"]),
            "ssim": mean_std(metrics["identity_B_ssim"]),
        },
        "classifier_translation_accuracy": (
            (cls_correct / cls_total) if cls_total > 0 else None
        ),
    }

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Conditional CycleGAN checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to training checkpoint (.pt)")
    p.add_argument("--domain_a_dir", type=str, default=None,
                   help="Override domain A folder path")
    p.add_argument("--domain_b_dir", type=str, default=None,
                   help="Override domain B folder path")
    p.add_argument("--max_images", type=int, default=200,
                   help="Randomly sample up to N images from each domain (<=0 means all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save full metrics JSON")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU evaluation")
    return p.parse_args()


def print_summary(summary: Dict[str, Any]) -> None:
    def _fmt(block: Dict[str, Dict[str, float]], key: str) -> str:
        m = block[key]["mean"]
        s = block[key]["std"]
        return f"{m:.4f} +/- {s:.4f}"

    print("\n[eval] ===== Summary =====")
    print(f"[eval] A samples: {summary['num_eval_A']}, B samples: {summary['num_eval_B']}")

    print("[eval] cycle A->B->A:")
    print(f"  L1   : {_fmt(summary['cycle_A'], 'l1')}")
    print(f"  PSNR : {_fmt(summary['cycle_A'], 'psnr')}")
    print(f"  SSIM : {_fmt(summary['cycle_A'], 'ssim')}")

    print("[eval] identity A->A:")
    print(f"  L1   : {_fmt(summary['identity_A'], 'l1')}")
    print(f"  PSNR : {_fmt(summary['identity_A'], 'psnr')}")
    print(f"  SSIM : {_fmt(summary['identity_A'], 'ssim')}")

    print("[eval] cycle B->A->B:")
    print(f"  L1   : {_fmt(summary['cycle_B'], 'l1')}")
    print(f"  PSNR : {_fmt(summary['cycle_B'], 'psnr')}")
    print(f"  SSIM : {_fmt(summary['cycle_B'], 'ssim')}")

    print("[eval] identity B->B:")
    print(f"  L1   : {_fmt(summary['identity_B'], 'l1')}")
    print(f"  PSNR : {_fmt(summary['identity_B'], 'psnr')}")
    print(f"  SSIM : {_fmt(summary['identity_B'], 'ssim')}")

    acc = summary["classifier_translation_accuracy"]
    if acc is None:
        print("[eval] classifier translation accuracy: n/a (classifier not available)")
    else:
        print(f"[eval] classifier translation accuracy: {acc:.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    domain_a_dir, domain_b_dir = resolve_domains(args, cfg)

    summary = evaluate(
        checkpoint_path=args.checkpoint,
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        max_images=args.max_images,
        seed=args.seed,
        device=device,
    )
    print_summary(summary)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
