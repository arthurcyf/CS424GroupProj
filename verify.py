# =============================================================================
# verify.py — Environment + model smoke-tests (upgraded architecture)
#
# Run with:
#   E:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe verify.py
# =============================================================================

import sys
import os

# Ensure this project's directory is first on sys.path so local modules
# (models, dataset, losses, utils) are found before any same-named packages
# that may be installed in the Python environment (e.g. ComfyUI packages).
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

print(f"Python: {sys.version}\n")

# ── Library versions ──────────────────────────────────────────────────────────
import torch
import torchvision
print(f"PyTorch:        {torch.__version__}")
print(f"Torchvision:    {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM (total):   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Import project modules ────────────────────────────────────────────────────
print("\n── Importing project modules ───────────────────────────────────────")
from models  import Generator, MultiScaleDiscriminator, Classifier
from dataset import JerseyDataset, ImageBuffer, sample_target_labels
from losses  import (LSGanLoss, CycleLoss, IdentityLoss,
                     ClassificationLoss, PerceptualLoss)
from utils   import verify_param_count, LossLogger
print("All imports OK.")

# ── Parameter counts ─────────────────────────────────────────────────────────
print("\n── Parameter counts ────────────────────────────────────────────────")
num_teams = 4
G   = Generator(num_teams, embed_dim=512, num_blocks=12, pretrained=False)
D_A = MultiScaleDiscriminator(num_teams, img_size=256, num_scales=2)
D_B = MultiScaleDiscriminator(num_teams, img_size=256, num_scales=2)
C   = Classifier(num_teams)
verify_param_count(G, D_A, D_B, C)
print("PASS: parameter count reported (no limit)")

# ── Forward pass (CPU, pretrained=False for speed) ────────────────────────────
print("\n── Forward pass smoke-test (CPU) ───────────────────────────────────")
B       = 2
x       = torch.randn(B, 3, 256, 256)
lbl_src = torch.tensor([0, 1], dtype=torch.long)
lbl_tgt = torch.tensor([2, 3], dtype=torch.long)

G.eval()
with torch.no_grad():
    fake_B = G(x, lbl_tgt)
    rec_A  = G(fake_B, lbl_src)
    idt_A  = G(x, lbl_src)

d_out = D_A(x, lbl_src)   # returns List[Tensor]
c_out = C(x)

assert fake_B.shape == (B, 3, 256, 256), f"G output shape wrong: {fake_B.shape}"
assert rec_A.shape  == (B, 3, 256, 256), f"rec_A shape wrong: {rec_A.shape}"
assert isinstance(d_out, list),           "D output should be a list"
assert d_out[0].shape[1] == 1,           f"D scale-0 channels wrong: {d_out[0].shape}"
assert d_out[1].shape[1] == 1,           f"D scale-1 channels wrong: {d_out[1].shape}"
assert c_out.shape  == (B, num_teams),    f"C output shape wrong: {c_out.shape}"
assert fake_B.min() >= -1-1e-3 and fake_B.max() <= 1+1e-3, \
    f"G output out of [-1,1]: [{fake_B.min():.3f}, {fake_B.max():.3f}]"

print(f"  G(x, tgt)       → {fake_B.shape}  range [{fake_B.min():.3f}, {fake_B.max():.3f}]")
print(f"  G(fake, src)    → {rec_A.shape}  (cycle)")
print(f"  D_A(x, src)     → List[{d_out[0].shape}, {d_out[1].shape}]  (2 scales)")
print(f"  C(x)            → {c_out.shape}")
print("PASS: all forward passes correct")

# ── Multi-scale discriminator spatial sizes ───────────────────────────────────
print("\n── Multi-scale discriminator output sizes ──────────────────────────")
print(f"  Scale 0 (256×256 input): patch map {d_out[0].shape}")
print(f"  Scale 1 (128×128 input): patch map {d_out[1].shape}")
# Scale 1 should have smaller spatial dims due to avg-pooling
assert d_out[1].shape[-1] < d_out[0].shape[-1], "Scale 1 should have smaller spatial dims"
print("PASS: multi-scale spatial sizes correct")

# ── Loss functions ────────────────────────────────────────────────────────────
print("\n── Loss function smoke-test ────────────────────────────────────────")
G.train()
gan_loss  = LSGanLoss()
cyc_loss  = CycleLoss(lambda_cycle=10.0)
idt_loss  = IdentityLoss(lambda_identity=5.0)
cls_loss  = ClassificationLoss(lambda_cls=2.0)
perc_loss = PerceptualLoss(lambda_perceptual=1.0)

# Generator losses
l_gan_B = gan_loss.generator_loss(D_B(fake_B, lbl_tgt))
l_gan_A = gan_loss.generator_loss(D_A(rec_A,  lbl_src))
l_cyc   = cyc_loss(rec_A, x)
l_idt   = idt_loss(idt_A, x)
l_cls   = cls_loss(c_out, lbl_src)
l_perc  = perc_loss(fake_B, x)

# Discriminator losses (list → averaged internally)
pred_real_A = D_A(x,      lbl_src)
pred_fake_A = D_A(rec_A,  lbl_src)
loss_D_A    = gan_loss.discriminator_loss(pred_real_A, pred_fake_A)

total_G = l_gan_A + l_gan_B + l_cyc + l_idt + l_cls + l_perc

print(f"  GAN-B generator loss:    {l_gan_B.item():.4f}")
print(f"  GAN-A generator loss:    {l_gan_A.item():.4f}")
print(f"  Cycle loss:              {l_cyc.item():.4f}")
print(f"  Identity loss:           {l_idt.item():.4f}")
print(f"  Classification loss:     {l_cls.item():.4f}")
print(f"  Perceptual loss:         {l_perc.item():.4f}")
print(f"  D_A discriminator loss:  {loss_D_A.item():.4f}")
print(f"  Total G loss:            {total_G.item():.4f}")
print("PASS: all losses computed correctly")

# ── AdaIN identity initialisation ────────────────────────────────────────────
print("\n── AdaIN identity initialisation check ─────────────────────────────")
label0 = torch.tensor([0], dtype=torch.long)
pairs  = G.label_embedding(label0)
for i, (g, b) in enumerate(pairs):
    assert abs(g.mean().item() - 1.0) < 0.05, \
        f"Block {i} gamma mean not ~1.0: {g.mean().item():.4f}"
    assert abs(b.mean().item() - 0.0) < 0.05, \
        f"Block {i} beta mean not ~0.0: {b.mean().item():.4f}"
print(f"  All {len(pairs)} blocks: gamma ≈ 1.0, beta ≈ 0.0")
print("PASS: AdaIN identity initialisation correct")

# ── Encoder train() keeps backbone BN in eval ────────────────────────────────
print("\n── Encoder BN eval-mode override check ─────────────────────────────")
G.train()
# The encoder's features (ResNet backbone) should be in eval mode
backbone_training = any(m.training for m in G.encoder.features.modules()
                        if isinstance(m, torch.nn.BatchNorm2d))
assert not backbone_training, "Backbone BN should be in eval mode during G.train()"
print("PASS: backbone BatchNorm stays in eval mode during training")

# ── ImageBuffer smoke-test ────────────────────────────────────────────────────
print("\n── ImageBuffer smoke-test ──────────────────────────────────────────")
buf  = ImageBuffer(max_size=5)
imgs = torch.randn(3, 3, 256, 256)
out  = buf.push_and_pop(imgs)
assert out.shape == imgs.shape
print(f"  push_and_pop({imgs.shape}) → {out.shape}")
print("PASS: ImageBuffer working")

# ── sample_target_labels ──────────────────────────────────────────────────────
print("\n── sample_target_labels check ──────────────────────────────────────")
src = torch.tensor([0, 0, 1, 2], dtype=torch.long)
tgt = sample_target_labels(src, num_domains=4)
for s, t in zip(src.tolist(), tgt.tolist()):
    assert s != t, f"Target same as source: {s} == {t}"
print(f"  src={src.tolist()} → tgt={tgt.tolist()}")
print("PASS: target labels always differ from source")

# ── AMP training step on CUDA ────────────────────────────────────────────────
if torch.cuda.is_available():
    print("\n── AMP training step (CUDA) ────────────────────────────────────────")
    dev = torch.device('cuda')

    # Reload with pretrained encoder for a realistic test
    G_gpu   = Generator(num_teams, embed_dim=512, num_blocks=12, pretrained=True).to(dev)
    D_B_gpu = MultiScaleDiscriminator(num_teams).to(dev)
    x_gpu   = torch.randn(2, 3, 256, 256, device=dev)
    lbl_gpu = torch.tensor([2, 3], dtype=torch.long, device=dev)
    src_gpu = torch.tensor([0, 1], dtype=torch.long, device=dev)

    scaler = torch.amp.GradScaler('cuda')
    opt    = torch.optim.Adam(
        [p for p in G_gpu.parameters() if p.requires_grad], lr=2e-4
    )
    opt.zero_grad()

    G_gpu.train()
    with torch.autocast('cuda', dtype=torch.float16):
        fake_gpu = G_gpu(x_gpu, lbl_gpu)
        l_gan    = gan_loss.generator_loss(D_B_gpu(fake_gpu, lbl_gpu))
        l_cyc_   = cyc_loss(G_gpu(fake_gpu, src_gpu), x_gpu)

    perc_gpu = PerceptualLoss(1.0).to(dev)
    with torch.autocast('cuda', enabled=False):
        l_perc_gpu = perc_gpu(fake_gpu.float(), x_gpu.float())

    total = l_gan + l_cyc_ + l_perc_gpu
    scaler.scale(total).backward()
    scaler.step(opt)
    scaler.update()

    print(f"  AMP step OK — loss={total.item():.4f}")
    print(f"  fake_B range: [{fake_gpu.min().item():.3f}, {fake_gpu.max().item():.3f}]")
    print("PASS: AMP training step (CUDA) successful")
else:
    print("\n[skip] CUDA not available — skipping AMP test")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
