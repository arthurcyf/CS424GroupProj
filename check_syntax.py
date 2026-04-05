"""Quick syntax + import check — run this first before verify.py."""
import ast, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

files = ['models.py', 'dataset.py', 'losses.py', 'utils.py', 'train.py', 'infer.py', 'evaluate.py', 'verify.py']
all_ok = True
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f"  SYNTAX OK  {f}")
    except SyntaxError as e:
        print(f"  SYNTAX ERR {f}: {e}")
        all_ok = False

if not all_ok:
    sys.exit(1)

print("\nAll files pass syntax check.\n")

# Try imports (requires torch + torchvision to be installed)
try:
    from models  import Generator, MultiScaleDiscriminator, Classifier
    from dataset import JerseyDataset, ImageBuffer, sample_target_labels
    from losses  import LSGanLoss, CycleLoss, IdentityLoss, ClassificationLoss, PerceptualLoss
    from utils   import verify_param_count, LossLogger, save_sample_grid
    print("All project imports OK.")
    print("\nReady to run:  python verify.py")
except ImportError as e:
    print(f"Import error (check torch/torchvision installation): {e}")
    sys.exit(1)
