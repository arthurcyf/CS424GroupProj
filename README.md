# CS424GroupProj
Group Project for CS424

## Checkpoint Evaluation Metrics

Use `evaluate.py` to score a trained checkpoint with CycleGAN-friendly metrics:

- Cycle consistency: `A->B->A` and `B->A->B` using `L1`, `PSNR`, `SSIM`
- Identity preservation: `A->A` and `B->B` using `L1`, `PSNR`, `SSIM`
- Optional translation-domain classifier accuracy (if `C_state` exists in checkpoint)

Run:

```bash
python evaluate.py --checkpoint checkpoints/epoch_040.pt --max_images 200
```

Optional JSON output:

```bash
python evaluate.py --checkpoint checkpoints/epoch_040.pt --max_images 200 --output_json logs/eval_epoch_040.json
```

Interpretation guide:

- Lower is better: `L1`
- Higher is better: `PSNR`, `SSIM`, classifier translation accuracy

## Annotation-Free Segmentation Pipeline

The notebook `segmentation_pipeline.ipynb` implements face + jersey segmentation **without using** `dataset/_annotations.coco.json`.

### What it does
- Reads all images from `dataset/`
- Runs pretrained inference for person + face
- Derives jersey region from person mask (upper-body heuristic)
- Exports `256x256` outputs for CycleGAN under `outputs/`

### Install (recommended)
```bash
pip install numpy pillow opencv-python matplotlib ultralytics mediapipe
```

Notes:
- If `ultralytics` is unavailable, person segmentation will not run.
- If `mediapipe` is unavailable, face detection falls back to OpenCV Haar cascade.

### Run
1. Open `segmentation_pipeline.ipynb`
2. Run cells top-to-bottom
3. Check generated files in:
	- `outputs/masked_rgb/`
	- `outputs/face_mask/`
	- `outputs/jersey_mask/`
	- `outputs/combined_mask/`

### Adjust settings
Edit the `PipelineConfig` cell in the notebook (for example `max_images`, `image_size`, thresholds, and output paths).
