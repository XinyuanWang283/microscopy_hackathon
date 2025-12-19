# MoS2 SR3 Super-Resolution (Hackathon)

Minimal, end-to-end pipeline for conditional diffusion (SR3) on MoS2 STEM patches. Only SR3 is included; no unconditional DDPM.

## What’s here
- `MoS2_Nanowire/generate_ddpm_patches.py`: build 128×128 float TIFF patches from `MoS2_0510_1` with resampling + percentile norm.
- `sr3_training_data_128_resampled_nofilt/`: prepared training patches (resampled, p1/p99 ~ 2/98).
- `train_sr3.py`: SR3 trainer (blur→downsample→shot noise→Gaussian noise→upsample as LR condition).
- `MoS2_SR3_walkthrough.ipynb`: English notebook showing data prep, quick viz, training, and triplet visualization.
- `sr3_runs/`: checkpoints and sample grids go here (e.g., `mos2_sr3_fast_rerun`).


## Train (SR3, x2)
```bash
cd /export/scratch2/xinyuan/microscopy_hackathon
python train_sr3.py \
  --data-dir /export/scratch2/xinyuan/microscopy_hackathon/sr3_training_data_128_resampled_nofilt \
  --output-dir microscopy_hackathon/sr3_runs \
  --run-name mos2_sr3_fast_rerun \
  --epochs 50 \
  --target-train-size 5000 \
  --batch-size 64 \
  --learning-rate 2e-4 \
  --num-timesteps 400 \
  --lr-scale 2 \
  --blur-sigma 0.8 \
  --shot-noise-scale 50 \
  --gaussian-noise-std 0.01 \
  --gaussian-noise-std-max 0.03 \
  --sample-every 10 \
  --save-every 10
```
Outputs: `sr3_runs/mos2_sr3_fast_rerun/model_epoch_XXX.pt` and `samples/epoch_XXX.png` (LR | SR | HR).

## Visualize
- Use the notebook (`MoS2_SR3_walkthrough.ipynb`) cells 4–5 to preview patches and plot latest triplets in color (viridis).
- Or manually open `sr3_runs/.../samples/epoch_*.png`.

## Notes
- Training/val patches are all from `MoS2_0510_1`; other folders are excluded.
- Data range is [-1,1] inside the model; TIFFs are stored as float [0,1].
