# Deepfake DDPM Face Generation

Denoising Diffusion Probabilistic Model for state-of-the-art face synthesis. Part of MPhil research on hybrid multi-modal deepfake detection.

## Architecture
- **U-Net** with time-step sinusoidal embeddings
- Residual blocks with adaptive GroupNorm (conditioned on t)
- Self-attention at 32x32 and 16x16 spatial resolutions
- EMA of model weights for smooth sampling
- Supports both DDPM (1000 steps) and DDIM (50-250 steps, ~10x faster)

## Diffusion Schedule
- Linear beta schedule: beta_1=1e-4 to beta_T=0.02, T=1000
- Simple epsilon-prediction loss (Ho et al. 2020)
- DDIM sampler for fast inference (Song et al. 2021)

## Training
```bash
# Standard training
python train.py --dataset celeba --data ./data --epochs 100 --batch_size 8
# With mixed precision (recommended for GPU)
python train.py --dataset celeba --data ./data --epochs 100 --batch_size 16 --fp16
# FFHQ 70K
python train.py --dataset ffhq --data ./data/ffhq --epochs 200 --batch_size 8 --fp16
```

## Generation
```bash
# Fast DDIM sampling (50 steps)
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --sampler ddim --steps 50 --n 16
# High quality DDPM (1000 steps, slow)
python generate.py --ckpt outputs/ckpt_epoch_0100.pt --sampler ddpm --n 4
```

## Diffusion Artifacts for Detection Research
- Pixel value distribution shift (Gaussian artifacts)
- Spectral signature from iterative denoising
- Inconsistent noise residuals across frequency bands
- Subtle grid-like patterns from U-Net architecture
- These are distinct from GAN/VAE artifacts — key for multi-modal detector

## Datasets
| Dataset | Resolution | Images | Notes |
|---------|-----------|--------|-------|
| CelebA | 178x218 | 202K | torchvision download |
| FFHQ | 1024x1024 | 70K | HuggingFace download |
| FaceForensics++ | varies | 1K videos | FF++ benchmark |

---
MPhil Research | [Dnshitobu](https://github.com/Dnshitobu)
