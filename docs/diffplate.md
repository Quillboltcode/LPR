# DiffPlate: Diffusion Model for License Plate Super-Resolution

## Overview

DiffPlate is a conditional diffusion model designed for super-resolution of license plate images. It uses a U-Net architecture with time embeddings to progressively denoise images conditioned on low-resolution inputs.

## Architecture

### Model Components

| Component | Description |
|-----------|-------------|
| `SinusoidalPositionEmbeddings` | Encodes timestep `t` into a vector using sine/cosine positional encoding |
| `Block` | Residual block with GroupNorm, SiLU activation, and time embedding injection |
| `DiffPlateUNet` | Conditional U-Net backbone for noise prediction |
| `DiffPlate` | Main wrapper managing noise schedule, training, and sampling |

### DiffPlateUNet Architecture

```
Input: [x_t (3ch) + lr_condition (3ch)] = 6 channels
      ↓
    Conv0 (6 → 64)
      ↓
    Encoder Path (downsampling):
    Block(64 → 128)  →  Block(128 → 256)  →  Block(256 → 512)  →  Block(512 → 1024)
      ↓
    Decoder Path (upsampling with skip connections):
    Block(1024 → 512)  →  Block(512 → 256)  →  Block(256 → 128)  →  Block(128 → 64)
      ↓
    Output Conv (64 → 3)
      ↓
Output: Predicted noise (3 channels)
```

### Channel Configuration

| Stage | Channels |
|-------|----------|
| Input | 6 (3 noisy + 3 condition) |
| Down 1 | 64 → 128 |
| Down 2 | 128 → 256 |
| Down 3 | 256 → 512 |
| Down 4 | 512 → 1024 |
| Up 1 | 1024 → 512 |
| Up 2 | 512 → 256 |
| Up 3 | 256 → 128 |
| Up 4 | 128 → 64 |
| Output | 3 |

**Total Parameters**: ~62.4M

## Image Dimensions

Based on EDA analysis of the dataset:

| Type | Height | Width | Aspect Ratio |
|------|--------|-------|--------------|
| LR Input | 32 | 64 | 2.0 |
| HR Output | 64 | 128 | 2.0 |

## Usage

### Training

```python
from src.diffplate import DiffPlateUNet, DiffPlate
import torch

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
unet = DiffPlateUNet(image_height=64, image_width=128)
diff_plate = DiffPlate(unet, image_height=64, image_width=128, device=device)

# Training step
hr_imgs = torch.randn(batch_size, 3, 64, 128).to(device)  # Ground truth
lr_imgs = torch.randn(batch_size, 3, 32, 64).to(device)   # Low-res input

loss = diff_plate(hr_imgs, lr_imgs)
loss.backward()
```

### Inference (Super-Resolution)

```python
# Generate high-res from low-res
sr_imgs = diff_plate.super_resolve(lr_imgs)
# Output shape: [batch_size, 3, 64, 128]
```

### Command Line Training

```bash
# Full training
uv run python train_diffplate.py --data_dir data/train --epochs 100 --batch_size 8

# Quick test with fewer diffusion steps
uv run python train_diffplate.py --data_dir data/train --epochs 1 --batch_size 8 --noise_steps 100
```

## Training Configuration

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Beta (Adam) | (0.9, 0.999) |
| Noise Steps | 1000 |
| Batch Size | 8 |
| Epochs | 100 |
| Early Stopping Patience | 10 |
| Gradient Clipping | 1.0 |

### Noise Schedule

- **Type**: Linear schedule
- **β range**: 1e-4 to 0.02
- **α**: 1 - β
- **α̂ (alpha_hat)**: Cumulative product of α

### Loss Function

Mean Squared Error (MSE) between predicted noise and actual noise:

```python
loss = F.mse_loss(predicted_noise, actual_noise)
```

## Sampling Process (DDPM)

The denoising process follows the standard DDPM formulation:

```
x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-α̂_t) * ε_θ(x_t, t)) + √β_t * z
```

Where:
- `ε_θ(x_t, t)` is the predicted noise from the U-Net
- `z ~ N(0, I)` is random noise (except at t=1 where z=0)

## Data Preprocessing

### Normalization

Dataset-specific normalization (from EDA):

```python
transforms.Normalize(
    mean=[0.251, 0.251, 0.251],
    std=[0.324, 0.323, 0.319]
)
```

### Image Transforms

```python
# Low-resolution input
transform_lr = transforms.Compose([
    transforms.Resize((32, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.251, 0.251, 0.251], std=[0.324, 0.323, 0.319])
])

# High-resolution target
transform_hr = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.251, 0.251, 0.251], std=[0.324, 0.323, 0.319])
])
```

## File Structure

```
src/
├── diffplate.py          # Model definitions
│   ├── SinusoidalPositionEmbeddings
│   ├── Block
│   ├── DiffPlateUNet
│   └── DiffPlate
train_diffplate.py        # Training script
weights/
├── diffplate_best.pth    # Best model checkpoint
└── diffplate_final.pth   # Final model checkpoint
```

## Checkpoint Format

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
}
```

## Performance Considerations

1. **Memory**: Model has 62.4M parameters - requires ~2GB GPU memory for inference
2. **Speed**: Full DDPM sampling (1000 steps) is slow - consider:
   - Fewer noise steps (100-200)
   - DDIM sampler for faster inference
   - Reduced channel dimensions
3. **Batch Size**: 8 is recommended for 16GB GPU

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Image Super-Resolution via Iterative Refinement (SR3)](https://arxiv.org/abs/2104.07636)
