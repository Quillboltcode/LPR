# Image Size EDA Report

## Dataset Overview

- **Total tracks analyzed**: 20,000
- **Total images**: 100,000 (50,000 LR + 50,000 HR)

## Image Size Statistics

### Low Resolution (LR) Images

| Metric | Width | Height |
|--------|-------|--------|
| Min | 24 | 13 |
| Max | 68 | 36 |
| Mean | 42.5 | 19.9 |
| Std | 7.7 | 1.9 |
| Median | 44.0 | 20.0 |

- **Aspect Ratio (W/H)**: 2.14
- **Total images**: 50,000
- **Sample sizes**: (49, 23), (26, 21), (38, 23), (63, 25), (30, 18), etc.

### High Resolution (HR) Images

| Metric | Width | Height |
|--------|-------|--------|
| Min | 50 | 148 |
| Max | 22 | 60 |
| Mean | 83.2 | 35.9 |
| Std | 18.9 | 5.5 |
| Median | 81.0 | 35.0 |

- **Aspect Ratio (W/H)**: 2.32
- **Total images**: 50,000
- **Sample sizes**: (98, 37), (121, 39), (90, 33), (79, 33), etc.

### Resolution Comparison

| Metric | Value |
|--------|-------|
| HR/LR Width Scale | 1.96x |
| HR/LR Height Scale | 1.81x |
| Average HR/LR Scale | ~2x |

## Pixel Value Statistics (Normalized 0-1)

### Low Resolution (LR)

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.3922 | 0.3168 |
| G | 0.3922 | 0.3151 |
| B | 0.3922 | 0.3081 |

- **Global Mean**: 0.4623
- **Global Std**: 0.3309

### High Resolution (HR)

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.1093 | 0.3306 |
| G | 0.1093 | 0.3306 |
| B | 0.1093 | 0.3306 |

- **Global Mean**: 0.5398
- **Global Std**: 0.3261

### Combined Dataset Statistics

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.2507 | 0.3237 |
| G | 0.2507 | 0.3228 |
| B | 0.2507 | 0.3193 |

## Recommendations

### Training Image Size

| Target Height | Recommended Width | Aspect Ratio |
|---------------|-------------------|--------------|
| 64 | 128 | 2.0 |
| 128 | 288 | 2.25 |

**Current Config**: (64, 192) with aspect ratio 3.0

**Issue**: Current aspect ratio (3.0) doesn't match actual data (~2.14). This causes horizontal stretching during resize.

**Recommended**: Use (64, 128) for better aspect ratio match.

### Normalization Comparison

| Source | Mean (R,G,B) | Std (R,G,B) |
|--------|--------------|-------------|
| **Dataset** | [0.251, 0.251, 0.251] | [0.324, 0.323, 0.319] |
| ImageNet (current) | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |

**Findings**:
- Dataset images are darker (mean ~0.25 vs ImageNet ~0.46)
- Dataset has higher std (~0.32 vs ImageNet ~0.23)
- Dataset is approximately grayscale (R≈G≈B)

### Suggested Configuration Update

```python
# In src/dataset.py
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128  # Changed from 192

# Dataset-specific normalization
transforms.Normalize(
    mean=[0.251, 0.251, 0.251],
    std=[0.324, 0.323, 0.319]
)
```

## Analysis Script

Run EDA analysis with:
```bash
uv run python eda_image_analysis.py --data_dir data/train
```
