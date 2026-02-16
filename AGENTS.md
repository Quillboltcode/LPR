# ICPR 2026 License Plate Recognition Project

## Project Overview

This project aims to predict license plate text from low-resolution (LR) images using a two-phase training approach. The training data contains both low-resolution (lr-*.png) and high-resolution (hr-*.png) images for each track.

### Problem Statement
- Input: Low-resolution license plate images (5 frames per track)
- Output: Predicted 7-character alphanumeric text (A-Z, 0-9)

### Data Structure

```
data/
├── train/
│   ├── Scenario-A/
│   │   ├── Brazilian/         # Brazilian license plates
│   │   │   └── track_XXXXX/
│   │   │       ├── annotations.json   # Contains plate_text, plate_layout, corners
│   │   │       ├── lr-001.png ... lr-005.png   # Low-resolution frames
│   │   │       └── hr-001.png ... hr-005.png   # High-resolution frames
│   │   └── Mercosur/          # Mercosur license plates
│   │       └── track_XXXXX/
│   └── Scenario-B/
│       └── (same structure as Scenario-A)
└── test/
    └── track_XXXXX/           # Test tracks (LR images only)
        └── lr-001.png ... lr-005.png
```

### Annotations Format (annotations.json)

```json
{
  "plate_layout": "Brazilian",
  "plate_text": "AVL5215",
  "corners": {
    "lr-001.png": {
      "top-left": [x, y],
      "top-right": [x, y],
      "bottom-right": [x, y],
      "bottom-left": [x, y]
    },
    ...
  }
}
```

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training script with two-phase training (HR pre-training → LR fine-tuning) |
| `train_fsrcnn.py` | FSRCNN super-resolution model training |
| `submit.py` | Generates submission.csv from test data |
| `src/model.py` | TemporalCRNN model (ResNet backbone + LSTM + CTC) |
| `src/dataset.py` | LicensePlateDataset for loading tracks |
| `src/FSRCNN.py` | FSRCNN super-resolution model with CBAM attention |
| `src/metric.py` | Recognition rate metrics |

## Commands

### Training

```bash
# Full training (Phase 1: HR pre-training, Phase 2: LR fine-tuning)
python train.py --data_dir data/train --hr_epochs 10 --lr_epochs 20 --batch_size 16

# Skip Phase 1 and use existing HR weights
python train.py --data_dir data/train --skip_phase1 --phase1_weights weights/hr_pretrained_weights.pth --lr_epochs 20

# Use FSRCNN super-resolution preprocessing
python train.py --data_dir data/train --use_sr --fsrcnn_weights weights/fsrcnn_best.pth --lr_epochs 20

# Train FSRCNN super-resolution model first
python train_fsrcnn.py --data_dir data/train --epochs 50
```

### Inference/Submission

```bash
# Generate submission.csv from test data
python submit.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_helper.py -v
```

### Linting & Type Checking

```bash
# Type check (if mypy available)
python -m mypy . --ignore-missing-imports

# Lint (if ruff available)
ruff check .
```

## Model Architecture

### TemporalCRNN

1. **Backbone**: ResNet-18 (pretrained) - extracts features from each frame
2. **CBAM Attention**: Focuses on important spatial regions
3. **Temporal Fusion**: Averages features across 5 frames
4. **LSTM**: Bidirectional 2-layer LSTM for sequence modeling
5. **CTC Decoder**: Beam search decoding for final text prediction

### Character Set
- 36 characters: A-Z (26) + 0-9 (10)
- Target length: 7 characters
- CTC blank token at index 0

## Training Strategy

### Phase 1: HR Pre-training
- Train on high-resolution images to learn character recognition
- Uses `is_lr=False` in dataset
- Default: 10 epochs, learning rate 1e-4

### Phase 2: LR Fine-tuning
- Fine-tune on low-resolution images with augmentation
- Applies Gaussian blur and noise to simulate LR conditions
- Can optionally use FSRCNN for super-resolution preprocessing
- Default: 20 epochs, learning rate 1e-4

### Data Augmentation (LR Training)
- Gaussian blur (radius 0.5-1.5)
- Gaussian noise (std=0.1)
- Standard ImageNet normalization

## Image Specifications

| Property | LR Image | HR Image | Model Input |
|----------|----------|----------|-------------|
| Width | ~30-35px | ~50-60px | 192px |
| Height | ~15-18px | ~25-30px | 64px |
| Frames per track | 5 | 5 | 5 |

## Output Format

Submission CSV format:
```csv
track_id,predicted_text,confidence
track_10005,ABC1234,0.9500
track_10015,XYZ5678,0.8700
...
```

## Dependencies

- Python 3.12+
- PyTorch 2.0+
- TorchVision
- torchaudio (for beam search decoder)
- PIL/Pillow
- NumPy
- OpenCV

Install with uv:
```bash
uv sync
```

## Weights Directory

Model weights are saved to `weights/`:
- `hr_pretrained_weights.pth` - Phase 1 HR pre-trained weights
- `final_lr_finetuned_weights.pth` - Final LR fine-tuned weights
- `fsrcnn_best.pth` - FSRCNN super-resolution weights

## Important Notes

1. **Character encoding**: Characters are 1-indexed (0 reserved for CTC blank)
2. **Early stopping**: Patience of 5 epochs for both phases
3. **Gradient clipping**: Max norm 5.0 to prevent exploding gradients
4. **Validation split**: 20% of training data held out for validation
5. **Random seed**: 42 for reproducible train/val splits