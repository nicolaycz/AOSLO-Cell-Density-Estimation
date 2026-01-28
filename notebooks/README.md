# Training Notebooks

This directory contains Jupyter notebooks for training and evaluating the cone density estimation models.

## Notebooks Overview

| Notebook | Model | Description | Recommended |
|----------|-------|-------------|-------------|
| `Baseline_Model.ipynb` | Baseline | CoDE reference implementation (2.06M params) | Reference only |
| `Model_A_Lightweight.ipynb` | Model A | Lightweight U-Net - best accuracy | Yes |
| `Model_B_LinearRegression_with_Metrics.ipynb` | Model B | U-Net + Linear correction | Yes |
| `Model_C_GlobalSum.ipynb` | Model C | U-Net + Global-Sum (unstable) | No |
| `Model_D_DirectRegression.ipynb` | Model D | Direct count regression | Yes |

## Quick Start

### 1. Activate Environment

```bash
cd /path/to/AOSLO-Cell-Density-Estimation
source .venv/bin/activate  # or create new venv
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Update Data Paths

The notebooks were originally developed in Google Colab. Update the data paths for local execution:

```python
# Original (Colab):
train_path = '/content/drive/MyDrive/.../Training+Density/'

# Local:
train_path = '../data/Training+Density/'
val_path = '../data/Validation+Density/'
```

## Model Descriptions

### Baseline Model
- **Architecture**: Xception-style residual encoder-decoder
- **Parameters**: 2.06M
- **Training**: 200 epochs with extensive augmentation
- **Purpose**: Reference benchmark for comparison

### Model A - Lightweight U-Net
- **Architecture**: 3-level U-Net encoder-decoder
- **Parameters**: 538K (74% reduction)
- **Output**: Density map (256x256x1)
- **Loss**: Pixel-wise MSE
- **Result**: MAE = 921.62 cones (40% better than baseline)

### Model B - U-Net + Linear Correction
- **Architecture**: Same as Model A + post-hoc linear regression
- **Two-stage training**:
  1. Train U-Net on density maps
  2. Fit linear correction: `corrected = slope * pred + intercept`
- **Result**: MAE = 977.05 cones

### Model C - Global-Sum Regression
- **Architecture**: U-Net with embedded global sum layer
- **Output**: Scalar count (end-to-end)
- **Loss**: Count-level MSE
- **Note**: Showed training instability (not recommended)

### Model D - Direct Regression
- **Architecture**: Encoder-only + GlobalAveragePooling + Dense
- **Parameters**: 380K (most lightweight)
- **Output**: Scalar count directly
- **Result**: MAE = 1043.95 cones

## Training Configuration

All models use consistent configuration:

```python
# Optimizer
optimizer = Adam(learning_rate=1e-3)

# Callbacks
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6)

# Training
epochs = 100
batch_size = 8
```

## Files to Ignore

The following notebooks are duplicates or intermediate versions:
- `Model_A_Lightweight_Fixed.ipynb` - Use `Model_A_Lightweight.ipynb` instead
- `Model_B_LinearRegression.ipynb` - Use `Model_B_LinearRegression_with_Metrics.ipynb` instead

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: GPU with 4GB+ VRAM (NVIDIA T4 or better)
- **Training time**: ~30 min per model on GPU T4
