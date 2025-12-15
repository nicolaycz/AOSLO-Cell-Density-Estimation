# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for the thesis "Cell Density Estimation in AOSLO Images Using Image Processing and Deep Learning." The project implements four lightweight deep learning models (Models A-D) for automated cone photoreceptor density estimation in Adaptive Optics Scanning Laser Ophthalmoscopy (AOSLO) retinal images.

The goal is to provide efficient, accurate alternatives to complex state-of-the-art models like CoDE (2.06M parameters), demonstrating that simpler architectures can achieve comparable or better performance while being more clinically deployable.

## Environment Setup

**Python Version**: 3.10+ (currently using 3.12)

**Virtual Environment**:
```bash
# Activate the existing virtual environment
source .venv/bin/activate

# Key dependencies (already installed)
# - tensorflow 2.20.0
# - keras 3.12.0
# - numpy 2.3.5
# - scikit-learn 1.8.0
# - matplotlib 3.10.8
# - pandas 2.3.3
# - imageio
# - statsmodels
# - PIL (Pillow)
```

**No package.json, setup.py, or requirements.txt exists** - dependencies are managed manually in the virtual environment.

## Running the Models

All model implementations are in Jupyter notebooks located in `code/originals/`:

**Training/Evaluation** (run in Jupyter or Google Colab):
```bash
# Launch Jupyter
jupyter notebook

# Then open one of:
# - code/originals/Model_A_Lightweight.ipynb
# - code/originals/Model_B_LinearRegression_with_Metrics.ipynb
# - code/originals/Model_C_GlobalSum.ipynb
# - code/originals/Model_D_DirectRegression.ipynb
# - code/originals/Baseline_Model.ipynb
```

**Note**: The notebooks were originally developed in Google Colab and reference paths like `/content/drive/MyDrive/...`. When running locally, update the data paths:
```python
# Original (Colab):
train_path = '/content/drive/MyDrive/NCZ/10 Projects/MAA-Cone-Density-Estimation/data/machine1_5/Training+Density/'

# Local:
train_path = './data/Training+Density/'
```

## Architecture: Four Model Variants

### Model A - Lightweight U-Net
- **Architecture**: 3-level U-Net encoder-decoder with skip connections
- **Filters**: 16 → 32 → 64 → 128 (bottleneck)
- **Output**: Density map (256×256×1) with ReLU activation
- **Loss**: MSE on pixel-wise density maps
- **Parameters**: ~538K (74% reduction vs. baseline)
- **Key Innovation**: Minimal architecture while maintaining accuracy

### Model B - Lightweight U-Net + Linear Correction
- **Architecture**: Same as Model A + post-hoc linear regression
- **Two-stage approach**:
  1. Train U-Net to predict density maps
  2. Fit linear regression: `corrected_count = slope × pred_count + intercept`
- **Purpose**: Corrects systematic bias in total cone counts
- **Evaluation**: Compares metrics before/after linear correction

### Model C - U-Net + Global-Sum Regression
- **Architecture**: U-Net encoder-decoder → density map → `tf.reduce_sum()` → Dense(1)
- **Key Difference**: Global sum embedded within the network (differentiable)
- **Output**: Scalar cone count (not a density map)
- **Loss**: MSE on total counts
- **Innovation**: End-to-end training for count estimation

### Model D - Encoder + Global Average Pooling (Direct Regression)
- **Architecture**: Encoder only (no decoder) → GlobalAveragePooling2D → Dense(64) → Dense(1)
- **Output**: Scalar cone count directly from features
- **Most Lightweight**: ~380K parameters
- **Key Innovation**: Bypasses density map entirely for maximum efficiency

## Data Structure

```
data/
├── Training+Density/     # 184 images + density maps
│   ├── image001.tif
│   ├── image001_Density.tif
│   └── ...
└── Validation+Density/   # 80 images + density maps
    ├── image001.tif
    ├── image001_Density.tif
    └── ...
```

**Data Preprocessing**:
- Images: RGB TIF normalized to [0,1]
- Density maps: Grayscale TIF (float32)
- Filtering: Only images with `sum(density_map)/100 < 400` are used
- Augmentation: Random horizontal/vertical flips + 90° rotations

**Important Scale Convention**:
- Density maps store values scaled by 100
- Ground truth counts computed as: `np.sum(density_map) / 100`
- Models A/B work with raw density maps
- Models C/D predict counts directly (already divided by 100)

## Pre-trained Models

Located in `models/`:
- `Model_A.h5` - Lightweight U-Net (6.7 MB)
- `ModelC.h5` - Global-Sum U-Net (6.7 MB)
- `ModelD.h5` - Direct Regression Encoder (3.8 MB)

**Loading a model**:
```python
from tensorflow.keras import models
model = models.load_model('models/Model_A.h5')
```

## Training Configuration

All models use consistent training setup:

```python
# Optimizer
optimizer = Adam(learning_rate=1e-3)  # Model A, C, D
optimizer = Adam(learning_rate=1e-4)  # Model B (fine-tuning)

# Callbacks
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6)
ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Training
epochs = 100
batch_size = 8
```

## Evaluation Metrics

Models are evaluated on two levels:

**1. Pixel-level (Models A/B only)**:
- MAE, MSE, RMSE, R², Correlation on density map values

**2. Count-level (All models)**:
- MAE (Mean Absolute Error in cones)
- MSE, RMSE
- MAPE (Mean Absolute Percentage Error)
- R² score
- Bland-Altman plots for agreement analysis

**Critical Note**: When comparing models, focus on count-level metrics since that's the clinical endpoint.

## Paper and LaTeX

The repository includes the thesis paper in IEEE conference format:

```bash
# Paper source
paper/main.tex              # Main LaTeX source
paper/bibliography.bib      # References
paper/attachments/          # Figures

# Compiled PDF
paper/conference_101719.pdf

# Build (if LaTeX tools installed)
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Code Patterns

**Data Loading**:
```python
def read_data(base_path, density_threshold=400):
    # Pairs .tif images with *_Density.tif annotations
    # Returns: (images, density_maps) as numpy arrays
```

**TensorFlow Dataset Pipeline**:
```python
dataset = tf.data.Dataset.from_tensor_slices((images, targets))
dataset = dataset.map(augment)  # Only for training
dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)
```

**Common Import Block**:
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm  # For Bland-Altman plots
```

## Important Conventions

1. **Image Size**: All models expect 256×256×3 RGB inputs
2. **Density Map Size**: 256×256×1 (single channel)
3. **Activation**: ReLU on final conv layer (prevents negative densities)
4. **Batch Normalization**: Used after every Conv2D layer
5. **Dropout**: 0.5 in bottleneck only (for regularization)
6. **Random Seeds**: Models use fixed seeds for reproducibility (check individual notebooks)

## Licensing

- **Code**: Apache License 2.0
- **Data**: CC BY-NC 4.0 (non-commercial research use only)
- **Disclaimer**: Research code not intended for clinical diagnostics

## Research Context

This work addresses three limitations of existing AOSLO analysis methods:
1. **Computational Complexity**: Baseline models (e.g., CoDE) have 2.06M parameters
2. **Limited Accuracy**: Mean biases of -159 to +519 cones/mm² with wide confidence intervals
3. **Clinical Deployability**: Heavy preprocessing, lack of interpretability

The four proposed models demonstrate that 74% parameter reduction is possible while maintaining or improving accuracy.
