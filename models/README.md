# Pre-trained Models

This directory contains pre-trained model weights for cone density estimation in AOSLO images.

## Available Models

| Model | File | Size | Parameters | MAE (cones) | Description |
|-------|------|------|------------|-------------|-------------|
| **Model A** | `Model_A.h5` | 6.4 MB | 538K | 921.62 | Lightweight U-Net (best accuracy) |
| **Model C** | `ModelC.h5` | 6.4 MB | ~540K | 3884.54 | U-Net + Global-Sum (unstable) |
| **Model D** | `ModelD.h5` | 3.6 MB | 380K | 1043.95 | Direct Regression (most efficient) |
| **Baseline** | `Baseline_Model.keras` | 16.0 MB | 2.06M | 1534.57 | CoDE reference implementation |

## Performance Comparison

| Model | MAE | RMSE | MAPE (%) | vs Baseline |
|-------|-----|------|----------|-------------|
| Baseline | 1534.57 | - | - | reference |
| **Model A** | **921.62** | 1255.19 | 13.64 | **-40%** |
| Model B | 977.05 | 1253.57 | 12.80 | -36% |
| Model C | 3884.54 | 4970.76 | 55.29 | +153% |
| Model D | 1043.95 | 1317.45 | 17.08 | -32% |

## Quick Start

### Loading a Model

```python
from tensorflow.keras.models import load_model

# Load Model A (recommended)
model = load_model('models/Model_A.h5')

# For .keras format (Baseline)
baseline = load_model('models/Baseline_Model.keras')
```

### Making Predictions

```python
import numpy as np
from PIL import Image

# Load and preprocess image
img = Image.open('path/to/aoslo_image.tif')
img = np.array(img) / 255.0  # Normalize to [0, 1]
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict density map (Models A, B, C)
density_map = model.predict(img)
cone_count = np.sum(density_map) / 100

# For Model D (direct count regression)
cone_count = model.predict(img)[0][0]

print(f"Estimated cone count: {cone_count:.0f}")
```

## Model Architectures

### Model A - Lightweight U-Net (Recommended)
- **Encoder**: 3 downsampling blocks (16 -> 32 -> 64 filters)
- **Bottleneck**: 128 filters + 0.5 Dropout
- **Decoder**: 3 upsampling blocks with skip connections
- **Output**: 256x256x1 density map (ReLU activation)
- **Loss**: Pixel-wise MSE on density maps

### Model D - Direct Regression
- **Encoder**: 3 blocks (16 -> 32 -> 64 filters)
- **Bottleneck**: 128 filters + 0.5 Dropout
- **Head**: GlobalAveragePooling2D -> Dense(64) -> Dense(1)
- **Output**: Scalar cone count
- **Loss**: MSE on counts

## Input Requirements

- **Size**: 256 x 256 pixels
- **Channels**: 3 (RGB)
- **Range**: Normalized to [0, 1]
- **Format**: NumPy array with shape `(batch, 256, 256, 3)`

## Notes

- **Model A** is recommended for most use cases (best MAE, interpretable output)
- **Model D** is recommended for resource-constrained deployments
- **Model C** showed training instability and is not recommended
- **Model B** weights are not included (same as Model A + linear regression)

## Citation

If you use these models, please cite:

```bibtex
@mastersthesis{cerda2024aoslo,
  title={Cone Density Estimation in AOSLO Images Using Image Processing and Deep Learning},
  author={Cerda Cortez, Nicolay Agustin},
  school={Universidad de La Sabana},
  year={2024},
  address={Chia, Colombia}
}
```
