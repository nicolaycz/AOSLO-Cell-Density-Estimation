# Dubis Dataset - AOSLO Images

This directory contains the Dubis dataset used for training and evaluating cone density estimation models.

## Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Images** | 264 split-detector AOSLO images |
| **Image Size** | 256 x 256 pixels |
| **Color Channels** | 3 (RGB) |
| **Format** | TIFF (.tif) |
| **Annotations** | Expert-marked cone centroids |

## Disease Distribution

| Condition | Samples |
|-----------|---------|
| Control (healthy) | 60 |
| Stargardt's disease | 65 |
| Retinitis pigmentosa | 139 |
| **Total** | **264** |

## Data Splits

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 140 (76%) | Model training |
| Validation | 44 (24%) | Hyperparameter tuning |
| Test | 80 | Final evaluation |

## Directory Structure

```
data/
├── Training+Density/          # 184 images (train + validation)
│   ├── image001.tif           # Original AOSLO image
│   ├── image001_Density.tif   # Corresponding density map
│   └── ...
└── Validation+Density/        # 80 images (test set)
    ├── image001.tif
    ├── image001_Density.tif
    └── ...
```

## File Naming Convention

- **Original images**: `imageXXX.tif` - RGB AOSLO images
- **Density maps**: `imageXXX_Density.tif` - Grayscale density annotations

## Density Map Format

Ground truth density maps are generated from expert annotations:

- **Format**: Single-channel grayscale TIFF (float32)
- **Scale**: Density values are multiplied by 100
- **Cone count**: `total_cones = sum(density_map) / 100`

## Preprocessing Applied

During training, the following preprocessing steps are applied:

1. **Normalization**: Images normalized to [0, 1] range
2. **Density filtering**: Only images with `sum(density_map)/100 < 400` used
3. **Data augmentation** (training only):
   - Random horizontal/vertical flips
   - Random 90-degree rotations
   - Intensity variations

## Data Source

The Dubis dataset contains split-detector AOSLO images with expert annotations marking the centroids of individual cone photoreceptors. Each annotation provides pixel-coordinate precision for supervised learning.

## License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)

This dataset is provided for **non-commercial research use only**. If you use this data in your research, please cite the original source and this repository.

## Citation

If you use this dataset, please cite:

```bibtex
@article{toledo2023deep,
  title={Deep density estimation for cone counting and diagnosis of genetic eye diseases from adaptive optics scanning light ophthalmoscope images},
  author={Toledo-Cort{\'e}s, Santiago and Dubis, Adam M and Gonz{\'a}lez, Fabio A and M{\"u}ller, Henning},
  journal={Translational Vision Science \& Technology},
  volume={12},
  number={11},
  pages={25},
  year={2023}
}
```
