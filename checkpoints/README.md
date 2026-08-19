# Model Checkpoints & Pretrained Weights

This directory holds the pretrained neural network weights and ensemble stacker for the triplanar brain age prediction model:

1. `model_axial_resnet18_soft.pt` (128 MB): ResNet-18 trained with soft age labels (0-99 bins).
2. `model_coronal_resnet34_smoothl1.pt` (219 MB): ResNet-34 trained with Smooth L1 regression.
3. `model_sagittal_resnet18_mse.pt` (128 MB): ResNet-18 trained with MSE regression.
4. `ridge_triplanar_ensemble.joblib` (1.7 KB): Multivariate Ridge Regression Stacker.

## Automatic Download

The pipeline automatically checks for these files and downloads them upon first run. You can also manually download them at any time:

```bash
python download_checkpoints.py
```
