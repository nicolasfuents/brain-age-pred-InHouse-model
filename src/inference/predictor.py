#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py

Triplanar Ensemble Inference Engine with Test-Time Augmentation (TTA).
Orchestrates loading specialist networks (Axial, Coronal, Sagittal) and Ridge Regression stacker.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import torch
import torch.nn as nn
import numpy as np
import joblib

# Suppress serialization and version warnings for clean CLI output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ..models.gl_transformer import GlobalLocalTransformer
from ..models.ridge_stacker import RidgeStacker

class TriplanarPredictor:
    def __init__(
        self, 
        checkpoints_dir: Path, 
        device: Optional[str] = None,
        use_tta: bool = True
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.use_tta = use_tta
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self._check_and_download_checkpoints()
        self._load_models()
        self._load_stacker()

    def _check_and_download_checkpoints(self):
        """Checks for checkpoint availability and triggers download if missing."""
        required = [
            "model_axial_resnet18_soft.pt",
            "model_coronal_resnet34_smoothl1.pt",
            "model_sagittal_resnet18_mse.pt",
            "ridge_triplanar_ensemble.joblib"
        ]
        missing = [f for f in required if not (self.checkpoints_dir / f).exists()]
        if missing:
            try:
                from ..utils.download_weights import ensure_checkpoints
                ensure_checkpoints(self.checkpoints_dir)
            except Exception as e:
                print(f"[!] Could not automatically download weights: {e}")
                print(f"    Please run 'python download_checkpoints.py' or copy weights into {self.checkpoints_dir}")

    def _load_models(self):
        """Initializes and loads weights for all 3 specialized plane models."""
        # 1. Axial (ResNet-18 Soft Labels, nblock=6)
        self.model_axial = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=6, backbone="resnet18", num_classes=100
        ).to(self.device)
        ax_ckpt = torch.load(self.checkpoints_dir / "model_axial_resnet18_soft.pt", map_location=self.device, weights_only=False)
        self.model_axial.load_state_dict(ax_ckpt["model_state_dict"] if "model_state_dict" in ax_ckpt else ax_ckpt)
        self.model_axial.eval()
        
        # 2. Coronal (ResNet-34 Smooth L1, nblock=8)
        self.model_coronal = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=8, backbone="resnet34", num_classes=1
        ).to(self.device)
        cor_ckpt = torch.load(self.checkpoints_dir / "model_coronal_resnet34_smoothl1.pt", map_location=self.device, weights_only=False)
        self.model_coronal.load_state_dict(cor_ckpt["model_state_dict"] if "model_state_dict" in cor_ckpt else cor_ckpt)
        self.model_coronal.eval()
        
        # 3. Sagittal (ResNet-18 MSE, nblock=6)
        self.model_sagittal = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=6, backbone="resnet18", num_classes=1
        ).to(self.device)
        sag_ckpt = torch.load(self.checkpoints_dir / "model_sagittal_resnet18_mse.pt", map_location=self.device, weights_only=False)
        self.model_sagittal.load_state_dict(sag_ckpt["model_state_dict"] if "model_state_dict" in sag_ckpt else sag_ckpt)
        self.model_sagittal.eval()

    def _load_stacker(self):
        """Loads trained Ridge Stacker regression model."""
        stacker_path = self.checkpoints_dir / "ridge_triplanar_ensemble.joblib"
        if not stacker_path.exists():
            raise FileNotFoundError(f"Ridge stacker not found at {stacker_path}")
        self.stacker = joblib.load(stacker_path)

    def _forward_axial(self, tensor: torch.Tensor) -> float:
        """Axial inference with soft-argmax expectation and optional TTA."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outs = self.model_axial(tensor)
            probs = torch.softmax(outs, dim=1)
            bins = torch.arange(100, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_base = float(torch.sum(probs * bins, dim=1).item())
            
            if self.use_tta:
                # Horizontal anatomical flip
                flipped = torch.flip(tensor, dims=[-1])
                outs_flip = self.model_axial(flipped)
                probs_flip = torch.softmax(outs_flip, dim=1)
                pred_flip = float(torch.sum(probs_flip * bins, dim=1).item())
                return (pred_base + pred_flip) / 2.0
                
            return pred_base

    def _forward_coronal(self, tensor: torch.Tensor) -> float:
        """Coronal inference with direct regression and optional TTA."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            pred_base = float(self.model_coronal(tensor).item())
            
            if self.use_tta:
                flipped = torch.flip(tensor, dims=[-1])
                pred_flip = float(self.model_coronal(flipped).item())
                return (pred_base + pred_flip) / 2.0
                
            return pred_base

    def _forward_sagittal(self, tensor: torch.Tensor) -> float:
        """Sagittal inference with direct regression."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            return float(self.model_sagittal(tensor).item())

    def predict(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes full triplanar ensemble inference on input tensor dictionary.
        Returns individual plane predictions and meta-learner stacked ensemble estimate.
        """
        pred_ax = self._forward_axial(tensors["axial"])
        pred_cor = self._forward_coronal(tensors["coronal"])
        pred_sag = self._forward_sagittal(tensors["sagittal"])
        
        X_stack = np.array([[pred_ax, pred_cor, pred_sag]], dtype=np.float32)
        pred_ens = float(self.stacker.predict(X_stack)[0])
        
        return {
            "pred_axial": pred_ax,
            "pred_coronal": pred_cor,
            "pred_sagittal": pred_sag,
            "pred_ensemble": pred_ens,
            "Pred_Axial": pred_ax,
            "Pred_Coronal": pred_cor,
            "Pred_Sagittal": pred_sag,
            "Pred_Ensemble": pred_ens
        }
