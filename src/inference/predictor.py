#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py

Motor de inferencia triplanar y ensamble Ridge.
Carga los modelos entrenados para cada plano anatómico, ejecuta la pasada hacia adelante con TTA
y calcula la predicción combinada de edad cerebral (Pred_Ensemble).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import joblib
from src.models.global_local_transformer import GlobalLocalTransformer

class TriplanarPredictor:
    def __init__(
        self, 
        checkpoints_dir: Path, 
        device: Optional[torch.device] = None,
        use_tta: bool = True
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_tta = use_tta
        
        # Bins para Soft Labels (1 a 100 años)
        self.bins = torch.arange(1, 101, dtype=torch.float32).to(self.device)
        
        self._check_and_download_checkpoints()
        self._load_models()
        self._load_stacker()
        
    def _check_and_download_checkpoints(self):
        """Verifica la existencia de los checkpoints; si faltan, intenta descargarlos."""
        required = [
            "model_axial_resnet18_soft.pt",
            "model_coronal_resnet34_smoothl1.pt",
            "model_sagittal_resnet18_mse.pt",
            "ridge_triplanar_ensemble.joblib"
        ]
        missing = [f for f in required if not (self.checkpoints_dir / f).exists()]
        if missing:
            print(f"[!] Checkpoints faltantes en {self.checkpoints_dir}: {missing}")
            try:
                from download_checkpoints import ensure_checkpoints
                ensure_checkpoints(self.checkpoints_dir)
            except Exception as e:
                print(f"[!] No se pudo descargar automáticamente: {e}")
                print(f"    Por favor ejecuta 'python download_checkpoints.py' o copia los archivos .pt en {self.checkpoints_dir}")

    def _load_models(self):
        """Instancia y carga los pesos de los 3 modelos de plano."""
        # 1. Axial (ResNet-18 Soft Labels, nblock=6)
        self.model_axial = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=6, backbone="resnet18", num_classes=100
        ).to(self.device)
        ax_ckpt = torch.load(self.checkpoints_dir / "model_axial_resnet18_soft.pt", map_location=self.device)
        self.model_axial.load_state_dict(ax_ckpt["model_state_dict"] if "model_state_dict" in ax_ckpt else ax_ckpt)
        self.model_axial.eval()
        
        # 2. Coronal (ResNet-34 Smooth L1, nblock=8)
        self.model_coronal = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=8, backbone="resnet34", num_classes=1
        ).to(self.device)
        cor_ckpt = torch.load(self.checkpoints_dir / "model_coronal_resnet34_smoothl1.pt", map_location=self.device)
        self.model_coronal.load_state_dict(cor_ckpt["model_state_dict"] if "model_state_dict" in cor_ckpt else cor_ckpt)
        self.model_coronal.eval()
        
        # 3. Sagital (ResNet-18 MSE, nblock=6)
        self.model_sagittal = GlobalLocalTransformer(
            inplace=5, patch_size=64, step=32, nblock=6, backbone="resnet18", num_classes=1
        ).to(self.device)
        sag_ckpt = torch.load(self.checkpoints_dir / "model_sagittal_resnet18_mse.pt", map_location=self.device)
        self.model_sagittal.load_state_dict(sag_ckpt["model_state_dict"] if "model_state_dict" in sag_ckpt else sag_ckpt)
        self.model_sagittal.eval()

    def _load_stacker(self):
        """Carga el modelo de regresión Ridge para el ensamble."""
        stacker_path = self.checkpoints_dir / "ridge_triplanar_ensemble.joblib"
        if not stacker_path.exists():
            raise FileNotFoundError(f"No se encontró el Ridge Stacker en {stacker_path}")
        self.stacker = joblib.load(stacker_path)

    def _forward_axial(self, tensor: torch.Tensor) -> float:
        """Inferencia Axial con Soft-Argmax y TTA opcional."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outs = self.model_axial(tensor)
            logits = outs[0] if isinstance(outs, (list, tuple)) else outs
            probs = torch.softmax(logits, dim=1)
            pred_orig = (probs * self.bins).sum(dim=1).item()
            
            if self.use_tta:
                # TTA: Flip horizontal (eje X / ancho)
                tensor_flip = torch.flip(tensor, dims=[-1])
                outs_flip = self.model_axial(tensor_flip)
                logits_flip = outs_flip[0] if isinstance(outs_flip, (list, tuple)) else outs_flip
                probs_flip = torch.softmax(logits_flip, dim=1)
                pred_flip = (probs_flip * self.bins).sum(dim=1).item()
                return (pred_orig + pred_flip) / 2.0
            return pred_orig

    def _forward_coronal(self, tensor: torch.Tensor) -> float:
        """Inferencia Coronal con Smooth L1 y TTA opcional."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outs = self.model_coronal(tensor)
            pred_orig = outs[0].view(-1).item() if isinstance(outs, (list, tuple)) else outs.view(-1).item()
            
            if self.use_tta:
                # TTA: Flip horizontal
                tensor_flip = torch.flip(tensor, dims=[-1])
                outs_flip = self.model_coronal(tensor_flip)
                pred_flip = outs_flip[0].view(-1).item() if isinstance(outs_flip, (list, tuple)) else outs_flip.view(-1).item()
                return (pred_orig + pred_flip) / 2.0
            return pred_orig

    def _forward_sagittal(self, tensor: torch.Tensor) -> float:
        """Inferencia Sagital continua con MSE (sin TTA para preservar asimetría hemisférica)."""
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outs = self.model_sagittal(tensor)
            return outs[0].view(-1).item() if isinstance(outs, (list, tuple)) else outs.view(-1).item()

    def predict(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Ejecuta la inferencia triplanar y el Ridge Stacker.
        Retorna las predicciones individuales y la predicción final consolidada.
        """
        pred_ax = self._forward_axial(tensors["axial"])
        pred_cor = self._forward_coronal(tensors["coronal"])
        pred_sag = self._forward_sagittal(tensors["sagittal"])
        
        X_stack = np.array([[pred_ax, pred_cor, pred_sag]], dtype=np.float32)
        
        if isinstance(self.stacker, dict) and "model" in self.stacker:
            ridge_model = self.stacker["model"]
        else:
            ridge_model = self.stacker
            
        pred_ensemble = float(ridge_model.predict(X_stack)[0])
        
        return {
            "Pred_Axial": pred_ax,
            "Pred_Coronal": pred_cor,
            "Pred_Sagittal": pred_sag,
            "Pred_Ensemble": pred_ensemble
        }
