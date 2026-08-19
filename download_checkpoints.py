#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_checkpoints.py

Script para la descarga automática de los checkpoints del ensamble triplanar
desde los Releases de GitHub / almacenamiento remoto institucional.
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

# URLs de descarga pública (GitHub Releases Assets)
RELEASE_TAG = "v1.0.0"
GITHUB_REPO = "nicolasfuents/brain-age-pred-InHouse-model" # Será actualizado con el repo del usuario
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

MODELS = {
    "model_axial_resnet18_soft.pt": {
        "url": f"{BASE_URL}/model_axial_resnet18_soft.pt",
        "size_mb": 128
    },
    "model_coronal_resnet34_smoothl1.pt": {
        "url": f"{BASE_URL}/model_coronal_resnet34_smoothl1.pt",
        "size_mb": 219
    },
    "model_sagittal_resnet18_mse.pt": {
        "url": f"{BASE_URL}/model_sagittal_resnet18_mse.pt",
        "size_mb": 128
    },
    "ridge_triplanar_ensemble.joblib": {
        "url": f"{BASE_URL}/ridge_triplanar_ensemble.joblib",
        "size_mb": 0.002
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_model(filename: str, info: dict, target_dir: Path):
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 1000:
        print(f"[✓] {filename} ya existe ({target_path.stat().st_size / (1024*1024):.1f} MB). Omitiendo descarga.")
        return

    print(f"[↓] Descargando {filename} (~{info['size_mb']} MB)...")
    url = info["url"]
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to)
        print(f"[✓] Descarga completada: {filename}")
    except Exception as e:
        print(f"[!] Error descargando {filename} desde {url}: {e}", file=sys.stderr)
        print(f"    Si el Release aún no está publicado, puedes copiar manualmente los archivos .pt en: {target_dir}")

def ensure_checkpoints(checkpoints_dir: Path = CHECKPOINTS_DIR):
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for filename, info in MODELS.items():
        download_model(filename, info, checkpoints_dir)

if __name__ == "__main__":
    ensure_checkpoints()
