# Brain Age Prediction (In-House Model & Medical XAI Framework)

<p align="center">
  <img src="assets/banner.png" alt="Brain Age Prediction & Medical XAI Banner" width="100%">
</p>

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Model Release](https://img.shields.io/badge/Release-v1.0.0-purple.svg)](https://github.com/nicolasfuents/brain-age-pred-InHouse-model/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

Este repositorio contiene todo lo necesario para el cálculo de estimación de **Edad Cerebral (Brain Age Gap, BAG)**, calibración local de escáneres externos, y generación de **mapas de explicabilidad diagnóstica (Medical XAI)** a partir de resonancias magnéticas estructurales T1 (compatibilidad nativa con carpetas/archivos `.zip` **DICOM**, volúmenes **NIfTI** y tensores preprocesados **`.pt`**).

El modelo implementa una arquitectura 2.5D optimizada que opera con solo 5 cortes representativos por cada plano anatómico (axial, coronal y sagital). Gracias a esto, la inferencia de las redes neuronales es sumamente rápida y liviana (aproximadamente 0.5 segundos por volumen en GPU). La mayor carga computacional del flujo de trabajo reside en la etapa de preprocesamiento (alineación a MNI152 y corrección de inhomogeneidad N4).

---

## Instalación y Requisitos

### 1. Clonar el repositorio
```bash
git clone https://github.com/nicolasfuents/brain-age-pred-InHouse-model.git
cd brain-age-pred-InHouse-model
```

### 2. Crear y activar el entorno Conda
```bash
conda env create -f environment.yml
conda activate brain_age_env
```

*Nota:* Asegúrese de tener instalado `dcm2niix` en su sistema para la conversión automática de estudios DICOM.

---

## Guía de Uso

### 1. Inferencia Individual Rápida (`run_pipeline.py`)
```bash
# Inferencia desde DICOM (extrae edad de cabecera automáticamente):
python run_pipeline.py --input_dicom /ruta/al/estudio_DICOM/

# Inferencia desde volumen NIfTI T1:
python run_pipeline.py --input_t1 /ruta/al/volumen_T1w.nii.gz --age 68.5

# Inferencia directa sobre volumen/tensor ya preprocesado (sin re-ejecutar registro/N4):
python run_pipeline.py --input_t1 /ruta/al/volumen_MNI_preprocesado.nii.gz --age 68.5 --skip-prep
```

### 2. Inferencia Completa con Explicabilidad XAI (`--all`)
```bash
# Inferencia + Generación de mapas IG, Occlusion, Grad-Attention y Panel Diagnóstico PNG:
python run_pipeline.py --input_dicom /ruta/al/estudio_DICOM/ --all
```

### 3. Inferencia en Lote (`batch_inference.py`)
Procesa un directorio completo de resonancias (NIfTIs, tensores `.pt` o carpetas/zips DICOM) y genera un CSV listo para calibración:
```bash
# Inferencia estándar en lote:
python batch_inference.py     --input_dir /ruta/a/directorio_de_escaneos/     --output_csv ./batch_predictions.csv

# Inferencia en lote rápida sobre datasets ya preprocesados:
python batch_inference.py     --input_dir /ruta/a/datasets_preprocesados/     --output_csv ./batch_predictions.csv     --skip-prep
```

### 4. Calibración Local del Resonador (`calibrate_local_scanner.py`)
Ajusta la regresión lineal sobre una cohorte local de Controles Sanos (CN) para eliminar el sesgo de regresión a la media y corregir la cohorte clínica:
```bash
python calibrate_local_scanner.py     --controls_csv ./controls_predictions.csv     --clinical_csv ./clinical_predictions.csv     --output_dir ./calibration_results
```
*(Ver detalles completos en [HOWTO_CALIBRATION.md](HOWTO_CALIBRATION.md))*

---

## Métodos de Interpretabilidad Médica (XAI) con `--all`

Al especificar la bandera opcional `--all`, el pipeline genera automáticamente los 3 métodos de interpretabilidad diagnóstica:

1. **Integrated Gradients (IG firmado):** Integración de gradientes a nivel de vóxel a lo largo de una trayectoria lineal desde una base neutra hasta la imagen real. Revela qué microestructuras específicas aceleran (+) o disminuyen (-) la edad predicha.
2. **Occlusion Sensitivity:** Perturbación sistemática mediante parches oclusivos deslizantes, cuantificando el impacto causal directo en la predicción.
3. **Grad-Attention (Transformer Rollout):** Matriz de auto-atención del último bloque Transformer multiplicada por los gradientes de retropropagación ($|\text{Atención} \odot \text{Gradiente}|$), aislando las redes anatómicas y circuitos a larga distancia determinantes.

---

## Salidas del Pipeline

* **`results.json` / `results.csv`:** Predicciones cuantitativas para cada plano, predicción consolidada del ensamble, edad cronológica, `Raw_BAG` y `bc_BAG` (calibrado).
* **`tensors/`:** Tensores PyTorch 2.5D (`tensor_axial.pt`, `tensor_coronal.pt`, `tensor_sagittal.pt`) extraídos y normalizados.
* **`xai/<PATIENT_ID>_xai_diagnostic_panel.png` (con `--all`):** Panel visual de alta resolución (300 DPI) con la anatomía T1 y los 3 mapas de explicabilidad (IG, Occlusión y Grad-Attention) por cada plano.

---

## Rendimiento y Benchmark

El tiempo de procesamiento y consumo de memoria del framework se divide en dos fases:

| Etapa | Hardware Evaluado | Tiempo por Sujeto | Huella de Memoria |
| :--- | :--- | :--- | :--- |
| **Inferencia Triplanar (3 Modelos + TTA)** | GPU (NVIDIA H100 80GB) | **~0.55 s** (~1.8 sujetos/s) | **< 1.0 GB VRAM** (pico 954 MB) |
| **Inferencia Triplanar (3 Modelos + TTA)** | CPU (AMD EPYC 9654) | **~10.8 s** | ~1.2 GB RAM |
| **Preprocesamiento Quasiraw (FLIRT + N4)** | CPU / GPU | **~45 – 60 s** | ~2.0 GB RAM |

*Nota:* Al requerir menos de 1 GB de VRAM durante la inferencia, el framework puede ejecutarse en GPUs comerciales de gama de entrada (e.g. GTX 1650, RTX 3050 de laptop) o en servidores que operen exclusivamente sobre CPU.
