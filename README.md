# Brain Age Prediction (In-House Triplanar Ensemble & Medical XAI Framework)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

Repositorio universal y autónomo para la estimación de la **Edad Cerebral (Brain Age Gap, BAG)**, calibración local de escáneres/resonadores externos, y generación de **mapas de explicabilidad diagnóstica (Medical XAI)** a partir de resonancias magnéticas estructurales T1 (compatibilidad nativa con carpetas/archivos `.zip` **DICOM**, volúmenes **NIfTI** y tensores preprocesados **`.pt`**).

---

## 🔬 Arquitectura y Fundamento Clínico

El sistema utiliza un ensamble triplanar basado en **Global-Local Transformers (GLT)** y fusión tardía mediante regresión de Ridge:

1. **Plano Axial:** Backbone **ResNet-18** entrenado con distribución Gaussiana de **Soft Labels** (100 bins de edad de 1 a 100 años + Soft-Argmax diferenciable) y 6 bloques de atención Transformer. Siempre evaluado con Test-Time Augmentation (**TTA**).
2. **Plano Coronal:** Backbone **ResNet-34** entrenado con función de pérdida **Smooth L1** continua y 8 bloques de atención Transformer. Siempre evaluado con Test-Time Augmentation (**TTA**).
3. **Plano Sagital:** Backbone **ResNet-18** entrenado con función de pérdida **MSE** continua y 6 bloques de atención Transformer.
4. **Fusión Tardía (Ensemble Stacker):** Regresión de Ridge multivariable optimizada sobre validación cruzada:
   $$\text{Pred\_Ensemble} = \beta_{\text{ax}} \cdot \text{Pred}_{\text{ax}} + \beta_{\text{cor}} \cdot \text{Pred}_{\text{cor}} + \beta_{\text{sag}} \cdot \text{Pred}_{\text{sag}} + \text{Intercept}$$
5. **Calibración del Sesgo Etario (bc-BAG):** Corrección lineal de la regresión a la media ajustada exclusivamente en controles sanos (CN) para garantizar ortogonalidad $r = 0.000$ frente a la edad cronológica:
   $$\text{bc-BAG} = \text{Raw BAG} - (\alpha \cdot \text{Edad} + \beta)$$

---

## 🧩 Métodos de Interpretabilidad Médica (XAI) con `--all`

Al especificar la bandera opcional `--all`, el pipeline genera automáticamente los 3 métodos de interpretabilidad diagnóstica:

1. **Integrated Gradients (IG firmado):** Integración de gradientes a nivel de vóxel a lo largo de una trayectoria lineal desde una base neutra hasta la imagen real. Revela qué microestructuras específicas aceleran (+) o disminuyen (-) la edad predicha.
2. **Occlusion Sensitivity:** Perturbación sistemática mediante parches oclusivos deslizantes ($32 \times 32$), cuantificando el impacto causal directo en la predicción ($\Delta = \text{Pred}_{\text{base}} - \text{Pred}_{\text{ocluida}}$).
3. **Grad-Attention (Transformer Rollout):** Matriz de auto-atención del último bloque Transformer multiplicada por los gradientes de retropropagación ($|\text{Atención} \odot \text{Gradiente}|$), aislando las redes anatómicas y circuitos a larga distancia determinantes.

---

## 📁 Estructura del Repositorio

```text
brain-age-pred-InHouse-model/
├── README.md                                     # Documentación general y guía de uso
├── HOWTO_CALIBRATION.md                          # Guía paso a paso de calibración para resonadores externos
├── environment.yml                               # Definición del entorno Conda
├── requirements.txt                              # Dependencias de Python vía pip
├── config.yaml                                   # Parámetros centrales y rutas relativas internas
├── .gitignore                                    # Exclusiones de control de versiones
│
├── data/
│   └── atlases/
│       ├── mni152_brain_mask_1mm.nii.gz          # Máscara intracraneal MNI152 (SOLID_v2)
│       └── mni152_t1_1mm_brain.nii.gz            # Atlas anatómico MNI152 1mm
│
├── checkpoints/
│   ├── model_axial_resnet18_soft.pt              # Modelo Axial (ResNet-18 Soft Labels, nblock=6)
│   ├── model_coronal_resnet34_smoothl1.pt        # Modelo Coronal (ResNet-34 Smooth L1, nblock=8)
│   ├── model_sagittal_resnet18_mse.pt            # Modelo Sagital (ResNet-18 MSE, nblock=6)
│   └── ridge_triplanar_ensemble.joblib           # Regresión Ridge para fusión triplanar
│
├── src/
│   ├── models/
│   │   ├── global_local_transformer.py          # Definición de la red Global-Local Transformer
│   │   ├── resnet_backbone.py                   # Extractor de características ResNet
│   │   └── vgg.py                               # Extractor de características VGG
│   │
│   ├── preprocessing/
│   │   ├── dicom_reader.py                      # Ingesta, parseo de edad DICOM y dcm2niix
│   │   ├── register_and_n4.sh                   # Script de registro a MNI152 y corrección N4
│   │   └── slice_extractor.py                   # Máscara SOLID_v2, normalización P1-P99 y corte 2.5D
│   │
│   ├── inference/
│   │   ├── predictor.py                         # Inferencia triplanar con TTA y ensamble Ridge
│   │   └── bias_correction.py                   # Calibración del sesgo etario (bc-BAG)
│   │
│   └── xai/
│       ├── xai_engine.py                        # Orquestador central de explicabilidad médica
│       ├── integrated_gradients.py              # Algoritmo de Integrated Gradients firmado
│       ├── occlusion_sensitivity.py             # Algoritmo de Occlusion Sensitivity
│       ├── grad_attention.py                    # Algoritmo de Grad-Attention Rollout
│       └── visualizer.py                        # Renderizado del panel diagnóstico PNG
│
├── run_pipeline.py                              # CLI Principal: Procesa T1 individual -> Predice Edad (+ XAI con --all)
├── batch_inference.py                           # Inferencia masiva para carpetas/cohortes completas
└── calibrate_local_scanner.py                   # Ajusta alpha/beta en controles y calibra la cohorte clínica
```

---

## 🚀 Instalación y Requisitos

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd brain-age-pred-InHouse-model
```

### 2. Crear y activar el entorno Conda
```bash
conda env create -f environment.yml
conda activate brain_age_env
```

*Nota:* Asegúrese de tener instalado `dcm2niix` en su sistema para la conversión automática de estudios DICOM.

---

## 💻 Guía de Uso

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
python batch_inference.py \
    --input_dir /ruta/a/directorio_de_escaneos/ \
    --output_csv ./batch_predictions.csv

# Inferencia en lote rápida sobre datasets ya preprocesados:
python batch_inference.py \
    --input_dir /ruta/a/datasets_preprocesados/ \
    --output_csv ./batch_predictions.csv \
    --skip-prep
```

### 4. Calibración Local del Resonador (`calibrate_local_scanner.py`)
Ajusta la regresión lineal sobre una cohorte local de Controles Sanos (CN) para eliminar el sesgo de regresión a la media y corregir la cohorte clínica:
```bash
python calibrate_local_scanner.py \
    --controls_csv ./controls_predictions.csv \
    --clinical_csv ./clinical_predictions.csv \
    --output_dir ./calibration_results
```
*(Ver detalles completos en [HOWTO_CALIBRATION.md](HOWTO_CALIBRATION.md))*

---

## 📊 Salidas del Pipeline

* **`results.json` / `results.csv`:** Predicciones cuantitativas para cada plano, predicción consolidada del ensamble, edad cronológica, `Raw_BAG` y `bc_BAG` (calibrado).
* **`tensors/`:** Tensores PyTorch 2.5D (`tensor_axial.pt`, `tensor_coronal.pt`, `tensor_sagittal.pt`) extraídos y normalizados.
* **`xai/<PATIENT_ID>_xai_diagnostic_panel.png` (con `--all`):** Panel visual de alta resolución (300 DPI) con la anatomía T1 y los 3 mapas de explicabilidad (IG, Occlusión y Grad-Attention) por cada plano.
