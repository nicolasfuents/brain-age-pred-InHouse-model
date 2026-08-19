"""
dicom_reader.py

Módulo de ingesta, sanitización, parseo de cabeceras DICOM y conversión a NIfTI mediante dcm2niix.
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple, Optional

def sanitize_name(s: str) -> str:
    """Limpia caracteres inválidos y espacios para nombres de directorios seguros."""
    s = s.strip().replace(" ", "_")
    keep = "".join([c if c.isalnum() or c in "._-+" else "_" for c in s])
    return keep or "UNKNOWN_PATIENT"

def rename_spaces_recursive(root: Path):
    """Reemplaza espacios por guiones bajos en nombres de archivos y carpetas recursivamente."""
    for current_root, dirs, files in os.walk(root, topdown=False):
        current_path = Path(current_root)
        for name in files:
            old = current_path / name
            new = current_path / name.replace(" ", "_")
            if new != old:
                try: old.rename(new)
                except Exception: pass
        for name in dirs:
            old = current_path / name
            new = current_path / name.replace(" ", "_")
            if new != old:
                try: old.rename(new)
                except Exception: pass

def handle_input_path(input_path: Path, work_dir: Path) -> Path:
    """
    Descomprime si es un archivo .zip o sanitiza si es un directorio.
    Retorna el directorio con los cortes DICOM válidos.
    """
    if input_path.is_file() and zipfile.is_zipfile(input_path):
        extracted_dir = work_dir / "extracted_dicom"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        rename_spaces_recursive(extracted_dir)
        dicom_dir = find_first_dicom_dir(extracted_dir)
        if not dicom_dir:
            raise FileNotFoundError(f"No se encontraron cortes .dcm en el zip extraído: {input_path}")
        return dicom_dir
    elif input_path.is_dir():
        dicom_dir = find_first_dicom_dir(input_path)
        if not dicom_dir:
            raise FileNotFoundError(f"No se encontraron archivos .dcm en el directorio: {input_path}")
        return dicom_dir
    else:
        raise ValueError(f"La ruta ingresada no es un archivo .zip ni un directorio válido: {input_path}")

def find_first_dicom_dir(root: Path) -> Optional[Path]:
    """Busca recursivamente el primer directorio que contenga archivos .dcm."""
    for dirpath, _, filenames in os.walk(root):
        if any(fn.lower().endswith(".dcm") for fn in filenames):
            return Path(dirpath)
    return None

def clean_dicom_age(age_str: str) -> Optional[float]:
    """Convierte cadenas como '068Y' o '68' en float validando el rango [0, 99] años."""
    if not age_str:
        return None
    numeric_age = "".join([c for c in str(age_str) if c.isdigit()])
    try:
        val = float(numeric_age)
        if 0 <= val <= 99:
            return val
        else:
            print(f"[WARN] La edad extraída ({val}) excede el rango válido [0, 99] años para el cual fue entrenado el modelo.", file=sys.stderr)
            return None
    except ValueError:
        return None

def extract_patient_info(dicom_dir: Path) -> Tuple[str, Optional[float]]:
    """
    Extrae PatientName y edad cronológica (con estrategia dual: PatientAge -> StudyDate - BirthDate).
    Valida que la edad esté dentro del dominio [0, 99] años del modelo.
    """
    try:
        import pydicom
    except ImportError:
        print("[WARN] pydicom no está instalado. Instálalo con 'pip install pydicom' para leer cabeceras DICOM.", file=sys.stderr)
        return "UNKNOWN_PATIENT", None

    for dirpath, _, filenames in os.walk(dicom_dir):
        for fn in filenames:
            if not fn.lower().endswith(".dcm"):
                continue
            fpath = Path(dirpath) / fn
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
                
                # 1. Nombre
                name = "UNKNOWN_PATIENT"
                if getattr(ds, "PatientName", None):
                    name = sanitize_name(str(ds.PatientName))
                
                # 2. Edad Primaria
                age = None
                if getattr(ds, "PatientAge", None):
                    age = clean_dicom_age(str(ds.PatientAge))
                
                # 3. Edad Secundaria (Fallback por fechas)
                if age is None:
                    bd = str(getattr(ds, "PatientBirthDate", ""))
                    sd = str(getattr(ds, "StudyDate", ""))
                    if len(bd) == 8 and len(sd) == 8:
                        calc_age = float(int(sd[:4]) - int(bd[:4]))
                        if 0 <= calc_age <= 99:
                            age = calc_age
                        else:
                            print(f"[WARN] La edad calculada por fechas ({calc_age}) excede el rango [0, 99] años.", file=sys.stderr)
                
                return name, age
            except Exception:
                continue
    return "UNKNOWN_PATIENT", None

def convert_dicom_to_nifti(dicom_dir: Path, out_dir: Path) -> Path:
    """Convierte serie DICOM a NIfTI (.nii.gz) utilizando dcm2niix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dcm2niix", "-z", "y", "-o", str(out_dir), str(dicom_dir)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al ejecutar dcm2niix: {e.stderr.decode('utf-8', errors='ignore')}")
    except FileNotFoundError:
        raise RuntimeError("No se encontró el ejecutable 'dcm2niix'. Por favor instálalo en el sistema para procesar DICOM.")
        
    niftis = sorted(list(out_dir.glob("*.nii.gz")) + list(out_dir.glob("*.nii")))
    if not niftis:
        raise FileNotFoundError(f"dcm2niix no generó volúmenes NIfTI en {out_dir}")
    return niftis[0]
