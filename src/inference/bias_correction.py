"""
bias_correction.py

Módulo de calibración y corrección del sesgo etario (bc-BAG).
Elimina el fenómeno de regresión a la media mediante el modelo lineal ajustado en controles sanos (CN).
"""

from typing import Optional, Dict

class AgeBiasCalibrator:
    def __init__(self, alpha: float = -0.226305, beta: float = 16.582412):
        """
        Parámetros pre-ajustados derivados de la cohorte normativa de controles sanos (CN).
        bc_BAG = Raw_BAG - (alpha * Chronological_Age + beta)
        """
        self.alpha = alpha
        self.beta = beta
        
    def calculate_bag(
        self, 
        pred_age: Optional[float] = None, 
        chronological_age: Optional[float] = None,
        pred_ensemble: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Optional[float]]:
        """
        Calcula el BAG crudo y el bc-BAG calibrado.
        Si chronological_age es None o NaN, retorna None para las brechas.
        """
        actual_pred = pred_age if pred_age is not None else pred_ensemble
        if actual_pred is None:
            raise ValueError("Must provide either pred_age or pred_ensemble.")
            
        if chronological_age is None or chronological_age != chronological_age: # NaN check
            return {
                "Chronological_Age": None,
                "Raw_BAG": None,
                "bc_BAG": None,
                "raw_bag": None,
                "bc_bag": None
            }
            
        raw_bag = float(actual_pred - chronological_age)
        bias = float(self.alpha * chronological_age + self.beta)
        bc_bag = float(raw_bag - bias)
        
        return {
            "Chronological_Age": float(chronological_age),
            "Raw_BAG": raw_bag,
            "bc_BAG": bc_bag,
            "raw_bag": raw_bag,
            "bc_bag": bc_bag
        }
