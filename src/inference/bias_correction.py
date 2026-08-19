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
        pred_age: float, 
        chronological_age: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calcula el BAG crudo y el bc-BAG calibrado.
        Si chronological_age es None o NaN, retorna None para las brechas.
        """
        if chronological_age is None or chronological_age != chronological_age: # NaN check
            return {
                "Chronological_Age": None,
                "Raw_BAG": None,
                "bc_BAG": None
            }
            
        raw_bag = pred_age - chronological_age
        bias = self.alpha * chronological_age + self.beta
        bc_bag = raw_bag - bias
        
        return {
            "Chronological_Age": float(chronological_age),
            "Raw_BAG": float(raw_bag),
            "bc_BAG": float(bc_bag)
        }
