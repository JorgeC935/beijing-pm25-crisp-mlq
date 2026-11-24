"""Inferencia del pronóstico único a seis horas."""
import numpy as np

from .preprocessing import prediction_features


def forecast(model, history) -> float:
    """Predice PM2.5 en t+6 usando una medición de origen y su historial."""
    features = prediction_features(history)
    value = float(model.predict(features)[0])
    if not np.isfinite(value):
        raise ValueError("El modelo produjo una predicción no finita.")
    return max(0.0, value)
