"""Características compartidas por el entrenamiento y la predicción a +6 horas.

Las variables se calculan usando únicamente información disponible en la hora
de origen t. El objetivo es el PM2.5 observado exactamente seis horas después.
"""
import numpy as np
import pandas as pd

from .schemas import MEASUREMENTS, NUMERIC

HORIZON_HOURS = 6

FEATURE_COLUMNS = [
    "pm10", "so2", "no2", "co", "o3", "temp", "pres", "dewp", "rain", "wspm",
    "wd", "pm25_t",
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6",
    "pm25_lag_12", "pm25_lag_24",
    "pm25_mean_6", "pm25_std_6",
    "pm25_mean_12", "pm25_std_12",
    "pm25_mean_24", "pm25_std_24",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "yearday_sin", "yearday_cos",
]
NUMERIC_FEATURES = [name for name in FEATURE_COLUMNS if name != "wd"]
CATEGORICAL_FEATURES = ["wd"]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construye las 32 características del notebook 04 para cada hora."""
    required = {"observed_at", *MEASUREMENTS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas para preparar datos: {sorted(missing)}")

    frame = data.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], errors="coerce")
    if frame["observed_at"].isna().any():
        raise ValueError("Hay fechas inválidas en observed_at.")

    frame = frame.sort_values("observed_at").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No hay observaciones para preparar.")
    if frame["observed_at"].duplicated().any():
        raise ValueError("La serie contiene fechas duplicadas.")
    if not frame["observed_at"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Las observaciones deben ser consecutivas, con una por hora.")

    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    frame["wd"] = frame["wd"].astype("object")
    frame["wd"] = frame["wd"].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    frame["wd"] = frame["wd"].replace(["", "Desconocido"], np.nan)

    frame["pm25_t"] = frame["pm25"]
    for lag in (1, 2, 3, 6, 12, 24):
        frame[f"pm25_lag_{lag}"] = frame["pm25"].shift(lag)

    for window in (6, 12, 24):
        rolling = frame["pm25"].rolling(
            window=window, min_periods=max(2, window // 2)
        )
        frame[f"pm25_mean_{window}"] = rolling.mean()
        frame[f"pm25_std_{window}"] = rolling.std()

    dates = pd.DatetimeIndex(frame["observed_at"])
    hour = dates.hour
    weekday = dates.day_of_week
    month = dates.month
    yearday = dates.day_of_year

    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    frame["yearday_sin"] = np.sin(2 * np.pi * (yearday - 1) / 365.25)
    frame["yearday_cos"] = np.cos(2 * np.pi * (yearday - 1) / 365.25)

    if "is_simulated" not in frame:
        frame["is_simulated"] = False
    frame["is_simulated"] = frame["is_simulated"].fillna(False).astype(bool)
    return frame


def prepare_supervised(data: pd.DataFrame) -> pd.DataFrame:
    """Crea un ejemplo por origen: X(t) -> PM2.5(t+6)."""
    frame = build_features(data)
    frame["target"] = frame["pm25"].shift(-HORIZON_HOURS)
    frame["target_at"] = frame["observed_at"].shift(-HORIZON_HOURS)
    frame["target_is_simulated"] = (
        frame["is_simulated"].shift(-HORIZON_HOURS).eq(True)
    )

    usable = (
        frame["pm25_t"].notna()
        & frame["target"].notna()
        & frame["target_at"].notna()
    )
    result = frame.loc[usable].copy().reset_index(drop=True)
    result["target"] = result["target"].astype(float)
    result["split"] = "unused"
    if "id" not in result:
        result["id"] = np.arange(1, len(result) + 1)
    return result


def prediction_features(history: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila de 32 características para el último origen disponible."""
    if len(history) < 25:
        raise ValueError(
            "Se necesitan al menos 25 observaciones horarias para calcular el rezago de 24 horas."
        )
    frame = build_features(history)
    latest = frame.tail(1)
    if latest["pm25_t"].isna().any():
        raise ValueError("La última observación necesita un PM2.5 válido.")
    return latest[FEATURE_COLUMNS]
