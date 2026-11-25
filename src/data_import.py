"""Validación e importación del CSV histórico UCI de la estación Wanliu."""
import io

import numpy as np
import pandas as pd

from .schemas import MEASUREMENTS

UCI_COLUMNS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3",
    "TEMP": "temp",
    "PRES": "pres",
    "DEWP": "dewp",
    "RAIN": "rain",
    "WSPM": "wspm",
    "wd": "wd",
}


def parse_historical(raw: bytes) -> pd.DataFrame:
    original = pd.read_csv(
        io.BytesIO(raw),
        encoding="utf-8-sig",
        low_memory=False,
    )
    required = set(UCI_COLUMNS) | {"year", "month", "day", "hour", "station"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"Faltan columnas del CSV UCI: {sorted(missing)}")
    if original.empty or not original["station"].astype(str).str.strip().eq("Wanliu").all():
        raise ValueError("El CSV histórico debe corresponder exclusivamente a la estación Wanliu.")

    dates = pd.to_datetime(
        original[["year", "month", "day", "hour"]],
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("El CSV contiene fechas inválidas.")

    frame = pd.DataFrame({"observed_at": dates})
    for source, destination in UCI_COLUMNS.items():
        if destination == "wd":
            frame[destination] = original[source].astype("object").map(
                lambda value: value.strip() if isinstance(value, str) else value
            ).replace("", np.nan)
            continue

        values = pd.to_numeric(original[source], errors="coerce")
        if (original[source].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f"Hay valores numéricos inválidos en {source}.")
        if destination not in ("temp", "dewp"):
            if (values < 0).any():
                raise ValueError(f"Hay valores negativos no permitidos en {source}.")
            if destination == "pres" and (values == 0).any():
                raise ValueError("La presión debe ser mayor que cero.")
        frame[destination] = values

    frame = frame.sort_values("observed_at").reset_index(drop=True)
    expected = pd.date_range(
        "2013-03-01 00:00:00",
        "2017-02-28 23:00:00",
        freq="h",
    )
    if (
        len(frame) != 35064
        or frame["observed_at"].duplicated().any()
        or not pd.DatetimeIndex(frame["observed_at"]).equals(expected)
    ):
        raise ValueError(
            "Se requiere el CSV UCI completo de Wanliu: 35.064 horas consecutivas "
            "entre 2013-03-01 y 2017-02-28."
        )

    return frame[["observed_at", *MEASUREMENTS]]
