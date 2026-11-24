from __future__ import annotations

import csv
import io
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from psycopg.types.json import Jsonb
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORIGINAL = ROOT / "models" / "wanliu_pm25_24h.joblib"
MODEL_UPDATED = ROOT / "models" / "wanliu_pm25_latest.joblib"
MODEL_LOCK = threading.RLock()
STATION_NAME = "Wanliu"
NUMERIC = ("pm25", "pm10", "so2", "no2", "co", "o3", "temp", "pres", "dewp", "rain", "wspm")
OBS_COLUMNS = ("observed_at",) + NUMERIC + ("wd",)

app = FastAPI(title="Pronóstico PM2.5 Wanliu", version="1.0", description="API para predecir PM2.5 a 24 horas, mantener historial y reentrenar el modelo.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_package():
    ruta = MODEL_UPDATED if MODEL_UPDATED.exists() else MODEL_ORIGINAL
    if not ruta.exists():
        raise HTTPException(503, f"No se encontró el modelo: {ruta}")
    return joblib.load(ruta)


def db_params():
    password = os.getenv("BEIJING_DB_PASSWORD")
    if not password:
        raise HTTPException(503, "Falta configurar BEIJING_DB_PASSWORD en la terminal que ejecuta la API")
    return dict(
        host=os.getenv("BEIJING_DB_HOST", "127.0.0.1"),
        port=int(os.getenv("BEIJING_DB_PORT", "5432")),
        dbname=os.getenv("BEIJING_DB_NAME", "beijing_pm25"),
        user=os.getenv("BEIJING_DB_USER", "beijing_app"),
        password=password,
        row_factory=psycopg.rows.dict_row,
    )


@contextmanager
def connect():
    try:
        with psycopg.connect(**db_params()) as conn:
            yield conn
    except (psycopg.OperationalError, psycopg.errors.UndefinedTable) as exc:
        raise HTTPException(503, f"Error de PostgreSQL o esquema sin crear: {exc}") from exc


def find_station(conn, lock=False):
    suffix = " FOR UPDATE" if lock else ""
    row = conn.execute("SELECT id FROM stations WHERE name = %s" + suffix, (STATION_NAME,)).fetchone()
    if row is None:
        raise HTTPException(404, "No existe la estación Wanliu en la base de datos")
    return row["id"]


class Observation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    observed_at: datetime
    pm25: float = Field(ge=0)
    pm10: float | None = Field(default=None, ge=0)
    so2: float | None = Field(default=None, ge=0)
    no2: float | None = Field(default=None, ge=0)
    co: float | None = Field(default=None, ge=0)
    o3: float | None = Field(default=None, ge=0)
    temp: float | None = None
    pres: float | None = Field(default=None, gt=0)
    dewp: float | None = None
    rain: float | None = Field(default=None, ge=0)
    wspm: float | None = Field(default=None, ge=0)
    wd: str = "Desconocido"

    @field_validator("observed_at")
    @classmethod
    def naive_beijing_local(cls, value):
        if value.tzinfo is not None:
            raise ValueError("Enviar hora local de Beijing sin zona horaria: YYYY-MM-DDTHH:MM:SS")
        if value.minute or value.second or value.microsecond:
            raise ValueError("La observación debe corresponder a una hora exacta")
        return value

    @field_validator("wd")
    @classmethod
    def valid_wind(cls, value):
        return value.strip() or "Desconocido"


def next_expected(conn, station_id):
    row = conn.execute("SELECT MAX(observed_at) AS last_time FROM observations WHERE station_id = %s", (station_id,)).fetchone()
    if row["last_time"] is None:
        raise HTTPException(409, "Wanliu no contiene historial previo")
    return row["last_time"] + timedelta(hours=1)


def insert_observation(conn, station_id, ob: Observation):
    expected = next_expected(conn, station_id)
    if ob.observed_at != expected:
        raise HTTPException(409, f"Fecha no consecutiva. Se esperaba {expected.isoformat(sep=' ')} (hora local de Beijing)")
    row = ob.model_dump()
    row_no = conn.execute(
        "SELECT COALESCE(MAX(source_row_no), 0) + 1 AS next_no FROM observations WHERE station_id = %s",
        (station_id,),
    ).fetchone()["next_no"]
    conn.execute(
        """INSERT INTO observations
        (station_id, observed_at, source_row_no,
         pm25, pm10, so2, no2, co, o3,
         temp, pres, dewp, rain, wspm, wd)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s)""",
        (station_id, row["observed_at"], row_no)
        + tuple(row[c] for c in NUMERIC)
        + (row["wd"],),
    )


def fetch_recent(conn, station_id, limit=48):
    rows = conn.execute(
        """SELECT observed_at, pm25, pm10, so2, no2, co, o3, temp, pres, dewp, rain, wd, wspm
        FROM observations WHERE station_id = %s ORDER BY observed_at DESC LIMIT %s""",
        (station_id, limit),
    ).fetchall()
    return list(reversed(rows))


def make_prediction_features(rows, columns):
    if len(rows) < 48:
        raise HTTPException(409, "Se necesitan al menos 48 mediciones horarias para crear el historial")
    times = [r["observed_at"] for r in rows]
    if any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
        raise HTTPException(409, "El historial más reciente presenta horas discontinuas")
    current = rows[-1]
    pm = pd.Series([r["pm25"] for r in rows], dtype="float64")
    base = dict(current)
    for i in range(48):
        base[f"pm25_hist_{i:02d}"] = rows[-i - 1]["pm25"]
    for window in (6, 12, 24, 48):
        sub = pm.iloc[-window:]
        base[f"pm25_media_{window}"] = sub.mean()
        base[f"pm25_cuenta_{window}"] = sub.count()
        base[f"pm25_std_{window}"] = sub.std()
    for lag in (1, 24):
        base[f"temp_lag_{lag}"] = rows[-lag - 1]["temp"]
        base[f"pm10_lag_{lag}"] = rows[-lag - 1]["pm10"]
    base["hora"] = current["observed_at"].hour
    base["mes"] = current["observed_at"].month
    base["dia_semana"] = current["observed_at"].weekday()
    base["wd"] = current["wd"] or "Desconocido"
    features = []
    for h in range(1, 25):
        future = current["observed_at"] + timedelta(hours=h)
        row = dict(base)
        row.update({
            "horizonte": h,
            "hora_objetivo": future.hour,
            "mes_objetivo": future.month,
            "dia_semana_objetivo": future.weekday(),
            "hora_sin": np.sin(2 * np.pi * future.hour / 24),
            "hora_cos": np.cos(2 * np.pi * future.hour / 24),
            "mes_sin": np.sin(2 * np.pi * (future.month - 1) / 12),
            "mes_cos": np.cos(2 * np.pi * (future.month - 1) / 12),
            "pm25_ayer": base[f"pm25_hist_{24 - h:02d}"],
        })
        features.append(row)
    df = pd.DataFrame(features)
    missing = [c for c in columns if c not in df]
    if missing:
        raise HTTPException(500, f"Faltan características requeridas por el modelo: {missing}")
    return df.loc[:, columns]


def forecast(package, rows):
    columns = package["feature_columns"]
    x = make_prediction_features(rows, columns)
    values = np.maximum(package.get("clip_min", 0.0), package["model"].predict(x))
    current = rows[-1]["observed_at"]
    predictions = [{
        "horizonte": i,
        "observed_at": (current + timedelta(hours=i)).isoformat(),
        "pm25_predicho": round(float(value), 3),
    } for i, value in enumerate(values, 1)]
    return predictions


@app.get("/")
def root():
    return {"service": "Wanliu PM2.5", "swagger": "/docs", "horizons": 24}


@app.get("/state")
def state():
    with connect() as conn:
        station_id = find_station(conn)
        return {"station": STATION_NAME, "next_expected_observed_at": next_expected(conn, station_id).isoformat()}


@app.post("/predict")
def predict(observation: Observation):
    with MODEL_LOCK:
        package = load_package()
        with connect() as conn:
            station_id = find_station(conn, lock=True)
            insert_observation(conn, station_id, observation)
            rows = fetch_recent(conn, station_id)
            predicted = forecast(package, rows)
            original = observation.model_dump(mode="json")
            record = conn.execute(
                """INSERT INTO prediction_runs
                (station_id, observed_at, input_data, forecast, model_version, model_name)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                (station_id, observation.observed_at, Jsonb(original), Jsonb(predicted),
                 package["version"], package.get("model_name", "modelo")),
            ).fetchone()
            return {"id": record["id"], "station": STATION_NAME,
                    "observed_at": observation.observed_at.isoformat(),
                    "model_version": package["version"], "input": original, "predictions": predicted}


@app.get("/predictions")
def predictions(limit: int = Query(20, ge=1, le=100)):
    with connect() as conn:
        station_id = find_station(conn)
        rows = conn.execute(
            """SELECT id, observed_at, input_data, forecast, model_version, model_name, created_at
            FROM prediction_runs WHERE station_id = %s ORDER BY id DESC LIMIT %s""",
            (station_id, limit),
        ).fetchall()
        return [{"id": r["id"], "observed_at": r["observed_at"],
                 "input": r["input_data"], "predictions": r["forecast"],
                 "model_version": r["model_version"], "model_name": r["model_name"],
                 "created_at": r["created_at"]} for r in rows]


@app.get("/metrics")
def metrics():
    with MODEL_LOCK:
        package = load_package()
        return {"station": package["station"], "version": package["version"],
                "model_name": package.get("model_name"), "feature_count": len(package["feature_columns"]),
                "feature_columns": package["feature_columns"],
                "validation": package.get("validation_metrics"),
                "test_initial": package.get("test_metrics"),
                "test_initial_6h": package.get("test_metrics_6h"),
                "retrain_evaluation": package.get("retrain_evaluation"),
                "trained_at_utc": package.get("trained_at_utc")}


@app.post("/training-data")
async def training_data(file: Annotated[UploadFile, File(description="CSV de observaciones nuevas y consecutivas de Wanliu")]):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Sube un archivo .csv")
    raw = await file.read(3_000_001)
    if len(raw) > 3_000_000:
        raise HTTPException(413, "El archivo debe tener como máximo 3 MB")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or not {"observed_at", "pm25"}.issubset(reader.fieldnames):
            raise ValueError("Faltan las columnas observed_at y pm25")
        items = []
        for n, row in enumerate(reader, 1):
            if row.get("station") not in (None, "", "Wanliu"):
                raise ValueError(f"La fila {n} no corresponde a Wanliu")
            obj = {}
            for key in OBS_COLUMNS:
                val = row.get(key)
                if val is None or str(val).strip() == "":
                    if key == "wd":
                        val = "Desconocido"
                    elif key not in ("observed_at", "pm25"):
                        val = None
                obj[key] = val
            items.append(Observation.model_validate(obj))
        if not items:
            raise ValueError("El CSV está vacío")
        if len(items) > 1000:
            raise ValueError("Máximo 1000 observaciones por archivo")
    except Exception as exc:
        raise HTTPException(422, f"CSV inválido: {exc}") from exc

    with MODEL_LOCK:
        with connect() as conn:
            station_id = find_station(conn, lock=True)
            expected = next_expected(conn, station_id)
            for idx, item in enumerate(items):
                if item.observed_at != expected + timedelta(hours=idx):
                    raise HTTPException(409, f"La fila {idx + 1} debe tener fecha {(expected + timedelta(hours=idx)).isoformat()}")
            for item in items:
                insert_observation(conn, station_id, item)
            conn.execute(
                "INSERT INTO training_uploads (filename, rows_inserted, first_time, last_time) VALUES (%s, %s, %s, %s)",
                (file.filename, len(items), items[0].observed_at, items[-1].observed_at),
            )
    return {"status": "ok", "inserted": len(items), "first": items[0].observed_at,
            "last": items[-1].observed_at, "note": "Los datos se incorporan al reentrenar manualmente."}


def build_retrain_matrix(data: pd.DataFrame, feature_columns):
    frame = data.copy().sort_values("observed_at").reset_index(drop=True)
    pm = frame["pm25"]
    frame["wd"] = frame["wd"].fillna("Desconocido")
    for lag in range(48):
        frame[f"pm25_hist_{lag:02d}"] = pm.shift(lag)
    for window in (6, 12, 24, 48):
        frame[f"pm25_media_{window}"] = pm.rolling(window, min_periods=1).mean()
        frame[f"pm25_cuenta_{window}"] = pm.rolling(window, min_periods=1).count()
        frame[f"pm25_std_{window}"] = pm.rolling(window, min_periods=2).std()
    for lag in (1, 24):
        for name in ("temp", "pm10"):
            frame[f"{name}_lag_{lag}"] = frame[name].shift(lag)
    frame["hora"] = frame["observed_at"].dt.hour
    frame["mes"] = frame["observed_at"].dt.month
    frame["dia_semana"] = frame["observed_at"].dt.day_of_week
    valid = (
        (frame["observed_at"].shift(47) == frame["observed_at"] - pd.Timedelta(hours=47)) &
        (frame["observed_at"].shift(-24) == frame["observed_at"] + pd.Timedelta(hours=24)) &
        frame["pm25"].notna()
    )
    for h in range(1, 25):
        frame[f"y_{h:02d}"] = pm.shift(-h)
        valid &= frame[f"y_{h:02d}"].notna()
    frame = frame.loc[valid].copy()
    if len(frame) < 100:
        raise HTTPException(409, "No hay suficientes orígenes con 48 horas de historial y 24 objetivos reales")
    parts = []
    for h in range(1, 25):
        portion = frame.copy()
        future = portion["observed_at"] + pd.Timedelta(hours=h)
        portion["horizonte"] = h
        portion["hora_objetivo"] = future.dt.hour
        portion["mes_objetivo"] = future.dt.month
        portion["dia_semana_objetivo"] = future.dt.day_of_week
        portion["hora_sin"] = np.sin(2 * np.pi * future.dt.hour / 24)
        portion["hora_cos"] = np.cos(2 * np.pi * future.dt.hour / 24)
        portion["mes_sin"] = np.sin(2 * np.pi * (future.dt.month - 1) / 12)
        portion["mes_cos"] = np.cos(2 * np.pi * (future.dt.month - 1) / 12)
        portion["pm25_ayer"] = portion[f"pm25_hist_{24 - h:02d}"]
        portion["target"] = portion[f"y_{h:02d}"]
        parts.append(portion[list(feature_columns) + ["target", "observed_at"]])
    return pd.concat(parts, ignore_index=True)



@app.post("/retrain")
def retrain():
    with MODEL_LOCK:
        current = load_package()

        with connect() as conn:
            station_id = find_station(conn, lock=True)

            rows = conn.execute(
                """SELECT observed_at, pm25, pm10, so2, no2,
                          co, o3, temp, pres, dewp, rain, wd, wspm
                   FROM observations
                   WHERE station_id = %s
                   ORDER BY observed_at""",
                (station_id,),
            ).fetchall()

            data = pd.DataFrame(rows)
            data["observed_at"] = pd.to_datetime(
                data["observed_at"]
            )

            # Excluir las nueve observaciones ficticias
            # utilizadas durante nuestras pruebas.
            demo = data["observed_at"].between(
                pd.Timestamp("2017-03-01 00:00:00"),
                pd.Timestamp("2017-03-01 08:00:00"),
            )
            data = data.loc[~demo].copy()

            # El dataset original terminó el 28/02/2017.
            # Exigir datos posteriores para evaluar un
            # reentrenamiento con información nueva.
            new_rows = data.loc[
                data["observed_at"] >
                pd.Timestamp("2017-02-28 23:00:00")
            ].sort_values("observed_at")

            if len(new_rows) < 168:
                return {
                    "status": "insufficient_new_data",
                    "new_rows": len(new_rows),
                    "required": 168,
                    "note": (
                        "No se reentrenó el modelo. Se necesitan "
                        "al menos 168 mediciones horarias nuevas, "
                        "consecutivas y auténticas. Las observaciones "
                        "ficticias de demostración están excluidas."
                    ),
                }

            if not new_rows["observed_at"].diff().dropna().eq(
                pd.Timedelta(hours=1)
            ).all():
                raise HTTPException(
                    409,
                    "Las observaciones nuevas presentan horas discontinuas",
                )

            cols = current["feature_columns"]
            samples = build_retrain_matrix(data, cols)

            # Reservar las últimas 48 horas nuevas
            # como período de evaluación.
            cutoff = new_rows.iloc[-48]["observed_at"]

            # Evitar que los objetivos futuros del
            # entrenamiento alcancen el período validado.
            train = samples.loc[
                samples["observed_at"] +
                pd.Timedelta(hours=24) < cutoff
            ]

            val = samples.loc[
                samples["observed_at"] >= cutoff
            ]

            if len(train) < 1000 or len(val) < 100:
                raise HTTPException(
                    409,
                    "No hay suficientes ejemplos para entrenar y validar",
                )

            # Entrenar candidato solo con el período permitido.
            candidate = clone(current["model"])
            candidate.fit(train[cols], train["target"])

            candidate_pred = np.maximum(
                0, candidate.predict(val[cols])
            )
            incumbent_pred = np.maximum(
                0, current["model"].predict(val[cols])
            )

            candidate_mae = float(
                mean_absolute_error(val["target"], candidate_pred)
            )
            incumbent_mae = float(
                mean_absolute_error(val["target"], incumbent_pred)
            )

            candidate_rmse = float(
                np.sqrt(mean_squared_error(
                    val["target"], candidate_pred
                ))
            )
            incumbent_rmse = float(
                np.sqrt(mean_squared_error(
                    val["target"], incumbent_pred
                ))
            )

            improved = candidate_mae < incumbent_mae

            if improved:
                # IMPORTANTE: no entrenar nuevamente con
                # los datos reservados para validación.
                updated = dict(current)
                updated["model"] = candidate
                updated["version"] = (
                    "wanliu-retrain-" +
                    datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    )
                )
                updated["trained_at_utc"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                updated["training_data_last_origin"] = str(
                    train["observed_at"].max()
                )
                updated["retrain_evaluation"] = {
                    "candidate_validation_MAE": candidate_mae,
                    "candidate_validation_RMSE": candidate_rmse,
                    "incumbent_validation_MAE": incumbent_mae,
                    "incumbent_validation_RMSE": incumbent_rmse,
                    "comparison": (
                        "Evaluación cronológica con datos posteriores "
                        "al período original; validación excluida "
                        "del ajuste del candidato"
                    ),
                }
                updated["validation_metrics"] = {
                    "MAE": candidate_mae,
                    "RMSE": candidate_rmse,
                }

                # No reutilizar métricas del test original
                # como si fueran de la nueva versión.
                updated["test_metrics"] = None
                updated["test_metrics_6h"] = None

                tmp = MODEL_UPDATED.with_suffix(".joblib.tmp")
                MODEL_UPDATED.parent.mkdir(
                    parents=True, exist_ok=True
                )
                joblib.dump(updated, tmp)
                os.replace(tmp, MODEL_UPDATED)

            conn.execute(
                """INSERT INTO retraining_runs
                   (promoted, candidate_mae, incumbent_mae,
                    sample_count, validation_start)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    improved,
                    candidate_mae,
                    incumbent_mae,
                    len(samples),
                    cutoff.to_pydatetime(),
                ),
            )

            return {
                "status": (
                    "promoted" if improved else "not_promoted"
                ),
                "candidate_validation": {
                    "MAE": candidate_mae,
                    "RMSE": candidate_rmse,
                },
                "incumbent_validation": {
                    "MAE": incumbent_mae,
                    "RMSE": incumbent_rmse,
                },
                "sample_count": len(samples),
                "validation_start": cutoff.isoformat(),
                "note": (
                    "La comparación usa datos no incluidos en "
                    "el entrenamiento del candidato. "
                    "El modelo promovido conserva ese período "
                    "sin utilizarlo para ajustarse."
                ),
            }
