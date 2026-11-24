"""API REST de Wanliu Air Forecast: pronóstico único de PM2.5 a +6 horas."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import psycopg
from psycopg.types.json import Jsonb
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import database as db
from .data_import import parse_historical
from .prediction import forecast
from .preprocessing import FEATURE_COLUMNS, HORIZON_HOURS
from .schemas import Observation
from .train_model import ALGORITHM, PARAMETERS, RETRAIN_PROTOCOL, train

app = FastAPI(
    title="Wanliu Air Forecast",
    version="3.0",
    description=(
        "Pronóstico académico de PM2.5 exactamente seis horas después "
        "de una medición de la estación Wanliu."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(psycopg.Error)
def database_error(request: Request, exc: psycopg.Error):
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Error de PostgreSQL. Verifica la conexión, los permisos y "
                "que se haya ejecutado schema.sql o migrate_to_6h.sql."
            )
        },
    )


@app.exception_handler(RuntimeError)
def runtime_error(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def read_csv(file: UploadFile, maximum: int) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(422, "Selecciona un archivo CSV.")
    raw = file.file.read(maximum + 1)
    if len(raw) > maximum:
        raise HTTPException(413, "El archivo es demasiado grande.")
    return raw


def require_model(conn):
    model = db.current_model(conn)
    if model is None:
        raise HTTPException(
            409,
            "Primero debes entrenar el modelo de PM2.5 a seis horas.",
        )
    return model


def load_model(record):
    if int(record["horizon_hours"]) != HORIZON_HOURS:
        raise HTTPException(409, "La versión seleccionada no es compatible con el horizonte de seis horas.")

    path = (db.ROOT / record["artifact_path"]).resolve()
    if not path.is_relative_to(db.MODEL_DIR.resolve()) or not path.is_file():
        raise HTTPException(
            503,
            "No se encuentra el archivo del modelo registrado. "
            "No se cargarán automáticamente modelos antiguos.",
        )

    package = joblib.load(path)
    if (
        package.get("version") != record["version"]
        or package.get("horizon_hours") != HORIZON_HOURS
        or package.get("feature_columns") != FEATURE_COLUMNS
    ):
        raise HTTPException(
            503,
            "El artefacto no coincide con la versión de seis horas registrada.",
        )
    return package["model"]


def train_and_persist(conn, kind: str, incumbent_record=None):
    all_data = db.observations(conn)
    simulated_count = int(all_data["is_simulated"].sum())

    # El entrenamiento inicial se basa únicamente en el dataset histórico UCI.
    # El reentrenamiento sí incorpora las mediciones nuevas registradas en la UI.
    if kind == "initial":
        data = all_data.loc[~all_data["is_simulated"]].copy()
        incumbent = None
    else:
        data = all_data.copy()
        incumbent = load_model(incumbent_record)

    if data.empty:
        raise ValueError("No hay observaciones históricas para entrenar.")

    run_id = db.create_run(
        conn,
        kind,
        "historical" if kind == "initial" else "simulation",
        PARAMETERS,
        len(data),
        simulated_count,
    )

    compatible = True
    if kind == "retrain":
        prior = conn.execute(
            "SELECT kind, split_metadata FROM training_runs WHERE id = %s",
            (incumbent_record["training_run_id"],),
        ).fetchone()
        protocol = (prior["split_metadata"] or {}).get("protocol")
        compatible = (
            (prior["kind"] == "initial" and protocol == "70_15_15_chronological_with_6h_gap")
            or (prior["kind"] == "retrain" and protocol == RETRAIN_PROTOCOL)
        )
    model, origins, metrics, metadata, comparison = train(
        data, incumbent=incumbent, incumbent_compatible=compatible
    )
    is_active = kind == "initial" or bool(comparison["promoted"])
    is_demo = kind == "retrain"

    version = (
        "wanliu-6h-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid4().hex[:8]
    )

    db.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = db.MODEL_DIR / f"{version}.joblib"
    temporary_path = artifact_path.with_suffix(".joblib.tmp")

    package = {
        "model": model,
        "version": version,
        "horizon_hours": HORIZON_HOURS,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": ALGORITHM,
        "parameters": PARAMETERS,
        "metrics": metrics,
        "split_metadata": metadata,
        "is_demo": is_demo,
    }

    try:
        db.store_processed(conn, run_id, origins)
        joblib.dump(package, temporary_path)
        temporary_path.replace(artifact_path)

        model_id = db.store_model(
            conn=conn,
            run_id=run_id,
            version=version,
            algorithm=ALGORITHM,
            artifact_path=artifact_path.relative_to(db.ROOT).as_posix(),
            feature_columns=FEATURE_COLUMNS,
            active=is_active,
            is_demo=is_demo,
            training_target_end=metadata["training_target_end"],
            metrics=metrics,
            split_metadata=metadata,
        )
    except BaseException:
        artifact_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)

    if kind == "initial":
        note = (
            "Modelo inicial entrenado con el histórico UCI de Wanliu. "
            "La prueba temporal se mantuvo separada del ajuste."
        )
        status = "trained"
    elif is_active:
        note = (
            comparison["promotion_reason"] +
            f" Ejemplos nuevos usados en el ajuste: {metadata['new_training_examples']}. "
            "No constituye validación de rendimiento futuro con datos reales."
        )
        status = "promoted"
    else:
        note = comparison.get("promotion_reason") or (
            "El candidato no cumplió los criterios de mejora."
        )
        note += (
            f" Ejemplos nuevos usados en el ajuste: "
            f"{metadata.get('new_training_examples', 0)}. "
            "La evaluación es demostrativa. Se conserva el modelo activo anterior."
        )
        status = "not_promoted"

    return {
        "status": status,
        "model_id": model_id,
        "model_version": version,
        "horizon_hours": HORIZON_HOURS,
        "is_active": is_active,
        "is_demo": is_demo,
        "observations": int(len(data)),
        "simulated_observations": simulated_count,
        "feature_count": len(FEATURE_COLUMNS),
        "split": metadata,
        "metrics": metrics,
        "baseline_metrics": metadata.get("baseline_metrics", {}),
        "comparison": comparison,
        "note": note,
    }


@app.get("/state")
def state():
    with db.connect() as conn:
        counts = conn.execute(
            """SELECT
                   count(*) AS observations,
                   count(*) FILTER (WHERE NOT is_simulated) AS historical,
                   count(*) FILTER (WHERE is_simulated) AS simulated
               FROM observations"""
        ).fetchone()
        model = db.current_model(conn)
        return {
            "station": "Wanliu",
            "trained": model is not None,
            "version": model["version"] if model else None,
            "horizon_hours": HORIZON_HOURS if model else None,
            "is_demo": model["is_demo"] if model else False,
            "next_expected_observed_at": db.next_time(conn),
            **counts,
        }


@app.post("/train-initial")
def initial_training(file: UploadFile | None = File(default=None)):
    with db.connect() as conn:
        db.lock(conn)

        if db.current_model(conn):
            raise HTTPException(409, "Ya existe un modelo de seis horas activo.")

        counts = conn.execute(
            """SELECT
                   count(*) FILTER (WHERE NOT is_simulated) AS historical,
                   count(*) FILTER (WHERE is_simulated) AS simulated
               FROM observations"""
        ).fetchone()

        if counts["historical"] == 0:
            if file is None:
                raise HTTPException(
                    422,
                    "Selecciona el CSV histórico de Wanliu para importarlo y entrenar.",
                )
            try:
                frame = parse_historical(read_csv(file, 5_000_000))
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
            db.import_historical(conn, frame, Path(file.filename).name)
        elif file is not None:
            raise HTTPException(
                409,
                "Ya existe un dataset histórico en PostgreSQL. "
                "El entrenamiento utilizará esos registros para evitar duplicarlos.",
            )

        try:
            return train_and_persist(conn, "initial")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post("/predict")
def predict(observation: Observation):
    with db.connect() as conn:
        db.lock(conn)
        model_record = require_model(conn)
        model = load_model(model_record)

        observed_at = db.next_time(conn)
        if observed_at is None:
            raise HTTPException(409, "No hay observaciones históricas.")

        observation_id = db.insert_observation(conn, observation, observed_at)
        history = db.observations(conn, limit=25)

        try:
            predicted = forecast(model, history)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

        target_at = observed_at + timedelta(hours=HORIZON_HOURS)
        input_data = {
            "observed_at": observed_at.isoformat(),
            **observation.model_dump(),
            "is_simulated": True,
        }

        prediction_run_id = conn.execute(
            """INSERT INTO prediction_runs(observation_id,model_id,input_data)
               VALUES (%s,%s,%s) RETURNING id""",
            (observation_id, model_record["id"], Jsonb(input_data)),
        ).fetchone()["id"]

        conn.execute(
            """INSERT INTO prediction_results(
                   prediction_run_id,horizon,target_at,pm25_predicted
               ) VALUES (%s,%s,%s,%s)""",
            (prediction_run_id, HORIZON_HOURS, target_at, predicted),
        )

        return {
            "id": prediction_run_id,
            "station": "Wanliu",
            "observed_at": observed_at.isoformat(),
            "target_at": target_at.isoformat(),
            "horizon_hours": HORIZON_HOURS,
            "input": input_data,
            "prediction": {
                "horizon_hours": HORIZON_HOURS,
                "target_at": target_at.isoformat(),
                "pm25_predicted": predicted,
            },
            "model_version": model_record["version"],
            "model_name": model_record["algorithm"],
            "is_demo": model_record["is_demo"],
        }


@app.get("/predictions")
def prediction_history(limit: int = Query(30, ge=1, le=100)):
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT
                   p.id,
                   p.input_data AS input,
                   p.created_at,
                   o.observed_at,
                   m.version AS model_version,
                   m.algorithm AS model_name,
                   m.is_demo,
                   r.target_at,
                   r.pm25_predicted,
                   target.pm25 AS observed_pm25,
                   target.is_simulated AS observed_is_simulated,
                   CASE
                       WHEN target.pm25 IS NOT NULL
                       THEN abs(target.pm25 - r.pm25_predicted)
                       ELSE NULL
                   END AS absolute_error
               FROM prediction_runs p
               JOIN observations o ON o.id = p.observation_id
               JOIN models m ON m.id = p.model_id
               LEFT JOIN prediction_results r
                   ON r.prediction_run_id = p.id AND r.horizon = %s
               LEFT JOIN observations target ON target.observed_at = r.target_at
               WHERE m.horizon_hours = %s
               ORDER BY p.id DESC
               LIMIT %s""",
            (HORIZON_HOURS, HORIZON_HOURS, limit),
        ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["prediction"] = {
                "horizon_hours": HORIZON_HOURS,
                "target_at": item.pop("target_at"),
                "pm25_predicted": item.pop("pm25_predicted"),
            }
            item["observed_pm25"] = item.pop("observed_pm25")
            item["observed_is_simulated"] = item.pop("observed_is_simulated")
            item["absolute_error"] = item.pop("absolute_error")
            result.append(item)
        return result


@app.get("/metrics")
def model_metrics(model_id: int | None = None):
    with db.connect() as conn:
        if model_id is None:
            record = db.current_model(conn)
        else:
            record = conn.execute(
                "SELECT * FROM models WHERE id = %s",
                (model_id,),
            ).fetchone()

        if record is None:
            raise HTTPException(404, "No hay un modelo de seis horas disponible.")
        if int(record["horizon_hours"]) != HORIZON_HOURS:
            raise HTTPException(
                409,
                "Esta versión pertenece al sistema anterior de 24 horas.",
            )

        rows = conn.execute(
            """SELECT evaluation_set,scope,horizon_start,horizon_end,
                      mae,rmse,r2,sample_count
               FROM model_metrics
               WHERE model_id = %s
               ORDER BY evaluation_set""",
            (record["id"],),
        ).fetchall()
        run = conn.execute(
            "SELECT split_metadata FROM training_runs WHERE id = %s",
            (record["training_run_id"],),
        ).fetchone()
        metadata = run["split_metadata"] or {}

        return {
            "station": "Wanliu",
            "model_id": record["id"],
            "version": record["version"],
            "model_name": record["algorithm"],
            "horizon_hours": HORIZON_HOURS,
            "is_demo": record["is_demo"],
            "is_active": record["is_active"],
            "feature_count": len(record["feature_columns"]),
            "feature_columns": record["feature_columns"],
            "metrics": rows,
            "baseline_metrics": metadata.get("baseline_metrics", {}),
            "comparison": metadata.get("comparison"),
            "split": metadata,
        }


@app.post("/retrain")
def retrain():
    with db.connect() as conn:
        db.lock(conn)
        record = require_model(conn)
        try:
            return train_and_persist(conn, "retrain", record)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc


@app.get("/models")
def model_versions():
    with db.connect() as conn:
        return conn.execute(
            """SELECT
                   m.id,m.version,m.algorithm,m.horizon_hours,
                   m.is_active,m.is_demo,m.created_at,m.training_target_end,
                   r.kind,r.mode,r.status,r.split_metadata,
                   r.observation_count,r.new_observation_count
               FROM models m
               JOIN training_runs r ON r.id = m.training_run_id
               WHERE m.horizon_hours = %s
               ORDER BY m.id DESC""",
            (HORIZON_HOURS,),
        ).fetchall()
