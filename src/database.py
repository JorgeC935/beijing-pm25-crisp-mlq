"""Persistencia PostgreSQL para el sistema de pronóstico PM2.5 a +6 horas."""
from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .preprocessing import FEATURE_COLUMNS, HORIZON_HOURS
from .schemas import MEASUREMENTS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "wanliu_6h"


def connect_params():
    password = os.getenv("BEIJING_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "Configura BEIJING_DB_PASSWORD en la terminal que ejecuta FastAPI."
        )
    return {
        "host": os.getenv("BEIJING_DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("BEIJING_DB_PORT", "5432")),
        "dbname": os.getenv("BEIJING_DB_NAME", "wanliu_pm25_6h"),
        "user": os.getenv("BEIJING_DB_USER", "beijing_app"),
        "password": password,
        "row_factory": dict_row,
    }


@contextmanager
def connect():
    with psycopg.connect(**connect_params()) as conn:
        yield conn


def lock(conn):
    """Serializa operaciones que modifican observaciones o versiones del modelo."""
    conn.execute("SELECT pg_advisory_xact_lock(20261124, 6)")


def clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def current_model(conn):
    return conn.execute(
        "SELECT * FROM models WHERE is_active AND horizon_hours = %s",
        (HORIZON_HOURS,),
    ).fetchone()


def next_time(conn):
    last = conn.execute(
        "SELECT max(observed_at) AS last FROM observations"
    ).fetchone()["last"]
    return last + timedelta(hours=1) if last is not None else None


def create_upload(conn, filename, kind, count, first, last):
    return conn.execute(
        """INSERT INTO data_uploads(filename,kind,row_count,first_time,last_time)
           VALUES (%s,%s,%s,%s,%s) RETURNING id""",
        (filename, kind, count, first, last),
    ).fetchone()["id"]


def import_historical(conn, frame, filename):
    upload_id = create_upload(
        conn,
        filename,
        "historical",
        len(frame),
        frame.observed_at.iloc[0].to_pydatetime(),
        frame.observed_at.iloc[-1].to_pydatetime(),
    )
    columns = ", ".join(
        ("observed_at", *MEASUREMENTS, "source", "is_simulated", "upload_id")
    )
    with conn.cursor().copy(f"COPY observations ({columns}) FROM STDIN") as copy:
        for row in frame[["observed_at", *MEASUREMENTS]].itertuples(
            index=False, name=None
        ):
            copy.write_row((
                row[0].to_pydatetime(),
                *(clean(value) for value in row[1:]),
                "historical",
                False,
                upload_id,
            ))
    return upload_id


def insert_observation(conn, item, observed_at):
    columns = ", ".join(
        ("observed_at", *MEASUREMENTS, "source", "is_simulated", "upload_id")
    )
    values = (
        observed_at,
        *(getattr(item, name) for name in MEASUREMENTS),
        "manual",
        True,
        None,
    )
    placeholders = ", ".join(["%s"] * len(values))
    return conn.execute(
        f"INSERT INTO observations ({columns}) VALUES ({placeholders}) RETURNING id",
        values,
    ).fetchone()["id"]


def observations(conn, limit=None):
    columns = ", ".join(
        ("id", "observed_at", *MEASUREMENTS, "is_simulated")
    )
    if limit is None:
        rows = conn.execute(
            f"SELECT {columns} FROM observations ORDER BY observed_at"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {columns} FROM observations ORDER BY observed_at DESC LIMIT %s",
            (limit,),
        ).fetchall()[::-1]
    return pd.DataFrame(
        rows,
        columns=["id", "observed_at", *MEASUREMENTS, "is_simulated"],
    )


def create_run(conn, kind, mode, parameters, observation_count, new_count):
    return conn.execute(
        """INSERT INTO training_runs(
               kind,mode,parameters,observation_count,new_observation_count
           ) VALUES (%s,%s,%s,%s,%s) RETURNING id""",
        (
            kind,
            mode,
            Jsonb(parameters),
            observation_count,
            new_count,
        ),
    ).fetchone()["id"]


def store_processed(conn, run_id, origins):
    columns = [
        "id",
        *FEATURE_COLUMNS,
        "target",
        "split",
    ]
    copy_sql = """
        COPY processed_observations
            (training_run_id, observation_id, features, target, split)
        FROM STDIN
    """
    with conn.cursor().copy(copy_sql) as copy:
        for row in origins[columns].itertuples(index=False, name=None):
            observation_id = int(row[0])
            features = {
                name: clean(value)
                for name, value in zip(FEATURE_COLUMNS, row[1:1 + len(FEATURE_COLUMNS)])
            }
            target = float(row[1 + len(FEATURE_COLUMNS)])
            split = row[-1]
            copy.write_row((
                run_id,
                observation_id,
                Jsonb(features),
                target,
                split,
            ))


def store_model(
    conn,
    run_id,
    version,
    algorithm,
    artifact_path,
    feature_columns,
    active,
    is_demo,
    training_target_end,
    metrics,
    split_metadata,
):
    if active:
        conn.execute("UPDATE models SET is_active=FALSE WHERE is_active")

    model_id = conn.execute(
        """INSERT INTO models(
               training_run_id,version,algorithm,artifact_path,feature_columns,
               is_active,is_demo,training_target_end,horizon_hours
           ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (
            run_id,
            version,
            algorithm,
            artifact_path,
            Jsonb(feature_columns),
            active,
            is_demo,
            pd.Timestamp(training_target_end).to_pydatetime(),
            HORIZON_HOURS,
        ),
    ).fetchone()["id"]

    with conn.cursor().copy(
        """COPY model_metrics(
               model_id,evaluation_set,scope,horizon_start,horizon_end,
               mae,rmse,r2,sample_count
           ) FROM STDIN"""
    ) as copy:
        for metric in metrics:
            copy.write_row((
                model_id,
                metric["evaluation_set"],
                metric["scope"],
                metric["horizon_start"],
                metric["horizon_end"],
                metric["mae"],
                metric["rmse"],
                metric["r2"],
                metric["sample_count"],
            ))

    conn.execute(
        """UPDATE training_runs
           SET status='succeeded',split_metadata=%s,finished_at=NOW()
           WHERE id=%s""",
        (Jsonb(split_metadata), run_id),
    )
    return model_id
