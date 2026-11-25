"""Entrenamiento, evaluación temporal y comparación con persistencia para +6 h.

La búsqueda de hiperparámetros se documenta en el notebook 04. La API utiliza
los parámetros seleccionados allí para que los entrenamientos sean reproducibles
y no repitan una búsqueda costosa en cada solicitud.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    NUMERIC_FEATURES,
    prepare_supervised,
)

PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.0,
    "random_state": 42,
}
ALGORITHM = "HistGradientBoostingRegressor"
RETRAIN_PROTOCOL = "fixed_historical_benchmark_v1"
MAX_LOOKBACK_HOURS = 24


def create_pipeline():
    """Reproduce el pipeline HistGradientBoosting seleccionado en el notebook."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparation = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    estimator = HistGradientBoostingRegressor(**PARAMETERS)
    return Pipeline([
        ("preprocess", preparation),
        ("model", estimator),
    ])


def calculate_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(y) == 0:
        raise ValueError("No hay ejemplos para calcular métricas.")
    r2 = float(r2_score(y, pred)) if len(y) > 1 and np.ptp(y) > 0 else None
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2,
        "sample_count": int(len(y)),
    }


def _metric_row(evaluation_set, values):
    return {
        "evaluation_set": evaluation_set,
        "scope": "global",
        "horizon_start": HORIZON_HOURS,
        "horizon_end": HORIZON_HOURS,
        "mae": values["mae"],
        "rmse": values["rmse"],
        "r2": values["r2"],
        "sample_count": values["sample_count"],
    }


def split_initial(origins):
    """70/15/15 cronológico, evitando que las etiquetas crucen los cortes."""
    if len(origins) < 100:
        raise ValueError("El histórico no contiene suficientes ejemplos supervisados.")

    n = len(origins)
    train_cutoff = pd.Timestamp(origins["observed_at"].iloc[int(n * 0.70)])
    validation_cutoff = pd.Timestamp(origins["observed_at"].iloc[int(n * 0.85)])

    train_mask = origins["target_at"] < train_cutoff
    validation_mask = (
        (origins["observed_at"] >= train_cutoff)
        & (origins["target_at"] < validation_cutoff)
    )
    test_mask = origins["observed_at"] >= validation_cutoff

    origins["split"] = "unused"
    origins.loc[train_mask, "split"] = "train"
    origins.loc[validation_mask, "split"] = "validation"
    origins.loc[test_mask, "split"] = "test"

    return {
        "protocol": "70_15_15_chronological_with_6h_gap",
        "train_cutoff": train_cutoff.isoformat(),
        "validation_cutoff": validation_cutoff.isoformat(),
        "horizon_hours": HORIZON_HOURS,
    }



def split_retrain(origins, simulated_observations, first_simulated_at=None):
    """Conserva el ajuste histórico original y añade objetivos nuevos conocidos.

    La referencia histórica es fija y retrospectiva. Se excluye su extremo
    final para que ni siquiera los rezagos de los ejemplos nuevos la invadan.
    """
    historical_mask = ~origins["is_simulated"] & ~origins["target_is_simulated"]
    historical = origins.loc[historical_mask].copy().sort_values("observed_at")
    original = split_initial(historical)
    new_mask = origins["target_is_simulated"]
    new_count = int(new_mask.sum())
    if new_count == 0:
        raise ValueError(
            "Aún no hay nuevos objetivos conocidos a +6 h. Registra una "
            "medición adicional con PM2.5 válido; debe existir un origen válido "
            "exactamente seis horas antes. El modelo activo se conserva."
        )
    if first_simulated_at is None:
        raise ValueError("Falta la fecha de inicio de las mediciones nuevas.")
    first_simulated_at = pd.Timestamp(first_simulated_at)
    reference_end = first_simulated_at - pd.Timedelta(
        hours=HORIZON_HOURS + MAX_LOOKBACK_HOURS
    )
    historical_train = historical.loc[historical["split"].isin(["train", "validation"])]
    reference = historical.loc[
        (historical["split"] == "test") & (historical["target_at"] < reference_end)
    ]
    if len(reference) < 2:
        raise ValueError("El histórico no permite reservar una referencia común aislada.")
    origins["split"] = "unused"
    origins.loc[historical_train.index, "split"] = "train"
    origins.loc[new_mask, "split"] = "train"
    origins.loc[reference.index, "split"] = "validation"
    return {
        "protocol": RETRAIN_PROTOCOL,
        "historical_validation_cutoff": original["validation_cutoff"],
        "validation_start": pd.Timestamp(reference["observed_at"].min()).isoformat(),
        "validation_end": pd.Timestamp(reference["target_at"].max()).isoformat(),
        "reference_end_exclusive": reference_end.isoformat(),
        "first_simulated_at": first_simulated_at.isoformat(),
        "validation_origins": int(len(reference)),
        "new_training_examples": new_count,
        "historical_training_examples": int(len(historical_train)),
        "promotion_eligible": True,
        "validation_status": "retrospective",
        "independent_test": False,
        "horizon_hours": HORIZON_HOURS,
        "note": (
            "Referencia histórica fija, fuera del ajuste de ambos modelos. "
            "Comparación retrospectiva para demostración; no es una nueva "
            "prueba de pronóstico futuro ni evidencia ambiental real."
        ),
    }


def train(data, incumbent=None, incumbent_compatible=True):
    """Entrena un modelo inicial o un candidato de reentrenamiento."""
    origins = prepare_supervised(data)
    simulated_count = int(data["is_simulated"].fillna(False).astype(bool).sum()) if "is_simulated" in data else 0

    if incumbent is None:
        meta = split_initial(origins)
        train_rows = origins.loc[origins["split"] == "train"]
        validation_rows = origins.loc[origins["split"] == "validation"]
        test_rows = origins.loc[origins["split"] == "test"]

        if min(len(train_rows), len(validation_rows), len(test_rows)) < 2:
            raise ValueError("La separación temporal produjo un conjunto demasiado pequeño.")

        validation_model = create_pipeline()
        validation_model.fit(train_rows[FEATURE_COLUMNS], train_rows["target"])
        validation_prediction = validation_model.predict(validation_rows[FEATURE_COLUMNS])
        validation_metrics = calculate_metrics(
            validation_rows["target"], validation_prediction
        )
        baseline_validation = calculate_metrics(
            validation_rows["target"], validation_rows["pm25_t"]
        )

        final_training = pd.concat([train_rows, validation_rows]).sort_values("observed_at")
        final_model = create_pipeline()
        final_model.fit(final_training[FEATURE_COLUMNS], final_training["target"])

        test_prediction = final_model.predict(test_rows[FEATURE_COLUMNS])
        test_metrics = calculate_metrics(test_rows["target"], test_prediction)
        baseline_test = calculate_metrics(test_rows["target"], test_rows["pm25_t"])

        metrics = [
            _metric_row("validation", validation_metrics),
            _metric_row("test", test_metrics),
        ]
        meta.update({
            "train_origins": int(len(train_rows)),
            "validation_origins": int(len(validation_rows)),
            "test_origins": int(len(test_rows)),
            "train_examples": int(len(train_rows)),
            "validation_examples": int(len(validation_rows)),
            "test_examples": int(len(test_rows)),
            "training_target_end": pd.Timestamp(final_training["target_at"].max()).isoformat(),
            "baseline_metrics": {
                "validation": baseline_validation,
                "test": baseline_test,
            },
            "model_parameters": PARAMETERS,
            "model_selection": (
                "HistGradientBoosting seleccionado en el notebook 04 mediante "
                "RandomizedSearchCV con TimeSeriesSplit; la API reutiliza sus parámetros."
            ),
            "simulated_observations": simulated_count,
        })
        return final_model, origins, metrics, meta, None

    first_simulated = data.loc[data["is_simulated"], "observed_at"].min()
    meta = split_retrain(origins, simulated_count, first_simulated)
    train_rows = origins.loc[origins["split"] == "train"]
    validation_rows = origins.loc[origins["split"] == "validation"]

    if len(train_rows) < 100 or len(validation_rows) < 2:
        raise ValueError(
            "No hay suficientes ejemplos para entrenar y evaluar el candidato."
        )

    candidate = create_pipeline()
    candidate.fit(train_rows[FEATURE_COLUMNS], train_rows["target"])

    candidate_metrics = calculate_metrics(
        validation_rows["target"],
        candidate.predict(validation_rows[FEATURE_COLUMNS]),
    )
    incumbent_metrics = (calculate_metrics(
        validation_rows["target"],
        incumbent.predict(validation_rows[FEATURE_COLUMNS]),
    ) if incumbent_compatible else None)
    baseline_metrics = calculate_metrics(
        validation_rows["target"],
        validation_rows["pm25_t"],
    )


    promotion_eligible = bool(meta["promotion_eligible"] and incumbent_compatible)

    promoted = (
        promotion_eligible
        and candidate_metrics["mae"] < incumbent_metrics["mae"]
        and candidate_metrics["rmse"] <= incumbent_metrics["rmse"]
        and candidate_metrics["mae"] < baseline_metrics["mae"]
    )

    if not promotion_eligible:
        promotion_reason = (
            "El modelo activo pertenece a un protocolo previo y podría haber "
            "utilizado la referencia histórica durante su ajuste. No se "
            "compara ni se promueve automáticamente."
        )
    elif promoted:
        promotion_reason = (
            "El candidato mejoró en la referencia histórica común frente al "
            "modelo activo y la persistencia; promoción solo demostrativa."
        )
    else:
        promotion_reason = (
            "El candidato no cumplió los criterios de mejora."
        )

    comparison = {
        "candidate": candidate_metrics,
        "incumbent": incumbent_metrics,
        "persistence": baseline_metrics,
        "promotion_eligible": promotion_eligible,
        "promoted": bool(promoted),
        "promotion_reason": promotion_reason,
        "mae_improvement_vs_incumbent_pct": (
            (incumbent_metrics["mae"] - candidate_metrics["mae"])
            / incumbent_metrics["mae"] * 100
            if incumbent_metrics and incumbent_metrics["mae"] else None
        ),
    }


    metrics = [_metric_row("retrain_validation", candidate_metrics)]
    meta.update({
        "train_origins": int(len(train_rows)),
        "validation_origins": int(len(validation_rows)),
        "train_examples": int(len(train_rows)),
        "validation_examples": int(len(validation_rows)),
        "training_target_end": pd.Timestamp(train_rows["target_at"].max()).isoformat(),
        "baseline_metrics": {"retrain_validation": baseline_metrics},
        "comparison": comparison,
        "model_parameters": PARAMETERS,
        "simulated_observations": simulated_count,
        "promotion_eligible": promotion_eligible,
        "promotion_reason": promotion_reason,
    })
    return candidate, origins, metrics, meta, comparison
