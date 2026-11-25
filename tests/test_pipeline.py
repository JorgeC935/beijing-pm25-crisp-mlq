import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS,
    HORIZON_HOURS,
    build_features,
    prediction_features,
    prepare_supervised,
)
from src.train_model import (
    calculate_metrics,
    split_initial,
    split_retrain,
)


def make_data(n=240, simulated_from=None):
    dates = pd.date_range("2017-01-01", periods=n, freq="h")
    frame = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "observed_at": dates,
        "pm25": 30 + 10 * np.sin(np.arange(n) / 8) + np.arange(n) * 0.02,
        "pm10": 50 + np.arange(n) * 0.01,
        "so2": 5.0,
        "no2": 20.0,
        "co": 400.0,
        "o3": 30.0,
        "temp": 10 + np.sin(np.arange(n) / 24),
        "pres": 1015.0,
        "dewp": 2.0,
        "rain": 0.0,
        "wd": "NW",
        "wspm": 2.0,
        "is_simulated": False,
    })
    if simulated_from is not None:
        frame.loc[frame.index >= simulated_from, "is_simulated"] = True
    return frame


def test_feature_contract_matches_six_hour_notebook():
    data = make_data(80)
    features = build_features(data)
    assert len(FEATURE_COLUMNS) == 32
    assert features.loc[30, "pm25_t"] == pytest.approx(data.loc[30, "pm25"])
    assert features.loc[30, "pm25_lag_6"] == pytest.approx(data.loc[24, "pm25"])
    assert features.loc[30, "pm25_lag_24"] == pytest.approx(data.loc[6, "pm25"])


def test_supervised_target_is_exactly_six_hours_ahead():
    data = make_data(80)
    origins = prepare_supervised(data)
    assert HORIZON_HOURS == 6
    assert len(origins) == 80 - 6
    first = origins.iloc[0]
    assert first["target"] == pytest.approx(data.loc[6, "pm25"])
    assert first["target_at"] == data.loc[6, "observed_at"]


def test_prediction_features_are_one_row_with_same_feature_order():
    data = make_data(80)
    latest = prediction_features(data.tail(25))
    assert latest.shape == (1, 32)
    assert latest.columns.tolist() == FEATURE_COLUMNS
    assert latest.iloc[0]["pm25_t"] == pytest.approx(data.iloc[-1]["pm25"])
    assert latest.iloc[0]["pm25_lag_6"] == pytest.approx(data.iloc[-7]["pm25"])


def test_prediction_requires_25_hourly_observations():
    with pytest.raises(ValueError, match="25 observaciones"):
        prediction_features(make_data(24))


def test_initial_split_does_not_leak_future_targets():
    origins = prepare_supervised(make_data(500))
    meta = split_initial(origins)
    train = origins[origins["split"] == "train"]
    validation = origins[origins["split"] == "validation"]
    test = origins[origins["split"] == "test"]
    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert train["target_at"].max() < validation["observed_at"].min()
    assert validation["target_at"].max() < test["observed_at"].min()
    assert meta["horizon_hours"] == 6


@pytest.mark.parametrize("new_count", [1, 2, 7, 37, 60])
def test_retraining_uses_fixed_reference_and_new_targets(new_count):
    historical_count = 600
    data = make_data(historical_count + new_count, simulated_from=historical_count)
    origins = prepare_supervised(data)
    meta = split_retrain(origins, new_count, data.loc[historical_count, "observed_at"])
    original = prepare_supervised(data.iloc[:historical_count])
    split_initial(original)
    train_rows = origins[origins["split"] == "train"]
    reference = origins[origins["split"] == "validation"]
    new_train = train_rows[train_rows["target_is_simulated"]]
    historical_train = train_rows[~train_rows["target_is_simulated"]]
    assert len(new_train) == new_count
    assert meta["new_training_examples"] == new_count
    assert set(historical_train.id) == set(original.loc[original["split"].isin(["train", "validation"]), "id"])
    assert set(reference.id).issubset(set(original.loc[original["split"] == "test", "id"]))
    assert not reference["target_is_simulated"].any()
    assert historical_train["target_at"].max() < reference["observed_at"].min()
    assert reference["target_at"].max() < new_train["observed_at"].min() - pd.Timedelta(hours=24)
    assert meta["promotion_eligible"] is True
    assert meta["independent_test"] is False
    if new_count == 1:
        assert not new_train.iloc[0]["is_simulated"]


def test_reference_does_not_change_when_new_data_arrives():
    references = []
    for n in (1, 7, 60):
        data = make_data(600 + n, simulated_from=600)
        origins = prepare_supervised(data)
        split_retrain(origins, n, data.loc[600, "observed_at"])
        references.append(origins.loc[origins["split"] == "validation", "id"].tolist())
    assert references[0] == references[1] == references[2]


def test_retraining_without_new_targets_is_rejected():
    data = make_data(600)
    with pytest.raises(ValueError, match="no hay nuevos objetivos"):
        split_retrain(prepare_supervised(data), 0)


def test_retraining_missing_origin_is_not_a_label():
    data = make_data(601, simulated_from=600)
    data.loc[594, "pm25"] = np.nan
    with pytest.raises(ValueError, match="no hay nuevos objetivos"):
        split_retrain(prepare_supervised(data), 1, data.loc[600, "observed_at"])


def test_metrics():
    actual = np.arange(20, dtype=float)
    predicted = actual + 2
    result = calculate_metrics(actual, predicted)
    assert result["mae"] == pytest.approx(2)
    assert result["rmse"] == pytest.approx(2)
    assert result["sample_count"] == 20
    assert result["r2"] < 1


def test_metrics_single_example():
    result = calculate_metrics([10], [12])
    assert result["r2"] is None
    assert result["mae"] == 2
    assert result["rmse"] == 2


@pytest.mark.parametrize("compatible", [True, False])
def test_candidate_trained_with_one_new_label_and_comparable_metrics(compatible):
    from src.train_model import train

    data = make_data(601, simulated_from=600)
    incumbent, _, _, _, _ = train(data.iloc[:600])
    candidate, origins, metrics, meta, comparison = train(
        data, incumbent=incumbent, incumbent_compatible=compatible
    )
    assert meta["new_training_examples"] == 1
    assert metrics[0]["sample_count"] == meta["validation_origins"]
    assert np.isfinite(candidate.predict(origins.loc[origins["split"] == "validation", FEATURE_COLUMNS])).all()
    assert comparison["promotion_eligible"] is compatible
    if compatible:
        assert comparison["incumbent"]["sample_count"] == metrics[0]["sample_count"]
        assert comparison["promoted"] == (
            comparison["candidate"]["mae"] < comparison["incumbent"]["mae"]
            and comparison["candidate"]["rmse"] <= comparison["incumbent"]["rmse"]
            and comparison["candidate"]["mae"] < comparison["persistence"]["mae"]
        )
    else:
        assert comparison["incumbent"] is None
        assert not comparison["promoted"]
