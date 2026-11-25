-- Esquema para Wanliu Air Forecast: pronóstico único de PM2.5 a +6 horas.
-- Ejecutar únicamente al crear una base nueva y vacía.
BEGIN;

CREATE TABLE IF NOT EXISTS data_uploads (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('historical', 'simulation')),
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    first_time TIMESTAMP NOT NULL,
    last_time TIMESTAMP NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (last_time >= first_time)
);

CREATE TABLE IF NOT EXISTS observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMP NOT NULL UNIQUE,
    pm25 DOUBLE PRECISION CHECK (pm25 >= 0),
    pm10 DOUBLE PRECISION CHECK (pm10 >= 0),
    so2 DOUBLE PRECISION CHECK (so2 >= 0),
    no2 DOUBLE PRECISION CHECK (no2 >= 0),
    co DOUBLE PRECISION CHECK (co >= 0),
    o3 DOUBLE PRECISION CHECK (o3 >= 0),
    temp DOUBLE PRECISION,
    pres DOUBLE PRECISION CHECK (pres > 0),
    dewp DOUBLE PRECISION,
    rain DOUBLE PRECISION CHECK (rain >= 0),
    wd TEXT,
    wspm DOUBLE PRECISION CHECK (wspm >= 0),
    source TEXT NOT NULL CHECK (source IN ('historical', 'manual', 'csv')),
    is_simulated BOOLEAN NOT NULL,
    upload_id BIGINT REFERENCES data_uploads(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (date_trunc('hour', observed_at) = observed_at),
    CHECK (
        (source = 'historical' AND NOT is_simulated) OR
        (source IN ('manual', 'csv') AND is_simulated AND pm25 IS NOT NULL)
    ),
    CHECK (
        (source = 'manual' AND upload_id IS NULL) OR
        (source IN ('historical', 'csv') AND upload_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS training_runs (
    id BIGSERIAL PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('initial', 'retrain')),
    mode TEXT NOT NULL CHECK (mode IN ('historical', 'simulation')),
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed')),
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    split_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    observation_count INTEGER CHECK (observation_count >= 0),
    new_observation_count INTEGER CHECK (new_observation_count >= 0),
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS processed_observations (
    training_run_id BIGINT NOT NULL REFERENCES training_runs(id),
    observation_id BIGINT NOT NULL REFERENCES observations(id),
    features JSONB NOT NULL CHECK (jsonb_typeof(features) = 'object'),
    target DOUBLE PRECISION NOT NULL CHECK (target >= 0),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test', 'unused')),
    PRIMARY KEY (training_run_id, observation_id)
);

CREATE TABLE IF NOT EXISTS models (
    id BIGSERIAL PRIMARY KEY,
    training_run_id BIGINT NOT NULL UNIQUE REFERENCES training_runs(id),
    version TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    artifact_path TEXT NOT NULL UNIQUE,
    feature_columns JSONB NOT NULL CHECK (jsonb_typeof(feature_columns) = 'array'),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    is_demo BOOLEAN NOT NULL DEFAULT FALSE,
    horizon_hours SMALLINT NOT NULL DEFAULT 6 CHECK (horizon_hours IN (6, 24)),
    training_target_end TIMESTAMP NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_models_active
    ON models (is_active) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_models_horizon
    ON models (horizon_hours, id DESC);

CREATE TABLE IF NOT EXISTS model_metrics (
    model_id BIGINT NOT NULL REFERENCES models(id),
    evaluation_set TEXT NOT NULL
        CHECK (evaluation_set IN ('validation', 'test', 'retrain_validation')),
    scope TEXT NOT NULL CHECK (scope IN ('global', 'block', 'horizon')),
    horizon_start SMALLINT NOT NULL CHECK (horizon_start BETWEEN 1 AND 24),
    horizon_end SMALLINT NOT NULL CHECK (horizon_end BETWEEN 1 AND 24),
    mae DOUBLE PRECISION NOT NULL CHECK (mae >= 0),
    rmse DOUBLE PRECISION NOT NULL CHECK (rmse >= 0),
    r2 DOUBLE PRECISION,
    sample_count INTEGER NOT NULL CHECK (sample_count > 0),
    PRIMARY KEY (model_id, evaluation_set, scope, horizon_start, horizon_end),
    CHECK (horizon_start <= horizon_end)
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id BIGSERIAL PRIMARY KEY,
    observation_id BIGINT NOT NULL REFERENCES observations(id),
    model_id BIGINT NOT NULL REFERENCES models(id),
    input_data JSONB NOT NULL CHECK (jsonb_typeof(input_data) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prediction_runs_origin
    ON prediction_runs (observation_id);

CREATE TABLE IF NOT EXISTS prediction_results (
    prediction_run_id BIGINT NOT NULL REFERENCES prediction_runs(id),
    horizon SMALLINT NOT NULL CHECK (horizon BETWEEN 1 AND 24),
    target_at TIMESTAMP NOT NULL,
    pm25_predicted DOUBLE PRECISION NOT NULL CHECK (pm25_predicted >= 0),
    PRIMARY KEY (prediction_run_id, horizon)
);
CREATE INDEX IF NOT EXISTS idx_prediction_results_target
    ON prediction_results (target_at);

COMMIT;
