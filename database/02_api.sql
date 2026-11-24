CREATE TABLE IF NOT EXISTS prediction_runs (
    id BIGSERIAL PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES stations(id),
    observed_at TIMESTAMP NOT NULL,
    input_data JSONB NOT NULL,
    forecast JSONB NOT NULL,
    model_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prediction_runs_station_id ON prediction_runs(station_id, id DESC);

CREATE TABLE IF NOT EXISTS training_uploads (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    rows_inserted INTEGER NOT NULL CHECK (rows_inserted > 0),
    first_time TIMESTAMP NOT NULL,
    last_time TIMESTAMP NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retraining_runs (
    id BIGSERIAL PRIMARY KEY,
    promoted BOOLEAN NOT NULL,
    candidate_mae DOUBLE PRECISION NOT NULL,
    incumbent_mae DOUBLE PRECISION NOT NULL,
    sample_count INTEGER NOT NULL,
    validation_start TIMESTAMP NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
