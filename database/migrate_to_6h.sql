-- Migración de la base v2 (pronóstico 24h) a pronóstico único a +6h.
-- Ejecutar UNA SOLA VEZ sobre la base wanliu_pm25 existente.
-- Conserva observaciones, predicciones, ejecuciones y modelos antiguos como historial.

BEGIN;

-- Los modelos existentes pertenecen al sistema de 24 horizontes.
ALTER TABLE models
    ADD COLUMN IF NOT EXISTS horizon_hours SMALLINT;

UPDATE models
SET horizon_hours = 24
WHERE horizon_hours IS NULL;

ALTER TABLE models
    ALTER COLUMN horizon_hours SET DEFAULT 6,
    ALTER COLUMN horizon_hours SET NOT NULL;

ALTER TABLE models
    DROP CONSTRAINT IF EXISTS models_horizon_hours_check;

ALTER TABLE models
    ADD CONSTRAINT models_horizon_hours_check
    CHECK (horizon_hours IN (6, 24));

-- Se desactiva el modelo antiguo de 24 horas sin borrar su registro ni su archivo.
UPDATE models
SET is_active = FALSE
WHERE horizon_hours <> 6 AND is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_models_horizon
    ON models (horizon_hours, id DESC);

-- En la tabla procesada antigua se guardaba un array y_01 ... y_24.
-- Conservamos el objetivo y_06 como un único valor y retiramos el array.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'processed_observations'
          AND column_name = 'targets'
    ) THEN
        ALTER TABLE processed_observations
            ADD COLUMN IF NOT EXISTS target DOUBLE PRECISION;

        UPDATE processed_observations
        SET target = targets[6]
        WHERE target IS NULL;

        ALTER TABLE processed_observations
            ALTER COLUMN target SET NOT NULL;

        ALTER TABLE processed_observations
            DROP COLUMN targets;
    END IF;
END $$;

-- La versión anterior obligaba a que el registro global cubriera 1..24.
-- Ahora las nuevas métricas globales corresponden únicamente a +6h.
ALTER TABLE model_metrics
    DROP CONSTRAINT IF EXISTS model_metrics_check;

ALTER TABLE model_metrics
    DROP CONSTRAINT IF EXISTS model_metrics_horizon_range_check;

ALTER TABLE model_metrics
    ADD CONSTRAINT model_metrics_horizon_range_check
    CHECK (
        horizon_start BETWEEN 1 AND 24
        AND horizon_end BETWEEN 1 AND 24
        AND horizon_start <= horizon_end
    );

COMMIT;
