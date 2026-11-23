# API Wanliu PM2.5 — Instrucciones de integración

Copia **solamente** los archivos de este paquete a la raíz existente de `beijing-pm25-crisp-mlq`. No reemplaces el directorio `src` ni `database` completo: combina sus contenidos. **No elimines** `database/01_schema.sql`, `src/importar_datos.py`, tus notebooks ni el archivo `models/wanliu_pm25_24h.joblib`.

## Paso A — Git

Después de fusionar el Pull Request `feature/modelado` → `develop` en GitHub:

```powershell
git switch develop
git pull --ff-only origin develop
git switch -c feature/api
git push -u origin feature/api
```

## Paso B — Paquetes

Desde la raíz del repositorio, con `.venv` activado:

```powershell
python -m pip install -r requirements_api.txt
```

## Paso C — Crear las tres tablas nuevas

No vuelve a crear ni modifica las tablas originales `stations` y `observations`.

```powershell
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -h 127.0.0.1 -p 5432 -U beijing_app -d beijing_pm25 -W -v ON_ERROR_STOP=1 -f ".\database\02_api.sql"
```

## Paso D — Iniciar FastAPI

En la MISMA terminal donde arrancarás el servidor, define la contraseña. Se puede introducir con `Read-Host` para no escribirla en el comando:

```powershell
$env:BEIJING_DB_PASSWORD = Read-Host "Contrasena de PostgreSQL"
python -m uvicorn src.api:app --reload
```

Abre http://127.0.0.1:8000/docs y verifica los endpoints `/state`, `/predict`, `/predictions`, `/metrics`, `/training-data`, `/retrain`.

## Paso E — Iniciar la interfaz web

En una SEGUNDA terminal, con el mismo directorio de proyecto:

```powershell
python -m http.server 5500 --directory frontend
```

Abre http://127.0.0.1:5500 y prueba los cinco botones. El botón «Comprobar API» rellena automáticamente la fecha de la próxima observación para Wanliu.

## Ejemplo de predicción con Swagger

Usa `GET /state` primero y coloca exactamente `next_expected_observed_at` en `observed_at`. La última hora original de Wanliu suele estar en febrero de 2017, así que **no** envíes la fecha actual de 2026. Ejemplo ilustrativo (ajusta la fecha a la que devuelve `/state`):

```json
{
  "observed_at": "2017-03-01T00:00:00",
  "pm25": 75,
  "pm10": 100,
  "so2": 10,
  "no2": 35,
  "co": 600,
  "o3": 45,
  "temp": 8,
  "pres": 1015,
  "dewp": -5,
  "rain": 0,
  "wspm": 2,
  "wd": "NW"
}
```

## CSV para carga nueva

Una fila por hora, fechas continuas **desde la hora que devuelve `/state`**, máximo 1.000 registros por archivo. Encabezado básico: `observed_at,pm25,pm10,so2,no2,co,o3,temp,pres,dewp,rain,wspm,wd`. Los valores faltantes de variables ambientales opcionales se imputan por el Pipeline entrenado; `pm25` y `observed_at` son obligatorios. `observed_at` usa la hora local de Beijing y no lleva zona horaria.

La carga CSV registra mediciones, pero no genera automáticamente una predicción. `POST /predict` agrega una medición y genera 24 pronósticos; `POST /retrain` intenta ajustar una nueva versión y solo la promueve si mejora el MAE de validación.

## Modelo y controles

- Se usa el artefacto versionado `models/wanliu_pm25_24h.joblib` para iniciar. Cuando se aprueba un reentrenamiento, se genera `models/wanliu_pm25_latest.joblib`. Es conveniente ignorar este segundo archivo en Git hasta decidir si se quiere publicar una nueva versión.
- El entrenamiento para reentrenar usa todos los orígenes que tienen 48 horas continuas previas y 24 PM2.5 futuros reales. Divide cronológicamente y compara el candidato con el modelo anterior en el mismo período. Si mejora, lo reajusta con todos los ejemplos y lo publica.
- La comparación de reentrenamiento es adecuada para la **demostración académica**, no una validación independiente de un sistema productivo, particularmente después de múltiples reentrenamientos.
- Los comandos de arranque usan un solo proceso de Uvicorn. No levantes varios `--workers` simultáneos mientras ejecutas reentrenamiento.
- El archivo original `.joblib` fue generado desde tus notebooks; una incompatibilidad de versiones scikit-learn requiere instalar una versión compatible con la del entrenamiento.

## Nota importante

Este paquete se revisó sintácticamente y está pensado para coincidir con la estructura de los notebooks que compartiste. **No se ha probado todavía contra tu instancia local de PostgreSQL ni contra tu archivo `.joblib` final**, porque esos servicios y archivos están en tu equipo. Antes de darlo por terminado, comprueba en Swagger el flujo completo y revisa cualquier error de compatibilidad con el esquema SQL real.
