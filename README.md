# Wanliu Air Forecast

Aplicación académica para estimar el PM2.5 de la estación Wanliu exactamente **6 horas después** de registrar una medición.

## Requisitos

- Python 3.13
- PostgreSQL 15 o compatible
- Git y navegador web
- Windows PowerShell (los comandos siguientes están escritos para Windows)

## 1. Instalar dependencias

```powershell
git clone https://github.com/JorgeC935/beijing-pm25-crisp-mlq.git
cd beijing-pm25-crisp-mlq
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_api.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Para abrir y ejecutar los notebooks de investigación, instala también:

```powershell
python -m pip install -r requirements.txt
```

## 2. Preparar PostgreSQL

### Base de datos nueva

Conéctate como administrador `postgres`:

```powershell
$psql = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
& $psql -h 127.0.0.1 -U postgres -d postgres -W
```

En `psql`:

```sql
CREATE ROLE beijing_app LOGIN;
\password beijing_app
CREATE DATABASE wanliu_pm25_6h OWNER beijing_app;
\q
```

Si el usuario o la base ya existen, no repitas sus instrucciones.

Desde la raíz del proyecto, crea las tablas:

```powershell
& $psql -h 127.0.0.1 -U beijing_app -d wanliu_pm25_6h -W -v ON_ERROR_STOP=1 -f ".\database\schema.sql"
```

Si tienes una base anterior de 24 horas (`wanliu_pm25`), consérvala como respaldo. No ejecutes la migración sobre `wanliu_pm25_6h` recién creada. Si ya existe esta última y tiene el esquema de seis horas, no repitas la creación ni la importación.

## 3. Ejecutar la aplicación

Abre dos terminales desde la carpeta del proyecto.

**Terminal 1 — API REST:**

Copia `.env.example` a `.env` si todavía no existe, y escribe tu contraseña local en `.env`. No subas este archivo a GitHub.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

En una terminal nueva:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api:app --reload --port 8000 --env-file .env
```

Alternativamente, puedes usar variables de entorno temporales (sin `.env`):

```powershell
$env:BEIJING_DB_NAME = "wanliu_pm25_6h"
$env:BEIJING_DB_PASSWORD = Read-Host "Contraseña de beijing_app"
python -m uvicorn src.api:app --reload --port 8000
```

Estas variables se pierden al cerrar PowerShell y tienen prioridad sobre el archivo `.env`.

Swagger: http://127.0.0.1:8000/docs

**Terminal 2 — interfaz web:**

```powershell
python -m http.server 5500 --directory frontend
```

Abre http://127.0.0.1:5500. Mantén ambas terminales abiertas.

## 4. Entrenamiento inicial

1. Abre la pestaña **Entrenamiento**.
2. Si PostgreSQL ya contiene el histórico de Wanliu, pulsa **Entrenar con histórico disponible**.
3. Si la base está vacía, selecciona el archivo `PRSA_Data_Wanliu_20130301-20170228.csv` y pulsa **Importar CSV y entrenar**.
4. Espera a que finalice el proceso. El modelo y sus métricas quedarán registrados en PostgreSQL y el artefacto se guardará en `models/wanliu_6h/`.

El entrenamiento inicial utiliza solamente el histórico original. Las mediciones simuladas que ya estén guardadas no se mezclan en este primer ajuste.

## 5. Uso

- **Pronóstico:** registra una medición individual. El sistema asigna automáticamente la siguiente fecha y hora y devuelve una estimación de PM2.5 a +6 horas.
- **Métricas:** consulta MAE, RMSE y R² del modelo y compáralos con persistencia.
- **Historial:** consulta cada medición ingresada y su único pronóstico. La observación posterior y el error aparecen cuando se registra el valor de la fecha objetivo.
- **Reentrenamiento:** desde una medición nueva, si existe un origen válido seis horas antes, se puede ajustar un candidato desde cero con el mismo histórico de ajuste inicial (entrenamiento + validación) y todos los ejemplos cuyo objetivo nuevo ya se conoce, incluidos los que tienen origen histórico. No se exige un mínimo de 7, 30 ni 37 mediciones. Se compara el candidato, el modelo activo compatible y persistencia sobre una referencia histórica común: parte fija del conjunto de prueba inicial, excluyendo las últimas 30 horas de contexto respecto a la primera medición simulada (6 h de horizonte + 24 h de rezagos). La referencia y sus objetivos no entran al ajuste. La promoción demostrativa requiere MAE menor que el activo y persistencia, y RMSE no superior al activo. Un modelo activo de protocolo anterior no compatible se conserva sin comparación contra él. No se garantiza promoción. La referencia es retrospectiva y se reutiliza en solicitudes sucesivas: no constituye una prueba independiente del rendimiento futuro ni una validación con datos ambientales reales. Todos los candidatos con datos simulados se identifican como demostrativos.

La interfaz no importa CSV de mediciones simuladas. El único CSV que se carga desde ella es el dataset histórico del entrenamiento inicial.

## 6. Notebooks

Los notebooks `01`, `02` y `03` documentan la exploración, preparación y modelado inicial. El notebook `04_Modelado_Wanliu_PM25_6h.ipynb` contiene la experimentación del objetivo a seis horas, comparación de modelos y búsqueda de hiperparámetros.

La API utiliza las 32 características y los hiperparámetros seleccionados en el notebook 04. No es necesario ejecutar Jupyter para utilizar la aplicación.

## 7. Endpoints principales

| Método | Ruta | Función |
|---|---|---|
| GET | `/state` | Estado del sistema y siguiente hora |
| POST | `/train-initial` | Importar histórico si hace falta y entrenar |
| POST | `/predict` | Registrar medición y pronosticar a +6 h |
| GET | `/predictions` | Historial de pronósticos a +6 h |
| GET | `/metrics` | Métricas del modelo |
| POST | `/retrain` | Entrenar y evaluar un candidato |
| GET | `/models` | Versiones de modelos de seis horas |

## 8. Pruebas

Con el entorno virtual activo:

```powershell
python -m compileall -q src tests
python -m pytest -q
```

## Solución de problemas

- Si falta `BEIJING_DB_PASSWORD`, verifica el archivo `.env` y el argumento `--env-file .env`, o define la variable en la misma terminal donde ejecutas Uvicorn.
- Si aparece un error por `horizon_hours` o `processed_observations.target`, verifica que la API apunta a `wanliu_pm25_6h`, no a la base anterior. No ejecutes una migración sin revisar primero el esquema.
- Si la interfaz no conecta, comprueba el puerto de Uvicorn y selecciona el mismo en la parte superior.
- Si falta el archivo del modelo activo, revisa el registro en PostgreSQL y `models/wanliu_6h/`.

Proyecto académico. Utiliza datos históricos y mediciones simuladas; no es un sistema conectado a sensores ambientales en tiempo real.
