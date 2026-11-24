# Pronóstico de PM2.5 — Beijing Wanliu

Aplicación web que utiliza un modelo de aprendizaje automático para pronosticar la concentración de PM2.5 durante las próximas 24 horas en la estación Wanliu, Beijing.

El proyecto incluye una API REST desarrollada con FastAPI, una interfaz web y una base de datos PostgreSQL.

## 1. Requisitos

Para ejecutar el proyecto se necesita:

- Python 3.13
- PostgreSQL 15
- Git
- Un navegador web

Los siguientes comandos están preparados para Windows PowerShell.

## 2. Descargar el proyecto

Abrir PowerShell y ejecutar:

```powershell
git clone --branch develop https://github.com/JorgeC935/beijing-pm25-crisp-mlq.git
cd beijing-pm25-crisp-mlq
```

### Crear el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Después, volver a activar el entorno virtual.

### Instalar las dependencias

```powershell
python -m pip install -r requirements_api.txt
```

Para trabajar también con los notebooks de entrenamiento:

```powershell
python -m pip install -r requirements.txt
```

## 3. Configurar PostgreSQL

La aplicación utiliza la base de datos `beijing_pm25` y el usuario `beijing_app`.

### Crear el usuario y la base de datos

Abrir PowerShell y ejecutar:

```powershell
& "C:\Program Files\PostgreSQL\15\bin\psql.exe" -h 127.0.0.1 -U postgres -d postgres -W
```

Dentro de PostgreSQL, ejecutar:

```sql
CREATE ROLE beijing_app LOGIN;
```

Establecer una contraseña:

```text
\password beijing_app
```

Crear la base de datos:

```sql
CREATE DATABASE beijing_pm25 OWNER beijing_app;
```

Salir de PostgreSQL:

```text
\q
```

Si el usuario y la base de datos ya existen, omitir este paso.

### Crear las tablas

Desde la carpeta principal del proyecto, ejecutar:

```powershell
$psql = "C:\Program Files\PostgreSQL\15\bin\psql.exe"

& $psql -h 127.0.0.1 -U beijing_app -d beijing_pm25 -W -v ON_ERROR_STOP=1 -f ".\database\01_schema.sql"

& $psql -h 127.0.0.1 -U beijing_app -d beijing_pm25 -W -v ON_ERROR_STOP=1 -f ".\database\02_api.sql"
```

PostgreSQL solicitará la contraseña del usuario `beijing_app`.

**Nota:** si PostgreSQL está instalado en otra ubicación, ajustar la ruta de `psql.exe`.

## 4. Descargar e importar los datos

Descargar el dataset **Beijing Multi-Site Air Quality** desde:

https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata

Guardar el archivo ZIP descargado en:

```text
data/raw/beijing+multi+site+air+quality+data.zip
```

Si la carpeta no existe, crearla:

```powershell
New-Item -ItemType Directory -Force ".\data\raw"
```

No es necesario descomprimir el archivo.

Para importar los datos a PostgreSQL:

```powershell
python src/importar_datos.py
```

El programa solicitará la contraseña de la base de datos.

Una importación correcta debe registrar **12 estaciones y 420.768 observaciones**.

La importación inicial se realiza una sola vez, sobre una base de datos sin observaciones.

## 5. Ejecutar la aplicación

Se deben abrir dos terminales desde la carpeta del proyecto.

### Terminal 1 — API REST

Activar el entorno virtual, configurar la contraseña e iniciar FastAPI:

```powershell
.\.venv\Scripts\Activate.ps1
$env:BEIJING_DB_PASSWORD = Read-Host "Contraseña de PostgreSQL"
python -m uvicorn src.api:app --reload
```

La API estará disponible en:

http://127.0.0.1:8000

La documentación interactiva de Swagger estará en:

http://127.0.0.1:8000/docs

### Terminal 2 — Interfaz web

Ejecutar:

```powershell
python -m http.server 5500 --directory frontend
```

Abrir la aplicación en:

**http://127.0.0.1:5500**

Mantener ambas terminales abiertas mientras se utiliza el sistema.

## 6. Utilizar la aplicación

Desde la interfaz web se pueden realizar las siguientes operaciones:

1. **Generar pronóstico:** ingresar una medición ambiental y obtener las predicciones de PM2.5 para las siguientes 24 horas.
2. **Consultar historial:** ver las mediciones ingresadas y los pronósticos anteriores.
3. **Ver métricas:** consultar el rendimiento y la versión del modelo.
4. **Cargar CSV:** agregar nuevas mediciones a la base de datos.
5. **Reentrenar:** solicitar la actualización del modelo cuando existan suficientes datos nuevos válidos.

La aplicación trabaja con una serie histórica. Por eso, antes de introducir una medición, se debe consultar la **siguiente fecha y hora esperada**.

En una base recién importada, normalmente será:

```text
2017-03-01T00:00:00
```

Las siguientes mediciones deben ingresarse en orden, avanzando exactamente una hora.

### Formato para cargar un CSV

El archivo debe incluir, como mínimo, `observed_at` y `pm25`. Se recomienda incluir todas las variables ambientales:

```csv
observed_at,pm25,pm10,so2,no2,co,o3,temp,pres,dewp,rain,wspm,wd
2017-03-01T00:00:00,75,100,10,35,600,45,8,1015,-5,0,2,NW
2017-03-01T01:00:00,76,102,11,36,610,44,8,1015,-5,0,2.1,NW
```

Los valores anteriores son solamente un ejemplo de prueba.

La primera fecha debe coincidir con la hora esperada por la aplicación.

El archivo debe contener como máximo 1.000 registros, no superar 3 MB y mantener horas consecutivas.

La carga CSV incorpora observaciones, pero no genera pronósticos ni reentrena automáticamente.

## 7. Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/state` | Consultar la siguiente hora esperada |
| POST | `/predict` | Generar un pronóstico de 24 horas |
| GET | `/predictions` | Consultar el historial |
| GET | `/metrics` | Consultar las métricas del modelo |
| POST | `/training-data` | Subir nuevas mediciones CSV |
| POST | `/retrain` | Solicitar reentrenamiento |

Todos pueden probarse desde Swagger:

http://127.0.0.1:8000/docs

## 8. Si ocurre algún problema

**No conecta con PostgreSQL:** verificar que el servicio esté iniciado, que la base de datos exista y que `BEIJING_DB_PASSWORD` se haya configurado en la terminal de FastAPI.

**La fecha ingresada no es válida:** consultar `GET /state` e ingresar exactamente la hora indicada.

**El frontend no obtiene resultados:** comprobar que FastAPI continúa ejecutándose en el puerto 8000.

**No se encuentra el modelo:** verificar que exista el archivo `models/wanliu_pm25_24h.joblib`.

---

**Proyecto académico — Taller de Sistemas Inteligentes**

Repositorio: https://github.com/JorgeC935/beijing-pm25-crisp-mlq
