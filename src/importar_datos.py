from pathlib import Path
from zipfile import ZipFile
from io import BytesIO, TextIOWrapper
from datetime import datetime
from getpass import getpass
import csv
import psycopg

proyecto = Path(__file__).resolve().parents[1]
ruta_zip = proyecto / "data" / "raw" / "beijing+multi+site+air+quality+data.zip"

with ZipFile(ruta_zip) as archivo_externo:
    contenido = archivo_externo.read(
        "PRSA2017_Data_20130301-20170228.zip"
    )

def numero(valor):
    if valor is None or valor.strip() in ("", "NA"):
        return None
    return float(valor)

columnas = """
    station_id, observed_at, source_row_no,
    pm25, pm10, so2, no2, co, o3,
    temp, pres, dewp, rain, wd, wspm
"""

password = getpass("Contraseña de beijing_app: ")

with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="beijing_pm25",
    user="beijing_app",
    password=password
) as conexion:

    with conexion.cursor() as cursor:

        cursor.execute("SELECT COUNT(*) FROM observations")
        if cursor.fetchone()[0] != 0:
            raise ValueError("La tabla observations ya contiene datos.")

        with ZipFile(BytesIO(contenido)) as archivo_interno:

            archivos_csv = sorted(
                nombre for nombre in archivo_interno.namelist()
                if nombre.lower().endswith(".csv")
            )

            if len(archivos_csv) != 12:
                raise ValueError("Se esperaban 12 archivos CSV.")

            for nombre in archivos_csv:

                with archivo_interno.open(nombre) as archivo:
                    lector = csv.DictReader(
                        TextIOWrapper(archivo, encoding="utf-8-sig")
                    )

                    filas = iter(lector)
                    primera = next(filas)
                    estacion = primera["station"]

                    cursor.execute(
                        "INSERT INTO stations (name) VALUES (%s) "
                        "ON CONFLICT (name) DO NOTHING",
                        (estacion,)
                    )

                    cursor.execute(
                        "SELECT id FROM stations WHERE name = %s",
                        (estacion,)
                    )
                    estacion_id = cursor.fetchone()[0]

                    from itertools import chain

                    cantidad = 0

                    with cursor.copy(
                        f"COPY observations ({columnas}) FROM STDIN"
                    ) as copia:

                        for fila in chain([primera], filas):

                            fecha = datetime(
                                int(fila["year"]),
                                int(fila["month"]),
                                int(fila["day"]),
                                int(fila["hour"])
                            )

                            viento = fila["wd"]
                            if viento in ("", "NA"):
                                viento = None

                            copia.write_row((
                                estacion_id,
                                fecha,
                                int(fila["No"]),
                                numero(fila["PM2.5"]),
                                numero(fila["PM10"]),
                                numero(fila["SO2"]),
                                numero(fila["NO2"]),
                                numero(fila["CO"]),
                                numero(fila["O3"]),
                                numero(fila["TEMP"]),
                                numero(fila["PRES"]),
                                numero(fila["DEWP"]),
                                numero(fila["RAIN"]),
                                viento,
                                numero(fila["WSPM"])
                            ))

                            cantidad += 1

                print(estacion, cantidad)

        cursor.execute("SELECT COUNT(*) FROM stations")
        total_estaciones = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM observations")
        total_observaciones = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM observations WHERE pm25 IS NULL"
        )
        total_nulos_pm25 = cursor.fetchone()[0]

        if (total_estaciones, total_observaciones, total_nulos_pm25) != (
            12, 420768, 8739
        ):
            raise ValueError("Las cantidades no coinciden con el dataset.")

print("Estaciones:", total_estaciones)
print("Observaciones:", total_observaciones)
print("PM2.5 faltantes:", total_nulos_pm25)