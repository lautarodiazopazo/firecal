# Pronóstico mensual de anomalías de fuego

Pregunta: **¿el mes que viene tendrá más fuego de lo normal en esta celda de 0,5°?** (probabilidad)

Correr desde la carpeta raíz del proyecto, con el entorno activado:

```bash
pip install -r pronostico/requirements.txt
python pronostico/era5_a_grilla.py --input Sudamerica/ERAS5-monthly --output data/era5_05.parquet
python pronostico/construir_tabla.py
python pronostico/entrenar.py
```

Resultados en `data/pronostico/`: `metricas.csv`, `importancias.csv`, `predicciones.parquet`,
`pronostico.parquet` (próximo mes), figuras `metricas.png`, `importancias.png`, `pronostico.png`
y los modelos (`lgb_fuego.txt`, `lgb_clima.txt`, `mlp.pt`).

Separación temporal: entrenamiento 2003–2019, validación 2020–2022, test 2023–2025.
Referencias: climatología y persistencia. Fuentes: serie armonizada FIRMS (este repo),
ERA5-Land monthly averaged (Copernicus C3S), ONI (NOAA CPC).
