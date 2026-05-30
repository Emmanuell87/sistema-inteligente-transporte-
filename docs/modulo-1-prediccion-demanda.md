# Módulo 1 - Predicción de Demanda

Este módulo entrena y evalúa modelos LSTM para demanda de BUS y METRO, y guarda métricas, predicciones y gráficas en `outputs/`.

## Archivos principales

- `main.py`
- `src/lstm_model.py`
- `src/data_loader.py`
- `src/exploratory.py`
- `src/utils.py`

## Por qué están separados

El Módulo 1 se divide por responsabilidades para mantener el código escalable y fácil de extender:

- `main.py` orquesta el flujo completo y define rutas de salida.
- `data_loader.py` se dedica a carga y preparación de series temporales.
- `exploratory.py` genera gráficas de análisis exploratorio.
- `lstm_model.py` concentra la lógica del modelo LSTM y el forecast.
- `utils.py` centraliza el guardado de métricas y predicciones.

## Entrenamiento y generación de salidas

Ejecutar el flujo completo:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe main.py
```

Solo regenerar gráficas:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe main.py --only-plots
```

Mostrar gráficas en ventana y además guardarlas:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe main.py --show-plots
```

## Salidas esperadas

- `outputs/metricas_modelos.csv`
- `outputs/prediccion_bus_30dias.csv`
- `outputs/prediccion_metro_30dias.csv`
- `outputs/img/bus/...`
- `outputs/img/metro/...`
