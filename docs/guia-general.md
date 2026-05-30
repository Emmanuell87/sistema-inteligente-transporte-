# Guía general del proyecto

## Objetivo

Este proyecto integra tres soluciones de aprendizaje automático y una interfaz web única:

- Predicción de demanda de transporte (BUS y METRO) con LSTM.
- Clasificación de conducción distractiva a partir de imágenes.
- Sistema de recomendación híbrido de destinos.

La interfaz web centraliza los resultados para facilitar la revisión y la presentación pública.

## Arquitectura general

| Componente | Archivo principal | Rol |
| --- | --- | --- |
| Módulo 1 | `main.py` | Orquestar el pipeline de predicción y generar salidas. |
| Módulo 2 | `src/classification.py` | Entrenar/validar el clasificador y exportar reportes. |
| Módulo 3 | `src/recommender.py` | Generar recomendaciones y métricas del sistema. |
| Interfaz web | `streamlit_app.py` | Visualizar resultados y permitir exploración interactiva. |

## Módulo 1: predicción de demanda (estructura interna)

Los archivos del Módulo 1 están separados por responsabilidades para escalar el proyecto sin mezclar tareas:

- `main.py`: orquesta rutas, ejecuta el flujo completo y registra salidas.
- `src/data_loader.py`: carga datos crudos y los transforma en series temporales.
- `src/exploratory.py`: genera gráficas de análisis exploratorio.
- `src/lstm_model.py`: define, entrena y aplica el modelo LSTM.
- `src/utils.py`: guarda métricas y predicciones en CSV.

Separar estas piezas permite reutilizar funciones (por ejemplo, solo generar gráficas o solo entrenar el modelo), facilitar pruebas y mantener un código más legible.

## Entradas de datos

- Módulo 1: `data/raw/Bus_NYC.pkl`, `data/raw/Metro_NYC.pkl`.
- Módulo 2: `data/raw/distraccion_images/<clase>/*`.
- Módulo 3: `data/raw/recomendacion/Final_Updated_Expanded_Users.csv`, `Final_Updated_Expanded_UserHistory.csv`, `Final_Updated_Expanded_Reviews.csv`, `Expanded_Destinations.csv`.

## Salidas y responsables

### Módulo 1 (predicción)

| Salida | Generada por | Propósito |
| --- | --- | --- |
| `outputs/metricas_modelos.csv` | `main.py` + `src/utils.py` | Comparar desempeño BUS vs METRO (RMSE, MAE, MAPE). |
| `outputs/prediccion_bus_30dias.csv` | `main.py` | Serie de predicción para BUS (30 días). |
| `outputs/prediccion_metro_30dias.csv` | `main.py` | Serie de predicción para METRO (30 días). |
| `outputs/img/bus/exploratorio/*` | `src/exploratory.py` | Evidencia visual del análisis exploratorio de BUS. |
| `outputs/img/metro/exploratorio/*` | `src/exploratory.py` | Evidencia visual del análisis exploratorio de METRO. |
| `outputs/img/bus/prediccion/01_prediccion_30_dias_bus.png` | `src/lstm_model.py` | Gráfico forecast BUS para 30 días. |
| `outputs/img/metro/prediccion/01_prediccion_30_dias_metro.png` | `src/lstm_model.py` | Gráfico forecast METRO para 30 días. |
| `models/bus_nyc_lstm.h5` | `src/lstm_model.py` | Modelo LSTM entrenado para BUS. |
| `models/metro_nyc_lstm.h5` | `src/lstm_model.py` | Modelo LSTM entrenado para METRO. |
| `models/scaler_bus_nyc.pkl` | `src/lstm_model.py` | Escalador MinMax para BUS. |
| `models/scaler_metro_nyc.pkl` | `src/lstm_model.py` | Escalador MinMax para METRO. |

### Módulo 2 (clasificación)

| Salida | Generada por | Propósito |
| --- | --- | --- |
| `outputs/metricas_distraccion.csv` | `src/classification.py` | Resumen de accuracy, f1, precisión y recall. |
| `outputs/reporte_clasificacion_distraccion.csv` | `src/classification.py` | Reporte por clase (precision, recall, f1, support). |
| `outputs/matriz_confusion_distraccion.csv` | `src/classification.py` | Matriz de confusión en CSV. |
| `outputs/matriz_confusion_distraccion.png` | `src/classification.py` | Matriz de confusión como imagen. |
| `outputs/clases_distraccion.json` | `src/classification.py` | Mapa índice -> clase para inferencia. |
| `outputs/distraccion_images_split/` | `src/classification.py` | Split estratificado 80/10/10 generado automáticamente. |
| `outputs/img/distraccion/correct/` | `src/classification.py` | Ejemplos clasificados correctamente. |
| `outputs/img/distraccion/wrong/` | `src/classification.py` | Ejemplos con errores de clasificación. |
| `models/distraccion_mobilenet_best.keras` | `src/classification.py` | Mejor checkpoint por validación. |
| `models/distraccion_mobilenet_final.keras` | `src/classification.py` | Modelo final entrenado. |

### Módulo 3 (recomendación)

| Salida | Generada por | Propósito |
| --- | --- | --- |
| `outputs/metricas_recomendacion.csv` | `src/recommender.py` | Precision@K y Recall@K del sistema. |
| `outputs/reporte_recomendacion.md` | `src/recommender.py` | Interpretación y resumen del sistema. |
| `outputs/recomendaciones_usuario_<id>.csv` | `src/recommender.py` | Recomendaciones personalizadas por usuario. |
| `outputs/recomendaciones_usuario_catalogo.csv` | `src/recommender.py` | Recomendaciones basadas en catálogo. |

### Interfaz web

`streamlit_app.py` consume las salidas anteriores para visualizarlas. No genera artefactos nuevos; su propósito es presentar resultados y facilitar demostraciones públicas.
