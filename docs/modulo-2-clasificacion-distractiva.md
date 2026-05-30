# Módulo 2 - Clasificación de Conducción Distractiva

Este módulo entrena un clasificador de imágenes con MobileNetV2, genera métricas, matriz de confusión, reporte por clase y ejemplos correctos/erróneos.

## Archivos principales

- `src/classification.py`

## Salidas principales

- `outputs/metricas_distraccion.csv`
- `outputs/reporte_clasificacion_distraccion.csv`
- `outputs/matriz_confusion_distraccion.csv`
- `outputs/matriz_confusion_distraccion.png`
- `outputs/clases_distraccion.json`
- `outputs/distraccion_images_split/`
- `outputs/img/distraccion/correct/`
- `outputs/img/distraccion/wrong/`
- `models/distraccion_mobilenet_best.keras`
- `models/distraccion_mobilenet_final.keras`

## Estructura de datos esperada

Ubica las imágenes en una carpeta por clase. Si solo tienes una carpeta cruda, el módulo crea el split estratificado 80/10/10 automáticamente.

Ruta sugerida:

```text
data/raw/distraccion_images/<clase>/*
```

Split generado:

```text
outputs/distraccion_images_split/train/<clase>/*
outputs/distraccion_images_split/val/<clase>/*
outputs/distraccion_images_split/test/<clase>/*
```

## Entrenamiento y predicción

Entrenar:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/classification.py --train
```

Forzar recreación del split:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/classification.py --train --force-split
```

Predecir una imagen:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/classification.py --predict-image "ruta\a\tu\imagen.jpg"
```
