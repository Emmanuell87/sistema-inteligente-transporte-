# Módulo 3 - Sistema de Recomendación de Destinos

Este módulo construye un recomendador híbrido con filtrado colaborativo item-item, contenido con TF-IDF y popularidad.

## Archivos principales

- `src/recommender.py`
- `outputs/metricas_recomendacion.csv`
- `outputs/reporte_recomendacion.md`
- `outputs/recomendaciones_usuario_<id>.csv`

## Datos esperados

Los archivos se leen desde `data/raw/recomendacion/`.

Entradas principales:

- `Final_Updated_Expanded_Users.csv`
- `Final_Updated_Expanded_UserHistory.csv`
- `Final_Updated_Expanded_Reviews.csv`
- `Expanded_Destinations.csv`

## Ejecución

Evaluar el sistema completo:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/recommender.py --eval --top-n 5
```

Recomendación por usuario:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/recommender.py --user-id 123 --top-n 5
```

Recomendación por catálogo:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe src/recommender.py --preferred-type Beach --top-n 5
```
