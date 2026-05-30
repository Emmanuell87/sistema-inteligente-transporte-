# Sistema Inteligente Integrado

Proyecto público organizado por módulos para predicción de demanda, clasificación de conducción distractiva y recomendación de destinos, con una interfaz web unificada para exploración.

## Documentación destacada

- [Guía general del proyecto](docs/guia-general.md)
- [Documentación por módulo](docs/README.md)
- [Interfaz Web](docs/interfaz-web.md)

## Arquitectura (resumen)

- Módulo 1: predicción de demanda con LSTM (BUS y METRO) y generación de métricas, series y forecasts.
- Módulo 2: clasificación de conducción distractiva con MobileNetV2 y reportes de desempeño.
- Módulo 3: recomendación híbrida (colaborativo + contenido + popularidad).
- Interfaz web: visualización y exploración de resultados desde Streamlit.

## Estructura principal

- [docs/](docs/): documentación general y por módulo.
- [main.py](main.py): punto de entrada del módulo 1.
- [src/](src/): lógica reutilizable y módulos 2-3.
- [streamlit_app.py](streamlit_app.py): interfaz web integrada.
- [models/](models/): modelos entrenados.
- [outputs/](outputs/): métricas, reportes, imágenes y CSV generados.
- [data/raw/](data/raw/): datos de entrada.

## Uso rápido

1. [Módulo 1 - Predicción de Demanda](docs/modulo-1-prediccion-demanda.md)
2. [Módulo 2 - Clasificación de Conducción Distractiva](docs/modulo-2-clasificacion-distractiva.md)
3. [Módulo 3 - Sistema de Recomendación de Destinos](docs/modulo-3-recomendacion-destinos.md)
4. [Interfaz Web](docs/interfaz-web.md)

## Requisitos

Instala dependencias con:

```powershell
c:/Users/jesus/Documents/Redes_Neuronales_trabajo_3/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

