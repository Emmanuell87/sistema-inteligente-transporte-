# Documentación del Proyecto

Este proyecto está organizado en tres módulos principales y una interfaz web unificada.

## Navegación rápida

- [Guía general del proyecto](guia-general.md)
- [Módulo 1 - Predicción de Demanda](modulo-1-prediccion-demanda.md)
- [Módulo 2 - Clasificación de Conducción Distractiva](modulo-2-clasificacion-distractiva.md)
- [Módulo 3 - Sistema de Recomendación de Destinos](modulo-3-recomendacion-destinos.md)
- [Interfaz Web](interfaz-web.md)

## Estructura general

- `main.py`: ejecución del módulo 1 (orquesta la predicción).
- `src/`: código reusable por módulos (Módulo 2 y Módulo 3, además de utilidades del Módulo 1).
- `models/`: modelos entrenados.
- `outputs/`: métricas, reportes, predicciones y figuras.
- `streamlit_app.py`: interfaz web integrada.
- `docs/`: instrucciones separadas por módulo y guía general.
