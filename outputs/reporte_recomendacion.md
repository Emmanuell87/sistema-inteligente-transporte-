# Reporte del sistema de recomendacion

## Resumen metodologico
- Se usa un enfoque hibrido: filtrado colaborativo por similitud item-item + contenido por preferencias y atributos de destino.
- Las reseñas y el historial se combinan para construir interacciones usuario-destino.
- El catálogo de destinos se usa como base de contenido y popularidad.

## Metricas de evaluacion
- Precision@K: 0.0017
- Recall@K: 0.0083

## Analisis de efectividad
- El sistema recupera una fraccion pequena de los destinos relevantes dentro del top-K.
- Esto indica que ya existe señal util, pero el ranking todavia es sensible al desbalance entre interacciones y catalogo amplio.
- Para una version de produccion se recomienda enriquecer el historial con mas interacciones y explorar un modelo de aprendizaje por ranking.
- No se cuenta con una variable directa de satisfaccion del usuario, asi que la efectividad se interpreta a partir de Precision@K, Recall@K y coherencia de los ejemplos generados.

## Ejemplos de recomendaciones
### Usuario 1
- 72 | Goa Beaches | Goa | Beach | score=2.0442
- 172 | Goa Beaches | Goa | Beach | score=2.0418
- 177 | Goa Beaches | Goa | Beach | score=2.0339
- 837 | Goa Beaches | Goa | Beach | score=2.0431
- 987 | Goa Beaches | Goa | Beach | score=2.0445

### Usuario 2
- 59 | Kerala Backwaters | Kerala | Nature | score=2.0401
- 114 | Kerala Backwaters | Kerala | Nature | score=2.0355
- 154 | Kerala Backwaters | Kerala | Nature | score=2.0390
- 515 | Leh Ladakh | Jammu and Kashmir | Adventure | score=2.0378
- 684 | Kerala Backwaters | Kerala | Nature | score=2.0352

### Usuario 3
- 8 | Jaipur City | Rajasthan | City | score=2.0634
- 468 | Jaipur City | Rajasthan | City | score=2.0675
- 593 | Jaipur City | Rajasthan | City | score=2.0654
- 653 | Jaipur City | Rajasthan | City | score=2.0711
- 661 | Taj Mahal | Uttar Pradesh | Historical | score=2.1511

## Medidas para mejorar
- Incorporar mas historial de interacciones y sesiones de viaje reales.
- Usar feedback explicito de clics, reservas o conversiones para medir satisfaccion.
- Probar un modelo de ranking mas avanzado o un sistema de filtrado por contenido con embeddings.
- Ajustar pesos del componente colaborativo y de contenido segun pruebas A/B.

## Notas
- Los usuarios satisfechos no se miden de forma directa en estos CSV; la evaluacion usa Precision@K y Recall@K como proxy.
- La baja precision/recall indica que el historial disponible aun es limitado para entrenar un ranking fuerte.