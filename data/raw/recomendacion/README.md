# Datos para el Modulo 3

Coloca aqui el archivo CSV del historial de viajes / interaccion usuario-destino.

Nombre recomendado:
- `interacciones_viajes.csv`

Columnas recomendadas:
- `user_id` o `usuario_id`
- `destination_id` o `destino_id`
- `rating` o `interaccion` (opcional, si existe)
- `fecha` (opcional)
- cualquier otra columna de contexto o preferencia de destino

Ejemplo de ruta esperada:
- `data/raw/recomendacion/interacciones_viajes.csv`

Si tu archivo usa otros nombres de columnas, luego se puede adaptar el cargador en `src/recommender.py`.