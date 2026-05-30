import pandas as pd
import numpy as np


def _convert_traffic_dict_to_dataframe(raw_dict):
    """Convierte un dataset tipo dict (TimeRange/TimeFitness/Node) a DataFrame temporal."""
    required_keys = {'TimeRange', 'TimeFitness', 'Node', 'LenTimeSlots'}
    if not required_keys.issubset(raw_dict.keys()):
        return None

    node_data = raw_dict.get('Node', {})
    traffic_node = node_data.get('TrafficNode') if isinstance(node_data, dict) else None
    if not isinstance(traffic_node, np.ndarray):
        return None

    start_date = raw_dict['TimeRange'][0]
    freq_minutes = int(raw_dict['TimeFitness'])
    n_slots = int(raw_dict['LenTimeSlots'])

    if traffic_node.shape[0] != n_slots:
        n_slots = traffic_node.shape[0]

    timestamps = pd.date_range(start=pd.to_datetime(start_date), periods=n_slots, freq=f'{freq_minutes}min')
    demand_total = traffic_node[:n_slots].sum(axis=1)

    return pd.DataFrame({
        'timestamp': timestamps,
        'demand_total': demand_total
    })

def load_pkl_dataset(file_path, dataset_name):
    """Carga un archivo .pkl y muestra información básica."""
    print(f"\n{'='*60}")
    print(f"Cargando: {dataset_name}")
    print(f"{'='*60}")
    df = pd.read_pickle(file_path)
    print(f"Tipo: {type(df)}")
    if isinstance(df, pd.DataFrame):
        print(f"Shape: {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
        print(df.head())
    elif isinstance(df, dict):
        print(f"Claves del diccionario: {list(df.keys())}")
        converted_df = _convert_traffic_dict_to_dataframe(df)
        if isinstance(converted_df, pd.DataFrame):
            df = converted_df
            print("Formato de tráfico detectado. Convertido a DataFrame temporal.")
            print(f"Shape: {df.shape}")
            print(f"Columnas: {df.columns.tolist()}")
            print(df.head())
            return df
        for k, v in df.items():
            if isinstance(v, pd.DataFrame):
                print(f"Extrayendo DataFrame de la clave '{k}'")
                df = v
                break
    return df

def prepare_time_series(df, date_col=None, demand_col=None, freq='D'):
    """
    Convierte un DataFrame genérico en serie temporal con índice datetime.
    Detecta automáticamente columnas de fecha y demanda si no se especifican.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El objeto no es DataFrame")
    
    # Buscar columna de fecha
    if date_col is None:
        for col in df.columns:
            if any(key in col.lower() for key in ['date', 'time', 'datetime', 'timestamp']):
                date_col = col
                break
        if date_col is None and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        elif date_col is None:
            raise ValueError("No se encontró columna de fecha. Especifica 'date_col'.")
    
    # Buscar columna de demanda
    if demand_col is None:
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                if any(key in col.lower() for key in ['passenger', 'demand', 'count', 'volume', 'trips', 'ridership']):
                    demand_col = col
                    break
        if demand_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                demand_col = numeric_cols[0]
            else:
                raise ValueError("No se encontró columna numérica para la demanda.")
    
    print(f"\nUsando fecha: '{date_col}' -> Demanda: '{demand_col}'")
    df_ts = df[[date_col, demand_col]].copy()
    df_ts.rename(columns={date_col: 'timestamp', demand_col: 'demand'}, inplace=True)
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
    df_ts = df_ts.sort_values('timestamp').set_index('timestamp')
    df_ts = df_ts.resample(freq).sum().dropna()
    print(f"Serie temporal creada: {len(df_ts)} registros de {df_ts.index[0]} a {df_ts.index[-1]}")
    return df_ts