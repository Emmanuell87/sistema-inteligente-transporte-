import os
import argparse
import pandas as pd
from src.data_loader import load_pkl_dataset, prepare_time_series
from src.exploratory import plot_series, plot_distribution, plot_seasonality
from src.lstm_model import train_lstm, plot_forecast
from src.utils import save_metrics, save_predictions


def build_output_paths(outputs_dir):
    """Define una estructura de salida estable para facilitar revisión y entregables."""
    paths = {
        'images_root': os.path.join(outputs_dir, 'img'),
        'bus_exploratory': os.path.join(outputs_dir, 'img', 'bus', 'exploratorio'),
        'bus_forecast': os.path.join(outputs_dir, 'img', 'bus', 'prediccion'),
        'metro_exploratory': os.path.join(outputs_dir, 'img', 'metro', 'exploratorio'),
        'metro_forecast': os.path.join(outputs_dir, 'img', 'metro', 'prediccion'),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def main(only_plots=False, show_plots=False):
    # 1) Rutas base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_raw_dir = os.path.join(base_dir, 'data', 'raw')
    models_dir = os.path.join(base_dir, 'models')
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(data_raw_dir, exist_ok=True)
    paths = build_output_paths(outputs_dir)
    
    # 2) Carga de datasets crudos
    bus_raw = load_pkl_dataset(os.path.join(data_raw_dir, 'Bus_NYC.pkl'), 'BUS NYC')
    metro_raw = load_pkl_dataset(os.path.join(data_raw_dir, 'Metro_NYC.pkl'), 'METRO NYC')
    
    # 3) Transformación a series temporales diarias
    bus_series = prepare_time_series(bus_raw, freq='D')
    metro_series = prepare_time_series(metro_raw, freq='D')
    
    # 4) Análisis exploratorio (siempre se guarda en disco)
    print("\nGenerando gráficos de análisis exploratorio...")
    plot_series(
        bus_series,
        'Demanda histórica - Bus NYC',
        save_path=os.path.join(paths['bus_exploratory'], '01_serie_historica_bus.png'),
        show_plot=show_plots
    )
    plot_distribution(
        bus_series,
        'Bus NYC',
        save_path=os.path.join(paths['bus_exploratory'], '02_distribucion_bus.png'),
        show_plot=show_plots
    )
    plot_seasonality(
        bus_series,
        'Bus NYC',
        save_path=os.path.join(paths['bus_exploratory'], '03_estacionalidad_bus.png'),
        show_plot=show_plots
    )
    
    plot_series(
        metro_series,
        'Demanda histórica - Metro NYC',
        save_path=os.path.join(paths['metro_exploratory'], '01_serie_historica_metro.png'),
        show_plot=show_plots
    )
    plot_distribution(
        metro_series,
        'Metro NYC',
        save_path=os.path.join(paths['metro_exploratory'], '02_distribucion_metro.png'),
        show_plot=show_plots
    )
    plot_seasonality(
        metro_series,
        'Metro NYC',
        save_path=os.path.join(paths['metro_exploratory'], '03_estacionalidad_metro.png'),
        show_plot=show_plots
    )

    # 5) Modo rápido: usa predicciones existentes para recrear gráficos sin entrenamiento
    if only_plots:
        bus_pred_path = os.path.join(outputs_dir, 'prediccion_bus_30dias.csv')
        metro_pred_path = os.path.join(outputs_dir, 'prediccion_metro_30dias.csv')

        if os.path.exists(bus_pred_path):
            pred_bus_df = pd.read_csv(bus_pred_path)
            dates_bus = pd.to_datetime(pred_bus_df['fecha'])
            plot_forecast(
                bus_series,
                dates_bus,
                pred_bus_df['demanda_predicha_bus'].values,
                'BUS NYC',
                save_path=os.path.join(paths['bus_forecast'], '01_prediccion_30_dias_bus.png'),
                show_plot=show_plots
            )
        else:
            print(f"Aviso: no existe {bus_pred_path}; no se pudo regenerar forecast de BUS.")

        if os.path.exists(metro_pred_path):
            pred_metro_df = pd.read_csv(metro_pred_path)
            dates_metro = pd.to_datetime(pred_metro_df['fecha'])
            plot_forecast(
                metro_series,
                dates_metro,
                pred_metro_df['demanda_predicha_metro'].values,
                'METRO NYC',
                save_path=os.path.join(paths['metro_forecast'], '01_prediccion_30_dias_metro.png'),
                show_plot=show_plots
            )
        else:
            print(f"Aviso: no existe {metro_pred_path}; no se pudo regenerar forecast de METRO.")

        print("\n✅ Gráficas guardadas en 'outputs/img' sin reentrenar modelos.")
        return
    
    # 6) Entrenamiento y predicción para BUS
    model_bus, scaler_bus, metrics_bus, pred_bus, dates_bus = train_lstm(
        bus_series, 'BUS_NYC', lookback=30, forecast_horizon=30, save_model_path=models_dir
    )
    plot_forecast(
        bus_series,
        dates_bus,
        pred_bus,
        'BUS NYC',
        save_path=os.path.join(paths['bus_forecast'], '01_prediccion_30_dias_bus.png'),
        show_plot=show_plots
    )
    
    # 7) Entrenamiento y predicción para METRO
    model_metro, scaler_metro, metrics_metro, pred_metro, dates_metro = train_lstm(
        metro_series, 'METRO_NYC', lookback=30, forecast_horizon=30, save_model_path=models_dir
    )
    plot_forecast(
        metro_series,
        dates_metro,
        pred_metro,
        'METRO NYC',
        save_path=os.path.join(paths['metro_forecast'], '01_prediccion_30_dias_metro.png'),
        show_plot=show_plots
    )
    
    # 8) Guardado de métricas y archivos de predicción
    metrics_df = pd.DataFrame([
        {'dataset': 'BUS_NYC', **metrics_bus},
        {'dataset': 'METRO_NYC', **metrics_metro}
    ])
    save_metrics(metrics_df, os.path.join(outputs_dir, 'metricas_modelos.csv'))
    
    pred_bus_df = pd.DataFrame({'fecha': dates_bus, 'demanda_predicha_bus': pred_bus})
    pred_metro_df = pd.DataFrame({'fecha': dates_metro, 'demanda_predicha_metro': pred_metro})
    save_predictions(pred_bus_df, os.path.join(outputs_dir, 'prediccion_bus_30dias.csv'))
    save_predictions(pred_metro_df, os.path.join(outputs_dir, 'prediccion_metro_30dias.csv'))
    
    print("\n✅ Módulo 1 completado. Resultados en carpeta 'outputs/' y modelos en 'models/'.")

if __name__ == '__main__':
    # CLI principal para dos flujos:
    # - Entrenamiento completo (por defecto)
    # - Solo regeneración de gráficas (sin reentrenar)
    parser = argparse.ArgumentParser(description='Pipeline de predicción de demanda de transporte.')
    parser.add_argument('--only-plots', action='store_true', help='Genera y guarda gráficas usando predicciones ya existentes.')
    parser.add_argument('--show-plots', action='store_true', help='Muestra las ventanas de las gráficas además de guardarlas.')
    args = parser.parse_args()
    main(only_plots=args.only_plots, show_plots=args.show_plots)