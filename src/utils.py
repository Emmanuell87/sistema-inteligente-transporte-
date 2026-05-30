import pandas as pd
import os

def save_metrics(metrics_df, output_path):
    metrics_df.to_csv(output_path, index=False)
    print(f"Métricas guardadas en {output_path}")

def save_predictions(predictions_df, output_path):
    predictions_df.to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")