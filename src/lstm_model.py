import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def prepare_lstm_data(series, lookback=30, forecast_horizon=30):
    data = series['demand'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(data_scaled) - forecast_horizon + 1):
        X.append(data_scaled[i-lookback:i])
        y.append(data_scaled[i:i+forecast_horizon].flatten())
    return np.array(X), np.array(y), scaler

def create_lstm_model(lookback, forecast_horizon):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(series, name, lookback=30, forecast_horizon=30, save_model_path=None):
    print(f"\n{'='*60}\nEntrenando LSTM para {name}\n{'='*60}")
    X, y, scaler = prepare_lstm_data(series, lookback, forecast_horizon)
    
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model = create_lstm_model(lookback, forecast_horizon)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluación en test
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\nMétricas en test: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    
    # Predicción a 30 días
    last_data = series['demand'].values[-lookback:].reshape(-1, 1)
    last_scaled = scaler.transform(last_data).reshape(1, lookback, 1)
    future_scaled = model.predict(last_scaled)
    future_pred = scaler.inverse_transform(future_scaled).flatten()
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    
    # Guardar modelo y scaler si se pide
    if save_model_path:
        model.save(os.path.join(save_model_path, f"{name.lower().replace(' ', '_')}_lstm.h5"))
        joblib.dump(scaler, os.path.join(save_model_path, f"scaler_{name.lower().replace(' ', '_')}.pkl"))
    
    return model, scaler, {'rmse': rmse, 'mae': mae, 'mape': mape}, future_pred, future_dates

def plot_forecast(series, future_dates, future_pred, name, save_dir=None, save_path=None, show_plot=True):
    plt.figure(figsize=(14, 5))
    plt.plot(series.index[-180:], series['demand'].values[-180:], label='Histórico (últimos 180 días)', color='blue')
    plt.plot(future_dates, future_pred, label='Predicción 30 días', color='red', linestyle='--', marker='o')
    plt.axvline(x=series.index[-1], color='gray', linestyle=':', alpha=0.7)
    plt.title(f'{name} - Predicción demanda próximos 30 días')
    plt.xlabel('Fecha')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
    elif save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.lower().replace(' ', '_')
        output = os.path.join(save_dir, f"{safe_name}_prediccion_30_dias.png")
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {output}")
    if show_plot:
        plt.show()
    else:
        plt.close()