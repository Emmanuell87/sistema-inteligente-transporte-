import matplotlib.pyplot as plt
import pandas as pd
import os


def _sanitize_filename(text):
    safe = ''.join(c.lower() if c.isalnum() else '_' for c in text)
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')


def _save_plot(save_dir=None, save_path=None, fallback_name='grafica.png'):
    """Guarda la figura activa en una ruta personalizada o en un directorio con nombre por defecto."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {save_path}")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output = os.path.join(save_dir, fallback_name)
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {output}")

def plot_series(serie, title, save_dir=None, save_path=None, show_plot=True):
    """Gráfico de evolución de la demanda."""
    plt.figure(figsize=(14, 5))
    plt.plot(serie.index, serie['demand'], color='steelblue')
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel('Demanda')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_plot(save_dir, save_path, f"{_sanitize_filename(title)}_serie.png")
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_distribution(serie, title, save_dir=None, save_path=None, show_plot=True):
    """Histograma y boxplot de la demanda."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(serie['demand'], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_title(f'{title} - Distribución')
    axes[0].set_xlabel('Demanda')
    axes[0].set_ylabel('Frecuencia')
    
    axes[1].boxplot(serie['demand'])
    axes[1].set_title(f'{title} - Boxplot')
    axes[1].set_ylabel('Demanda')
    plt.tight_layout()
    _save_plot(save_dir, save_path, f"{_sanitize_filename(title)}_distribucion.png")
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_seasonality(serie, title, save_dir=None, save_path=None, show_plot=True):
    """Estacionalidad por mes y día de la semana."""
    serie_copy = serie.copy()
    serie_copy['mes'] = serie_copy.index.month
    serie_copy['dia_semana'] = serie_copy.index.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    serie_copy.boxplot(column='demand', by='mes', ax=axes[0])
    axes[0].set_title(f'{title} - Demanda por mes')
    axes[0].set_xlabel('Mes')
    
    serie_copy.boxplot(column='demand', by='dia_semana', ax=axes[1])
    axes[1].set_title(f'{title} - Demanda por día semana (0=Lun, 6=Dom)')
    axes[1].set_xlabel('Día semana')
    plt.suptitle('')
    plt.tight_layout()
    _save_plot(save_dir, save_path, f"{_sanitize_filename(title)}_estacionalidad.png")
    if show_plot:
        plt.show()
    else:
        plt.close()