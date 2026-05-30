from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.classification import load_class_map
from src.recommender import (
    build_interactions,
    build_text_profiles,
    build_user_item_matrix,
    load_destinations,
    load_history,
    load_reviews,
    load_users,
    recommend_from_catalog,
    recommend_hybrid,
    train_item_similarity_model,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / 'outputs'
MODELS_DIR = BASE_DIR / 'models'

BUS_PREDICTION_CSV = OUTPUTS_DIR / 'prediccion_bus_30dias.csv'
METRO_PREDICTION_CSV = OUTPUTS_DIR / 'prediccion_metro_30dias.csv'
METRICS_MODEL_CSV = OUTPUTS_DIR / 'metricas_modelos.csv'
METRICS_CLASSIFICATION_CSV = OUTPUTS_DIR / 'metricas_distraccion.csv'
CLASSIFICATION_REPORT_CSV = OUTPUTS_DIR / 'reporte_clasificacion_distraccion.csv'
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / 'matriz_confusion_distraccion.png'
RECOMMENDATION_METRICS_CSV = OUTPUTS_DIR / 'metricas_recomendacion.csv'
RECOMMENDATION_REPORT_MD = OUTPUTS_DIR / 'reporte_recomendacion.md'

BUS_PREDICTION_IMAGE = OUTPUTS_DIR / 'img' / 'bus' / 'prediccion' / '01_prediccion_30_dias_bus.png'
METRO_PREDICTION_IMAGE = OUTPUTS_DIR / 'img' / 'metro' / 'prediccion' / '01_prediccion_30_dias_metro.png'
BUS_EXPLORATORY_IMAGES = [
    OUTPUTS_DIR / 'img' / 'bus' / 'exploratorio' / '01_serie_historica_bus.png',
    OUTPUTS_DIR / 'img' / 'bus' / 'exploratorio' / '02_distribucion_bus.png',
    OUTPUTS_DIR / 'img' / 'bus' / 'exploratorio' / '03_estacionalidad_bus.png',
]

CLASS_LABELS_ES = {
    'other_activities': 'Otras actividades',
    'safe_driving': 'Conducción segura',
    'talking_phone': 'Hablando por teléfono',
    'texting_phone': 'Enviando mensajes',
    'turning': 'Girando',
}

REPORT_COLUMN_LABELS_ES = {
    'precision': 'Precisión',
    'recall': 'Cobertura',
    'f1-score': 'F1',
    'support': 'Casos',
}
METRO_EXPLORATORY_IMAGES = [
    OUTPUTS_DIR / 'img' / 'metro' / 'exploratorio' / '01_serie_historica_metro.png',
    OUTPUTS_DIR / 'img' / 'metro' / 'exploratorio' / '02_distribucion_metro.png',
    OUTPUTS_DIR / 'img' / 'metro' / 'exploratorio' / '03_estacionalidad_metro.png',
]


st.set_page_config(
    page_title='Sistema Inteligente Integrado',
    page_icon='🚦',
    layout='wide',
)


st.markdown(
    '''
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f4ef 0%, #f3efe8 48%, #edf2f7 100%);
        color: #18304b;
    }
    .hero {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(24, 48, 75, 0.09);
        color: #18304b;
        padding: 1.4rem 1.6rem;
        border-radius: 1.2rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #18304b;
        margin-bottom: 0.25rem;
        font-size: 2rem;
    }
    .hero p {
        color: #41566d;
        margin-bottom: 0;
        font-size: 1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(24, 48, 75, 0.08);
        border-radius: 1rem;
        padding: 1rem 1rem 0.85rem;
        box-shadow: 0 10px 24px rgba(24, 48, 75, 0.08);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.4rem;
        color: #18304b;
    }
    div[data-testid="stTabs"] button {
        color: #18304b;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #d94b45;
    }
    /* Forzar color legible en controles (slider, checkbox, select, inputs) */
    div[data-testid="stSlider"] *,
    div[data-testid="stCheckbox"] *,
    div[data-testid="stSelectbox"] *,
    div[data-testid="stNumberInput"] *,
    div[data-testid="stTextInput"] *,
    div[data-testid="stFileUploader"] *,
    div[data-testid="stMetric"] * {
        color: #18304b !important;
    }
    .stCheckbox label, .stTextInput label {
        color: #18304b !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Asegurar que en recuadros editables de fondo oscuro el texto sea legible (blanco).
st.markdown(
    '''
    <style>
    /* Texto blanco en campos editables por defecto (mejor legibilidad sobre fondos oscuros) */
    input, textarea, select, div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        color: #ffffff !important;
    }
    ::placeholder { color: rgba(255,255,255,0.72) !important; }

    /* Restaurar texto oscuro en zonas claramente de fondo claro (hero y tarjetas) */
    .hero input, .metric-card input, .hero textarea, .metric-card textarea, .metric-card select {
        color: #18304b !important;
    }
    .hero ::placeholder, .metric-card ::placeholder { color: rgba(24,48,75,0.4) !important; }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Asegurar selectbox legible en fondos oscuros y devolver texto de markdown a color oscuro
st.markdown(
    '''
    <style>
    /* Forzar texto blanco solo dentro del control del selectbox, no en la etiqueta */
    div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        color: #ffffff !important;
    }

    /* Asegurar que el contenido markdown/expander use color oscuro por defecto */
    .markdown-text-container, .stMarkdown, .stExpanderContent, .stExpanderContent * {
        color: #18304b !important;
    }

    /* Specific: make the selectbox arrow and placeholder visible */
    div[data-testid="stSelectbox"] svg, div[data-testid="stSelectbox"] [role="button"] { fill: #ffffff !important; }
    /* Hacer todos los selectbox opacos y con fondo oscuro para legibilidad */
    div[data-testid="stSelectbox"] > div, div[data-testid="stSelectbox"] .css-1v3fvcr, div[data-testid="stSelectbox"] .st-ae {
        background-color: #222226 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    /* Etiqueta del selectbox: oscura para que contraste con el fondo claro */
    div[data-testid="stSelectbox"] label, div[data-testid="stSelectbox"] div[role="group"] > label {
        color: #18304b !important;
    }
    /* Si el navegador resalta texto dentro de esa zona, usar negro */
    #reco-personalizada ::selection {
        background: #000000 !important;
        color: #ffffff !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Forzar texto oscuro en bloques de texto y pre/code dentro de expanders (reporte completo)
st.markdown(
    '''
    <style>
    pre, code, .stText, .stText * { color: #18304b !important; }
    /* Priorizar contenido dentro del expander específico */
    div[data-testid="stExpander"] pre, div[data-testid="stExpander"] code, div[data-testid="stExpander"] .stText, div[data-testid="stExpander"] .stMarkdown {
        color: #18304b !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# CSS específico para el selectbox de 'Seleccionar usuario' dentro de la sección de recomendaciones
st.markdown(
    '''
    <style>
    /* Apuntar al selectbox que aparece después del header con id reco-personalizada */
    #reco-personalizada ~ div [data-testid="stSelectbox"],
    #reco-personalizada ~ div [data-testid="stSelectbox"] * {
        color: #ffffff !important;
    }
    /* Hacer el fondo del selectbox oscuro para visibilidad */
    #reco-personalizada ~ div [data-testid="stSelectbox"] .css-1v3fvcr, 
    #reco-personalizada ~ div [data-testid="stSelectbox"] .css-1v3fvcr * {
        background-color: #222226 !important;
        color: #ffffff !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding='utf-8')
    return 'No se encontró el archivo solicitado.'


def show_metric_row(metrics: dict[str, str | float], columns: int = 3) -> None:
    if not metrics:
        return
    items = list(metrics.items())
    cols = st.columns(columns)
    for index, (label, value) in enumerate(items):
        with cols[index % columns]:
            st.markdown(
                f'''
                <div class="metric-card">
                    <div style="font-size:0.84rem;color:#5b6b7a;">{label}</div>
                    <div style="font-size:1.45rem;font-weight:800;color:#18304b;">{value}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )


def translate_class_label(label: str) -> str:
    return CLASS_LABELS_ES.get(label, str(label).replace('_', ' ').title())


def load_classification_report() -> pd.DataFrame:
    report_df = read_csv_if_exists(CLASSIFICATION_REPORT_CSV)
    if report_df.empty:
        return report_df

    report_df = report_df.rename(columns=lambda col: str(col).strip())
    if 'Unnamed: 0' in report_df.columns:
        report_df = report_df.rename(columns={'Unnamed: 0': 'clase'})
    elif report_df.columns[0] != 'clase':
        report_df = report_df.rename(columns={report_df.columns[0]: 'clase'})

    rename_map = {old: new for old, new in REPORT_COLUMN_LABELS_ES.items()}
    report_df = report_df.rename(columns=rename_map)
    if 'clase' in report_df.columns:
        # Traducir nombres de clase y de filas resumen
        report_df['clase'] = report_df['clase'].astype(str)
        report_df['clase'] = report_df['clase'].replace({
            'accuracy': 'Exactitud',
            'macro avg': 'Promedio macro',
            'weighted avg': 'Promedio ponderado',
        })
        report_df['clase'] = report_df['clase'].map(lambda x: translate_class_label(x) if x not in ['Exactitud', 'Promedio macro', 'Promedio ponderado'] else x)
    return report_df


def build_distraction_report(report_df: pd.DataFrame) -> tuple[str, str, str]:
    if report_df.empty or 'Casos' not in report_df.columns or 'clase' not in report_df.columns:
        return (
            'No hay datos suficientes para generar el informe de distracciones.',
            'Con los archivos actuales no se pudo identificar la frecuencia de clases.',
            '- Revisar la distribución del dataset antes de sacar conclusiones.',
        )

    class_rows = report_df[~report_df['clase'].isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
    class_rows['Casos'] = pd.to_numeric(class_rows['Casos'], errors='coerce')
    class_rows = class_rows.dropna(subset=['Casos'])
    if class_rows.empty:
        return (
            'No hay clases válidas para resumir.',
            'Las métricas existen, pero no fue posible ordenar por frecuencia.',
            '- Verificar el formato del reporte exportado por el módulo 2.',
        )

    distraction_rows = class_rows[class_rows['clase'] != 'Conducción segura'].copy()
    distraction_rows = distraction_rows.sort_values('Casos', ascending=False)
    top_distractions = distraction_rows.head(3)
    most_common = distraction_rows.iloc[0] if not distraction_rows.empty else class_rows.sort_values('Casos', ascending=False).iloc[0]

    summary = (
        f"La clase más frecuente en el conjunto es {most_common['clase']} con {int(most_common['Casos'])} casos. "
        f"Entre las distracciones, las más repetidas son {', '.join(top_distractions['clase'].tolist())}."
    )

    preventive = (
        '- Reforzar la atención visual con pausas de seguridad y recordatorios en el vehículo.\n'
        '- Desactivar o restringir el uso del teléfono al conducir.\n'
        '- Aumentar la señalización y la capacitación sobre maniobras como girar o cambiar de carril.\n'
        '- Supervisar los casos de mayor confusión para ajustar el entrenamiento con ejemplos más difíciles.'
    )

    details_lines = []
    for _, row in top_distractions.iterrows():
        details_lines.append(
            f"- {row['clase']}: {int(row['Casos'])} casos, precisión {float(row.get('Precisión', 0.0)):.2f}, cobertura {float(row.get('Cobertura', 0.0)):.2f}."
        )
    details = '\n'.join(details_lines)
    return summary, details, preventive


def forecast_line_chart(csv_path: Path, date_col: str, value_col: str, title: str) -> None:
    df = read_csv_if_exists(csv_path)
    if df.empty or date_col not in df.columns or value_col not in df.columns:
        st.info(f'No se encontró información suficiente para {title}.')
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if df.empty:
        st.info(f'No se pudo graficar {title}.')
        return

    chart_df = df.set_index(date_col)[[value_col]]
    st.line_chart(chart_df, height=280)
    st.dataframe(df.tail(10), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def load_recommendation_artifacts() -> dict:
    users_df = load_users(str(BASE_DIR / 'data' / 'raw' / 'recomendacion' / 'Final_Updated_Expanded_Users.csv'))
    history_df = load_history(str(BASE_DIR / 'data' / 'raw' / 'recomendacion' / 'Final_Updated_Expanded_UserHistory.csv'))
    reviews_df = load_reviews(str(BASE_DIR / 'data' / 'raw' / 'recomendacion' / 'Final_Updated_Expanded_Reviews.csv'))
    destinations_df = load_destinations(str(BASE_DIR / 'data' / 'raw' / 'recomendacion' / 'Expanded_Destinations.csv'))
    interactions_df = build_interactions(users_df, history_df, reviews_df)
    user_item_matrix = build_user_item_matrix(interactions_df) if not interactions_df.empty else pd.DataFrame()
    similarity_df = train_item_similarity_model(user_item_matrix) if not user_item_matrix.empty else pd.DataFrame()
    users_df, destinations_df, user_vectors, destination_vectors, _ = build_text_profiles(users_df, destinations_df)
    return {
        'users_df': users_df,
        'destinations_df': destinations_df,
        'interactions_df': interactions_df,
        'user_item_matrix': user_item_matrix,
        'similarity_df': similarity_df,
        'user_vectors': user_vectors,
        'destination_vectors': destination_vectors,
    }


@st.cache_resource(show_spinner=False)
def load_classification_model(model_path: str, class_map_path: str, split_dir: str):
    model = load_model(model_path)
    class_map = load_class_map(class_map_path, split_dir=split_dir)
    return model, class_map


def classify_uploaded_image(uploaded_file, model_path: str, class_map_path: str, split_dir: str):
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_image_path = temp_file.name

    model, class_map = load_classification_model(model_path, class_map_path, split_dir)
    image = load_img(temp_image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = image_array[None, ...]
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(probabilities.argmax())
    predicted_label = class_map[predicted_index]
    confidence = float(probabilities[predicted_index])
    ranking = pd.DataFrame(
        {
            'Clase': [class_map[int(index)] for index in probabilities.argsort()[::-1]],
            'Probabilidad': [float(probabilities[int(index)]) for index in probabilities.argsort()[::-1]],
        }
    )
    return temp_image_path, predicted_label, confidence, ranking


def image_gallery(paths: list[Path], captions: list[str] | None = None, columns: int = 3) -> None:
    available = [path for path in paths if path.exists()]
    if not available:
        st.info('No hay imágenes disponibles en esta sección.')
        return

    rows = [available[index:index + columns] for index in range(0, len(available), columns)]
    for row_index, row in enumerate(rows):
        cols = st.columns(len(row))
        for col_index, image_path in enumerate(row):
            with cols[col_index]:
                st.image(Image.open(image_path), use_container_width=True)
                if captions and row_index * columns + col_index < len(captions):
                    st.caption(captions[row_index * columns + col_index])
                else:
                    st.caption(image_path.name)


def demand_tab() -> None:
    st.markdown('<div class="section-title">Predicción de demanda</div>', unsafe_allow_html=True)
    metrics_df = read_csv_if_exists(METRICS_MODEL_CSV)
    if not metrics_df.empty:
        metrics_df = metrics_df.rename(columns=str.lower)
        if {'dataset', 'rmse', 'mae', 'mape'}.issubset(metrics_df.columns):
            metrics_summary = {}
            for _, row in metrics_df.iterrows():
                dataset = 'BUS' if 'bus' in str(row['dataset']).lower() else 'METRO'
                metrics_summary[f'{dataset} RMSE'] = f"{float(row['rmse']):,.2f}"
                metrics_summary[f'{dataset} MAE'] = f"{float(row['mae']):,.2f}"
                metrics_summary[f'{dataset} MAPE'] = f"{float(row['mape']):.2f}%"
            show_metric_row(metrics_summary, columns=3)
            st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader('BUS')
        forecast_line_chart(BUS_PREDICTION_CSV, 'fecha', 'demanda_predicha_bus', 'predicción BUS')
        if BUS_PREDICTION_IMAGE.exists():
            st.image(Image.open(BUS_PREDICTION_IMAGE), caption='Gráfico guardado de predicción BUS', use_container_width=True)
        with st.expander('Exploración BUS', expanded=False):
            image_gallery(BUS_EXPLORATORY_IMAGES, columns=1)

    with col_right:
        st.subheader('METRO')
        forecast_line_chart(METRO_PREDICTION_CSV, 'fecha', 'demanda_predicha_metro', 'predicción METRO')
        if METRO_PREDICTION_IMAGE.exists():
            st.image(Image.open(METRO_PREDICTION_IMAGE), caption='Gráfico guardado de predicción METRO', use_container_width=True)
        with st.expander('Exploración METRO', expanded=False):
            image_gallery(METRO_EXPLORATORY_IMAGES, columns=1)


def classification_tab() -> None:
    st.markdown('<div class="section-title">Clasificación de conducción distractiva</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        metrics_df = read_csv_if_exists(METRICS_CLASSIFICATION_CSV)
        if not metrics_df.empty:
            st.markdown('<div class="section-title">Métricas del modelo</div>', unsafe_allow_html=True)
            metrics_df = metrics_df.rename(
                columns={
                    'accuracy': 'Exactitud',
                    'f1_weighted': 'F1 ponderado',
                    'precision_weighted': 'Precisión ponderada',
                    'recall_weighted': 'Cobertura ponderada',
                }
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        report_df = load_classification_report()
        if not report_df.empty:
            st.markdown('<div class="section-title">Reporte por clase</div>', unsafe_allow_html=True)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            with st.expander('¿Qué significan estas métricas?', expanded=False):
                st.markdown("""
- **Precisión**: proporción de predicciones correctas entre las predicciones para una clase.
- **Cobertura (Recall)**: proporción de instancias reales de la clase que fueron correctamente detectadas.
- **F1**: media armónica entre precisión y cobertura; útil cuando hay desequilibrio entre clases.
- **Promedio macro**: promedio simple de la métrica por clase.
- **Promedio ponderado**: promedio de la métrica ponderado por el número de casos por clase.
""")

            summary_text, details_text, preventive_text = build_distraction_report(report_df)
            st.markdown('<div class="section-title">Informe sobre distracciones frecuentes y medidas preventivas</div>', unsafe_allow_html=True)
            st.info(summary_text)
            st.markdown(details_text)
            st.markdown('**Medidas preventivas**')
            st.markdown(preventive_text)

        if CONFUSION_MATRIX_PNG.exists():
            st.markdown('<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True)
            st.image(Image.open(CONFUSION_MATRIX_PNG), use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Probar una imagen</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('Sube una imagen para clasificarla', type=['png', 'jpg', 'jpeg', 'webp'])
        model_path = str(MODELS_DIR / 'distraccion_mobilenet_final.keras')
        class_map_path = str(OUTPUTS_DIR / 'clases_distraccion.json')
        split_dir = str(OUTPUTS_DIR / 'distraccion_images_split')

        if uploaded_file is not None:
            temp_result = classify_uploaded_image(uploaded_file, model_path, class_map_path, split_dir)
            if temp_result is not None:
                temp_image_path, predicted_label, confidence, ranking = temp_result
                st.image(Image.open(temp_image_path), caption='Imagen cargada', use_container_width=True)
                st.success(f'Categoría asignada: {translate_class_label(predicted_label)}')
                st.metric('Confianza', f'{confidence:.2%}')
                ranking = ranking.copy()
                ranking['Clase'] = ranking['Clase'].map(translate_class_label)
                ranking = ranking.rename(columns={'Clase': 'Clase', 'Probabilidad': 'Probabilidad'})
                st.bar_chart(ranking.set_index('Clase')['Probabilidad'], height=240)
                st.dataframe(ranking, use_container_width=True, hide_index=True)

        with st.expander('Ejemplos de correctos y errores', expanded=False):
            correct_dir = OUTPUTS_DIR / 'img' / 'distraccion' / 'correct'
            wrong_dir = OUTPUTS_DIR / 'img' / 'distraccion' / 'wrong'
            st.caption('Casos correctamente clasificados')
            image_gallery(sorted(correct_dir.glob('*'))[:6], columns=2)
            st.caption('Casos con error de clasificación')
            image_gallery(sorted(wrong_dir.glob('*'))[:6], columns=2)


def recommendation_tab() -> None:
    st.markdown('<div class="section-title">Sistema de recomendación de destinos</div>', unsafe_allow_html=True)
    artifacts = load_recommendation_artifacts()
    users_df = artifacts['users_df']
    destinations_df = artifacts['destinations_df']
    user_item_matrix = artifacts['user_item_matrix']
    similarity_df = artifacts['similarity_df']
    user_vectors = artifacts['user_vectors']
    destination_vectors = artifacts['destination_vectors']

    metrics_df = read_csv_if_exists(RECOMMENDATION_METRICS_CSV)
    if not metrics_df.empty:
        metrics_row = metrics_df.iloc[0].to_dict()
        show_metric_row(
            {
                'Precisión@K': f"{float(metrics_row.get('precision_at_k', 0.0)):.4f}",
                'Cobertura@K': f"{float(metrics_row.get('recall_at_k', 0.0)):.4f}",
                'Usuarios': f"{len(users_df):,}",
            },
            columns=3,
        )

    st.markdown('<div id="reco-personalizada" class="section-title">Recomendación personalizada</div>', unsafe_allow_html=True)
    user_options = users_df['user_id'].astype(str).tolist()[:50] if not users_df.empty else []
    fallback_user = user_options[0] if user_options else ''
    # Selectbox for user (styling for visibility handled by CSS targeting #reco-personalizada)
    selected_user_id = st.selectbox('Seleccionar usuario', options=user_options or [fallback_user], index=0 if user_options else None)
    top_n = st.slider('Cantidad de destinos sugeridos', min_value=3, max_value=10, value=5, step=1)
    show_profile = st.checkbox('Mostrar perfil del usuario', value=True)

    if show_profile and not users_df.empty and selected_user_id:
        profile_row = users_df[users_df['user_id'].astype(str) == str(selected_user_id)]
        if not profile_row.empty:
            st.dataframe(profile_row.head(1), use_container_width=True, hide_index=True)

    if selected_user_id:
        personalized_df = recommend_hybrid(
            user_id=selected_user_id,
            users_df=users_df,
            destinations_df=destinations_df,
            user_item_matrix=user_item_matrix,
            similarity_df=similarity_df,
            user_vectors=user_vectors,
            destination_vectors=destination_vectors,
            top_n=top_n,
        )
        if not personalized_df.empty:
            personalized_df = personalized_df.rename(
                columns={
                    'destination_id': 'ID destino',
                    'name': 'Destino',
                    'state': 'Estado/Región',
                    'type': 'Tipo',
                    'popularity': 'Popularidad',
                    'best_time_to_visit': 'Mejor época',
                    'final_score': 'Puntaje final',
                }
            )
            st.dataframe(
                personalized_df[['ID destino', 'Destino', 'Estado/Región', 'Tipo', 'Popularidad', 'Mejor época', 'Puntaje final']],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info('No se pudieron generar recomendaciones personalizadas para ese usuario.')

    st.divider()
    st.markdown('<div class="section-title">Recomendación por catálogo</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        preferred_type = st.text_input('Tipo de destino preferido', value='')
    with col_b:
        preferred_state = st.text_input('Estado o región preferida', value='')
    with col_c:
        min_popularity = st.number_input('Popularidad mínima', min_value=0.0, value=0.0, step=1.0)

    catalog_df = recommend_from_catalog(
        destinations_df,
        top_n=top_n,
        preferred_type=preferred_type or None,
        preferred_state=preferred_state or None,
        min_popularity=min_popularity if min_popularity > 0 else None,
    )
    catalog_df = catalog_df.rename(
        columns={
            'destination_id': 'ID destino',
            'name': 'Destino',
            'state': 'Estado/Región',
            'type': 'Tipo',
            'popularity': 'Popularidad',
            'best_time_to_visit': 'Mejor época',
            'final_score': 'Puntaje final',
        }
    )
    st.dataframe(
        catalog_df[['ID destino', 'Destino', 'Estado/Región', 'Tipo', 'Popularidad', 'Mejor época', 'Puntaje final']],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander('Reporte completo del sistema de recomendación', expanded=False):
        st.text(read_text_if_exists(RECOMMENDATION_REPORT_MD))


def main() -> None:
    st.markdown(
        '''
        <div class="hero">
            <h1>Sistema Inteligente Integrado</h1>
            <p>Interfaz web para explorar la predicción de demanda, clasificar imágenes y probar recomendaciones personalizadas.</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.caption('La app usa los modelos y salidas ya generadas en outputs/ y models/.')
    tabs = st.tabs(['Demanda', 'Clasificación', 'Recomendación'])
    with tabs[0]:
        demand_tab()
    with tabs[1]:
        classification_tab()
    with tabs[2]:
        recommendation_tab()


if __name__ == '__main__':
    main()