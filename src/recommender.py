import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_USERS_PATH = 'data/raw/recomendacion/Final_Updated_Expanded_Users.csv'
DEFAULT_HISTORY_PATH = 'data/raw/recomendacion/Final_Updated_Expanded_UserHistory.csv'
DEFAULT_REVIEWS_PATH = 'data/raw/recomendacion/Final_Updated_Expanded_Reviews.csv'
DEFAULT_DESTINATIONS_PATH = 'data/raw/recomendacion/Expanded_Destinations.csv'
DEFAULT_OUTPUT_DIR = 'outputs'


def _lower_columns(df):
    return df.rename(columns={c: c.strip() for c in df.columns})


def load_users(csv_path):
    """Carga la tabla de usuarios y normaliza preferencias."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['user_id', 'name', 'email', 'preferences', 'gender', 'number_of_adults', 'number_of_children'])

    df = pd.read_csv(csv_path)
    df = _lower_columns(df)
    rename_map = {}
    columns = {col.lower(): col for col in df.columns}

    for candidate in ['userid', 'user_id']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'user_id'
            break
    for candidate in ['name', 'nombre']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'name'
            break
    for candidate in ['email', 'correo']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'email'
            break
    for candidate in ['preferences', 'preferencias']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'preferences'
            break
    for candidate in ['gender', 'sexo']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'gender'
            break
    for candidate in ['numberofadults', 'adultos']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'number_of_adults'
            break
    for candidate in ['numberofchildren', 'children', 'ninos', 'niños']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'number_of_children'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'user_id' not in df.columns:
        raise ValueError('La tabla de usuarios debe incluir UserID.')

    if 'preferences' not in df.columns:
        df['preferences'] = ''

    return df


def load_history(csv_path):
    """Carga el historial de viajes y lo convierte a interacciones usuario-destino."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['user_id', 'item_id', 'score'])

    df = pd.read_csv(csv_path)
    df = _lower_columns(df)
    rename_map = {}
    columns = {col.lower(): col for col in df.columns}

    for candidate in ['userid', 'user_id']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'user_id'
            break
    for candidate in ['destinationid', 'destination_id', 'destinoid']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'item_id'
            break
    for candidate in ['experiencerating', 'rating', 'score', 'interaccion']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'score'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'user_id' not in df.columns or 'item_id' not in df.columns:
        raise ValueError('El historial debe incluir UserID y DestinationID.')

    if 'score' not in df.columns:
        df['score'] = 1.0

    df = df[['user_id', 'item_id', 'score']].copy()
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(1.0)
    return df


def load_reviews(csv_path):
    """Carga reseñas como refuerzo de score para el historial."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['user_id', 'item_id', 'score'])

    df = pd.read_csv(csv_path)
    df = _lower_columns(df)
    rename_map = {}
    columns = {col.lower(): col for col in df.columns}

    for candidate in ['userid', 'user_id']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'user_id'
            break
    for candidate in ['destinationid', 'destination_id', 'destinoid']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'item_id'
            break
    for candidate in ['rating', 'score']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'score'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'user_id' not in df.columns or 'item_id' not in df.columns:
        return pd.DataFrame(columns=['user_id', 'item_id', 'score'])

    if 'score' not in df.columns:
        df['score'] = 1.0

    df = df[['user_id', 'item_id', 'score']].copy()
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(1.0)
    return df


def load_destinations(csv_path):
    """Carga el catalogo de destinos de viaje."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'No existe el archivo: {csv_path}')

    df = pd.read_csv(csv_path)
    df = _lower_columns(df)
    rename_map = {}
    columns = {col.lower(): col for col in df.columns}

    for candidate in ['destinationid', 'destination_id', 'item_id']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'destination_id'
            break
    for candidate in ['name', 'nombre']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'name'
            break
    for candidate in ['state', 'estado']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'state'
            break
    for candidate in ['type', 'tipo']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'type'
            break
    for candidate in ['popularity', 'popularidad']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'popularity'
            break
    for candidate in ['besttimetovisit', 'best_time_to_visit']:
        if candidate in columns:
            rename_map[columns[candidate]] = 'best_time_to_visit'
            break

    if rename_map:
        df = df.rename(columns=rename_map)

    if 'destination_id' not in df.columns:
        raise ValueError('El catalogo debe incluir DestinationID.')

    if 'name' not in df.columns:
        df['name'] = df['destination_id'].astype(str)
    if 'state' not in df.columns:
        df['state'] = ''
    if 'type' not in df.columns:
        df['type'] = ''
    if 'popularity' not in df.columns:
        df['popularity'] = 0.0
    if 'best_time_to_visit' not in df.columns:
        df['best_time_to_visit'] = ''

    df['destination_id'] = df['destination_id'].astype(str)
    df['name'] = df['name'].astype(str)
    df['state'] = df['state'].astype(str)
    df['type'] = df['type'].astype(str)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0)
    df['best_time_to_visit'] = df['best_time_to_visit'].astype(str)
    return df[['destination_id', 'name', 'state', 'type', 'popularity', 'best_time_to_visit']].copy()


def build_interactions(users_df, history_df, reviews_df):
    """Une historial y reseñas en una sola tabla de interacciones."""
    interactions = []

    if not history_df.empty:
        history = history_df.copy()
        history['user_id'] = history['user_id'].astype(str)
        history['item_id'] = history['item_id'].astype(str)
        interactions.append(history)

    if not reviews_df.empty:
        reviews = reviews_df.copy()
        reviews['user_id'] = reviews['user_id'].astype(str)
        reviews['item_id'] = reviews['item_id'].astype(str)
        interactions.append(reviews)

    if not interactions:
        return pd.DataFrame(columns=['user_id', 'item_id', 'score'])

    merged = pd.concat(interactions, ignore_index=True)
    merged['score'] = pd.to_numeric(merged['score'], errors='coerce').fillna(1.0)
    merged = merged.groupby(['user_id', 'item_id'], as_index=False)['score'].mean()
    return merged


def build_user_item_matrix(interactions_df):
    return interactions_df.pivot_table(index='user_id', columns='item_id', values='score', aggfunc='sum', fill_value=0)


def train_item_similarity_model(user_item_matrix):
    item_matrix = user_item_matrix.T
    similarity = cosine_similarity(item_matrix)
    return pd.DataFrame(similarity, index=item_matrix.index, columns=item_matrix.index)


def build_text_profiles(users_df, destinations_df):
    """Crea perfiles de texto para usuarios y destinos usando preferencias y atributos."""
    users = users_df.copy()
    users['user_text'] = (
        users.get('preferences', '').fillna('').astype(str)
        + ' '
        + users.get('gender', '').fillna('').astype(str)
    ).str.strip()

    destinations = destinations_df.copy()
    destinations['destination_text'] = (
        destinations['name'].fillna('').astype(str)
        + ' '
        + destinations['state'].fillna('').astype(str)
        + ' '
        + destinations['type'].fillna('').astype(str)
        + ' '
        + destinations['best_time_to_visit'].fillna('').astype(str)
    ).str.strip()

    corpus = pd.concat([users['user_text'], destinations['destination_text']], ignore_index=True).fillna('')
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)

    user_vectors = tfidf[: len(users)]
    destination_vectors = tfidf[len(users) :]
    return users, destinations, user_vectors, destination_vectors, vectorizer


def extract_preference_tokens(users_df):
    """Convierte preferencias textuales a una lista de tokens simples por usuario."""
    preference_map = {}
    if 'preferences' not in users_df.columns:
        return preference_map

    for _, row in users_df.iterrows():
        user_id = str(row['user_id'])
        raw_preferences = str(row.get('preferences', ''))
        tokens = [token.strip().lower() for token in raw_preferences.replace(';', ',').split(',') if token.strip()]
        preference_map[user_id] = tokens
    return preference_map


def content_scores_for_user(user_id, users_df, destinations_df, user_vectors, destination_vectors):
    user_id = str(user_id)
    if user_id not in users_df['user_id'].astype(str).values:
        return pd.Series(0.0, index=destinations_df['destination_id'].astype(str))

    user_idx = users_df.index[users_df['user_id'].astype(str) == user_id][0]
    similarities = cosine_similarity(user_vectors[user_idx], destination_vectors)[0]
    return pd.Series(similarities, index=destinations_df['destination_id'].astype(str))


def recommend_hybrid(user_id, users_df, destinations_df, user_item_matrix, similarity_df, user_vectors, destination_vectors, top_n=5, alpha=0.6, beta=0.4):
    """Combina colaborativo e informacion de contenido."""
    user_id = str(user_id)
    collaborative_scores = pd.Series(0.0, index=destinations_df['destination_id'].astype(str))

    if user_id in user_item_matrix.index:
        user_vector = user_item_matrix.loc[user_id]
        seen_items = set(user_vector[user_vector > 0].index)
        scores = defaultdict(float)
        for seen_item in seen_items:
            if seen_item not in similarity_df.columns:
                continue
            item_similarities = similarity_df[seen_item].sort_values(ascending=False)
            for candidate_item, similarity in item_similarities.items():
                if candidate_item in seen_items:
                    continue
                scores[candidate_item] += float(similarity)
        if scores:
            collaborative_scores = pd.Series(scores)

    content_scores = content_scores_for_user(user_id, users_df, destinations_df, user_vectors, destination_vectors)

    preference_map = extract_preference_tokens(users_df)
    preference_tokens = preference_map.get(user_id, [])
    if preference_tokens:
        preference_boost = pd.Series(0.0, index=destinations_df['destination_id'].astype(str))
        for token in preference_tokens:
            matches = (
                destinations_df['name'].str.contains(token, case=False, na=False)
                | destinations_df['state'].str.contains(token, case=False, na=False)
                | destinations_df['type'].str.contains(token, case=False, na=False)
                | destinations_df['best_time_to_visit'].str.contains(token, case=False, na=False)
            )
            preference_boost = preference_boost + np.where(matches, 1.0, 0.0)
        if preference_boost.max() > 0:
            preference_boost = preference_boost / preference_boost.max()
        content_scores = 0.7 * content_scores.reindex(preference_boost.index).fillna(0.0) + 0.3 * preference_boost

    all_items = destinations_df['destination_id'].astype(str).tolist()
    collaborative_scores = collaborative_scores.reindex(all_items).fillna(0.0)
    content_scores = content_scores.reindex(all_items).fillna(0.0)

    popularity = destinations_df.set_index('destination_id')['popularity'].reindex(all_items).fillna(0.0)
    final_scores = (alpha * collaborative_scores) + (beta * content_scores) + (0.20 * popularity)

    ranked = final_scores.sort_values(ascending=False).head(top_n).index.tolist()
    return destinations_df[destinations_df['destination_id'].astype(str).isin(ranked)].copy().assign(final_score=lambda x: x['destination_id'].astype(str).map(final_scores))


def recommend_from_catalog(destinations_df, top_n=5, preferred_type=None, preferred_state=None, min_popularity=None):
    candidates = destinations_df.copy()
    candidates['type_boost'] = 0.0
    candidates['state_boost'] = 0.0

    if preferred_type:
        candidates['type_boost'] = np.where(candidates['type'].str.contains(str(preferred_type), case=False, na=False), 1.0, 0.0)
    if preferred_state:
        candidates['state_boost'] = np.where(candidates['state'].str.contains(str(preferred_state), case=False, na=False), 1.0, 0.0)
    if min_popularity is not None:
        candidates = candidates[candidates['popularity'] >= float(min_popularity)]

    candidates['final_score'] = candidates['popularity'] + (0.75 * candidates['type_boost']) + (0.5 * candidates['state_boost'])
    return candidates.sort_values(['final_score', 'popularity'], ascending=False).head(top_n)


def leave_one_out_split(interactions_df, seed=42):
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []
    for _, user_df in interactions_df.groupby('user_id'):
        if len(user_df) < 2:
            train_rows.append(user_df)
            continue
        test_index = rng.choice(user_df.index)
        test_rows.append(user_df.loc[[test_index]])
        train_rows.append(user_df.drop(index=test_index))
    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=interactions_df.columns)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=interactions_df.columns)
    return train_df, test_df


def evaluate_recommendations(train_df, test_df, destinations_df, users_df, top_n=5, alpha=0.6, beta=0.4):
    if test_df.empty:
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0}

    user_item_matrix = build_user_item_matrix(train_df)
    similarity_df = train_item_similarity_model(user_item_matrix)
    users_df, destinations_df, user_vectors, destination_vectors, _ = build_text_profiles(users_df, destinations_df)

    precision_scores = []
    recall_scores = []

    for user_id, user_test_df in test_df.groupby('user_id'):
        relevant = set(user_test_df['item_id'].astype(str).tolist())
        if not relevant:
            continue
        recommended_df = recommend_hybrid(
            user_id=user_id,
            users_df=users_df,
            destinations_df=destinations_df,
            user_item_matrix=user_item_matrix,
            similarity_df=similarity_df,
            user_vectors=user_vectors,
            destination_vectors=destination_vectors,
            top_n=top_n,
            alpha=alpha,
            beta=beta,
        )
        recommended = set(recommended_df['destination_id'].astype(str).tolist())
        hits = len(recommended & relevant)
        precision_scores.append(hits / top_n)
        recall_scores.append(hits / len(relevant))

    return {
        'precision_at_k': float(np.mean(precision_scores)) if precision_scores else 0.0,
        'recall_at_k': float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metricas_recomendacion.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f'Metricas guardadas en {metrics_path}')


def save_recommendations(recommendations_df, output_dir, user_label):
    os.makedirs(output_dir, exist_ok=True)
    rec_path = os.path.join(output_dir, f'recomendaciones_usuario_{user_label}.csv')
    recommendations_df.to_csv(rec_path, index=False)
    print(f'Recomendaciones guardadas en {rec_path}')


def save_recommendation_report(output_dir, metrics=None, example_recommendations=None, notes=None):
    """Guarda un reporte markdown con interpretación del sistema de recomendación."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'reporte_recomendacion.md')

    lines = []
    lines.append('# Reporte del sistema de recomendacion')
    lines.append('')
    lines.append('## Resumen metodologico')
    lines.append('- Se usa un enfoque hibrido: filtrado colaborativo por similitud item-item + contenido por preferencias y atributos de destino.')
    lines.append('- Las reseñas y el historial se combinan para construir interacciones usuario-destino.')
    lines.append('- El catálogo de destinos se usa como base de contenido y popularidad.')
    lines.append('')

    if metrics:
        lines.append('## Metricas de evaluacion')
        lines.append(f"- Precision@K: {metrics.get('precision_at_k', 0.0):.4f}")
        lines.append(f"- Recall@K: {metrics.get('recall_at_k', 0.0):.4f}")
        lines.append('')

    lines.append('## Analisis de efectividad')
    if metrics and metrics.get('precision_at_k', 0.0) > 0 and metrics.get('recall_at_k', 0.0) > 0:
        lines.append('- El sistema recupera una fraccion pequena de los destinos relevantes dentro del top-K.')
        lines.append('- Esto indica que ya existe señal util, pero el ranking todavia es sensible al desbalance entre interacciones y catalogo amplio.')
        lines.append('- Para una version de produccion se recomienda enriquecer el historial con mas interacciones y explorar un modelo de aprendizaje por ranking.')
    else:
        lines.append('- No fue posible medir una señal robusta de efectividad con el historial disponible.')
    lines.append('- No se cuenta con una variable directa de satisfaccion del usuario, asi que la efectividad se interpreta a partir de Precision@K, Recall@K y coherencia de los ejemplos generados.')
    lines.append('')

    if example_recommendations:
        lines.append('## Ejemplos de recomendaciones')
        for user_label, rec_df in example_recommendations.items():
            lines.append(f'### Usuario {user_label}')
            preview = rec_df[['destination_id', 'name', 'state', 'type', 'popularity', 'best_time_to_visit', 'final_score']].head(5)
            for _, row in preview.iterrows():
                lines.append(f"- {row['destination_id']} | {row['name']} | {row['state']} | {row['type']} | score={row['final_score']:.4f}")
            lines.append('')

    lines.append('## Medidas para mejorar')
    lines.append('- Incorporar mas historial de interacciones y sesiones de viaje reales.')
    lines.append('- Usar feedback explicito de clics, reservas o conversiones para medir satisfaccion.')
    lines.append('- Probar un modelo de ranking mas avanzado o un sistema de filtrado por contenido con embeddings.')
    lines.append('- Ajustar pesos del componente colaborativo y de contenido segun pruebas A/B.')
    lines.append('')

    if notes:
        lines.append('## Notas')
        lines.extend([f'- {note}' for note in notes])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Reporte guardado en {report_path}')
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Sistema de recomendacion de destinos de viaje')
    parser.add_argument('--users-file', default=DEFAULT_USERS_PATH, help='CSV de usuarios')
    parser.add_argument('--history-file', default=DEFAULT_HISTORY_PATH, help='CSV de historial de viajes')
    parser.add_argument('--reviews-file', default=DEFAULT_REVIEWS_PATH, help='CSV de reseñas')
    parser.add_argument('--destinations-file', default=DEFAULT_DESTINATIONS_PATH, help='CSV del catalogo de destinos')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Carpeta de salida')
    parser.add_argument('--user-id', default=None, help='Usuario para recomendaciones puntuales')
    parser.add_argument('--preferred-type', default=None, help='Tipo de destino preferido para modo contenido')
    parser.add_argument('--preferred-state', default=None, help='Estado/region preferida para modo contenido')
    parser.add_argument('--min-popularity', type=float, default=None, help='Popularidad minima para filtrar destinos')
    parser.add_argument('--top-n', type=int, default=5, help='Cantidad de recomendaciones')
    parser.add_argument('--alpha', type=float, default=0.6, help='Peso del componente colaborativo')
    parser.add_argument('--beta', type=float, default=0.4, help='Peso del componente de contenido')
    parser.add_argument('--eval', action='store_true', help='Evalua Precision@K y Recall@K si existe historial')
    args = parser.parse_args()

    users_df = load_users(args.users_file)
    history_df = load_history(args.history_file)
    reviews_df = load_reviews(args.reviews_file)
    destinations_df = load_destinations(args.destinations_file)

    interactions_df = build_interactions(users_df, history_df, reviews_df)
    train_df, test_df = leave_one_out_split(interactions_df) if not interactions_df.empty else (interactions_df, interactions_df)

    train_matrix = build_user_item_matrix(train_df) if not train_df.empty else pd.DataFrame()
    similarity_df = train_item_similarity_model(train_matrix) if not train_matrix.empty else pd.DataFrame()
    users_df, destinations_df, user_vectors, destination_vectors, _ = build_text_profiles(users_df, destinations_df)

    if args.eval and not interactions_df.empty:
        metrics = evaluate_recommendations(train_df, test_df, destinations_df, users_df, top_n=args.top_n, alpha=args.alpha, beta=args.beta)
        print(f"Precision@{args.top_n}: {metrics['precision_at_k']:.4f}")
        print(f"Recall@{args.top_n}: {metrics['recall_at_k']:.4f}")
        save_metrics(metrics, args.output_dir)

        example_recommendations = {}
        for sample_user_id in users_df['user_id'].astype(str).head(3).tolist():
            rec_df = recommend_hybrid(
                user_id=sample_user_id,
                users_df=users_df,
                destinations_df=destinations_df,
                user_item_matrix=train_matrix if not train_matrix.empty else pd.DataFrame(),
                similarity_df=similarity_df if not similarity_df.empty else pd.DataFrame(),
                user_vectors=user_vectors,
                destination_vectors=destination_vectors,
                top_n=args.top_n,
                alpha=args.alpha,
                beta=args.beta,
            )
            if not rec_df.empty:
                example_recommendations[sample_user_id] = rec_df
                save_recommendations(rec_df[['destination_id', 'name', 'state', 'type', 'popularity', 'best_time_to_visit', 'final_score']], args.output_dir, sample_user_id)

        save_recommendation_report(args.output_dir, metrics=metrics, example_recommendations=example_recommendations, notes=[
            'Los usuarios satisfechos no se miden de forma directa en estos CSV; la evaluacion usa Precision@K y Recall@K como proxy.',
            'La baja precision/recall indica que el historial disponible aun es limitado para entrenar un ranking fuerte.'
        ])
    elif args.eval:
        print('No hay historial suficiente para evaluar Precision@K y Recall@K. Sube Final_Updated_Expanded_UserHistory.csv.')

    if args.user_id is not None:
        recommendations_df = recommend_hybrid(
            user_id=args.user_id,
            users_df=users_df,
            destinations_df=destinations_df,
            user_item_matrix=train_matrix if not train_matrix.empty else pd.DataFrame(),
            similarity_df=similarity_df if not similarity_df.empty else pd.DataFrame(),
            user_vectors=user_vectors,
            destination_vectors=destination_vectors,
            top_n=args.top_n,
            alpha=args.alpha,
            beta=args.beta,
        )
        if recommendations_df.empty:
            print(f'No se pudieron generar recomendaciones para el usuario {args.user_id}.')
        else:
            print(f'Recomendaciones para usuario {args.user_id}:')
            print(recommendations_df[['destination_id', 'name', 'state', 'type', 'popularity', 'best_time_to_visit', 'final_score']].to_string(index=False))
            save_recommendations(recommendations_df[['destination_id', 'name', 'state', 'type', 'popularity', 'best_time_to_visit', 'final_score']], args.output_dir, args.user_id)
        return

    if os.path.exists(args.destinations_file):
        content_recommendations = recommend_from_catalog(
            destinations_df,
            top_n=args.top_n,
            preferred_type=args.preferred_type,
            preferred_state=args.preferred_state,
            min_popularity=args.min_popularity,
        )
        print('Recomendaciones basadas en catálogo:')
        print(content_recommendations.to_string(index=False))
        save_recommendations(content_recommendations, args.output_dir, 'catalogo')
        return

    print('No se encontraron archivos válidos para recomendación.')


if __name__ == '__main__':
    main()
