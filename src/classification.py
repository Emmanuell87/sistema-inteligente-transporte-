import os
import argparse
import math
import json
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _folder_has_images_or_subfolders(folder):
    """Indica si una carpeta contiene imágenes o subcarpetas útiles."""
    folder = Path(folder)
    if not folder.exists():
        return False
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            return True
        if item.is_dir():
            return True
    return False


def discover_class_folders(dataset_root):
    """Detecta carpetas de clases dentro del dataset crudo."""
    root = Path(dataset_root)

    train_dir = root / 'train'
    val_dir = root / 'val'
    test_dir = root / 'test'
    if all(_folder_has_images_or_subfolders(path) for path in [train_dir, val_dir, test_dir]):
        # Ya está organizado en splits reales.
        return None

    class_dirs = [p for p in root.iterdir() if p.is_dir()]

    # Caso común: la carpeta raíz contiene un directorio envoltorio con las clases reales.
    best_candidate = None
    best_count = 0
    for candidate in class_dirs:
        nested_classes = [p for p in candidate.iterdir() if p.is_dir()]
        if len(nested_classes) > best_count:
            best_candidate = candidate
            best_count = len(nested_classes)

    if best_candidate is not None and best_count > 0:
        return best_candidate

    # Si no hay subdirectorios internos con clases, se asume que la raíz contiene las clases directamente.
    if class_dirs:
        return root

    return None


def list_images(class_dir):
    """Lista todas las imágenes válidas dentro de una carpeta de clase."""
    images = []
    for item in Path(class_dir).iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(item)
    return sorted(images)


def create_stratified_split(dataset_root, output_root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, force=False):
    """Crea un split 80/10/10 por clase copiando imágenes a carpetas train/val/test.

    La división se hace por clase para conservar la distribución original.
    Si el split ya existe y force=False, se reutiliza.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError('Las proporciones train/val/test deben sumar 1.0')

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    if output_root.exists() and not force:
        has_train = (output_root / 'train').exists()
        has_val = (output_root / 'val').exists()
        has_test = (output_root / 'test').exists()
        if has_train and has_val and has_test:
            print(f'Usando split existente en {output_root}')
            return str(output_root)

    source_root = discover_class_folders(dataset_root)
    if source_root is None:
        raise ValueError(
            f'No se encontraron carpetas de clase en {dataset_root}. '
            'Debe existir una carpeta raíz con subcarpetas por clase.'
        )

    class_dirs = [p for p in source_root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        (output_root / split_name).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    print(f'Creando split estratificado en {output_root}')

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            print(f'Advertencia: la clase {class_dir.name} no tiene imágenes válidas.')
            continue

        rng.shuffle(images)
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        split_map = {
            'train': train_files,
            'val': val_files,
            'test': test_files,
        }

        print(
            f'Clase {class_dir.name}: total={total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}'
        )

        for split_name, files in split_map.items():
            target_dir = output_root / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, target_dir / src.name)

    return str(output_root)


def prepare_generators(base_dir, img_size=(224, 224), batch_size=32):
    """Crea ImageDataGenerators para train/val/test.

    Espera la estructura:
      base_dir/train/<class>/*.jpg
      base_dir/val/<class>/*.jpg
      base_dir/test/<class>/*.jpg

    Retorna (train_gen, val_gen, test_gen, class_indices)
    """
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_aug = ImageDataGenerator(rescale=1./255)
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_aug.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = test_aug.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen, train_gen.class_indices


def build_model(num_classes, img_size=(224, 224), lr=1e-4, dropout=0.4):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)

    # Congelar la base al inicio
    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


def compute_class_weights_from_generator(train_gen):
    """Calcula pesos de clase para mitigar desbalance entre categorías."""
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_gen.classes)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def unfreeze_last_layers(base_model, last_layers=40):
    """Descongela las últimas capas del backbone para fine-tuning suave."""
    for layer in base_model.layers:
        layer.trainable = False

    if last_layers <= 0:
        return

    for layer in base_model.layers[-last_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def save_confusion_matrix(y_true, y_pred, class_names, outputs_dir):
    """Guarda la matriz de confusión en CSV y PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = os.path.join(outputs_dir, 'matriz_confusion_distraccion.csv')
    cm_df.to_csv(cm_csv)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Matriz de confusión')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > threshold else 'black')

    plt.tight_layout()
    cm_png = os.path.join(outputs_dir, 'matriz_confusion_distraccion.png')
    plt.savefig(cm_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en {cm_csv} y {cm_png}")


def train_model(base_dir, models_dir, outputs_dir, img_size=(224, 224), batch_size=32, epochs=30, split_dir=None, force_split=False, fine_tune_layers=40):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Si el dataset viene como carpeta cruda con clases, se crea automáticamente el split 80/10/10.
    if split_dir is None:
        split_dir = os.path.join(outputs_dir, 'distraccion_images_split')

    prepared_dir = create_stratified_split(base_dir, split_dir, force=force_split)

    train_gen, val_gen, test_gen, class_indices = prepare_generators(prepared_dir, img_size, batch_size)
    num_classes = len(class_indices)

    class_map_path = os.path.join(outputs_dir, 'clases_distraccion.json')
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump({str(v): k for k, v in class_indices.items()}, f, ensure_ascii=False, indent=2)
    print(f'Mapa de clases guardado en {class_map_path}')

    model, base_model = build_model(num_classes, img_size)
    class_weights = compute_class_weights_from_generator(train_gen)
    print(f'Pesos de clase: {class_weights}')

    checkpoint_path = os.path.join(models_dir, 'distraccion_mobilenet_best.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)

    steps_per_epoch = max(1, math.ceil(train_gen.samples / batch_size))
    val_steps = max(1, math.ceil(val_gen.samples / batch_size))

    head_epochs = max(5, epochs // 2)
    fine_tune_epochs = max(5, epochs - head_epochs)

    print(f'Fase 1 (head entrenable): {head_epochs} epochs')
    history_head = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=head_epochs,
        callbacks=[checkpoint, early],
        class_weight=class_weights
    )

    print(f'Fase 2 (fine-tuning): últimas {fine_tune_layers} capas del backbone')
    unfreeze_last_layers(base_model, last_layers=fine_tune_layers)
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_ft = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_ft = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history_ft = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=fine_tune_epochs,
        callbacks=[checkpoint_ft, early_ft],
        class_weight=class_weights
    )

    # Evaluar en conjunto de test
    test_steps = max(1, math.ceil(test_gen.samples / batch_size))
    test_gen.reset()
    preds = model.predict(test_gen, steps=test_steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)

    # Obtener verdaderas etiquetas en el mismo orden del generador de test.
    y_true = test_gen.classes[:len(y_pred)]

    # Map indices -> labels
    idx2label = {v: k for k, v in train_gen.class_indices.items()}

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=[idx2label[i] for i in sorted(idx2label.keys())], zero_division=0)

    metrics = {'accuracy': acc, 'f1_weighted': f1, 'precision_weighted': prec, 'recall_weighted': rec}
    metrics_df = pd.DataFrame([{'dataset': 'distraccion_images', **metrics}])
    metrics_csv = os.path.join(outputs_dir, 'metricas_distraccion.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Métricas guardadas en {metrics_csv}")
    print(report)

    report_csv = os.path.join(outputs_dir, 'reporte_clasificacion_distraccion.csv')
    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=[idx2label[i] for i in sorted(idx2label.keys())],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(report_csv)
    print(f"Reporte de clasificación guardado en {report_csv}")

    save_confusion_matrix(y_true, y_pred, [idx2label[i] for i in sorted(idx2label.keys())], outputs_dir)

    # Guardar ejemplos (correctos y errores)
    save_examples(test_gen, y_true, y_pred, idx2label, outputs_dir)

    # Guardar modelo final
    final_model_path = os.path.join(models_dir, 'distraccion_mobilenet_final.keras')
    model.save(final_model_path)
    print(f"Modelo guardado en {final_model_path}")

    return model, metrics


def build_class_map_from_split(split_dir):
    """Reconstruye el mapa indice->clase a partir del split estratificado guardado."""
    train_dir = Path(split_dir) / 'train'
    if not train_dir.exists():
        raise FileNotFoundError(f'No existe el directorio de entrenamiento: {train_dir}')

    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise FileNotFoundError(f'No se encontraron clases dentro de {train_dir}')

    return {idx: name for idx, name in enumerate(class_names)}


def load_class_map(class_map_path, split_dir=None):
    """Carga el mapa indice->clase o lo reconstruye desde el split si el JSON no existe."""
    if class_map_path and os.path.exists(class_map_path):
        with open(class_map_path, 'r', encoding='utf-8') as f:
            raw_map = json.load(f)
        return {int(k): v for k, v in raw_map.items()}

    if split_dir:
        print(f'Aviso: no se encontró {class_map_path}. Se reconstruirá el mapa desde {split_dir}.')
        return build_class_map_from_split(split_dir)

    raise FileNotFoundError(
        f'No se encontró el mapa de clases en {class_map_path} y no se proporcionó split_dir.'
    )


def predict_single_image(model_path, image_path, class_map_path, img_size=(224, 224), split_dir=None):
    """Predice una imagen individual y muestra la clase más probable."""
    model = load_model(model_path)
    idx2label = load_class_map(class_map_path, split_dir=split_dir)

    image = load_img(image_path, target_size=img_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probs = model.predict(image_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx2label[pred_idx]
    confidence = float(probs[pred_idx])

    print(f'Imagen: {image_path}')
    print(f'Predicción: {pred_label} (confianza={confidence:.4f})')
    print('Probabilidades por clase:')
    for idx in np.argsort(probs)[::-1]:
        print(f'  - {idx2label[int(idx)]}: {float(probs[idx]):.4f}')

    return pred_label, confidence, probs


def save_examples(generator, y_true, y_pred, idx2label, outputs_dir, max_examples=30):
    """Guarda ejemplos clasificados correctamente y los errores."""
    correct_dir = os.path.join(outputs_dir, 'img', 'distraccion', 'correct')
    wrong_dir = os.path.join(outputs_dir, 'img', 'distraccion', 'wrong')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    # generator.filenames corresponde a la lista de archivos relativa al directorio del generator
    filenames = generator.filenames[:len(y_pred)]
    base_dir = generator.directory

    saved_correct = 0
    saved_wrong = 0
    for i, fname in enumerate(filenames):
        src = os.path.join(base_dir, fname)
        true_label = idx2label[y_true[i]]
        pred_label = idx2label[y_pred[i]]
        dst_dir = correct_dir if true_label == pred_label else wrong_dir
        dst_name = f"{i}_{true_label}_pred_{pred_label}_{os.path.basename(fname)}"
        dst = os.path.join(dst_dir, dst_name)
        try:
            from shutil import copyfile
            copyfile(src, dst)
            if true_label == pred_label:
                saved_correct += 1
            else:
                saved_wrong += 1
        except Exception:
            continue
        if saved_correct >= max_examples and saved_wrong >= max_examples:
            break
    print(f"Guardados ejemplos: correct={saved_correct}, wrong={saved_wrong}")


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento clasificación de conducción distractiva')
    parser.add_argument('--data-dir', default='data/raw/distraccion_images', help='Carpeta base con clases o con train/val/test')
    parser.add_argument('--models-dir', default='models', help='Carpeta para guardar modelos')
    parser.add_argument('--outputs-dir', default='outputs', help='Carpeta para guardar métricas/ejemplos')
    parser.add_argument('--model-path', default='models/distraccion_mobilenet_final.keras', help='Ruta del modelo entrenado para inferencia')
    parser.add_argument('--class-map', default='outputs/clases_distraccion.json', help='Ruta del mapa de clases índice->nombre')
    parser.add_argument('--split-dir', default='outputs/distraccion_images_split', help='Carpeta del split estratificado para reconstruir clases si falta el JSON')
    parser.add_argument('--predict-image', default=None, help='Ruta de una imagen individual para probar el modelo')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--fine-tune-layers', type=int, default=40, help='Número de capas finales del backbone a descongelar')
    parser.add_argument('--force-split', action='store_true', help='Regenera el split aunque ya exista')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    args = parser.parse_args()

    if args.predict_image:
        predict_single_image(
            args.model_path,
            args.predict_image,
            args.class_map,
            img_size=(args.img_size, args.img_size),
            split_dir=args.split_dir,
        )
    elif args.train:
        train_model(
            args.data_dir,
            args.models_dir,
            args.outputs_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            split_dir=args.split_dir,
            force_split=args.force_split,
            fine_tune_layers=args.fine_tune_layers
        )
    else:
        print('Modo interactivo: use --train para entrenar o --predict-image para probar una imagen.')


if __name__ == '__main__':
    main()
