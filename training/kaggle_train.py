"""Kaggle GPU training entry point for kidney disease CT classification.

How to use in a Kaggle notebook:
1. Add the CT kidney dataset to the notebook.
2. Enable GPU: Notebook settings -> Accelerator -> GPU.
3. Upload/copy this repo or this script, then run: !python training/kaggle_train.py

Outputs are written to /kaggle/working/artifacts on Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf


IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", "10"))
SEED = int(os.getenv("SEED", "42"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))

DEFAULT_KAGGLE_INPUT = Path("/kaggle/input")
DEFAULT_OUTPUT = Path("/kaggle/working/artifacts")
LOCAL_OUTPUT = Path("artifacts")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected. Training will run on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("GPU detected. Enabled mixed precision for faster training.")
    except Exception as exc:  # pragma: no cover - defensive for older TF builds.
        print(f"GPU detected, but mixed precision was not enabled: {exc}")


def contains_images(path: Path) -> bool:
    try:
        return any(file.suffix.lower() in IMAGE_EXTENSIONS for file in path.rglob("*"))
    except PermissionError:
        return False


def class_dir_count(path: Path) -> int:
    try:
        return sum(1 for child in path.iterdir() if child.is_dir() and contains_images(child))
    except PermissionError:
        return 0


def find_dataset_path() -> Path:
    configured = os.getenv("DATASET_PATH")
    if configured:
        path = Path(configured)
        if not path.exists():
            raise FileNotFoundError(f"DATASET_PATH does not exist: {path}")
        return path

    if DEFAULT_KAGGLE_INPUT.exists():
        candidates = [path for path in DEFAULT_KAGGLE_INPUT.rglob("*") if path.is_dir() and class_dir_count(path) >= 2]
        if candidates:
            # Prefer the shallowest valid folder so class directories stay directly below it.
            return sorted(candidates, key=lambda p: len(p.parts))[0]

    local_candidates = [path for path in Path.cwd().iterdir() if path.is_dir() and class_dir_count(path) >= 2]
    if local_candidates:
        return sorted(local_candidates, key=lambda p: len(p.parts))[0]

    raise FileNotFoundError(
        "Could not find a dataset folder. Set DATASET_PATH to the folder containing class subfolders."
    )


def build_datasets(dataset_path: Path):
    common = dict(
        directory=str(dataset_path),
        labels="inferred",
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **common)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    preprocess = tf.keras.applications.xception.preprocess_input
    train_ds = train_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def callbacks(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=0.001,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.weights.h5"),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
        ),
    ]


def fine_tune(model: tf.keras.Model, train_ds, val_ds, output_dir: Path) -> None:
    if FINE_TUNE_EPOCHS <= 0:
        return

    base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks(output_dir))


def main() -> None:
    configure_gpu()
    dataset_path = find_dataset_path()
    output_dir = DEFAULT_OUTPUT if Path("/kaggle/working").exists() else LOCAL_OUTPUT

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_dir}")

    train_ds, val_ds, class_names = build_datasets(dataset_path)
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names))
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks(output_dir))
    fine_tune(model, train_ds, val_ds, output_dir)

    model.save(output_dir / "KidneyDiseaseDetection.keras")
    model.save(output_dir / "KidneyDiseaseDetection.h5")
    print("Saved model files:")
    print(output_dir / "KidneyDiseaseDetection.keras")
    print(output_dir / "KidneyDiseaseDetection.h5")


if __name__ == "__main__":
    main()

