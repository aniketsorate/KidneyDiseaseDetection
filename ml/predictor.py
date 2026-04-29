from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

KERAS_MODEL_PATH = Path("artifacts") / "KidneyDiseaseDetection.keras"
H5_MODEL_PATH = Path("artifacts") / "KidneyDiseaseDetection.h5"
MODEL_PATH = KERAS_MODEL_PATH if KERAS_MODEL_PATH.exists() else H5_MODEL_PATH


def _build_model_from_legacy_weights():
    base = tf.keras.applications.Xception(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
    )
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    model.load_weights(H5_MODEL_PATH)
    return model


def _load_prediction_model():
    try:
        return load_model(MODEL_PATH, compile=False)
    except ValueError:
        return _build_model_from_legacy_weights()


model = _load_prediction_model()

class_names = ["Cyst", "Normal", "Stone", "Tumor"]


def _get_model_image_size():
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])

    return 256, 256


IMAGE_SIZE = _get_model_image_size()

def preprocess_image(img):
    """Preprocesses the image for model prediction."""
    img = img.convert("RGB").resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.xception.preprocess_input(img_array)

def predict_class(img):
    """Predicts the class of an input image."""
    result = predict_image(img)
    return result["diagnosis"], result["confidence"], np.array(result["raw_predictions"])

def predict_image(img):
    """Returns a structured prediction payload for API and UI use."""
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    probabilities = {
        class_name: float(score)
        for class_name, score in zip(class_names, predictions)
    }

    return {
        "diagnosis": class_names[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_predictions": predictions.tolist(),
        "image_size": IMAGE_SIZE,
    }
