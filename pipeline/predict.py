import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
MODEL_PATH = "artifacts\KidneyDiseaseDetection.h5"  # Ensure the file exists in the same directory
model = load_model(MODEL_PATH)

# Define class names (modify as per your dataset)
class_names = ["Cyst", "Normal", "Stone", "Tumor"]

def preprocess_image(img):
    """Preprocesses the image for model prediction."""
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_class(img):
    """Predicts the class of an input image."""
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)  # Get highest probability index
    confidence = np.max(predictions)  # Get confidence score
    return class_names[predicted_class], confidence, predictions[0]
