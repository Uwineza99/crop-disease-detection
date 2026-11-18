import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/crop_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.txt"

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load the class names
with open(CLASS_NAMES_PATH, "r") as file:
    class_names = [line.strip() for line in file.readlines()]


def predict_image(image_file):
    # Open the uploaded image
    image = Image.open(image_file).convert("RGB")

    # Resize the image
    image = image.resize((128, 128))

    # Convert the image to an array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }