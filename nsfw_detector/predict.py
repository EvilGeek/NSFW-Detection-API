import io
import numpy as np
from os.path import exists
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

IMAGE_DIM = 224   # required/default image dimensionality


def load_image_from_bytes(image_bytes, image_size):
    try:
        # Open the image from byte stream
        image = Image.open(io.BytesIO(image_bytes))

        # Resize the image to the required dimensions
        image = image.resize(image_size)

        # Convert the image to a numpy array and normalize it
        image_array = img_to_array(image)
        image_array = image_array / 255.0  # Normalize to [0, 1]

        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as ex:
        print(f"Failed to process image from bytes: {ex}")
        return None

def classify_bytes(model, image_bytes, image_dim=IMAGE_DIM):
    '''
    Classify the image provided as bytes using the given model.

    inputs:
        model: the loaded model to use for prediction
        image_bytes: image in bytes format to classify
        image_dim: dimension to which the image should be resized

    output:
        predictions: model predictions for the provided image in the same format as 'classify'
    '''
    # Load image from bytes and resize
    image_array = load_image_from_bytes(image_bytes, (image_dim, image_dim))

    if image_array is None:
        return {"error": "Could not process the image"}

    # No need to add batch dimension again, image_array already has it
    # Make predictions on the loaded image
    probs = model.predict(image_array)

    # Categories to map the probabilities
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    #print(probs)
    # Convert predictions into a dictionary format with labels as keys
    predictions = {}
    for i, category in enumerate(categories):
        predictions[category] = round(float(probs[0][i]), 6) * 100

    # Return predictions as a dictionary with the key 'data' to match the 'classify' function
    return {'data': predictions}


def load_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError(
            "saved_model_path must be the valid directory of a saved model to load.")

    model = tf.keras.models.load_model(model_path, custom_objects={
                                       'KerasLayer': hub.KerasLayer})
    return model
