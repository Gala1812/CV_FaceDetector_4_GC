from dotenv import load_dotenv
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from image_preprocessing import preprocess

#model_path = "models/siamese_model.h5"

#model = load_model(model_path)

# loading of .env variables for the path
load_dotenv()
model_path = os.getenv('MODEL_PATH')
application_data_path = os.getenv('APPLICATION_DATA_PATH')
verification_images_path = os.path.join(application_data_path, "verification_images")
input_image_path = os.path.join(application_data_path, "input_image", "input_image.jpg")


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(verification_images_path):
        input_image = preprocess(input_image_path)
        validation_img = preprocess(os.path.join(verification_images_path, image))


    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(verification_images_path))
    verified = verification > verification_threshold

    return results, verified