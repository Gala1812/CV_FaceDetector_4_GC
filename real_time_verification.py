import cv2
import numpy as np
import os
from dotenv import load_dotenv
from verification import verify 
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.layers import L1Dist
from tensorflow.keras.utils import custom_object_scope

# loading of .env variables for the path
load_dotenv()


# Path to the saved model
#model_path = os.path.join('Models', 'siamesemodelv2.h5')
model_path = os.getenv('MODEL_PATH')
application_data_path = os.getenv('APPLICATION_DATA_PATH')


with custom_object_scope({'L1Dist': L1Dist}):
    model = load_model(model_path)
print("Modelo cargado con éxito")

# Load the model
#siamese_model = load_model(model_path)
# Initialize the camera
cap = cv2.VideoCapture(0)


while cap.isOpened():
    # Read the frame from the camera
    ret, frame = cap.read()

    # Crop the frame based on the provided coordinates
    resized_frame = frame[50:50 + 218, 120:120 + 178, :]

    # Show the frame in a window named "Verification"
    cv2.imshow("Verification", resized_frame)

    # Wait for the 'v' key to save the image and perform verification
    if cv2.waitKey(10) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join(application_data_path, 'input_image', 'input_image.jpg'), resized_frame)

        # Perform verification
        results, verified = verify(model, 0.5, 0.5)
        print(verified)

         # Wait for the 'q' key to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

np.sum(np.squeeze(results) > 0.9)

# Display results after exiting the loop (adjust as needed)
print(results)