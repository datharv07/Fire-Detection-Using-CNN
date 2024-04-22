import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the saved model
loaded_model = load_model("fire_detection_model.h5")

# Function to preprocess input image
def preprocess_image(image_path):
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (256, 256))
    resized_array = np.expand_dims(resized_image, axis=0)
    return resized_array

# Function to make predictions using the loaded model
def make_predictions(input_data):
    predictions = loaded_model.predict(input_data)
    class_probabilities = predictions.flatten()
    predicted_class = "Fire" if (class_probabilities[0] > class_probabilities[1]) or (0.43 < class_probabilities[0]) else "Non-Fire"
    return predicted_class, class_probabilities

# Function to display the image with title and prediction
def display_image(image_rgb, predicted_class, class_probabilities):
    st.markdown(f"<p style='text-align: center; font-size: 18px;'>Predicted For Fire or Non-Fire : {predicted_class}</p>", unsafe_allow_html=True)
    st.image(image_rgb, caption='', use_column_width=True, width=300)



# Streamlit app
def main():
    st.title("Fire Detection App")
    st.write("Upload an image and let's detect if it contains fire or not!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        

        # Preprocess the image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_rgb = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(cv2.resize(image_rgb, (256, 256)), axis=0)

        # Make predictions
        predicted_class, class_probabilities = make_predictions(input_data)

        # Display the image with predictions
        display_image(image_rgb, predicted_class, class_probabilities)

        

if __name__ == "__main__":
    main()
