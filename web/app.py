import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time
from keras.models import model_from_json

#load model
model = model_from_json(open("./output/model.json", "r").read())
#load weights
model.load_weights('./output/model.h5')

#load cascade
cascade_path = "./output/haarcascade_frontalface_default.xml"
face_haar_cascade = cv2.CascadeClassifier(cascade_path)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the input size of the model
    resized = cv2.resize(gray, (48, 48))
    # Normalize the image
    normalized = resized / 255.0
    # Reshape the image to match the input shape of the model
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# Function to predict the emotion
def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    # Make the prediction using the pre-trained model
    predictions = model.predict(preprocessed_image)
    # Get the index of the predicted emotion
    predicted_index = np.argmax(predictions)
    # Get the predicted emotion label
    predicted_emotion = emotion_labels[predicted_index]
    return predicted_emotion

# Streamlit app
def main():
    st.title("Real-time Emotion Recognition")
    st.write("Upload an image or use your webcam to detect emotions.")

    # Choose the input source
    input_source = st.radio("Select input source:", ("Upload Image", "Webcam"))

    if input_source == "Upload Image":
        # Upload image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Convert the image to numpy array
            image_array = np.array(image)
            # Predict the emotion
            predicted_emotion = predict_emotion(image_array)
            # Display the predicted emotion
            st.write("Predicted Emotion:", predicted_emotion)

    elif input_source == "Webcam":
        # Open the webcam
        cap = cv2.VideoCapture(0)
        st.text("Please note emotion is predicted every 3 seconds.")
        # Create a placeholder for the image
        image_placeholder = st.empty()
        # Create a placeholder for the text
        text_placeholder = st.empty()
        # Read and display frames from the webcam
        counter = 0
        while True:
            ret, frame = cap.read()
            # Update the image in the placeholder
            image_placeholder.image(frame, channels="BGR", use_column_width=True)
            # Convert the frame to numpy array
            frame_array = np.array(frame)
            
            # Predict the emotion every 3 seconds
            if counter % 30 == 0:
                # Predict the emotion
                predicted_emotion = predict_emotion(frame_array)
                # Update the text in the placeholder
                text_placeholder.text("Predicted Emotion: " + predicted_emotion)

            counter += 1
            time.sleep(0.1)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam
        cap.release()

if __name__ == '__main__':
    main()
