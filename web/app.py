import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer,WebRtcMode
from collections import deque
from typing import List
import queue

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

    # Define the video processor class
    # this class is used to process the video frames in real time
    class VideoProcessor:
        result_queue: queue.Queue = queue.Queue()
        def recv(self, frame):
            # Convert the frame to numpy array
            frame_array = np.array(frame.to_ndarray(format="bgr24"))
            # Predict the emotion
            predicted_emotion = predict_emotion(frame_array)
            print(predicted_emotion)
            # Update the text in the placeholder
            self.result_queue.put(predicted_emotion)
            # Send the predicted emotion to the Streamlit UI
            return frame
    
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
        labels_placeholder = st.empty() 
        # we use webrtc_streamer to capture video frames from webcam 
        #  if we use opencv to capture video frames, we cannot use the 
        #  get user video only server video
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
             video_processor_factory=VideoProcessor,
            rtc_configuration={  # Add this line
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": True},
        )
        st.text("Please note emotion is predicted every second.")
        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.state.playing: 
                # NOTE: The video transformation with object detection and 
                # this loop displaying the result labels are running 
                # in different threads asynchronously. 
                # Then the rendered video frames and the labels displayed here 
                # are not strictly synchronized. 
                while True: 
                    if webrtc_ctx.video_processor.result_queue: 
                        try: 
                            print(webrtc_ctx.video_processor.result_queue.qsize())
                            result = webrtc_ctx.video_processor.result_queue.get( 
                                timeout=1.0 
                            ) 
                        except queue.Empty: 
                            result = None 
                        labels_placeholder.markdown('<h3>Predicted Emotion: '+ result +'<h3>', unsafe_allow_html=True) 
                    else: 
                        break 

if __name__ == '__main__':
    main()
