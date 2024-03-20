import time
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


cascade_path = "./output/haarcascade_frontalface_default.xml"
face_haar_cascade = cv2.CascadeClassifier(cascade_path)

isProd = st.secrets[
    "isProd"
]  # True if in production, False if in development soo that we can disable real-time detection in production


def get_emotion(gray_img, model_source, uploaded_image, model):
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for x, y, w, h in faces_detected:
        cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[
            y : y + w, x : x + h
        ]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        if model_source == "Bi-lstm":
            img_pixels = img_pixels.reshape(
                1, 48, 48, 1
            )  # needed for bi lstm as input shape while training was (1,48,48,1)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
        predicted_emotion = emotions[max_index]
        return predicted_emotion, x, y
    return "No face detected", 0, 0


def real_time_detection(model_source, model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = (
            cap.read()
        )  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        try:
            predicted_emotion, x, y = get_emotion(
                gray_img, model_source, uploaded_image=test_img, model=model
            )
        except:
            predicted_emotion = "try again"
            x = 0
            y = 0
        cv2.putText(
            test_img,
            predicted_emotion,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow("Face Emotion Recognition", resized_img)

        if cv2.waitKey(10) == ord("q"):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows


def get_image_emotion(uploaded_image, model_source, model):
    gray_img = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    return get_emotion(
        gray_img, model_source, uploaded_image=uploaded_image, model=model
    )[0]


def load_model_function(model_source):
    if model_source == "Bi-lstm":
        # load model
        model = model_from_json(
            open("./output/model_ck_fer.json", "r").read(),
        )
        # load weights
        model.load_weights("./output/model_ck_fer.h5")
    else:
        # load model
        model = model_from_json(
            open("./output/model375.json", "r").read(),
        )
        # load weights
        model.load_weights("./output/model375.h5")
    return model


# Streamlit app
def main():
    st.title("Face Detection and Emotion Recognition")
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "./public/faceImage.jpg",
        )
    with col2:
        st.info(
            "The models are trained using different image datasets. Using tensorflow and keras. Using kaggle and google colab."
            "We used Sequential model from keras to build the model. We used the VGG-16 model and Bi-lstm model."
        )

        st.info(
            "The VGG-16 model is trained on the FER-2013 dataset and has an accuracy of 62.7%. "
            "The Bi-lstm model is trained on the CK+ dataset and fer-2013 dataset and has an accuracy of 68%."
        )

    st.write("""### Model Accuracy and Loss""")

    accuracy_and_loss_image = st.radio(
        label="Select model:",
        options=("hide", "VGG-16", "Bi-lstm", "Both"),
        horizontal=True,
    )

    if accuracy_and_loss_image == "VGG-16" or accuracy_and_loss_image == "Both":
        st.image(
            "./public/epoch_history_vgg-16.png",
        )
        st.write("#### Accuracy: 78%  Loss: 0.6")
    if accuracy_and_loss_image == "Bi-lstm" or accuracy_and_loss_image == "Both":
        st.image(
            "./public/epoch_history_bilstm_final.png",
        )
        st.write("#### Accuracy: 86%  Loss: 0.3")
        st.write("#### Validation Accuracy: 83%  Validation Loss: 0.5")

    st.write("""## Upload an image or use your webcam to detect emotions.""")
    st.write("""#### Choose the model and input source.""")
    col11, col12 = st.columns(2)
    with col11:
        # Choose the model
        model_source = st.radio(
            "Select model:", ("VGG-16(Accuracy: 62.7%)", "Bi-lstm(Accuracy: 68%)")
        )
    with col12:
        # Choose the input source
        input_source = st.radio("Select input source:", ("Upload Image", "Webcam"))

    if input_source == "Upload Image":
        model = load_model_function(model_source)
        # Upload image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Predict the emotion
            predicted_emotion = get_image_emotion(
                np.array(image), model_source, model=model
            )
            if predicted_emotion == "No face detected":
                st.warning("No face detected in the image.")
            # Display the predicted emotion
            st.write(f"""## Predicted Emotion: {predicted_emotion}""")

    elif input_source == "Webcam":
        if isProd:
            st.warning(
                "Real-time detection is disabled in production. Due to WebRTC requirement, it only works in development/local enviroment."
            )
        else:
            model = load_model_function(model_source)
            real_time_detection(model_source, model=model)

    # footer
    link_text = "Made with ❤️ by Aaron Jevil Nazareth, Aaron Francis Dsouza, Akshatha SM, and Adril Vaz"
    link = f'<div style="display: block; text-align: center; padding: 10px; border: 2px solid #172c43; background-color:#172c43; border-radius:5px;">{link_text}</div>'
    st.write(link, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
