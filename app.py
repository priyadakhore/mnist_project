import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

st.title("MNIST Digit Classification Web App")
st.write("Upload an image of a **handwritten digit** (28x28) to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")   # Grayscale
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess
    img = image.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array                     # Invert colors (white → black)
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.write(f"### Predicted Digit: **{digit}**")
