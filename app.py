import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("our_model.h5")

st.title("🩺 Pneumonia Detection App")
st.write("Upload a chest X-ray image and the model will predict if it's Pneumonia or Normal.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # same size as training
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # normalize if used during training

    # Prediction
    prediction = model.predict(x)
    if prediction[0][0] > 0.5:
        st.error("⚠️ Pneumonia detected")
    else:
        st.success("✅ Normal")
