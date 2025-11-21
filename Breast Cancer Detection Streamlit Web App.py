import streamlit as st
import numpy as np
# Import the specific preprocessing for ResNet
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------------------
# Loading the trained CNN model
# ------------------------------
try:
    # Using st.cache_resource for Streamlit to load the model only once
    @st.cache_resource
    def load_my_model():
        return load_model("best_model.h5")

    model = load_my_model()

except FileNotFoundError:
    st.error("Error: Model file 'best_model.h5' not found. Please ensure it is in the same directory as app.py.")
    st.stop()  # Stopping the app if the model can't be loaded

# ---------------------------------------------
# Streamlit UI setup
# ---------------------------------------------
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🧬", layout="centered")
st.title("🧬 Breast Cancer Detection App")
st.write("Upload a microscopic image of breast tissue to check whether cancerous cells are detected.")

# ---------------------------------------------
# File uploader
# ---------------------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Displaying uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------------------------------------
    # Preprocessing image (with correct ResNet normalization)
    # ---------------------------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Applying ResNet's ImageNet normalization
    img_array = preprocess_input(img_array)

    # ---------------------------------------------
    # Making prediction and interpret
    # ---------------------------------------------
    prediction = model.predict(img_array)

    # Class '0' = Benign (Index 0), Class '1' = Malignant (Index 1)
    prob_benign = float(prediction[0][0])
    prob_malignant = 1.0 - prob_benign

    if prob_benign > 0.5:
        confidence = prob_benign
        st.markdown(f"### 🌱 Prediction: **Benign** (Confidence: {confidence:.2f})")
    else:
        confidence = prob_malignant
        st.markdown(f"### 🩸 Prediction: **Malignant** (Confidence: {confidence:.2f})")

# ---------------------------------------------
# Footer (always visible)
# ---------------------------------------------
st.markdown("---")
st.caption("Developed by Innocent Makaya | Powered by TensorFlow & Streamlit")
