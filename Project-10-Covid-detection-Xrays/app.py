import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="COVID-19 Detection from Chest X-ray",
    page_icon="🩻",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_covid_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.keras")

    if not os.path.exists(model_path):
        st.error("❌ model.keras file not found.")
        st.stop()

    return load_model(model_path)

try:
    model = load_covid_model()
except Exception as e:
    st.error("❌ Error loading model.")
    st.exception(e)
    st.stop()

# -------------------------------
# Title
# -------------------------------
st.title("🩻 COVID-19 Detection from Chest X-ray")
st.write(
    "Upload a Chest X-ray image to predict whether it is **COVID** or **NORMAL**."
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = image.resize((299, 299))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img, verbose=0)

    probability = float(prediction[0][0])

    # Your model:
    # sigmoid -> probability
    # >0.5 = COVID
    # <=0.5 = NORMAL

    if probability > 0.5:
        label = "COVID"
        confidence = probability * 100
        st.error(f"🦠 Prediction: **{label}**")
    else:
        label = "NORMAL"
        confidence = (1 - probability) * 100
        st.success(f"✅ Prediction: **{label}**")

    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(min(confidence / 100, 1.0))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed using TensorFlow & Streamlit")
