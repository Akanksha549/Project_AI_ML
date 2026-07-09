import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Gender Classification AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# THEME TOGGLE
# -------------------------------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    bg_color = "#0F172A"
    card_color = "#1E293B"
    text_color = "#F8FAFC"
    accent = "#60A5FA"
else:
    bg_color = "#F8FAFC"
    card_color = "#FFFFFF"
    text_color = "#1E293B"
    accent = "#2563EB"

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown(
    f"""
    <style>

    html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        background-color: {bg_color};
        color: {text_color};
    }}

    .stApp {{
        background-color: {bg_color};
    }}

    h1 {{
        text-align: center;
        color: {text_color};
        font-weight: 700;
    }}

    .glass {{
        background: {card_color};
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }}

    .footer {{
        text-align: center;
        font-size: 14px;
        color: gray;
        padding-top: 30px;
    }}

    div.stButton > button {{
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        background-color: {accent};
        color: white;
        border: none;
    }}

    div.stButton > button:hover {{
        background-color: #1D4ED8;
        transition: 0.3s;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("---")

    st.subheader("👩‍💻 Developer")

    st.link_button(
        "💻 GitHub",
        "https://github.com/Akanksha549/"
    )

    st.link_button(
        "🔗 LinkedIn",
        "https://www.linkedin.com/in/akanksha-mishra-7894912bb"
    )

    st.markdown("---")

    st.info(
        "Upload a face image and let the CNN model predict "
        "whether it belongs to a Male or Female."
    )

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("🧠 Gender Classification using CNN")

st.caption(
    "Deep Learning • TensorFlow • Keras • Streamlit"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("male_female_eye_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()

# -------------------------------------------------
# IMAGE SETTINGS
# -------------------------------------------------
IMG_SIZE = 128

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("###  Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:

        st.markdown("### 🤖 Prediction")

        if st.button("🚀 Predict"):

            with st.spinner("Analyzing image..."):

                start = time.time()

                img = preprocess_image(image)

                prediction = model.predict(img, verbose=0)[0][0]

                end = time.time()

                inference = end - start

            if prediction >= 0.5:
                label = " Male"
                confidence = prediction
            else:
                label = " Female"
                confidence = 1 - prediction

            st.success(f"Prediction: **{label}**")

            st.metric(
                label="Confidence",
                value=f"{confidence*100:.2f}%"
            )

            st.metric(
                label="Inference Time",
                value=f"{inference:.3f} sec"
            )

            st.progress(float(confidence))

            st.divider()

            st.write("### Prediction Summary")

            st.write(f"**Gender :** {label}")

            st.write(
                f"**Confidence :** {confidence*100:.2f}%"
            )

            st.write(
                f"**Inference Time :** {inference:.3f} sec"
            )
# -------------------------------------------------
# PROBABILITY CHART
# -------------------------------------------------
            st.markdown("### 📊 Prediction Probability")

            female_prob = float(1 - prediction)
            male_prob = float(prediction)

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Female", "Male"],
                        y=[female_prob, male_prob],
                        text=[
                            f"{female_prob*100:.1f}%",
                            f"{male_prob*100:.1f}%"
                        ],
                        textposition="auto",
                        marker_color=["#EC4899", "#3B82F6"]
                    )
                ]
            )

            fig.update_layout(
                height=350,
                template="plotly_white" if not dark_mode else "plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Class",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# ABOUT PROJECT
# -------------------------------------------------
st.divider()

with st.expander("ℹ️ About this Project"):

    st.markdown("""
This application predicts whether the uploaded face belongs to a **Male** or **Female**
using a **Convolutional Neural Network (CNN)** trained with TensorFlow/Keras.

The project demonstrates a complete Deep Learning pipeline:

- Data preprocessing
- CNN model training
- Model evaluation
- Streamlit deployment
- Interactive prediction interface
""")

# -------------------------------------------------
# DATASET
# -------------------------------------------------
with st.expander("📂 Dataset"):

    st.markdown("""
**Dataset Source**

Kaggle Gender Classification Dataset

The dataset contains thousands of labeled face images
used to train the CNN model.

Images were resized and normalized before training.
""")

# -------------------------------------------------
# MODEL DETAILS
# -------------------------------------------------
with st.expander(" Model Details"):

    st.markdown("""
**Framework:** TensorFlow / Keras

**Model:** Convolutional Neural Network (CNN)

**Input Size:** 128 × 128 × 3

**Output:** Binary Classification

**Classes**

- Female
- Male
""")

# -------------------------------------------------
# TECHNOLOGIES
# -------------------------------------------------
with st.expander("⚙️ Technologies Used"):

    st.markdown("""
- Python
- Streamlit
- TensorFlow
- Keras
- NumPy
- Pillow
- Plotly
""")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.link_button(
        "💻 GitHub",
        "https://github.com/Akanksha549/"
    )

with col2:
    st.link_button(
        "🔗 LinkedIn",
        "https://www.linkedin.com/in/akanksha-mishra-7894912bb"
    )

st.markdown(
    """
    <div class="footer">
        <br>
        <b>Gender Classification AI</b><br>
        Built with ❤️ using Streamlit, TensorFlow & Keras<br><br>
        © 2026 Akanksha Mishra
    </div>
    """,
    unsafe_allow_html=True
)
