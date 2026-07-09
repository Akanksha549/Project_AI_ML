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
# APP THEME
# -------------------------------------------------
bg_color = "#F4F9FF"
card_color = "#FFFFFF"     
sidebar_color = "#EAF4FF"  
text_color = "#1E3A5F"     
accent = "#3B82F6"          
accent_hover = "#2563EB"   
border = "#D6E8FF"         

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
# -------------------------------------------------
# CALM BLUE THEME
# -------------------------------------------------

bg_color = "#F4F9FF"          # Page background
card_color = "#FFFFFF"        # Cards
sidebar_color = "#EAF4FF"     # Sidebar
text_color = "#1E3A5F"        # Text
accent = "#3B82F6"            # Primary blue
accent_hover = "#2563EB"      # Hover blue
border = "#D6E8FF"            # Borders

st.markdown(
    f"""
<style>

/* ---------------- APP ---------------- */

html, body {{
    background-color: {bg_color};
}}

.stApp {{
    background-color: {bg_color};
}}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {{
    background-color: {sidebar_color};
    border-right: 1px solid {border};
}}

/* ---------------- TEXT ---------------- */

h1, h2, h3, h4, h5, h6,
p, span, label,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {{
    color: {text_color} !important;
}}

/* ---------------- TITLES ---------------- */

h1 {{
    text-align: center;
    font-weight: 700;
}}

/* ---------------- BUTTONS ---------------- */

div.stButton > button {{
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background-color: {accent};
    color: white;
    font-size: 17px;
    font-weight: 600;
}}

div.stButton > button:hover {{
    background-color: {accent_hover};
    color: white;
}}

/* ---------------- LINK BUTTONS ---------------- */

div[data-testid="stLinkButton"] button {{
    width: 100%;
    border-radius: 10px;
    background-color: white;
    color: {text_color};
    border: 1px solid {border};
    font-weight: 600;
}}

div[data-testid="stLinkButton"] button:hover {{
    background-color: {accent};
    color: white;
}}

div[data-testid="stLinkButton"] button * {{
    color: inherit !important;
}}

/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {{
    background-color: white;
    border: 2px dashed {accent};
    border-radius: 15px;
    padding: 10px;
}}

[data-testid="stFileUploader"] * {{
    color: {text_color} !important;
}}

[data-testid="stFileUploader"] button {{
    background-color: {accent} !important;
    color: white !important;
    border-radius: 10px !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background-color: {accent_hover} !important;
}}

/* ---------------- EXPANDERS ---------------- */

div[data-testid="stExpander"] {{
    background-color: white;
    border-radius: 12px;
    border: 1px solid {border};
}}

div[data-testid="stExpander"] * {{
    color: {text_color};
}}

/* ---------------- METRICS ---------------- */

div[data-testid="stMetric"] {{
    background-color: white;
    border: 1px solid {border};
    border-radius: 12px;
    padding: 10px;
}}

/* ---------------- PROGRESS BAR ---------------- */

div[data-testid="stProgressBar"] > div {{
    background-color: {accent};
}}

/* ---------------- FOOTER ---------------- */

.footer {{
    text-align: center;
    color: #64748B;
    font-size: 14px;
}}

</style>
""",
    unsafe_allow_html=True
)
# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("📘 Overview")

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

                prediction = float(model.predict(img, verbose=0)[0][0])

                end = time.time()

                inference = end - start

            #Debug info
            st.write("Raw Prediction Value:", prediction)

            
            if prediction >= 0.5:
                label = " Female"
                confidence = prediction
            else:
                label = " Male"
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
                       marker_color=["#93C5FD", "#2563EB"]
                    )
                ]
            )

            fig.update_layout(
                height=350,
                template="plotly_white",
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
