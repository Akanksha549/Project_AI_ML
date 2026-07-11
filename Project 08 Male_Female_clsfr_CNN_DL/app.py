import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Male/Female Eye Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# COLORS
# ----------------------------------------------------
bg_color = "#F4F9FF"
card_color = "#FFFFFF"
sidebar_color = "#EAF4FF"
text_color = "#1E3A5F"
accent = "#2563EB"
border = "#D6E8FF"

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown(f"""
<style>

.stApp {{
    background-color:{bg_color};
}}

section[data-testid="stSidebar"] {{
    background:{sidebar_color};
}}

h1,h2,h3,h4,p,label,span {{
    color:{text_color};
}}

div.stButton > button {{
    width:100%;
    height:50px;
    border-radius:12px;
    background:{accent};
    color:white;
    font-size:17px;
    font-weight:bold;
    border:none;
}}

div.stButton > button:hover {{
    background:#1D4ED8;
}}

[data-testid="stFileUploader"] {{
    background:white;
    border:2px dashed {accent};
    border-radius:15px;
}}

.footer {{
text-align:center;
color:gray;
font-size:14px;
}}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:

    st.title("👁️ Eye Classifier")

    st.markdown("---")

    st.success(
        """
Upload a **human eye image**.

The CNN model predicts whether the eye belongs to a

✅ Male

✅ Female
"""
    )

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

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.title("👁️ Male & Female Eye Classification using CNN")

st.caption(
    "Deep Learning • TensorFlow • Keras • Streamlit"
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("male_female_eye_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model.\n\n{e}")
    st.stop()

# ----------------------------------------------------
# IMAGE SETTINGS
# ----------------------------------------------------
IMG_SIZE = 128

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# ----------------------------------------------------
# IMAGE UPLOADER
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:

        st.subheader("📷 Uploaded Eye Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🤖 CNN Prediction")

        if st.button("🚀 Predict Gender"):

            with st.spinner("Analyzing eye image..."):

                start = time.time()

                img = preprocess_image(image)

                prediction = float(
                    model.predict(img, verbose=0)[0][0]
                )

                end = time.time()

                inference = end - start

            # ------------------------------------------------
            # CLASS MAPPING
            # femaleeyes = 0
            # maleeyes = 1
            # ------------------------------------------------

            if prediction >= 0.5:

                label = "👨 Male"

                confidence = prediction

            else:

                label = "👩 Female"

                confidence = 1 - prediction

            st.success(f"### Prediction : {label}")

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

            with c2:

                st.metric(
                    "Inference Time",
                    f"{inference:.3f} sec"
                )

            st.progress(float(confidence))

            st.divider()

            st.subheader("📊 Prediction Probability")

            female_prob = 1 - prediction
            male_prob = prediction

            fig = go.Figure()

            fig.add_trace(

                go.Bar(

                    x=["Female","Male"],

                    y=[female_prob,male_prob],

                    text=[
                        f"{female_prob*100:.1f}%",
                        f"{male_prob*100:.1f}%"
                    ],

                    textposition="outside",

                    marker_color=[
                        "#EC4899",
                        "#2563EB"
                    ]

                )

            )

            fig.update_layout(

                height=380,

                template="plotly_white",

                showlegend=False,

                yaxis=dict(range=[0,1]),

                xaxis_title="Gender",

                yaxis_title="Probability"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader("📋 Prediction Summary")

            st.write(f"### Gender : {label}")

            st.write(
                f"Confidence : {confidence*100:.2f}%"
            )

            st.write(
                f"Inference Time : {inference:.3f} sec"
            )

# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

with st.expander("ℹ️ About this Project", expanded=False):

    st.markdown("""
### 👁️ Male & Female Eye Classification using CNN

This application predicts whether an uploaded **human eye image**
belongs to a **Male** or **Female** using a trained
**Convolutional Neural Network (CNN)**.

### Features

- Eye Image Classification
- Deep Learning Prediction
- Real-time Inference
- Probability Visualization
- Confidence Score
- Streamlit Deployment

The model was developed using **TensorFlow/Keras**
and deployed with **Streamlit**.
""")

# ============================================================
# DATASET
# ============================================================

with st.expander("📂 Dataset"):

    st.markdown("""

### Dataset Information

The CNN model was trained on a labeled eye image dataset
containing images from both male and female subjects.

#### Image Processing

- RGB Images
- Image Size: **128 × 128**
- Pixel Normalization
- Binary Classification

Classes

- Female Eyes
- Male Eyes

""")

# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("🧠 CNN Model"):

    st.markdown("""

### Architecture

- TensorFlow
- Keras
- Convolutional Neural Network
- Binary Classification

### Input Shape

128 × 128 × 3

### Output

Sigmoid Activation

Prediction Range

0 → Female

1 → Male

""")

# ============================================================
# TECHNOLOGIES
# ============================================================

with st.expander("⚙️ Technologies Used"):

    st.markdown("""

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Plotly
- Streamlit

""")

# ============================================================
# PROJECT WORKFLOW
# ============================================================

with st.expander("🔄 Workflow"):

    st.markdown("""

1. Upload Eye Image

↓

2. Image Preprocessing

↓

3. CNN Feature Extraction

↓

4. Gender Prediction

↓

5. Display Confidence Score

""")

# ============================================================
# DISCLAIMER
# ============================================================

with st.expander("⚠ Disclaimer"):

    st.info(
        """
This project is developed for educational and research
purposes only.

Predictions depend on image quality, lighting conditions,
and the data used during training.
"""
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
"""
<div style='text-align:center;color:gray;'>

### 👁️ Male & Female Eye Classification using CNN

Built with ❤️ using

TensorFlow • Keras • Streamlit

© 2026 Akanksha Mishra

</div>
""",
unsafe_allow_html=True
)
