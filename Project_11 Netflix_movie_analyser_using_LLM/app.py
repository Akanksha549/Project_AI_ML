import streamlit as st
import pandas as pd
from transformers import pipeline

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analyzer",
    page_icon="🎥",
    layout="wide"
)

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix movie KGF 2.csv", sep=";")

df = load_data()

# -------------------------
# Load HuggingFace Model
# -------------------------
@st.cache_resource
def load_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return classifier

classifier = load_model()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🎥 Movie Review Analyzer")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Statistics",
        "About"
    ]
)

# -------------------------
# HOME
# -------------------------
if menu == "Home":

    st.title("🎬 Netflix Movie Review Sentiment Analysis")

    st.write(
        """
        This application uses a Hugging Face Transformer model to
        classify movie reviews as **Positive** or **Negative**.
        """
    )

    review = st.text_area(
        "Enter your movie review",
        height=180
    )

    if st.button("Analyze Sentiment"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:

            result = classifier(review)[0]

            sentiment = result["label"]
            confidence = result["score"]

            if sentiment == "POSITIVE":
                st.success("😊 Positive Review")
            else:
                st.error("☹️ Negative Review")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(float(confidence))

# -------------------------
# DATASET
# -------------------------
elif menu == "Dataset":

    st.title("Dataset")

    st.write(df)

    st.write("Shape:", df.shape)

# -------------------------
# STATISTICS
# -------------------------
elif menu == "Statistics":

    st.title("Dataset Statistics")

    st.write(df.describe(include="all"))

    if "Class" in df.columns:

        st.subheader("Sentiment Distribution")

        st.bar_chart(df["Class"].value_counts())

# -------------------------
# ABOUT
# -------------------------
else:

    st.title("About")

    st.info(
        """
        **Movie Review Sentiment Analysis**

        Developed using:

        • Streamlit

        • Hugging Face Transformers

        • DistilBERT

        • Pandas

        • Python
        """
    )
