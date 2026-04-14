import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import model, vectorizer
from src.preprocessing import clean_text

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📌 Navigation")

option = st.sidebar.radio(
    "Go to:",
    ["🏠 Home (Live Analysis)", "📊 Dataset Dashboard", "🤖 Model Info"]
)

# =========================
# Title
# =========================
st.title("📊 Sentiment Analysis Dashboard (ANN + NLP)")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/IMDB Dataset.csv")

df = load_data()

# =========================
# 🏠 HOME (Updated - Binary Only)
# =========================
if option == "🏠 Home (Live Analysis)":

    st.header("🎬 Live Sentiment Analyzer")

    text = st.text_area("✍️ Enter your review:", height=150)

    if st.button("🔍 Analyze Sentiment"):

        if text.strip():

            with st.spinner("Analyzing..."):

                cleaned = clean_text(text)
                vector = vectorizer.transform([cleaned]).toarray()
                prob = model.predict(vector)[0][0]

                positive = float(prob)
                negative = float(1 - prob)

            # Metrics
            st.subheader("📊 Sentiment Confidence")

            col1, col2 = st.columns(2)
            col1.metric("😊 Positive", f"{positive:.2f}")
            col2.metric("😞 Negative", f"{negative:.2f}")


            # PIE CHART (FIXED)
            # =========================
            st.subheader("🥧 Sentiment Distribution")

            # Ensure values are correct
            labels = ["Positive 😊", "Negative 😞"]
            sizes = [positive, negative]

            # Debug print (optional)
            st.write("Debug:", sizes)

            fig, ax = plt.subplots()

            ax.pie(
                sizes,
                labels=labels,
                autopct=lambda p: f'{p:.1f}%',
                startangle=90
            )

            ax.axis('equal')

            st.pyplot(fig)

# =========================
# 📊 DATASET DASHBOARD
# =========================
elif option == "📊 Dataset Dashboard":

    st.header("📈 Dataset Insights")

    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    df['length'] = df['review'].apply(len)

    fig2, ax2 = plt.subplots()
    ax2.hist(df['length'], bins=50)
    st.pyplot(fig2)

    st.dataframe(df.sample(10))

# =========================
# 🤖 MODEL INFO
# =========================
elif option == "🤖 Model Info":

    st.header("🤖 Model Details")

    st.markdown("""
    ### Model Used
    - Artificial Neural Network (ANN)
    - Binary Classification (Positive / Negative)

    ### Pipeline
    1. Data Cleaning
    2. TF-IDF Feature Extraction
    3. ANN Training
    4. Prediction

    ### Technologies
    - Python
    - TensorFlow / Keras
    - Scikit-learn
    - Streamlit
    """)

    st.success("✅ Model trained successfully!")