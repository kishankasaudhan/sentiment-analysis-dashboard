import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import nltk
from nltk.corpus import stopwords

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
  }
  /* Cards */
  .card {
    background: #1a1d27;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid #2a2d3e;
  }
  .card-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #7c83a0;
    margin-bottom: 6px;
  }
  /* Metric row */
  .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #e0e0e0;
  }
  /* Progress bar */
  .bar-wrap {
    background: #2a2d3e;
    border-radius: 99px;
    height: 10px;
    margin-top: 6px;
    margin-bottom: 14px;
  }
  .bar-fill {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #5c6bc0, #7e57c2);
  }
  /* Prediction result */
  .result-positive {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    border: 1px solid #40916c;
    border-radius: 14px;
    padding: 22px 28px;
    font-size: 22px;
    font-weight: 700;
    color: #95d5b2;
    text-align: center;
    margin-top: 12px;
  }
  .result-negative {
    background: linear-gradient(135deg, #3b0d0d, #6b1a1a);
    border: 1px solid #c1121f;
    border-radius: 14px;
    padding: 22px 28px;
    font-size: 22px;
    font-weight: 700;
    color: #f4a3a3;
    text-align: center;
    margin-top: 12px;
  }
  /* Section headers */
  .section-header {
    font-size: 18px;
    font-weight: 700;
    color: #c5cae9;
    margin: 24px 0 10px;
    border-left: 4px solid #7e57c2;
    padding-left: 10px;
  }
  /* Pipeline step */
  .pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
  }
  .step-badge {
    background: #2a2d3e;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    color: #7e57c2;
    white-space: nowrap;
    min-width: 28px;
    text-align: center;
    margin-top: 2px;
  }
  /* Footer */
  .footer {
    text-align: center;
    color: #4a4f6a;
    font-size: 13px;
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #2a2d3e;
  }
  /* Override Streamlit button */
  div.stButton > button {
    background: #2a2d3e;
    color: #c5cae9;
    border: 1px solid #3a3f5c;
    border-radius: 8px;
    font-weight: 600;
    transition: background .2s;
  }
  div.stButton > button:hover {
    background: #3a3f5c;
    color: #ffffff;
    border-color: #7e57c2;
  }
  /* Textarea */
  textarea {
    background: #1a1d27 !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── NLTK setup ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

download_nltk()


# ── Preprocessing ─────────────────────────────────────────────────────────────
KEEP_WORDS = {"not", "no"}

@st.cache_resource(show_spinner=False)
def get_stopwords():
    sw = set(stopwords.words("english")) - KEEP_WORDS
    return sw

STOP_WORDS = get_stopwords()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(tokens)


# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    try:
        df = pd.read_csv("IMDB Dataset.csv")
    except FileNotFoundError:
        return None, "❌ **Dataset not found.** Place `IMDB Dataset.csv` in the same directory as `app.py`."
    except Exception as e:
        return None, f"❌ **Error loading dataset:** {e}"

    if "review" not in df.columns or "sentiment" not in df.columns:
        return None, "❌ **CSV must contain `review` and `sentiment` columns.**"

    # Keep only binary labels
    df = df[df["sentiment"].isin(["positive", "negative"])].dropna(subset=["review", "sentiment"]).copy()
    if df.empty:
        return None, "❌ **No valid rows found in dataset.**"

    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df["clean"] = df["review"].apply(preprocess)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model":      model,
        "vectorizer": vectorizer,
        "metrics":    metrics,
        "cm":         cm,
        "y_pred":     y_pred,
        "y_test":     y_test,
    }, None


# ── Sample reviews ────────────────────────────────────────────────────────────
SAMPLE_POSITIVE = (
    "This movie was absolutely fantastic! The storyline kept me engaged "
    "throughout, and the performances were outstanding. A must-watch!"
)
SAMPLE_NEGATIVE = (
    "I was really disappointed. The plot made no sense, the acting was terrible, "
    "and I almost fell asleep. Not worth your time at all."
)


# ── Session state defaults ────────────────────────────────────────────────────
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None


# ── Load model ────────────────────────────────────────────────────────────────
result, error = load_and_train()


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.6], gap="large")


# ─── LEFT PANEL ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🎬 Sentiment Detector")
    st.markdown(
        "<p style='color:#7c83a0;font-size:14px;'>Binary sentiment classification "
        "powered by TF-IDF + Naive Bayes, trained on 50,000 IMDB movie reviews.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Preprocessing", "Lowercase · strip specials · remove stopwords (keep 'not', 'no')"),
        ("2", "TF-IDF", "TfidfVectorizer — max 5,000 features"),
        ("3", "Naive Bayes", "MultinomialNB classifier"),
        ("4", "Split", "80 / 20 train-test · random_state = 42"),
    ]
    for num, title, detail in steps:
        st.markdown(f"""
        <div class="pipeline-step">
          <span class="step-badge">{num}</span>
          <div>
            <span style="font-weight:600;color:#c5cae9;">{title}</span><br>
            <span style="font-size:12px;color:#7c83a0;">{detail}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)

    if result:
        m = result["metrics"]
        metric_items = [
            ("Accuracy",  m["accuracy"],  "#5c6bc0"),
            ("Precision", m["precision"], "#7e57c2"),
            ("Recall",    m["recall"],    "#26a69a"),
            ("F1-Score",  m["f1"],        "#ec407a"),
        ]
        for label, val, color in metric_items:
            pct = int(val * 100)
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;font-size:13px;">
                <span style="color:#c5cae9;font-weight:600;">{label}</span>
                <span style="color:{color};font-weight:700;">{val:.4f}</span>
              </div>
              <div class="bar-wrap">
                <div class="bar-fill" style="width:{pct}%;background:{color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Metrics unavailable — dataset not loaded.")

    st.markdown('<div class="footer">Developed as a Machine Learning Project</div>', unsafe_allow_html=True)


# ─── RIGHT PANEL ─────────────────────────────────────────────────────────────
with right:
    st.markdown("## Sentiment Analysis System")
    st.markdown(
        "<p style='color:#7c83a0;margin-top:-10px;'>Type or paste a movie review "
        "below and let the model predict its sentiment.</p>",
        unsafe_allow_html=True,
    )

    if error:
        st.error(error)
        st.stop()

    # ── Input area ──
    st.markdown('<div class="section-header">Input Review</div>', unsafe_allow_html=True)

    review_input = st.text_area(
        label="Review text",
        value=st.session_state["review_text"],
        placeholder="Enter your review...",
        height=140,
        label_visibility="collapsed",
        key="text_area",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Sample Positive"):
            st.session_state["review_text"] = SAMPLE_POSITIVE
            st.session_state["prediction"]  = None
            st.rerun()
    with c2:
        if st.button("❌ Sample Negative"):
            st.session_state["review_text"] = SAMPLE_NEGATIVE
            st.session_state["prediction"]  = None
            st.rerun()
    with c3:
        if st.button("🗑️ Clear"):
            st.session_state["review_text"] = ""
            st.session_state["prediction"]  = None
            st.rerun()

    # ── Predict button ──
    predict_clicked = st.button("🔍 Predict Sentiment", use_container_width=True)

    current_text = review_input.strip()

    if predict_clicked:
        if not current_text:
            st.warning("⚠️ Please enter a review before predicting.")
            st.session_state["prediction"] = None
        else:
            clean   = preprocess(current_text)
            vec     = result["vectorizer"].transform([clean])
            pred    = result["model"].predict(vec)[0]
            st.session_state["prediction"]  = pred
            st.session_state["review_text"] = current_text

    # ── Prediction result ──
    if st.session_state["prediction"] is not None:
        if st.session_state["prediction"] == 1:
            st.markdown(
                '<div class="result-positive">😊 &nbsp; Positive Sentiment</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-negative">😡 &nbsp; Negative Sentiment</div>',
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # MODEL PERFORMANCE SECTION
    # ════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    m        = result["metrics"]
    cm       = result["cm"]
    y_pred   = result["y_pred"]

    chart_col1, chart_col2 = st.columns(2)

    # ── 1. Bar chart ──
    with chart_col1:
        st.markdown("**Metrics Overview**")
        fig_bar, ax_bar = plt.subplots(figsize=(4.5, 3.2))
        fig_bar.patch.set_facecolor("#1a1d27")
        ax_bar.set_facecolor("#1a1d27")

        labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
        colors = ["#5c6bc0", "#7e57c2", "#26a69a", "#ec407a"]

        bars = ax_bar.bar(labels, values, color=colors, width=0.55, zorder=3)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_yticks(np.arange(0, 1.1, 0.2))
        ax_bar.tick_params(colors="#9e9e9e", labelsize=9)
        ax_bar.yaxis.label.set_color("#9e9e9e")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#2a2d3e")
        ax_bar.grid(axis="y", color="#2a2d3e", linewidth=0.8, zorder=0)
        for bar, val in zip(bars, values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8.5, color="#e0e0e0", fontweight="bold"
            )
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # ── 2. Confusion matrix ──
    with chart_col2:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
        fig_cm.patch.set_facecolor("#1a1d27")
        ax_cm.set_facecolor("#1a1d27")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            linewidths=0.5,
            linecolor="#0e1117",
            ax=ax_cm,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax_cm.set_xlabel("Predicted", color="#9e9e9e", fontsize=10)
        ax_cm.set_ylabel("Actual",    color="#9e9e9e", fontsize=10)
        ax_cm.tick_params(colors="#9e9e9e", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # ── 3. Pie chart (y_pred only) ──
    st.markdown("**Prediction Distribution (Test Set)**")

    unique, counts = np.unique(y_pred, return_counts=True)
    label_map = {0: "Negative", 1: "Positive"}
    pie_labels = [label_map[u] for u in unique]
    pie_colors = ["#c62828" if u == 0 else "#2e7d32" for u in unique]

    fig_pie, ax_pie = plt.subplots(figsize=(5, 3.5))
    fig_pie.patch.set_facecolor("#1a1d27")
    ax_pie.set_facecolor("#1a1d27")

    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=pie_labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=140,
        wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
        textprops={"color": "#e0e0e0", "fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("#ffffff")

    ax_pie.set_title(
        f"Total predictions: {len(y_pred):,}",
        color="#7c83a0", fontsize=11, pad=10
    )
    plt.tight_layout()
    st.pyplot(fig_pie)
    plt.close(fig_pie)
