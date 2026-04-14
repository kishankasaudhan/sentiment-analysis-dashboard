import os
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
# ANN IMPORTS  (your original src/ structure)
# ─────────────────────────────────────────────
from src.predict import model, vectorizer
from src.preprocessing import clean_text
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
# GLOBAL STYLES  (dark UI from template)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
 
    /* ── background ── */
    .stApp {
        background: #0d0f14;
        color: #e8e8f0;
    }
 
    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background: #13151c !important;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * {
        color: #c8c8d8 !important;
    }
 
    /* ── radio labels ── */
    .stRadio > label { color: #9090b0 !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
    div[role="radiogroup"] label { padding: 0.45rem 0.75rem; border-radius: 8px; transition: background 0.2s; }
    div[role="radiogroup"] label:hover { background: #1e2130; }
 
    /* ── metric cards ── */
    div[data-testid="metric-container"] {
        background: #13151c;
        border: 1px solid #1e2130;
        border-radius: 14px;
        padding: 1rem 1.4rem;
    }
    div[data-testid="metric-container"] label { color: #6666aa !important; font-size: 0.78rem !important; letter-spacing: 0.07em; text-transform: uppercase; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-family: 'DM Serif Display', serif; font-size: 2.2rem !important; color: #e8e8f0 !important; }
 
    /* ── text area ── */
    textarea {
        background: #13151c !important;
        border: 1px solid #2a2d40 !important;
        border-radius: 12px !important;
        color: #e8e8f0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }
    textarea:focus { border-color: #5c6ef8 !important; box-shadow: 0 0 0 3px rgba(92,110,248,0.15) !important; }
 
    /* ── primary button ── */
    div.stButton > button {
        background: linear-gradient(135deg, #5c6ef8, #8b5cf6);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        transition: opacity 0.2s, transform 0.15s;
        width: 100%;
    }
    div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
    div.stButton > button:active { transform: translateY(0); }
 
    /* ── sentiment result box ── */
    .result-box {
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-top: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .result-positive { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35); }
    .result-negative { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35); }
 
    .result-emoji { font-size: 3rem; line-height: 1; }
    .result-label { font-family: 'DM Serif Display', serif; font-size: 1.9rem; }
    .result-positive .result-label { color: #4ade80; }
    .result-negative .result-label { color: #f87171; }
 
    /* ── section headers ── */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        color: #e8e8f0;
        margin-bottom: 0.25rem;
    }
    .section-sub {
        color: #6666aa;
        font-size: 0.85rem;
        margin-bottom: 1.4rem;
    }
    hr.divider { border: none; border-top: 1px solid #1e2130; margin: 2rem 0; }
 
    /* ── confidence bar ── */
    .conf-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem; }
    .conf-label { width: 80px; font-size: 0.82rem; color: #9090b0; text-align: right; }
    .conf-bar-bg { flex: 1; background: #1e2130; border-radius: 999px; height: 8px; overflow: hidden; }
    .conf-bar-fill { height: 8px; border-radius: 999px; }
    .conf-pct { width: 46px; font-size: 0.82rem; color: #c8c8d8; }
 
    /* ── dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
 
    /* ── footer ── */
    .footer {
        text-align: center;
        color: #3a3a5a;
        font-size: 0.78rem;
        margin-top: 4rem;
        padding-top: 1rem;
        border-top: 1px solid #1a1c28;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ─────────────────────────────────────────────
# LOAD IMDB DATASET  (for Model Insights page)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_imdb_data(path: str = "data/IMDB Dataset.csv"):
    """
    Loads the IMDB dataset, cleans text, runs the ANN model on a sample,
    and returns (y_true, y_pred) for evaluation metrics.
    Uses a 2 000-row sample to keep the dashboard snappy; adjust as needed.
    """
    df = pd.read_csv(path)
 
    # Normalise the sentiment column to title-case ("Positive" / "Negative")
    df["sentiment"] = df["sentiment"].str.strip().str.title()
 
    # Work on a reproducible sample
    sample = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
 
    # Clean text using your src.preprocessing function
    sample["clean"] = sample["review"].apply(clean_text)
 
    # Vectorise
    X = vectorizer.transform(sample["clean"])
 
    # ANN predict  (model.predict returns probabilities for binary classification)
    probs  = model.predict(X)                          # shape (n, 1) or (n, 2)
    if probs.ndim == 2 and probs.shape[1] == 2:
        # Two-output softmax
        y_pred_idx = np.argmax(probs, axis=1)
    else:
        # Single sigmoid output
        y_pred_idx = (probs.ravel() >= 0.5).astype(int)
 
    label_map = {0: "Negative", 1: "Positive"}
    y_pred = np.array([label_map[i] for i in y_pred_idx])
    y_true = sample["sentiment"].values
 
    return y_true, y_pred
 
 
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:DM Serif Display,serif;font-size:1.35rem;"
        "color:#e8e8f0;margin-bottom:0.15rem;'>🧠 SentiScope</div>"
        "<div style='color:#4444aa;font-size:0.75rem;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-bottom:1.8rem;'>ANN · NLP Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6666aa;font-size:0.72rem;letter-spacing:0.09em;"
        "text-transform:uppercase;margin-bottom:0.4rem;'>Navigation</p>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        label="Navigation",
        options=["🏠  Home", "📊  Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#3a3a5a;font-size:0.72rem;'>TF-IDF · ANN (Keras) · IMDB Dataset</p>",
        unsafe_allow_html=True,
    )
 
# ─────────────────────────────────────────────
# HELPER — matplotlib dark theme
# ─────────────────────────────────────────────
def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#13151c")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2130")
    ax.tick_params(colors="#6666aa")
    ax.xaxis.label.set_color("#9090b0")
    ax.yaxis.label.set_color("#9090b0")
    ax.title.set_color("#c8c8d8")
    return fig, ax
 
PALETTE = {
    "Positive": "#4ade80",
    "Negative": "#f87171",
}
 
# ═══════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown(
        "<h1 style='font-family:DM Serif Display,serif;font-size:2.6rem;"
        "color:#e8e8f0;margin-bottom:0.2rem;'>Sentiment Analysis Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6666aa;font-size:1rem;max-width:620px;margin-bottom:2rem;'>"
        "Enter any piece of text and the ANN model will classify its emotional tone — "
        "Positive or Negative — using TF-IDF features and a Keras neural network."
        "</p>",
        unsafe_allow_html=True,
    )
 
    st.success("✅ ANN model loaded from **model.h5** and **vectorizer.pkl**.", icon="📦")
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
 
    # ── Input ────────────────────────────────────
    col_input, col_spacer = st.columns([2, 1])
    with col_input:
        st.markdown(
            "<p class='section-title'>Analyse Text</p>"
            "<p class='section-sub'>Paste a review, comment, or any text below.</p>",
            unsafe_allow_html=True,
        )
        user_text = st.text_area(
            label="Input Text",
            placeholder='e.g. "The movie was absolutely brilliant — gripping story and outstanding performances!"',
            height=150,
            label_visibility="collapsed",
        )
        predict_btn = st.button("Predict Sentiment", use_container_width=True)
 
    # ── Prediction ───────────────────────────────
    if predict_btn:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before predicting.", icon="✍️")
        else:
            # Use YOUR clean_text from src.preprocessing
            cleaned = clean_text(user_text)
            if not cleaned.strip():
                st.warning(
                    "⚠️ The input contains only stopwords or special characters. Please try different text.",
                    icon="✍️",
                )
            else:
                # Vectorise + ANN predict
                vec_input = vectorizer.transform([cleaned])
                probs = model.predict(vec_input)
 
                if probs.ndim == 2 and probs.shape[1] == 2:
                    # Softmax two-class output
                    pos_prob = float(probs[0][1])
                    neg_prob = float(probs[0][0])
                else:
                    # Sigmoid single output
                    pos_prob = float(probs.ravel()[0])
                    neg_prob = 1.0 - pos_prob
 
                prediction = "Positive" if pos_prob >= 0.5 else "Negative"
 
                emoji_map = {"Positive": "😊", "Negative": "😡"}
                class_map = {"Positive": "result-positive", "Negative": "result-negative"}
 
                st.markdown(
                    f"""
                    <div class="result-box {class_map[prediction]}">
                        <div class="result-emoji">{emoji_map[prediction]}</div>
                        <div>
                            <div class="result-label">{prediction}</div>
                            <div style="color:#6666aa;font-size:0.82rem;margin-top:0.2rem;">
                                Predicted sentiment
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
 
                st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#6666aa;font-size:0.8rem;letter-spacing:0.08em;"
                    "text-transform:uppercase;margin-bottom:0.6rem;'>Confidence Scores</p>",
                    unsafe_allow_html=True,
                )
 
                color_map  = {"Positive": "#4ade80", "Negative": "#f87171"}
                conf_pairs = sorted(
                    [("Positive", pos_prob), ("Negative", neg_prob)],
                    key=lambda x: -x[1],
                )
                bars_html = ""
                for cls, prob in conf_pairs:
                    pct   = prob * 100
                    color = color_map[cls]
                    bars_html += (
                        f"<div class='conf-row'>"
                        f"  <div class='conf-label'>{cls}</div>"
                        f"  <div class='conf-bar-bg'>"
                        f"    <div class='conf-bar-fill' style='width:{pct:.1f}%;background:{color};'></div>"
                        f"  </div>"
                        f"  <div class='conf-pct'>{pct:.1f}%</div>"
                        f"</div>"
                    )
                st.markdown(bars_html, unsafe_allow_html=True)
 
    # ── Footer ────────────────────────────────────
    st.markdown(
        "<div class='footer'>Developed as a Machine Learning Project · ANN Backend</div>",
        unsafe_allow_html=True,
    )
 
# ═══════════════════════════════════════════════
# MODEL INSIGHTS PAGE
# ═══════════════════════════════════════════════
else:
    st.markdown(
        "<h1 style='font-family:DM Serif Display,serif;font-size:2.6rem;"
        "color:#e8e8f0;margin-bottom:0.2rem;'>Model Insights</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#6666aa;font-size:1rem;max-width:620px;margin-bottom:2rem;'>"
        "Metrics and visualisations computed from a 2 000-row sample of the IMDB dataset "
        "using your trained ANN model."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
 
    with st.spinner("Running ANN predictions on IMDB sample …"):
        try:
            y_test, y_pred = load_imdb_data()
        except FileNotFoundError:
            st.error(
                "❌ Could not find **data/IMDB Dataset.csv**. "
                "Make sure the file exists relative to your working directory.",
                icon="📂",
            )
            st.stop()
 
    unique_labels = sorted(set(y_test) | set(y_pred))
    acc = accuracy_score(y_test, y_pred)
 
    # ── Metrics ──────────────────────────────────
    col_acc, col_n, col_classes = st.columns(3)
    with col_acc:
        st.metric("Accuracy", f"{acc * 100:.1f}%")
    with col_n:
        st.metric("Test Samples", len(y_test))
    with col_classes:
        st.metric("Classes", len(unique_labels))
 
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
 
    # ── Confusion Matrix + Pie Chart ─────────────
    st.markdown(
        "<p class='section-title'>Confusion Matrix &amp; Prediction Distribution</p>"
        "<p class='section-sub'>Left: actual vs predicted labels — Right: breakdown of predicted counts.</p>",
        unsafe_allow_html=True,
    )
 
    col_cm, col_pie = st.columns([1, 1])
 
    with col_cm:
        labels_order = sorted(unique_labels)
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
 
        fig_cm, ax_cm = dark_fig(5, 4)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_order,
            yticklabels=labels_order,
            ax=ax_cm,
            linewidths=0.5,
            linecolor="#0d0f14",
            cbar=False,
            annot_kws={"size": 13, "color": "#e8e8f0", "weight": "bold"},
        )
        ax_cm.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax_cm.set_ylabel("Actual Label",    fontsize=10, labelpad=8)
        ax_cm.set_title("Confusion Matrix", fontsize=12, pad=10)
        ax_cm.tick_params(axis="x", labelsize=9, colors="#9090b0")
        ax_cm.tick_params(axis="y", labelsize=9, colors="#9090b0", rotation=0)
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
 
    with col_pie:
        pred_series = pd.Series(y_pred)
        pred_counts = pred_series.value_counts()
        pie_labels  = pred_counts.index.tolist()
        pie_sizes   = pred_counts.values.tolist()
        pie_colors  = [PALETTE.get(lbl, "#5c6ef8") for lbl in pie_labels]
 
        fig_pie, ax_pie = dark_fig(5, 4)
        wedges, texts, autotexts = ax_pie.pie(
            pie_sizes,
            labels=None,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.78,
            wedgeprops={"linewidth": 2, "edgecolor": "#0d0f14"},
        )
        for at in autotexts:
            at.set_color("#0d0f14")
            at.set_fontsize(10)
            at.set_fontweight("bold")
 
        legend_patches = [
            mpatches.Patch(color=pie_colors[i], label=f"{pie_labels[i]}  ({pie_sizes[i]})")
            for i in range(len(pie_labels))
        ]
        ax_pie.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=len(pie_labels),
            frameon=False,
            fontsize=9,
            labelcolor="#9090b0",
        )
        ax_pie.set_title("Predicted Sentiment Distribution", fontsize=12, pad=10, color="#c8c8d8")
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)
 
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
 
    # ── Classification Report ────────────────────
    st.markdown(
        "<p class='section-title'>Classification Report</p>"
        "<p class='section-sub'>Per-class precision, recall, and F1-score from IMDB test predictions.</p>",
        unsafe_allow_html=True,
    )
 
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
 
    rows = []
    for label in unique_labels:
        if label in report_dict:
            m = report_dict[label]
            rows.append(
                {
                    "Class":     label,
                    "Precision": round(m["precision"], 3),
                    "Recall":    round(m["recall"],    3),
                    "F1-Score":  round(m["f1-score"],  3),
                    "Support":   int(m["support"]),
                }
            )
    for avg in ["macro avg", "weighted avg"]:
        if avg in report_dict:
            m = report_dict[avg]
            rows.append(
                {
                    "Class":     avg.title(),
                    "Precision": round(m["precision"], 3),
                    "Recall":    round(m["recall"],    3),
                    "F1-Score":  round(m["f1-score"],  3),
                    "Support":   int(m["support"]),
                }
            )
 
    report_df = pd.DataFrame(rows).set_index("Class")
    st.dataframe(
        report_df.style
            .format("{:.3f}", subset=["Precision", "Recall", "F1-Score"])
            .set_properties(**{"text-align": "center"})
            .background_gradient(
                cmap="Blues",
                subset=["Precision", "Recall", "F1-Score"],
                vmin=0,
                vmax=1,
            ),
        use_container_width=True,
        height=min(50 + 36 * len(report_df), 400),
    )
 
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
 
    # ── Per-class bar chart ──────────────────────
    st.markdown(
        "<p class='section-title'>Per-Class F1-Score</p>"
        "<p class='section-sub'>Visual comparison of F1-scores across sentiment classes.</p>",
        unsafe_allow_html=True,
    )
 
    class_rows = [r for r in rows if r["Class"] in unique_labels]
    f1_classes = [r["Class"]    for r in class_rows]
    f1_scores  = [r["F1-Score"] for r in class_rows]
    bar_colors = [PALETTE.get(c, "#5c6ef8") for c in f1_classes]
 
    fig_bar, ax_bar = dark_fig(7, 3.5)
    bars = ax_bar.bar(f1_classes, f1_scores, color=bar_colors, width=0.45, zorder=3)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_ylabel("F1-Score", fontsize=10)
    ax_bar.set_title("F1-Score by Class", fontsize=12, pad=10)
    ax_bar.yaxis.grid(True, color="#1e2130", linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    for bar, score in zip(bars, f1_scores):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.03,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#c8c8d8",
            fontweight="bold",
        )
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)
 
    # ── Footer ────────────────────────────────────
    st.markdown(
        "<div class='footer'>Developed as a Machine Learning Project · ANN Backend</div>",
        unsafe_allow_html=True,
    )
 