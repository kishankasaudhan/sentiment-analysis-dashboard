import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Detector | ML Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — DARK PRO DASHBOARD
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root & Reset ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0c10 !important;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0d1520 0%, #0a0c10 60%);
    min-height: 100vh;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0c10; }
::-webkit-scrollbar-thumb { background: #2a3f5f; border-radius: 10px; }

/* ── Remove default Streamlit padding ── */
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── LEFT PANEL ── */
.left-panel {
    background: linear-gradient(160deg, #0f1923 0%, #0d1520 100%);
    border: 1px solid #1a2d44;
    border-radius: 16px;
    padding: 28px 22px;
    height: 100%;
    box-shadow: 0 0 40px rgba(0,160,255,0.04);
}

.panel-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}

.panel-subtitle {
    font-size: 0.78rem;
    color: #4a6fa5;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 20px;
    letter-spacing: 0.5px;
}

.section-header {
    font-size: 0.65rem;
    font-weight: 700;
    color: #2a7abf;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a2d44;
}

.pipeline-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 7px 0;
    font-size: 0.82rem;
    color: #94a3b8;
}

.pipeline-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2a7abf;
    margin-top: 5px;
    flex-shrink: 0;
}

.pipeline-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    margin-bottom: 2px;
}

.pipeline-value {
    color: #cbd5e1;
    font-size: 0.83rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #111b2a 0%, #0f1923 100%);
    border: 1px solid #1e3450;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1a6abf, #0ea5e9);
}

.metric-label {
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    color: #4a6fa5;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #e0f0ff;
    line-height: 1.2;
}

.metric-bar-bg {
    height: 4px;
    background: #1a2d44;
    border-radius: 10px;
    margin-top: 6px;
    overflow: hidden;
}

.metric-bar-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #1a6abf, #0ea5e9);
}

/* ── RIGHT PANEL ── */
.right-panel {
    background: linear-gradient(160deg, #0c1520 0%, #0a0c10 100%);
    border: 1px solid #1a2d44;
    border-radius: 16px;
    padding: 28px 26px;
    box-shadow: 0 0 40px rgba(0,0,0,0.3);
}

.main-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.main-subtitle {
    font-size: 0.8rem;
    color: #4a6fa5;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 24px;
}

/* ── Input Section ── */
.input-label {
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    color: #2a7abf;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

textarea {
    background: #0d1520 !important;
    border: 1px solid #1e3450 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    resize: vertical !important;
}

textarea:focus {
    border-color: #1a6abf !important;
    box-shadow: 0 0 0 2px rgba(26,106,191,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid #1e3450 !important;
    background: #111b2a !important;
    color: #94a3b8 !important;
    padding: 0.45rem 0.9rem !important;
}

.stButton > button:hover {
    background: #1a2d44 !important;
    color: #e2e8f0 !important;
    border-color: #2a7abf !important;
    transform: translateY(-1px) !important;
}

.predict-btn > button {
    background: linear-gradient(135deg, #1a6abf 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
}

.predict-btn > button:hover {
    box-shadow: 0 6px 28px rgba(14,165,233,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── Prediction Cards ── */
.pred-card-positive {
    background: linear-gradient(135deg, #052015 0%, #071f14 100%);
    border: 1px solid #0d4a2a;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 20px 22px;
    margin: 16px 0;
}

.pred-card-negative {
    background: linear-gradient(135deg, #1a0508 0%, #15040a 100%);
    border: 1px solid #4a0d1a;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 20px 22px;
    margin: 16px 0;
}

.pred-emoji { font-size: 2.2rem; line-height: 1; }

.pred-label {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.pred-label-pos { color: #22c55e; }
.pred-label-neg { color: #ef4444; }

.pred-confidence {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ── Confidence Bars ── */
.conf-section { margin-top: 14px; }

.conf-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}

.conf-name {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #94a3b8;
}

.conf-pct {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

.conf-bar-bg {
    height: 6px;
    background: #1a2d44;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 10px;
}

.conf-bar-pos {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #16a34a, #22c55e);
}

.conf-bar-neg {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #b91c1c, #ef4444);
}

/* ── Performance Section ── */
.perf-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 24px 0 16px 0;
    padding-top: 20px;
    border-top: 1px solid #1a2d44;
    letter-spacing: -0.3px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #2a3f5f;
    padding: 24px 0 8px 0;
    letter-spacing: 0.5px;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3450, transparent);
    margin: 18px 0;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #052015;
    border: 1px solid #0d4a2a;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    color: #22c55e;
    letter-spacing: 1px;
}

.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Fix plot backgrounds ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
import nltk

@st.cache_resource(show_spinner=False)
def download_nltk():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        pass

download_nltk()

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ─────────────────────────────────────────────
# TRAINING DATA
# ─────────────────────────────────────────────
TRAINING_DATA = {
    "texts": [
        # POSITIVE
        "This product is absolutely fantastic and exceeded my expectations",
        "I love this item so much it works perfectly every single time",
        "Amazing quality and super fast delivery highly recommend to everyone",
        "Best purchase I have ever made completely worth every single penny",
        "Outstanding customer service and the product quality is top notch",
        "I am very happy with this product it is exactly what I wanted",
        "Excellent value for money and works exactly as described great",
        "Perfect in every way I will definitely buy this again soon",
        "This is a wonderful product that does everything it promises",
        "Really impressed with the build quality and the performance is great",
        "Superb experience from start to finish would highly recommend",
        "Brilliant product that changed my life for the better completely",
        "Very satisfied with this purchase the quality is exceptional",
        "Great product love it would buy again without hesitation",
        "Extremely happy with the results absolutely love this product",
        "This item is perfect for what I needed and exceeded all expectations",
        "Wonderful quality and delivered on time very pleased overall",
        "Top quality product that works exactly as advertised love it",
        "Fantastic experience with this product so glad I purchased it",
        "Really great product works well and looks amazing highly satisfied",
        "The performance is outstanding I could not be happier with this",
        "Incredible value highly recommend this to anyone looking for quality",
        "Absolutely brilliant product that delivers on every single promise",
        "I am delighted with this purchase the quality is simply superb",
        "This is genuinely the best product I have used in years love it",
        "Phenomenal quality and design I am thoroughly impressed by this",
        "Such a great buy excellent quality and very fast delivery",
        "Love everything about this product it is simply perfect for me",
        "A truly outstanding product that I will recommend to all my friends",
        "Magnificent product exceeded all my expectations would buy again",
        # NEGATIVE
        "This product is terrible and completely broke after two days",
        "Worst purchase I have ever made do not waste your money on this",
        "Absolutely terrible quality fell apart within a week of use",
        "Complete waste of money nothing worked as advertised at all",
        "Very disappointed with this product it is total garbage",
        "Poor quality and the customer service was absolutely dreadful",
        "Do not buy this it is a scam and the product is awful",
        "Terrible experience the product stopped working immediately",
        "I hate this product it is cheaply made and useless",
        "This is a complete disappointment not worth the money spent",
        "Really bad quality broke on first use totally unacceptable",
        "Dreadful product that fails to deliver on any of its promises",
        "Very unsatisfied with this purchase it is absolutely awful",
        "Horrible quality and arrived damaged will not buy again ever",
        "The worst product I have encountered in my entire life",
        "Absolutely useless product that does not work as described",
        "So disappointed with this purchase it is nothing but junk",
        "Terrible build quality and performance is shockingly bad",
        "This product is a disgrace and should not be sold to anyone",
        "Complete rubbish that failed to work from the very first day",
        "Dreadful experience I am extremely angry about this purchase",
        "Very poor product and the delivery was extremely slow and late",
        "Awful quality and the product smells bad and feels cheap",
        "This is the worst thing I have ever bought such a terrible product",
        "Total garbage waste of money and the support team was useless",
        "Extremely disappointed with this it failed within hours of use",
        "Such a bad product it does not do anything it claims to do",
        "Absolutely dreadful quality I regret buying this completely",
        "A terrible product that broke immediately would not recommend",
        "Shocking poor quality and a complete waste of time and money",
    ],
    "labels": [1]*30 + [0]*30
}

# ─────────────────────────────────────────────
# MODEL LOADING / TRAINING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    model_path = "model.pkl"
    vec_path   = "vectorizer.pkl"

    texts_raw = TRAINING_DATA["texts"]
    labels    = TRAINING_DATA["labels"]
    texts_clean = [preprocess(t) for t in texts_raw]

    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
    )

    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            loaded = True
        except Exception:
            loaded = False
    else:
        loaded = False

    if not loaded:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_tr_tfidf = vectorizer.fit_transform(X_train)
        X_te_tfidf = vectorizer.transform(X_test)
        model = MultinomialNB(alpha=0.3)
        model.fit(X_tr_tfidf, y_train)
    else:
        X_te_tfidf = vectorizer.transform(X_test)

    y_pred = model.predict(X_te_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='binary', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    return model, vectorizer, acc, prec, rec, f1, cm, y_test, y_pred, loaded


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_sentiment(text: str, model, vectorizer):
    cleaned   = preprocess(text)
    vec       = vectorizer.transform([cleaned])
    pred      = model.predict(vec)[0]       # 0 = Negative, 1 = Positive

    try:
        proba = model.predict_proba(vec)[0]
        if len(proba) == 2:
            prob_neg = float(proba[0])
            prob_pos = float(proba[1])
        else:
            prob_pos = float(proba[0])
            prob_neg = 1.0 - prob_pos
    except Exception:
        prob_pos = 1.0 if pred == 1 else 0.0
        prob_neg = 1.0 - prob_pos

    label = "Positive" if pred == 1 else "Negative"
    return label, prob_pos, prob_neg


# ─────────────────────────────────────────────
# CHART HELPERS  (dark theme)
# ─────────────────────────────────────────────
DARK_BG    = "#0a0c10"
PANEL_BG   = "#0d1520"
ACCENT     = "#0ea5e9"
ACCENT2    = "#1a6abf"
GREEN      = "#22c55e"
RED        = "#ef4444"
TEXT_COLOR = "#94a3b8"
GRID_COLOR = "#1a2d44"


def make_bar_chart(acc, prec, rec, f1):
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values  = [acc, prec, rec, f1]
    colors  = [ACCENT2, ACCENT, "#06b6d4", "#38bdf8"]
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                  edgecolor='none', linewidth=0)

    # value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.012,
                f'{val:.1%}', ha='center', va='bottom',
                color='#e2e8f0', fontsize=8, fontweight='bold',
                fontfamily='monospace')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=TEXT_COLOR, fontsize=8.5)
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    ax.set_yticklabels([f'{v:.0%}' for v in np.arange(0, 1.1, 0.25)],
                       color=TEXT_COLOR, fontsize=7.5, fontfamily='monospace')
    ax.set_ylim(0, 1.12)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.8)
    return fig


def make_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3.2), facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    labels = ['Negative', 'Positive']
    cmap   = sns.color_palette([PANEL_BG, ACCENT2, ACCENT], as_cmap=False)
    custom_cmap = sns.light_palette(ACCENT, as_cmap=True)

    sns.heatmap(
        cm, annot=True, fmt='d', cmap=custom_cmap,
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={'size': 14, 'weight': 'bold', 'color': '#ffffff'},
        linewidths=2, linecolor=DARK_BG
    )
    ax.set_xlabel('Predicted', color=TEXT_COLOR, fontsize=9, labelpad=8)
    ax.set_ylabel('Actual',    color=TEXT_COLOR, fontsize=9, labelpad=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.8)
    return fig


def make_pie_chart(y_pred):
    pos_count = int(np.sum(np.array(y_pred) == 1))
    neg_count = int(np.sum(np.array(y_pred) == 0))
    labels  = ['Positive', 'Negative']
    sizes   = [pos_count, neg_count]
    colors  = [GREEN, RED]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor=PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2)
    )
    for at in autotexts:
        at.set_color('#ffffff')
        at.set_fontsize(9)
        at.set_fontweight('bold')

    legend_patches = [
        mpatches.Patch(facecolor=GREEN, label=f'Positive  ({pos_count})'),
        mpatches.Patch(facecolor=RED,   label=f'Negative  ({neg_count})'),
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=False, fontsize=8,
              labelcolor=TEXT_COLOR)
    fig.tight_layout(pad=0.6)
    return fig


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Initialising ML pipeline…"):
    model, vectorizer, ACC, PREC, REC, F1, CM, y_test, y_pred, model_loaded = load_or_train_model()

# ─────────────────────────────────────────────
# SAMPLE TEXTS
# ─────────────────────────────────────────────
SAMPLE_POS = "This product is absolutely fantastic and exceeded all my expectations. The quality is superb and I would highly recommend it to everyone!"
SAMPLE_NEG = "Terrible product that completely broke after two days. Worst purchase I have ever made. Complete waste of money, do not buy this."

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.6], gap="medium")

# ══════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════
with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    # Title + badge
    st.markdown("""
        <div class="panel-title">🧠 Sentiment<br>Detector</div>
        <div class="panel-subtitle">// NLP BINARY CLASSIFIER</div>
        <span class="status-badge">
            <span class="status-dot"></span>MODEL ACTIVE
        </span>
    """, unsafe_allow_html=True)

    src = "Loaded from file" if model_loaded else "Trained in-session"
    st.markdown(f"""
        <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                    color:#2a3f5f;margin-top:6px;">Source: {src}</div>
        <div class="custom-divider"></div>
    """, unsafe_allow_html=True)

    # ── PIPELINE SECTION ──
    st.markdown('<div class="section-header">⚙ Pipeline</div>', unsafe_allow_html=True)

    pipeline_items = [
        ("Preprocessing",  "Lowercasing → Remove special chars → Stopword removal"),
        ("Feature",        "TF-IDF (n-gram 1-2, max 5000 features)"),
        ("Model",          "Multinomial Naive Bayes (α=0.3)"),
        ("Train / Test",   "80 % / 20 % stratified split"),
    ]
    for label, value in pipeline_items:
        st.markdown(f"""
            <div class="pipeline-item">
                <span class="pipeline-dot"></span>
                <div>
                    <div class="pipeline-label">{label}</div>
                    <div class="pipeline-value">{value}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── METRICS SECTION ──
    st.markdown('<div class="section-header">📊 Metrics</div>', unsafe_allow_html=True)

    metric_list = [
        ("ACCURACY",  ACC),
        ("PRECISION", PREC),
        ("RECALL",    REC),
        ("F1-SCORE",  F1),
    ]
    for m_label, m_val in metric_list:
        bar_pct = int(m_val * 100)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{m_label}</div>
                <div class="metric-value">{m_val:.1%}</div>
                <div class="metric-bar-bg">
                    <div class="metric-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── CLASSES ──
    st.markdown('<div class="section-header">🔖 Classes</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px;">
            <span style="background:#052015;border:1px solid #0d4a2a;color:#22c55e;
                         padding:4px 12px;border-radius:20px;font-size:0.72rem;
                         font-family:'JetBrains Mono',monospace;">😊 Positive</span>
            <span style="background:#1a0508;border:1px solid #4a0d1a;color:#ef4444;
                         padding:4px 12px;border-radius:20px;font-size:0.72rem;
                         font-family:'JetBrains Mono',monospace;">😡 Negative</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # end left-panel


# ══════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════
with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="main-title">Sentiment Analysis System</div>
        <div class="main-subtitle">// TF-IDF + NAIVE BAYES · BINARY CLASSIFICATION · REAL-TIME INFERENCE</div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── INPUT SECTION ──
    st.markdown('<div class="input-label">✏ Input Text</div>', unsafe_allow_html=True)

    input_text = st.text_area(
        label="",
        value=st.session_state["input_text"],
        placeholder="Paste your review or text here…",
        height=120,
        key="text_input_area",
        label_visibility="collapsed",
    )

    # Quick-fill buttons
    btn_c1, btn_c2, btn_c3, _ = st.columns([1.2, 1.2, 0.8, 2.8])
    with btn_c1:
        if st.button("😊 Sample Positive", key="btn_pos"):
            st.session_state["input_text"] = SAMPLE_POS
            st.rerun()
    with btn_c2:
        if st.button("😡 Sample Negative", key="btn_neg"):
            st.session_state["input_text"] = SAMPLE_NEG
            st.rerun()
    with btn_c3:
        if st.button("✕ Clear", key="btn_clear"):
            st.session_state["input_text"] = ""
            st.session_state["prediction"] = None
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Predict button
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Sentiment", key="btn_predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle prediction
    final_text = input_text or st.session_state.get("input_text", "")
    if predict_clicked:
        if not final_text.strip():
            st.warning("⚠️  Please enter some text before predicting.")
        else:
            label, prob_pos, prob_neg = predict_sentiment(final_text, model, vectorizer)
            st.session_state["prediction"] = (label, prob_pos, prob_neg, final_text)

    # Show result
    if st.session_state["prediction"] is not None:
        label, prob_pos, prob_neg, _ = st.session_state["prediction"]
        conf = prob_pos if label == "Positive" else prob_neg
        card_class   = "pred-card-positive" if label == "Positive" else "pred-card-negative"
        emoji        = "😊" if label == "Positive" else "😡"
        label_class  = "pred-label-pos" if label == "Positive" else "pred-label-neg"

        pos_pct = int(round(prob_pos * 100))
        neg_pct = int(round(prob_neg * 100))

        st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
                    <span class="pred-emoji">{emoji}</span>
                    <div>
                        <div class="pred-label {label_class}">{label}</div>
                        <div class="pred-confidence">Confidence: {conf:.1%}</div>
                    </div>
                </div>

                <div class="conf-section">
                    <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                                color:#4a6fa5;letter-spacing:1.5px;
                                text-transform:uppercase;margin-bottom:8px;">
                        Confidence Scores
                    </div>

                    <div class="conf-label-row">
                        <span class="conf-name">😊 Positive</span>
                        <span class="conf-pct" style="color:#22c55e;">{pos_pct}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-pos" style="width:{pos_pct}%;"></div>
                    </div>

                    <div class="conf-label-row">
                        <span class="conf-name">😡 Negative</span>
                        <span class="conf-pct" style="color:#ef4444;">{neg_pct}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-neg" style="width:{neg_pct}%;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── PERFORMANCE SECTION ──
    st.markdown('<div class="perf-title">📈 Model Performance</div>', unsafe_allow_html=True)

    ch_left, ch_mid, ch_right = st.columns([1.6, 1.2, 1.2], gap="small")

    with ch_left:
        st.markdown("""
            <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                        color:#4a6fa5;letter-spacing:1.5px;text-transform:uppercase;
                        margin-bottom:8px;">Metric Bar Chart</div>
        """, unsafe_allow_html=True)
        fig_bar = make_bar_chart(ACC, PREC, REC, F1)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    with ch_mid:
        st.markdown("""
            <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                        color:#4a6fa5;letter-spacing:1.5px;text-transform:uppercase;
                        margin-bottom:8px;">Confusion Matrix</div>
        """, unsafe_allow_html=True)
        fig_cm = make_confusion_matrix(CM)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with ch_right:
        st.markdown("""
            <div style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                        color:#4a6fa5;letter-spacing:1.5px;text-transform:uppercase;
                        margin-bottom:8px;">Prediction Distribution</div>
        """, unsafe_allow_html=True)
        fig_pie = make_pie_chart(y_pred)
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    # Stat row below charts
    pos_pred_count = int(np.sum(np.array(y_pred) == 1))
    neg_pred_count = int(np.sum(np.array(y_pred) == 0))
    total_samples  = len(y_pred)

    st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;">
            <div style="background:#0d1520;border:1px solid #1a2d44;border-radius:8px;
                        padding:10px 16px;font-family:'JetBrains Mono',monospace;">
                <div style="font-size:0.6rem;color:#4a6fa5;letter-spacing:1px;text-transform:uppercase;">
                    Test Samples</div>
                <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{total_samples}</div>
            </div>
            <div style="background:#0d1520;border:1px solid #1a2d44;border-radius:8px;
                        padding:10px 16px;font-family:'JetBrains Mono',monospace;">
                <div style="font-size:0.6rem;color:#4a6fa5;letter-spacing:1px;text-transform:uppercase;">
                    Positive Preds</div>
                <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">{pos_pred_count}</div>
            </div>
            <div style="background:#0d1520;border:1px solid #1a2d44;border-radius:8px;
                        padding:10px 16px;font-family:'JetBrains Mono',monospace;">
                <div style="font-size:0.6rem;color:#4a6fa5;letter-spacing:1px;text-transform:uppercase;">
                    Negative Preds</div>
                <div style="font-size:1.1rem;font-weight:700;color:#ef4444;">{neg_pred_count}</div>
            </div>
            <div style="background:#0d1520;border:1px solid #1a2d44;border-radius:8px;
                        padding:10px 16px;font-family:'JetBrains Mono',monospace;">
                <div style="font-size:0.6rem;color:#4a6fa5;letter-spacing:1px;text-transform:uppercase;">
                    Accuracy</div>
                <div style="font-size:1.1rem;font-weight:700;color:#0ea5e9;">{ACC:.1%}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # end right-panel

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        ──────────────────────────────────────────────────────────<br>
        Developed as a Machine Learning Project &nbsp;·&nbsp;
        TF-IDF + Naive Bayes &nbsp;·&nbsp; Binary Sentiment Classification
        <br>──────────────────────────────────────────────────────────
    </div>
""", unsafe_allow_html=True)
