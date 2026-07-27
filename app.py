"""
Mental Health AI Assistant — Premium Dashboard
------------------------------------------------
A Streamlit application that serves a BiLSTM text-classification model
(trained with FastText embeddings) to predict a mental-health status
label from a free-text statement.

IMPORTANT: This file's UI layer was redesigned into a premium AI SaaS
dashboard. The machine-learning logic (model loading, tokenizer, label
encoder, preprocessing, and prediction) is UNCHANGED from the original
implementation — only presentation/UX code was added or modified.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import string
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# If a local nltk_data folder exists next to this file, use it first so the
# app never depends on internet access at runtime, and uses the exact same
# tokenizer/lemmatizer data that training used.
_LOCAL_NLTK_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
if os.path.isdir(_LOCAL_NLTK_DATA):
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

nltk.download('wordnet')
nltk.download('omw-1.4')

# =============================================================================
# Configuration  (UNCHANGED — backend/model configuration)
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Static configuration for the app and model."""

    model_path: str = "bi_lstm_model.keras"
    tokenizer_path: str = "tokenizer.pkl"
    label_encoder_path: str = "label_encoder.pkl"
    max_sequence_length: int = 300
    nltk_resources: tuple[str, ...] = ("stopwords", "punkt", "wordnet", "omw-1.4", "punkt_tab")


CONFIG: Final = Config()

PAGE_TITLE: Final = "Mental Health AI Assistant"
PAGE_ICON: Final = "🧠"

_NLTK_LOOKUP_PATHS: Final[dict[str, str]] = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}

# punkt_tab is a newer NLTK resource (added in nltk >= 3.9). On older nltk
# versions it doesn't exist and will never resolve — that's fine, since
# word_tokenize works with punkt alone in that case. Don't block on it.
_OPTIONAL_NLTK_RESOURCES: Final[set[str]] = {"punkt_tab"}

DISCLAIMER_TEXT: Final = (
    "This tool is a machine-learning demo and **not a diagnostic or clinical "
    "instrument**. It should not inform decisions about your mental health "
    "or anyone else's. If you or someone you know is struggling, please "
    "reach out to a licensed mental health professional or a local crisis line."
)

# -----------------------------------------------------------------------------
# Front-end only presentation metadata — icons / colors / risk tiers per label.
# Purely cosmetic: does not touch model output in any way.
# -----------------------------------------------------------------------------
STATUS_STYLES: Final[dict[str, dict[str, str]]] = {
    "Normal":               {"emoji": "🌿", "color": "#22C55E", "risk": "Low",      "risk_color": "#22C55E"},
    "Depression":           {"emoji": "🌧️", "color": "#38BDF8", "risk": "Elevated", "risk_color": "#F59E0B"},
    "Suicidal":             {"emoji": "🚨", "color": "#EF4444", "risk": "Critical", "risk_color": "#EF4444"},
    "Anxiety":              {"emoji": "⚡", "color": "#F59E0B", "risk": "Elevated", "risk_color": "#F59E0B"},
    "Bipolar":              {"emoji": "🌓", "color": "#A78BFA", "risk": "Elevated", "risk_color": "#F59E0B"},
    "Stress":               {"emoji": "🧩", "color": "#FB923C", "risk": "Elevated", "risk_color": "#F59E0B"},
    "Personality disorder": {"emoji": "🎭", "color": "#818CF8", "risk": "Elevated", "risk_color": "#F59E0B"},
}
DEFAULT_STATUS_STYLE: Final[dict[str, str]] = {
    "emoji": "🔍", "color": "#94A3B8", "risk": "Unknown", "risk_color": "#94A3B8",
}

STATUS_EXPLANATIONS: Final[dict[str, str]] = {
    "Normal": "The language patterns in this statement most closely resemble everyday, stable mood expression.",
    "Depression": "The statement contains language patterns the model associates with low mood, fatigue, or hopelessness.",
    "Suicidal": "The model detected language patterns associated with suicidal ideation. Please treat this result seriously and see the resources below.",
    "Anxiety": "The statement contains language patterns the model associates with worry, nervousness, or racing thoughts.",
    "Bipolar": "The statement contains language patterns the model associates with mood-swing or high/low energy cycles.",
    "Stress": "The statement contains language patterns the model associates with pressure, overwhelm, or tension.",
    "Personality disorder": "The statement contains language patterns the model associates with identity or interpersonal-pattern themes.",
}

EXAMPLE_STATEMENTS: Final[list[str]] = [
    "Choose an example statement...",
    "I feel like nothing I do matters anymore.",
    "I've been so anxious about work I can't sleep at night.",
    "Today was actually a pretty good day, I felt productive.",
    "My mood swings between extreme highs and lows within the same week.",
]

CRISIS_RESOURCES: Final = (
    "**If you are in crisis:** in the US, call or text **988** (Suicide & Crisis "
    "Lifeline). Outside the US, please look up your local emergency or crisis line. "
    "You deserve support from a real person, right now, if you need it."
)


# =============================================================================
# Resource loading (cached — runs once per server session)  — UNCHANGED
# =============================================================================

@st.cache_resource(show_spinner=False)
def ensure_nltk_resources(resources: tuple[str, ...]) -> list[str]:
    """Make sure required NLTK corpora are available.

    Checks bundled/local data first; only downloads what's genuinely
    missing. Returns any resources that couldn't be resolved either way.
    """
    unresolved: list[str] = []

    for resource in resources:
        lookup_path = _NLTK_LOOKUP_PATHS.get(resource, resource)
        try:
            nltk.data.find(lookup_path)
            continue
        except LookupError:
            pass

        try:
            nltk.download(resource, quiet=True, raise_on_error=True)
            nltk.data.find(lookup_path)
        except LookupError:
            if resource not in _OPTIONAL_NLTK_RESOURCES:
                unresolved.append(resource)
        except Exception:
            if resource not in _OPTIONAL_NLTK_RESOURCES:
                unresolved.append(resource)

            return unresolved


@st.cache_resource(show_spinner=False)
def load_model_artifacts(config: Config):
    """Load and cache the trained model, tokenizer, and label encoder."""
    missing = [
        path
        for path in (config.model_path, config.tokenizer_path, config.label_encoder_path)
        if not Path(path).exists()
    ]
    if missing:
        st.error(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Make sure they sit alongside app.py."
        )
        st.stop()

    model = load_model(config.model_path)

    with open(config.tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    with open(config.label_encoder_path, "rb") as file:
        label_encoder = pickle.load(file)

    return model, tokenizer, label_encoder


# =============================================================================
# Text preprocessing  — UNCHANGED
# =============================================================================

class TextPreprocessor:
    """Applies the same cleaning pipeline used during model training."""

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)

        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]
        return " ".join(tokens)


# =============================================================================
# Prediction  — UNCHANGED
# =============================================================================

@dataclass
class PredictionResult:
    label: str
    confidence: float
    cleaned_text: str
    class_probabilities: dict[str, float]


def predict_status(
    raw_text: str,
    model,
    tokenizer,
    label_encoder,
    preprocessor: TextPreprocessor,
    max_length: int,
) -> PredictionResult:
    """Run the full inference pipeline on a single raw text statement."""
    cleaned_text = preprocessor.clean(raw_text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")

    probabilities = model(padded_sequence, training=False).numpy()[0]
    predicted_index = int(np.argmax(probabilities))

    return PredictionResult(
        label=label_encoder.inverse_transform([predicted_index])[0],
        confidence=float(probabilities[predicted_index]),
        cleaned_text=cleaned_text,
        class_probabilities=dict(zip(label_encoder.classes_, probabilities)),
    )


# =============================================================================
# THEME / DESIGN TOKENS  (front-end only)
# =============================================================================

BG_COLOR = "#0B1220"
CARD_COLOR = "#1E293B"
CARD_COLOR_GLASS = "rgba(30, 41, 59, 0.55)"
PRIMARY = "#7C3AED"
SECONDARY = "#06B6D4"
BORDER = "rgba(148, 163, 184, 0.15)"
TEXT_MUTED = "#94A3B8"


def inject_css() -> None:
    """Injects every custom style rule for the premium dashboard look.

    Kept in a single function so all visual styling lives in one place —
    no styling logic is scattered through the render functions below.
    """
    st.markdown(
        f"""
        <style>
        /* ---------- Global canvas ---------- */
        .stApp {{
            background: radial-gradient(circle at 15% 0%, #131C2E 0%, {BG_COLOR} 45%, #060A12 100%);
            color: #F1F5F9;
        }}
        .block-container {{
            padding-top: 0rem !important;
            margin-top: 0rem !important;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}
        html, body, [class*="css"] {{
            font-family: "Inter", "Segoe UI", sans-serif;
        }}
        #MainMenu, footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{
            display: none !important;
        }}

        [data-testid="stToolbar"] {{
            display: none !important;
        }}

        /* ---------- Fade-in animation for main content ---------- */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .fade-in {{ animation: fadeIn 0.5s ease-out; }}

        /* ---------- Glassmorphism card ---------- */
        .glass-card {{
            background: {CARD_COLOR_GLASS};
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid {BORDER};
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
            margin-bottom: 1.2rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 14px 40px rgba(124, 58, 237, 0.18);
        }}

        /* ---------- Hero banner ---------- */
        .hero-banner {{
            background: linear-gradient(120deg, {PRIMARY} 0%, #4338CA 45%, {SECONDARY} 100%);
            border-radius: 22px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 16px 45px rgba(124, 58, 237, 0.35);
            position: relative;
            overflow: hidden;
            height: 100%;
        }}
        .hero-banner::after {{
            content: "";
            position: absolute; top: -60px; right: -60px;
            width: 220px; height: 220px; border-radius: 50%;
            background: rgba(255,255,255,0.08);
        }}
        .hero-title {{
            font-size: 2.1rem; font-weight: 800; color: white; margin: 0;
        }}
        .hero-subtitle {{
            font-size: 1.05rem; color: rgba(255,255,255,0.9); margin-top: 0.35rem;
        }}
        .hero-badges span {{
            display: inline-block;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.25);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-top: 0.9rem;
            margin-right: 0.5rem;
            font-weight: 600;
        }}
        .hero-icon {{
            font-size: 5rem;
            text-align: center;
            filter: drop-shadow(0 6px 18px rgba(0,0,0,0.35));
            animation: float 3.5s ease-in-out infinite;
        }}
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-12px); }}
        }}

        /* ---------- Risk badge ---------- */
        .risk-badge {{
            display: inline-block;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }}

        /* ---------- Section headings inside cards ---------- */
        .card-heading {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 0.6rem;
        }}
        .card-subtext {{ color: {TEXT_MUTED}; font-size: 0.88rem; }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            background: linear-gradient(90deg, {PRIMARY} 0%, {SECONDARY} 100%);
            color: white;
            border: none;
            border-radius: 14px;
            font-weight: 700;
            padding: 0.7rem 1rem;
            width: 100%;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            filter: brightness(1.08);
            box-shadow: 0 10px 26px rgba(124, 58, 237, 0.5);
        }}
        .stButton > button:active {{ transform: translateY(0px) scale(0.98); }}

        /* ---------- Text area ---------- */
        .stTextArea textarea {{
            background-color: rgba(15, 23, 42, 0.6) !important;
            border-radius: 14px !important;
            border: 1px solid {BORDER} !important;
            color: #F1F5F9 !important;
        }}

        /* ---------- Selectbox ---------- */
        .stSelectbox > div > div {{
            background-color: rgba(15, 23, 42, 0.6) !important;
            border-radius: 12px !important;
            border: 1px solid {BORDER} !important;
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #0B1220 100%);
            border-right: 1px solid {BORDER};
        }}
        .sidebar-brand {{
            font-size: 1.2rem; font-weight: 800; color: white;
            padding: 0.6rem 0 1rem 0;
        }}
        .sidebar-footer-box {{
            background: rgba(124, 58, 237, 0.12);
            border: 1px solid rgba(124, 58, 237, 0.35);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-size: 0.82rem;
            color: #E2E8F0;
            margin-top: 1rem;
        }}
        .status-dot {{
            height: 8px; width: 8px; border-radius: 50%;
            background: #22C55E; display: inline-block; margin-right: 6px;
            box-shadow: 0 0 8px #22C55E;
        }}

        /* ---------- Progress bar styling ---------- */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
        }}

        /* ---------- Custom history table ---------- */
        .history-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
        .history-table th {{
            text-align: left; color: {TEXT_MUTED}; font-weight: 600;
            padding: 0.5rem 0.6rem; border-bottom: 1px solid {BORDER};
        }}
        .history-table td {{
            padding: 0.55rem 0.6rem; border-bottom: 1px solid {BORDER}; color: #E2E8F0;
        }}
        .history-table tr:hover td {{ background: rgba(124, 58, 237, 0.08); }}

        /* ---------- Divider ---------- */
        hr {{ border-color: {BORDER} !important; }}

        /* ---------- Responsive tweaks ---------- */
        @media (max-width: 1100px) {{
            .hero-title {{ font-size: 1.6rem; }}
            .hero-icon {{ font-size: 3rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Renders the navigation sidebar. Returns the selected page name."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🧠 MH&nbsp;AI&nbsp;Studio</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            options=["🏠 Dashboard", "📜 Prediction History", "📊 Analytics", "🧠 About Model", "📖 Instructions"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-footer-box">
                <b>Model:</b> BiLSTM<br>
                <b>Embedding:</b> FastText<br>
                <span class="status-dot"></span><b>Status:</b> Model Loaded
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# =============================================================================
# HERO SECTION
# =============================================================================

def render_hero() -> None:
    
    
        st.markdown(
            """
            <div class="hero-banner">
                <div class="hero-title">🧠 Mental Health AI Assistant</div>
                <div class="hero-subtitle">AI-powered Mental Health Status Classification</div>
                <div class="hero-badges">
                    <span>BiLSTM</span><span>FastText</span><span>TensorFlow</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 


# =============================================================================
# INPUT CARD
# =============================================================================

def render_input_card() -> tuple[str, bool]:
    """Renders the statement-entry card. Returns (text_to_analyze, submit_clicked)."""
    
    st.markdown('<div class="card-heading">✍️ Enter Your Statement</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtext">Describe how you\'re feeling in a sentence or two — '
        'the model will analyze the language patterns.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    selected_example = st.selectbox(
        "Try an example (optional)",
        EXAMPLE_STATEMENTS,
        label_visibility="collapsed",
    )
    prefill = "" if selected_example == EXAMPLE_STATEMENTS[0] else selected_example

    user_input = st.text_area(
        "Enter a statement",
        value=prefill,
        height=150,
        max_chars=1000,
        placeholder='e.g. "I haven\'t felt like myself in weeks."',
        label_visibility="collapsed",
    )

    char_count = len(user_input)
    st.markdown(
        f'<div class="card-subtext" style="text-align:right;">{char_count} / 1000 characters</div>',
        unsafe_allow_html=True,
    )

    submit_clicked = st.button("✨ Analyze Statement", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    return user_input, submit_clicked


def render_empty_result_card() -> None:
    st.markdown("### 📈 Analysis Result")
    st.write("🧭 Your result will appear here once you analyze a statement.")


# =============================================================================
# CHARTS  (Plotly)
# =============================================================================

def build_gauge_figure(confidence: float, color: str) -> go.Figure:
    """Builds a circular confidence gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#F1F5F9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(148,163,184,0.12)"},
                    {"range": [50, 80], "color": "rgba(148,163,184,0.18)"},
                    {"range": [80, 100], "color": "rgba(148,163,184,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F1F5F9"},
        height=230,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def build_probability_bar_chart(class_probabilities: dict[str, float]) -> go.Figure:
    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1])
    labels = [item[0] for item in ranked]
    values = [item[1] * 100 for item in ranked]
    colors = [STATUS_STYLES.get(label, DEFAULT_STATUS_STYLE)["color"] for label in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F1F5F9"},
        xaxis=dict(range=[0, max(values) * 1.25 if values else 100], showgrid=False, visible=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=30, t=10, b=10),
        height=320,
        transition={"duration": 400, "easing": "cubic-in-out"},
    )
    return fig


def build_donut_chart(class_probabilities: dict[str, float]) -> go.Figure:
    labels = list(class_probabilities.keys())
    values = [v * 100 for v in class_probabilities.values()]
    colors = [STATUS_STYLES.get(label, DEFAULT_STATUS_STYLE)["color"] for label in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
            textinfo="percent",
            hoverinfo="label+percent",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F1F5F9"},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        transition={"duration": 400, "easing": "cubic-in-out"},
    )
    return fig


# =============================================================================
# PREDICTION CARD
# =============================================================================

def render_result_card(result: PredictionResult) -> None:
    style = STATUS_STYLES.get(result.label, DEFAULT_STATUS_STYLE)
    explanation = STATUS_EXPLANATIONS.get(result.label, "The model matched this statement to the label shown above.")

    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📈 Analysis Result</div>', unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.markdown(
            f"""
            <div style="font-size: 2.6rem; line-height: 1;">{style['emoji']}</div>
            <div style="font-size: 1.6rem; font-weight: 800; color:{style['color']}; margin-top: 0.3rem;">
                {result.label}
            </div>
            <span class="risk-badge" style="background: {style['risk_color']}22; color: {style['risk_color']};
                  border: 1px solid {style['risk_color']}55;">
                {style['risk']} risk
            </span>
            <div class="card-subtext" style="margin-top: 0.8rem;">{explanation}</div>
            """,
            unsafe_allow_html=True,
        )
        if result.label == "Suicidal":
            st.warning(CRISIS_RESOURCES)

    with top_right:
        st.plotly_chart(
            build_gauge_figure(result.confidence, style["color"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(
            '<div class="card-subtext" style="text-align:center;">Model confidence</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card-heading" style="font-size:1rem;">📊 Full Probability Breakdown</div>', unsafe_allow_html=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(build_probability_bar_chart(result.class_probabilities), use_container_width=True, config={"displayModeBar": False})
    with chart_right:
        st.plotly_chart(build_donut_chart(result.class_probabilities), use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 View preprocessed text sent to the model"):
        st.code(result.cleaned_text or "(empty after preprocessing)")


# =============================================================================
# HISTORY
# =============================================================================

def add_to_history(raw_text: str, result: PredictionResult) -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    style = STATUS_STYLES.get(result.label, DEFAULT_STATUS_STYLE)
    preview = raw_text.strip().replace("\n", " ")
    preview = (preview[:60] + "…") if len(preview) > 60 else preview

    st.session_state["history"].insert(
        0,
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Statement Preview": preview,
            "Prediction": f"{style['emoji']} {result.label}",
            "Confidence": f"{result.confidence:.0%}",
            "Risk": style["risk"],
            "risk_color": style["risk_color"],
        },
    )


def render_history_table() -> None:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📜 Prediction History</div>', unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    if not history:
        st.markdown('<div class="card-subtext">No predictions yet — analyze a statement to build history.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows_html = ""
    for row in history:
        rows_html += (
            "<tr>"
            f"<td>{row['Time']}</td>"
            f"<td>{row['Statement Preview']}</td>"
            f"<td>{row['Prediction']}</td>"
            f"<td>{row['Confidence']}</td>"
            f"<td><span class='risk-badge' style='background:{row['risk_color']}22; "
            f"color:{row['risk_color']}; border:1px solid {row['risk_color']}55;'>{row['Risk']}</span></td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <table class="history-table">
            <thead>
                <tr><th>Time</th><th>Statement Preview</th><th>Prediction</th><th>Confidence</th><th>Risk</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear history"):
        st.session_state["history"] = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# ANALYTICS PAGE
# =============================================================================

def render_analytics() -> None:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📊 Session Analytics</div>', unsafe_allow_html=True)

    history = st.session_state.get("history", [])
    if not history:
        st.markdown(
            '<div class="card-subtext">Run a few predictions first — analytics are computed from your session history.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(history)
    df["Label"] = df["Prediction"].str.split(" ", n=1).str[1]
    counts = df["Label"].value_counts()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Total predictions", len(df))
    metric_cols[1].metric("Most common status", counts.idxmax())
    avg_conf = df["Confidence"].str.rstrip("%").astype(float).mean()
    metric_cols[2].metric("Average confidence", f"{avg_conf:.0f}%")

    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color=[STATUS_STYLES.get(l, DEFAULT_STATUS_STYLE)["color"] for l in counts.index]),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F1F5F9"},
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="Count"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# ABOUT MODEL / INSTRUCTIONS PAGES
# =============================================================================

def render_about_model() -> None:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">🧠 About the Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-subtext">
        This assistant is powered by a <b>Bidirectional LSTM (BiLSTM)</b> neural network
        trained on labeled mental-health statements. Text is first embedded using
        pretrained <b>FastText</b> word vectors, then passed through the BiLSTM layers
        to produce a probability distribution over mental-health status categories.
        <br><br>
        <b>Pipeline:</b> raw text → cleaning &amp; lemmatization → tokenization →
        padding (300 tokens) → BiLSTM inference → softmax probabilities → label decode.
        <br><br>
        <b>Categories recognized:</b> Normal, Depression, Suicidal, Anxiety, Bipolar,
        Stress, Personality disorder.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_instructions() -> None:
    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">📖 How to Use</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-subtext">
        1. Go to <b>🏠 Dashboard</b> and type a statement, or pick an example.<br>
        2. Click <b>✨ Analyze Statement</b> to run the model.<br>
        3. Review the predicted status, confidence gauge, and full probability breakdown.<br>
        4. Check <b>📜 Prediction History</b> to revisit earlier analyses this session.<br>
        5. Visit <b>📊 Analytics</b> for a summary of your session's predictions.<br><br>
        This tool is for exploration and demonstration only — see the disclaimer at
        the bottom of the Dashboard page.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# DISCLAIMER / FOOTER
# =============================================================================

def render_disclaimer() -> None:
    st.warning(f"⚠️ {DISCLAIMER_TEXT}")


def render_footer() -> None:
    st.markdown(
        f"""
        <div style="text-align:center; color:{TEXT_MUTED}; font-size:0.85rem; padding-top: 1rem;">
            Made with ❤️ using Python · TensorFlow · FastText · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DASHBOARD PAGE (input + result)
# =============================================================================

def render_dashboard(model, tokenizer, label_encoder, preprocessor: TextPreprocessor) -> None:
    render_hero()

    input_col, result_col = st.columns([5, 6], gap="medium")

    with input_col:
        user_input, submit_clicked = render_input_card()

    result: PredictionResult | None = None
    if submit_clicked:
        if not user_input.strip():
            st.warning("Please enter a statement first.")
        else:
            with st.spinner("Analyzing statement..."):
                result = predict_status(
                    raw_text=user_input,
                    model=model,
                    tokenizer=tokenizer,
                    label_encoder=label_encoder,
                    preprocessor=preprocessor,
                    max_length=CONFIG.max_sequence_length,
                )
            add_to_history(user_input, result)
            st.session_state["last_result"] = result

    with result_col:
        display_result = result or st.session_state.get("last_result")
        if display_result is not None:
            render_result_card(display_result)
        else:
            render_empty_result_card()

    render_disclaimer()
    render_footer()


# =============================================================================
# App entry point
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    missing_nltk = ensure_nltk_resources(CONFIG.nltk_resources)
    if missing_nltk:
        st.error(
            "Could not load NLTK data for: "
            + ", ".join(missing_nltk)
            + ". Predictions may not exactly match the trained model. "
            "Bundle an `nltk_data` folder next to app.py to fix this permanently."
        )

    model, tokenizer, label_encoder = load_model_artifacts(CONFIG)
    preprocessor = TextPreprocessor()

    page = render_sidebar()

    if page == "🏠 Dashboard":
        render_dashboard(model, tokenizer, label_encoder, preprocessor)
    elif page == "📜 Prediction History":
        render_history_table()
        render_footer()
    elif page == "📊 Analytics":
        render_analytics()
        render_footer()
    elif page == "🧠 About Model":
        render_about_model()
        render_footer()
    elif page == "📖 Instructions":
        render_instructions()
        render_footer()


if __name__ == "__main__":
    main()