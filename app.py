"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXPLAINABLE AI HEALTHCARE - Multiple Disease Prediction              ║
║         Powered by ML + SHAP for Human-Readable Explanations                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author  : Upgraded from jayanthreddymandadi/Multiple-Disease-Prediction-using-SHAP
Purpose : Predict diseases AND explain WHY in plain English using SHAP values
"""

import os
import json
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# ── [AUTH] Import authentication and database modules ─────────────────────────
from auth import (
    init_session, is_authenticated,
    current_user_id, current_user_email,
    render_auth_page, render_sidebar_user_widget,
)
import db as database

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi Disease Prediction Using SHAP",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

# ── [AUTH] Initialise session state keys before anything else ─────────────────
init_session()

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean medical aesthetic with clear typography
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.main { background-color: #f0f4f8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #e8edf3;
}

/* Prediction result banners */
.result-positive {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-left: 6px solid #ef4444;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
}
.result-negative {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 6px solid #22c55e;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
}
.result-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.result-subtitle {
    font-size: 0.9rem;
    opacity: 0.75;
    margin: 0;
    font-weight: 400;
}

/* Explanation bullets */
.explanation-container {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
    border: 1px solid #e2e8f0;
}
.factor-positive {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    background: #fff5f5;
    border-radius: 8px;
    border-left: 4px solid #f87171;
    font-size: 0.92rem;
    line-height: 1.5;
}
.factor-negative {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    background: #f0fdf4;
    border-radius: 8px;
    border-left: 4px solid #4ade80;
    font-size: 0.92rem;
    line-height: 1.5;
}
.factor-icon { font-size: 1.1rem; margin-top: 1px; }
.factor-text { flex: 1; color: #1e293b; }
.factor-badge {
    background: #e2e8f0;
    color: #475569;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
    align-self: center;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
    margin: 20px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Input styling */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1.5px solid #e2e8f0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 10px 14px;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 12px 32px;
    border-radius: 10px;
    border: none;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    box-shadow: 0 6px 20px rgba(37,99,235,0.4);
    transform: translateY(-1px);
}

/* Disclaimer */
.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #92400e;
    margin-top: 16px;
}

/* Page title */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -1px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 24px;
    font-weight: 400;
}

/* Divider */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* SHAP chart container */
.shap-container {
    background: white;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e2e8f0;
}

/* Metric chips */
.metric-chip {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all trained ML models from disk (cached for performance)."""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}
    model_paths = {
        "diabetes":    f"{working_dir}/notebooks/saved_models/diabetes_model.pkl",
        "heart":       f"{working_dir}/notebooks/saved_models/heart_disease_model.pkl",
        "parkinsons":  f"{working_dir}/notebooks/saved_models/parkinsons_model.pkl",
    }
    for name, path in model_paths.items():
        try:
            models[name] = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            models[name] = None
            st.warning(f"⚠️ Model file not found: {path}  — run the training notebook first.")
    return models

models = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# ── PREDICTION HISTORY ENGINE  [UPDATED → Supabase]  ─────────────────────────
# Old JSON-based functions are replaced by thin wrappers around db.py.
# build_history_record() is unchanged — it still assembles the record dict.
# save_history() and load_history() now talk to Supabase instead of a file.
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    """Fetch current user's predictions from Supabase (newest first)."""
    return database.load_history(current_user_id())


def save_history(record: dict) -> None:
    """Insert one prediction record into Supabase for the current user."""
    database.save_history(record, current_user_id())


def clear_history() -> None:
    """Delete all predictions for the current user from Supabase."""
    database.clear_history(current_user_id())


def build_history_record(
    disease_name: str,
    prediction: int,
    feature_names: list,
    input_values: np.ndarray,
    explanations: list[dict],
) -> dict:
    """
    Assemble a single history record from prediction outputs.
    Called inside run_prediction_pipeline() after SHAP explanations are ready.
    Only the top 3 explanations (by SHAP magnitude) are stored as key_reasons.
    """
    # Top-3 reasons: compact human-readable string
    top3 = explanations[:3]
    reason_parts = []
    for e in top3:
        arrow = "↑" if e["direction"] == "increases" else "↓"
        reason_parts.append(f"{e['label']} ({e['magnitude']}{arrow})")
    key_reasons = "; ".join(reason_parts) if reason_parts else "N/A"

    # Store all input values as a plain dict (feature → rounded value)
    input_dict = {
        feat: round(float(val), 6)
        for feat, val in zip(feature_names, input_values)
    }

    return {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disease":      disease_name,
        "prediction":   "Yes " if prediction == 1 else "No ",
        "key_reasons":  key_reasons,
        "input_values": input_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── CORE EXPLAINABILITY ENGINE ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Friendly names for technical feature labels
FEATURE_LABELS = {
    # Diabetes
    "Pregnancies":               "Number of Pregnancies",
    "Glucose":                   "Blood Glucose Level",
    "BloodPressure":             "Blood Pressure",
    "SkinThickness":             "Skin Fold Thickness",
    "Insulin":                   "Insulin Level",
    "BMI":                       "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction":  "Diabetes Family History Score",
    "Age":                       "Age",
    # Heart
    "Age":                       "Age",
    "Sex":                       "Sex",
    "Chest Pain types":          "Chest Pain Type",
    "Resting Blood Pressure":    "Resting Blood Pressure",
    "Serum Cholestoral":         "Serum Cholesterol",
    "Fasting Blood Sugar > 120 mg/dl": "Fasting Blood Sugar",
    "Resting ECG results":       "Resting ECG Result",
    "Max Heart Rate achieved":   "Maximum Heart Rate",
    "Exercise Induced Angina":   "Exercise-Induced Angina",
    "Oldpeak":                   "ST Depression (Oldpeak)",
    "Slope of Peak Exercise ST": "ST Slope",
    "Number of Major Vessels":   "Major Vessels (0–3)",
    "Thalassemia":               "Thalassemia Type",
    # Parkinson's — abbreviated MDVP measurements
    "MDVP:Fo(Hz)":      "Average Vocal Frequency",
    "MDVP:Fhi(Hz)":     "Maximum Vocal Frequency",
    "MDVP:Flo(Hz)":     "Minimum Vocal Frequency",
    "MDVP:Jitter(%)":   "Voice Jitter (%)",
    "MDVP:Jitter(Abs)": "Absolute Voice Jitter",
    "MDVP:RAP":         "Relative Amplitude Perturbation",
    "MDVP:PPQ":         "Five-Point Period Perturbation",
    "Jitter:DDP":       "Average Absolute Jitter Difference",
    "MDVP:Shimmer":     "Voice Shimmer",
    "MDVP:Shimmer(dB)": "Shimmer in dB",
    "Shimmer:APQ3":     "Three-Point Amplitude Perturbation",
    "Shimmer:APQ5":     "Five-Point Amplitude Perturbation",
    "MDVP:APQ":         "Average Amplitude Perturbation",
    "Shimmer:DDA":      "Average Absolute Shimmer Difference",
    "NHR":              "Noise-to-Harmonic Ratio",
    "HNR":              "Harmonic-to-Noise Ratio",
    "RPDE":             "Recurrence Period Density Entropy",
    "DFA":              "Detrended Fluctuation Analysis",
    "spread1":          "Nonlinear Frequency Spread (1)",
    "spread2":          "Nonlinear Frequency Spread (2)",
    "D2":               "Correlation Dimension",
    "PPE":              "Pitch Period Entropy",
}


def friendly_name(feature: str) -> str:
    """Return a human-friendly label for any feature name."""
    return FEATURE_LABELS.get(feature, feature)


def compute_shap_values(model, input_array: np.ndarray, n_features: int) -> np.ndarray:
    """
    Compute SHAP values for a single prediction using KernelExplainer.
    Uses a zero-baseline for speed; sufficient for directional explanation.
    """
    background = np.zeros((1, n_features))
    explainer  = shap.KernelExplainer(model.predict, background)
    shap_vals  = explainer.shap_values(input_array, nsamples=100)
    return shap_vals, explainer.expected_value


def generate_text_explanation(
    shap_values: np.ndarray,
    feature_names: list,
    input_values: np.ndarray,
    disease_name: str,
    prediction: int,
    top_n: int = 5,
) -> list[dict]:
    """
    Convert raw SHAP values into a ranked list of plain-English explanations.

    Returns a list of dicts:
        {
          "feature":      original feature name,
          "label":        friendly label,
          "value":        user-entered value,
          "shap":         SHAP value (float),
          "direction":    "increases" | "decreases",
          "magnitude":    "strongly" | "moderately" | "slightly",
          "sentence":     full plain-English sentence,
        }
    """
    shap_flat = np.array(shap_values).flatten()

    # Rank by absolute SHAP value (most important first)
    ranked_idx = np.argsort(np.abs(shap_flat))[::-1][:top_n]

    max_abs = np.max(np.abs(shap_flat)) if np.max(np.abs(shap_flat)) > 0 else 1.0

    explanations = []
    for idx in ranked_idx:
        sv    = float(shap_flat[idx])
        fname = feature_names[idx]
        label = friendly_name(fname)
        val   = float(input_values[idx])

        # Direction
        if sv > 0:
            direction = "increases"
            risk_word = "elevated" if val > 0 else "present"
        else:
            direction = "decreases"
            risk_word = "lower" if val >= 0 else "absent"

        # Magnitude (relative to max)
        ratio = abs(sv) / max_abs
        if ratio >= 0.60:
            magnitude = "strongly"
            strength  = "major"
        elif ratio >= 0.30:
            magnitude = "moderately"
            strength  = "notable"
        else:
            magnitude = "slightly"
            strength  = "minor"

        # Build sentence
        val_str = f"{val:.4g}"  # compact number display
        if direction == "increases":
            sentence = (
                f"Your {label} (value: {val_str}) {magnitude} {direction} "
                f"the likelihood of {disease_name}."
            )
        else:
            sentence = (
                f"Your {label} (value: {val_str}) {magnitude} {direction} "
                f"the likelihood of {disease_name}, which is a protective factor."
            )

        explanations.append({
            "feature":   fname,
            "label":     label,
            "value":     val,
            "shap":      sv,
            "direction": direction,
            "magnitude": magnitude,
            "strength":  strength,
            "sentence":  sentence,
        })

    return explanations


def generate_summary_sentence(
    explanations: list[dict],
    disease_name: str,
    prediction: int,
) -> str:
    """
    Produce a single paragraph that sums up the prediction and its top drivers.
    """
    risk_factors     = [e for e in explanations if e["direction"] == "increases"]
    protective       = [e for e in explanations if e["direction"] == "decreases"]

    if prediction == 1:
        if risk_factors:
            top_risks = ", ".join(
                [f"**{e['label']}**" for e in risk_factors[:2]]
            )
            summary = (
                f"The model predicts **{disease_name} is likely** primarily due to "
                f"{top_risks}"
            )
            if protective:
                prot = f"**{protective[0]['label']}**"
                summary += f", although {prot} partially offsets the risk."
            else:
                summary += "."
        else:
            summary = f"The model predicts **{disease_name} is likely** based on the combination of your health metrics."
    else:
        if protective:
            top_prot = ", ".join(
                [f"**{e['label']}**" for e in protective[:2]]
            )
            summary = (
                f"The model predicts **{disease_name} is unlikely**. Your "
                f"{top_prot} appear within healthy ranges and are the strongest "
                f"protective factors."
            )
        else:
            summary = f"The model predicts **{disease_name} is unlikely** based on your overall health profile."

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ── UI RENDERING HELPERS ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def render_prediction_banner(prediction: int, disease_name: str):
    """Render a coloured result banner."""
    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <p class="result-title">🔴 {disease_name} Detected</p>
            <p class="result-subtitle">
                The model indicates risk factors consistent with {disease_name}.
                Please consult a qualified healthcare professional.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <p class="result-title">🟢 No {disease_name} Detected</p>
            <p class="result-subtitle">
                The model does not detect significant risk factors for {disease_name}
                based on the values provided.
            </p>
        </div>""", unsafe_allow_html=True)


def render_explanation(
    explanations: list[dict],
    summary: str,
    prediction: int,
    disease_name: str,
):
    """Render the full human-readable explanation panel."""
    st.markdown("---")
    st.markdown(
        '<p class="section-header">🧠 Why did the model make this prediction?</p>',
        unsafe_allow_html=True,
    )

    # Summary paragraph
    st.info(summary)

    # Split into risk-raising vs risk-lowering
    risk_factors = [e for e in explanations if e["direction"] == "increases"]
    protective   = [e for e in explanations if e["direction"] == "decreases"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<p class="section-header">🔺 Factors Increasing Risk</p>',
            unsafe_allow_html=True,
        )
        if risk_factors:
            for e in risk_factors:
                badge_color = (
                    "#fee2e2" if e["magnitude"] == "strongly"
                    else "#fef3c7" if e["magnitude"] == "moderately"
                    else "#f1f5f9"
                )
                st.markdown(f"""
                <div class="factor-positive">
                    <span class="factor-icon">
                        {"🔴" if e["magnitude"] == "strongly"
                          else "🟠" if e["magnitude"] == "moderately"
                          else "🟡"}
                    </span>
                    <span class="factor-text">{e['sentence']}</span>
                    <span class="factor-badge">SHAP: {e['shap']:+.3f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="explanation-container">✅ No significant risk-raising factors identified.</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            '<p class="section-header">🔻 Protective / Reducing Factors</p>',
            unsafe_allow_html=True,
        )
        if protective:
            for e in protective:
                st.markdown(f"""
                <div class="factor-negative">
                    <span class="factor-icon">
                        {"🟢" if e["magnitude"] == "strongly"
                          else "🔵" if e["magnitude"] == "moderately"
                          else "⚪"}
                    </span>
                    <span class="factor-text">{e['sentence']}</span>
                    <span class="factor-badge">SHAP: {e['shap']:+.3f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="explanation-container">⚠️ No significant protective factors identified.</div>',
                unsafe_allow_html=True,
            )


def render_shap_bar_chart(
    explanations: list[dict],
    disease_name: str,
):
    """Render a clean horizontal SHAP bar chart."""
    st.markdown(
        '<p class="section-header">📊 Feature Impact Chart (SHAP Values)</p>',
        unsafe_allow_html=True,
    )

    labels  = [e["label"] for e in reversed(explanations)]
    values  = [e["shap"]  for e in reversed(explanations)]
    colors  = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.55)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            color="#374151",
        )

    ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP Value  (← decreases risk  |  increases risk →)", fontsize=9, color="#64748b")
    ax.set_title(f"Feature contributions for {disease_name} prediction",
                 fontsize=10, color="#0f172a", pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9, colors="#374151")
    ax.tick_params(axis="x", labelsize=8, colors="#64748b")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def render_disclaimer():
    st.markdown("""
    
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PREDICTION PIPELINE ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction_pipeline(
    model,
    input_array: np.ndarray,
    feature_names: list,
    disease_name: str,
    top_n: int = 5,
):
    """
    Full pipeline:
      1. Predict
      2. Compute SHAP values
      3. Generate text explanations
      4. Render UI
      5. [NEW] Save to prediction history
    """
    prediction = int(model.predict(input_array)[0])

    # Render result banner immediately
    render_prediction_banner(prediction, disease_name)

    # SHAP computation (with progress spinner)
    with st.spinner("🔍 Computing SHAP explanations…"):
        shap_values, expected_val = compute_shap_values(
            model, input_array, len(feature_names)
        )

    # Generate text explanations
    explanations = generate_text_explanation(
        shap_values, feature_names,
        input_array[0], disease_name, prediction, top_n=top_n,
    )

    summary = generate_summary_sentence(explanations, disease_name, prediction)

    # Render explanation panel
    render_explanation(explanations, summary, prediction, disease_name)

    # SHAP bar chart
    with st.expander("📊 View Feature Impact Chart", expanded=True):
        render_shap_bar_chart(explanations, disease_name)

    # Medical disclaimer
    render_disclaimer()

    # ── [NEW] Save this prediction to history ─────────────────────────────────
    record = build_history_record(
        disease_name  = disease_name,
        prediction    = prediction,
        feature_names = feature_names,
        input_values  = input_array[0],
        explanations  = explanations,
    )
    save_history(record)
    st.toast("📋 Prediction saved to history", icon="✅")


# ─────────────────────────────────────────────────────────────────────────────
# ── INPUT HELPERS ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def input_grid(features: list, cols: int = 3, hints: dict = None) -> list:
    """
    Render text inputs in a responsive grid.
    Returns a list of raw string values.
    """
    hints   = hints or {}
    values  = []
    columns = st.columns(cols)
    for i, feat in enumerate(features):
        col = columns[i % cols]
        with col:
            hint = hints.get(feat, "")
            val  = st.text_input(
                feat,
                placeholder=hint,
                help=f"Enter value for **{friendly_name(feat)}**. {hint}",
                key=f"input_{feat}",
            )
            values.append(val)
    return values


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.5rem;'></div>
        <div style='font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-top:8px;'>
            Multi Diseases Prediction Using SHAP
        </div>
        <div style='font-size:0.78rem; color:#94a3b8; margin-top:4px;'>
            Explainable Disease Prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── [AUTH] Show user widget + logout only when authenticated ──────────────
    if is_authenticated():
        render_sidebar_user_widget()
        st.markdown("<hr style='border-color:#334155; margin:12px 0'>",
                    unsafe_allow_html=True)

        selected = option_menu(
            menu_title=None,
            options=["Diabetes", "Heart Disease", "Parkinson's", "History"],
            icons=["droplet-fill", "heart-pulse-fill",
                   "person-lines-fill", "clock-history"],
            default_index=0,
            styles={
                "container":    {"background-color": "transparent", "padding": "0"},
                "icon":         {"color": "#60a5fa", "font-size": "16px"},
                "nav-link":     {
                    "font-size": "0.9rem",
                    "color": "#cbd5e1",
                    "padding": "12px 16px",
                    "border-radius": "8px",
                    "margin": "2px 0",
                },
                "nav-link-selected": {
                    "background-color": "#1d4ed8",
                    "color": "white",
                    "font-weight": "600",
                },
            },
        )

        st.markdown("<hr style='border-color:#334155; margin:16px 0'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.78rem; color:#64748b; padding:0 4px;'>
            <b style='color:#94a3b8;'>How it works:</b><br><br>
            1️⃣ Enter your health values<br><br>
            2️⃣ Click <b>Predict</b><br><br>
            3️⃣ Get prediction + plain-English explanation powered by SHAP<br><br>
            4️⃣ View all past predictions in <b>History</b>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Not logged in — show minimal sidebar
        selected = None
        st.markdown("""
        <div style='font-size:0.82rem; color:#64748b; padding:8px 4px;'>
            Please <b style='color:#94a3b8;'>log in</b> or
            <b style='color:#94a3b8;'>sign up</b> to use the app.
        </div>
        """, unsafe_allow_html=True)


# ── [AUTH] Gate: show auth page if not logged in ──────────────────────────────
if not is_authenticated():
    render_auth_page()
    st.stop()   # Stop executing the rest of the app


# ═════════════════════════════════════════════════════════════════════════════
# ── DISEASE PAGES ─────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────
# DIABETES
# ─────────────────────────────────────
if selected == "Diabetes":
    st.markdown('<p class="page-title">Diabetes Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Enter patient health metrics below. '
        'The AI will predict diabetes risk and explain the key contributing factors.</p>',
        unsafe_allow_html=True,
    )

    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]

    hints = {
        "Pregnancies":              "0–17  (e.g. 3)",
        "Glucose":                  "70–200 mg/dL  (e.g. 120)",
        "BloodPressure":            "40–130 mmHg  (e.g. 72)",
        "SkinThickness":            "0–100 mm  (e.g. 23)",
        "Insulin":                  "0–846 μU/mL  (e.g. 80)",
        "BMI":                      "10–70  (e.g. 32.0)",
        "DiabetesPedigreeFunction": "0.07–2.5  (e.g. 0.47)",
        "Age":                      "21–81 years",
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Patient Information**")
    user_vals = input_grid(feature_names, cols=4, hints=hints)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict Diabetes Risk", key="btn_diabetes"):
        try:
            arr = np.array([float(v) for v in user_vals]).reshape(1, -1)
        except ValueError:
            st.error("⚠️ Please fill in all fields with valid numbers before predicting.")
        else:
            if models["diabetes"] is None:
                st.error("Diabetes model could not be loaded. Check model file path.")
            else:
                run_prediction_pipeline(
                    models["diabetes"], arr, feature_names,
                    disease_name="Diabetes", top_n=5,
                )

# ─────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────
elif selected == "Heart Disease":
    st.markdown('<p class="page-title"> Heart Disease Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Enter cardiac health metrics. '
        'The AI will predict heart disease risk and explain the reasoning.</p>',
        unsafe_allow_html=True,
    )

    feature_names = [
        "Age", "Sex", "Chest Pain types", "Resting Blood Pressure",
        "Serum Cholestoral", "Fasting Blood Sugar > 120 mg/dl",
        "Resting ECG results", "Max Heart Rate achieved",
        "Exercise Induced Angina", "Oldpeak",
        "Slope of Peak Exercise ST", "Number of Major Vessels", "Thalassemia",
    ]

    hints = {
        "Age":                              "29–77 years",
        "Sex":                              "1=Male, 0=Female",
        "Chest Pain types":                 "0–3 (0=typical angina, 3=asymptomatic)",
        "Resting Blood Pressure":           "94–200 mmHg",
        "Serum Cholestoral":                "126–564 mg/dL",
        "Fasting Blood Sugar > 120 mg/dl":  "1=True, 0=False",
        "Resting ECG results":              "0–2 (0=Normal, 1=ST-T wave, 2=LV hypertrophy)",
        "Max Heart Rate achieved":          "71–202 bpm",
        "Exercise Induced Angina":          "1=Yes, 0=No",
        "Oldpeak":                          "0.0–6.2",
        "Slope of Peak Exercise ST":        "0–2 (0=Upsloping, 1=Flat, 2=Downsloping)",
        "Number of Major Vessels":          "0–4",
        "Thalassemia":                      "0–3 (1=Normal, 2=Fixed defect, 3=Reversible)",
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Cardiac Health Metrics**")
    user_vals = input_grid(feature_names, cols=3, hints=hints)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict Heart Disease Risk", key="btn_heart"):
        try:
            arr = np.array([float(v) for v in user_vals]).reshape(1, -1)
        except ValueError:
            st.error("⚠️ Please fill in all fields with valid numbers before predicting.")
        else:
            if models["heart"] is None:
                st.error("Heart disease model could not be loaded. Check model file path.")
            else:
                run_prediction_pipeline(
                    models["heart"], arr, feature_names,
                    disease_name="Heart Disease", top_n=5,
                )

# ─────────────────────────────────────
# PARKINSON'S DISEASE
# ─────────────────────────────────────
elif selected == "Parkinson's":
    st.markdown("<p class='page-title'>Parkinson's Disease Prediction</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>Enter vocal measurement features. "
        "The AI will predict Parkinson's risk and explain the contributing voice biomarkers.</p>",
        unsafe_allow_html=True,
    )

    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE",
    ]

    hints = {
        "MDVP:Fo(Hz)":      "88–260 Hz (e.g. 154.2)",
        "MDVP:Fhi(Hz)":     "102–592 Hz (e.g. 197.1)",
        "MDVP:Flo(Hz)":     "65–239 Hz (e.g. 116.3)",
        "MDVP:Jitter(%)":   "0.001–0.033 (e.g. 0.006)",
        "MDVP:Jitter(Abs)": "0–0.00026 (e.g. 0.00004)",
        "MDVP:RAP":         "0–0.021 (e.g. 0.003)",
        "MDVP:PPQ":         "0–0.020 (e.g. 0.003)",
        "Jitter:DDP":       "0–0.064 (e.g. 0.010)",
        "MDVP:Shimmer":     "0.009–0.12 (e.g. 0.030)",
        "MDVP:Shimmer(dB)": "0.085–1.3 (e.g. 0.28)",
        "Shimmer:APQ3":     "(e.g. 0.016)",
        "Shimmer:APQ5":     "(e.g. 0.019)",
        "MDVP:APQ":         "(e.g. 0.024)",
        "Shimmer:DDA":      "(e.g. 0.047)",
        "NHR":              "0–0.315 (e.g. 0.025)",
        "HNR":              "8–33 dB (e.g. 21.9)",
        "RPDE":             "0.26–0.69 (e.g. 0.50)",
        "DFA":              "0.57–0.83 (e.g. 0.72)",
        "spread1":          "-8 to -2 (e.g. -5.68)",
        "spread2":          "0.006–0.45 (e.g. 0.23)",
        "D2":               "1.4–3.7 (e.g. 2.38)",
        "PPE":              "0.04–0.53 (e.g. 0.21)",
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Vocal Biomarker Measurements**")
    st.caption(
        "These measurements are derived from sustained phonation recordings. "
        "You can obtain them using voice analysis software."
    )
    user_vals = input_grid(feature_names, cols=4, hints=hints)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick-fill example button for demo purposes
    col_btn, col_ex = st.columns([2, 1])
    with col_btn:
        predict_clicked = st.button("🔍 Predict Parkinson's Risk", key="btn_park")

    if predict_clicked:
        try:
            arr = np.array([float(v) for v in user_vals]).reshape(1, -1)
        except ValueError:
            st.error("⚠️ Please fill in all 22 fields with valid numbers before predicting.")
        else:
            if models["parkinsons"] is None:
                st.error("Parkinson's model could not be loaded. Check model file path.")
            else:
                run_prediction_pipeline(
                    models["parkinsons"], arr, feature_names,
                    disease_name="Parkinson's Disease", top_n=6,
                )


# ═════════════════════════════════════════════════════════════════════════════
# ── HISTORY PAGE  [NEW]  ──────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
elif selected == "History":

    st.markdown('<p class="page-title">📋 Prediction History</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">All past predictions are stored here automatically. '
        'Latest entries appear first.</p>',
        unsafe_allow_html=True,
    )

    records = load_history()

    # ── Empty state ────────────────────────────────────────────────────────────
    if not records:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 48px 32px;">
            <div style="font-size:3rem; margin-bottom:12px;">🗂️</div>
            <div style="font-size:1.1rem; font-weight:600; color:#0f172a; margin-bottom:8px;">
                No predictions yet
            </div>
            <div style="font-size:0.9rem; color:#64748b;">
                Run a prediction from Diabetes, Heart Disease, or Parkinson's — 
                it will appear here automatically.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Summary metric row ─────────────────────────────────────────────────
        total     = len(records)
        positives = sum(1 for r in records if r.get("prediction", "").startswith("Yes"))
        negatives = total - positives
        diseases  = list({r.get("disease", "") for r in records})

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Predictions", total)
        mc2.metric("Positive Results",  positives)
        mc3.metric("Negative Results",  negatives)
        mc4.metric("Diseases Tracked",  len(diseases))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filter + controls row ──────────────────────────────────────────────
        col_filter, col_spacer, col_clear = st.columns([2, 3, 2])

        with col_filter:
            disease_options = ["All"] + sorted(
                list({r.get("disease", "Unknown") for r in records})
            )
            selected_filter = st.selectbox(
                "🔍 Filter by disease",
                options=disease_options,
                key="history_filter",
            )

        with col_clear:
            st.markdown("<br>", unsafe_allow_html=True)  # vertical align
            if st.button("🗑️ Clear All History", key="btn_clear"):
                st.session_state["confirm_clear"] = True

        # Confirmation dialog (avoids accidental deletion)
        if st.session_state.get("confirm_clear"):
            st.warning("⚠️ This will permanently delete all prediction history. Are you sure?")
            conf_col1, conf_col2, _ = st.columns([1, 1, 4])
            with conf_col1:
                if st.button("✅ Yes, clear", key="btn_confirm_clear"):
                    clear_history()
                    st.session_state["confirm_clear"] = False
                    st.success("History cleared.")
                    st.rerun()
            with conf_col2:
                if st.button("❌ Cancel", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Apply filter + reverse (latest first) ─────────────────────────────
        filtered = records if selected_filter == "All" else [
            r for r in records if r.get("disease") == selected_filter
        ]
        filtered = list(reversed(filtered))  # latest first

        if not filtered:
            st.info(f"No records found for **{selected_filter}**.")
        else:
            # ── Summary table ──────────────────────────────────────────────────
            table_data = []
            for r in filtered:
                table_data.append({
                    "Date & Time":  r.get("timestamp",    "—"),
                    "Disease":      r.get("disease",      "—"),
                    "Prediction":   r.get("prediction",   "—"),
                    "Key Reasons":  r.get("key_reasons",  "—"),
                })

            df = pd.DataFrame(table_data)

            # Colour the Prediction column text for visual clarity
            def style_prediction(val):
                if "Yes" in str(val):
                    return "color: #dc2626; font-weight: 600;"
                elif "No" in str(val):
                    return "color: #16a34a; font-weight: 600;"
                return ""

            styled_df = df.style.applymap(style_prediction, subset=["Prediction"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # ── Expandable detail rows ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<p class="section-header">🔎 Expand a Record for Full Input Details</p>',
                unsafe_allow_html=True,
            )

            for i, r in enumerate(filtered):
                pred_icon = "🔴" if r.get("prediction", "").startswith("Yes") else "🟢"
                label = (
                    f"{pred_icon}  {r.get('timestamp','—')}  ·  "
                    f"{r.get('disease','—')}  ·  {r.get('prediction','—')}"
                )
                with st.expander(label, expanded=False):
                    st.markdown(f"**Key Reasons:** {r.get('key_reasons','—')}")
                    st.markdown("**Full Input Values:**")

                    input_vals = r.get("input_values", {})
                    if input_vals:
                        # Display as a neat 3-column grid of metric cards
                        items  = list(input_vals.items())
                        n_cols = 3
                        rows   = [items[j:j+n_cols] for j in range(0, len(items), n_cols)]
                        for row in rows:
                            cols = st.columns(n_cols)
                            for col, (feat, val) in zip(cols, row):
                                col.metric(
                                    label=friendly_name(feat),
                                    value=round(float(val), 4),
                                )
                    else:
                        st.caption("No input data stored for this record.")

            # ── Export as CSV ──────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download History as CSV",
                data=csv_bytes,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_history_csv",
            )
