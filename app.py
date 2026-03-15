# This will create (or overwrite) the file app.py
print("Your Streamlit app will go here!")
# -----------------------------
# AI-Driven Lifestyle Disease Predictor — Streamlit App
# -----------------------------
# Expects files created in your notebooks:
#   ../models/model.pkl              -> fitted sklearn Pipeline (imputer+scaler+clf)
#   ../models/feature_order.pkl      -> list of feature column names used for training (X.columns)
#   ../models/threshold.json         -> {"best_threshold": float}, optional (defaults to 0.5)
#   ../models/encoders.pkl           -> mapping dicts for categoricals, optional (we provide safe defaults)
#
# You can run locally from the /app folder with:
#   streamlit run app.py
# -----------------------------
import zipfile
import os

if not os.path.exists("model.pkl"):
    with zipfile.ZipFile("model.zip", "r") as zip_ref:
        zip_ref.extractall()
        
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_order.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"  # optional

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_feature_order():
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_threshold(default=0.5):
    if THRESHOLD_PATH.exists():
        try:
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            thr = float(data.get("best_threshold", default))
            return min(max(thr, 0.0), 1.0)  # clamp
        except Exception:
            return default
    return default

@st.cache_resource(show_spinner=False)
def load_encoders():
    """
    Load optional encoders. If not found, provide safe defaults that match
    the mappings you used during preprocessing.
    """
    defaults = {
        "gender_map": {"Male": 0, "Female": 1, "Other": 2},
        "smoker_map": {"No": 0, "Yes": 1},
        "exercise_map": {"None": 0, "1-2 Times/Week": 1, "3-5 Times/Week": 2, "Daily": 3},
        "diet_map": {"Poor": 0, "Good": 1, "Excellent": 2},
        "alcohol_map": {"None": 0, "Low": 1, "Moderate": 2, "High": 3},
    }
    if ENCODERS_PATH.exists():
        try:
            with open(ENCODERS_PATH, "rb") as f:
                loaded = pickle.load(f)
            # Merge in anything missing
            for k, v in defaults.items():
                loaded.setdefault(k, v)
            return loaded
        except Exception:
            return defaults
    return defaults

model = load_model()
feature_order = load_feature_order()           # list of feature names (X.columns)
best_threshold = load_threshold(0.5)           # decision threshold
encoders = load_encoders()                     # category mappings

# Useful lookups
gender_opts    = list(encoders["gender_map"].keys())
smoker_opts    = list(encoders["smoker_map"].keys())
exercise_opts  = list(encoders["exercise_map"].keys())
diet_opts      = list(encoders["diet_map"].keys())
alcohol_opts   = list(encoders["alcohol_map"].keys())

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Lifestyle Disease Predictor", page_icon="🧬", layout="centered")
st.title("🧬 AI-Driven Lifestyle Disease Predictor")
st.caption("Enter your details to estimate the risk of a lifestyle-related chronic disease.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a machine-learning model you trained in the notebooks. "
        "Inputs are encoded and scaled exactly like training (using the saved pipeline)."
    )
    st.write(f"**Decision threshold**: {best_threshold:.3f}")
    st.write("✅ Model & metadata loaded successfully.")

# -----------------------------
# Input form
# -----------------------------
with st.form("user_inputs"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=35, step=1)
        height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=35.0, max_value=180.0, value=70.0, step=0.5)
        # BMI will be calculated but we allow manual tweak if desired
        auto_bmi = weight_kg / ((height_cm / 100) ** 2)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(np.round(auto_bmi, 1)), step=0.1)

    with col2:
        gender = st.selectbox("Gender", gender_opts, index=gender_opts.index("Male") if "Male" in gender_opts else 0)
        smoker = st.selectbox("Smoker", smoker_opts, index=smoker_opts.index("No") if "No" in smoker_opts else 0)
        exercise = st.selectbox("Exercise Frequency", exercise_opts, index=exercise_opts.index("3-5 Times/Week") if "3-5 Times/Week" in exercise_opts else 1)
        diet_quality = st.selectbox("Diet Quality", diet_opts, index=diet_opts.index("Good") if "Good" in diet_opts else 1)
        alcohol = st.selectbox("Alcohol Consumption", alcohol_opts, index=alcohol_opts.index("Moderate") if "Moderate" in alcohol_opts else 2)

    stress_level = st.slider("Stress Level (1–10)", min_value=1, max_value=10, value=4, step=1)
    sleep_hours  = st.slider("Sleep Hours (per day)", min_value=3.0, max_value=12.0, value=7.0, step=0.5)

    submitted = st.form_submit_button("Predict risk")

# -----------------------------
# Build input row aligned to training feature order
# -----------------------------
def assemble_input_row() -> pd.DataFrame:
    # raw categorical values -> mapped codes
    row = {
        "age": age,
        "gender": encoders["gender_map"].get(gender, encoders["gender_map"].get("Other", 2)),
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "smoker": encoders["smoker_map"].get(smoker, 0),
        "exercise_freq": encoders["exercise_map"].get(exercise, 1),
        "diet_quality": encoders["diet_map"].get(diet_quality, 1),
        "alcohol_consumption": encoders["alcohol_map"].get(alcohol, 2),
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        # NOTE: do NOT include target ("chronic_disease") in features
    }

    # Ensure the DataFrame exactly matches the feature order used in training
    df = pd.DataFrame([row])
    missing = [c for c in feature_order if c not in df.columns]
    for c in missing:
        # If a column existed during training but isn't collected in the UI,
        # set to a reasonable default (zeros). Adjust if you know better defaults.
        df[c] = 0

    # Drop any unexpected columns not present at training time, then order
    df = df[[c for c in feature_order if c in df.columns]]
    return df

# -----------------------------
# Predict
# -----------------------------
def predict_one(df_row: pd.DataFrame):
    # model is a Pipeline: [Imputer -> Scaler -> Classifier]
    prob = float(model.predict_proba(df_row)[:, 1][0])
    pred_label = int(prob >= best_threshold)
    return prob, pred_label

if submitted:
    X_row = assemble_input_row()
    prob, label = predict_one(X_row)

    st.subheader("Prediction")
    colA, colB = st.columns([1, 2], vertical_alignment="center")
    with colA:
        st.metric(label="Estimated Risk (probability)", value=f"{prob:.3f}")
        st.metric(label="Decision (thresholded)", value="High Risk" if label == 1 else "Low Risk")
    with colB:
        st.progress(min(max(prob, 0.0), 1.0), text="Higher bar → higher risk")

    # Friendly guidance (lightweight rules of thumb from your EDA)
    tips = []
    if bmi >= 30:
        tips.append("Your BMI is in the obese range — consider a physician-approved weight plan.")
    if sleep_hours < 6:
        tips.append("Increase sleep to 7–8 hours where possible.")
    if encoders["smoker_map"].get(smoker, 0) == 1:
        tips.append("Quitting smoking greatly reduces long-term disease risk.")
    if encoders["alcohol_map"].get(alcohol, 2) >= 2:
        tips.append("Reduce alcohol intake to Low/None levels.")
    if encoders["exercise_map"].get(exercise, 1) <= 1:
        tips.append("Add more weekly activity — aim for ≥ 150 mins moderate exercise.")

    if tips:
        st.subheader("Personalized, non-medical suggestions")
        for t in tips:
            st.write("•", t)

    # Show feature importances when available (tree models like RandomForest)
    try:
        clf = model.named_steps.get("clf", None)
        if clf is not None and hasattr(clf, "feature_importances_"):
            st.subheader("Top global feature importances")
            importances = clf.feature_importances_
            top = sorted(zip(feature_order, importances), key=lambda x: -x[1])[:10]
            imp_df = pd.DataFrame(top, columns=["Feature", "Importance"])
            st.dataframe(imp_df, use_container_width=True)
    except Exception:
        pass

# Footer
st.write("---")
st.caption("⚠️ This tool is for educational purposes only and is **not** a medical device.")
