"""Streamlit frontend for the Diabetes Risk Prediction course project.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURES, MODEL_DIR, transform

MODEL_DIR = Path(MODEL_DIR)

MODEL_FILES = {
    "Logistic Regression": "logistic",
    "K-Nearest Neighbours": "knn",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}

GENDER_OPTIONS = ["Female", "Male"]
SMOKING_OPTIONS = ["No Info", "never", "current", "former", "not current"]


@st.cache_resource(show_spinner=False, max_entries=1)
def load_encoder_scaler():
    """Keep shared, read-only preprocessing objects in server memory."""
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return encoder, scaler


@st.cache_resource(show_spinner=False, max_entries=8)
def load_model(key: str):
    """Load an original or tuned model on first use, then reuse it."""
    return joblib.load(MODEL_DIR / f"{key}.pkl")


@st.cache_data(show_spinner=False, ttl=3600)
def load_metrics():
    """Performance metrics are optional; predictions do not depend on this file."""
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # Supports both {model_name: {metric: value}} and column-oriented JSON.
        if all(isinstance(value, dict) for value in data.values()):
            frame = pd.DataFrame.from_dict(data, orient="index")
            frame.index.name = "Model"
            return frame.reset_index()
        return pd.DataFrame([data])
    return pd.DataFrame()


def get_model_key(model_name: str, tuned: bool) -> str:
    base = MODEL_FILES[model_name]
    return f"{base}_tuned" if tuned else base


def build_input_row(age, gender, bmi, hba1c, glucose, hypertension, heart_disease, smoking):
    row = {
        "gender": gender,
        "smoking_history": smoking,
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
    }
    return pd.DataFrame([row], columns=FEATURES)


def predict_one(model_key: str, patient_values, encoder, scaler):
    """Run inference. Don't globally cache patient health data across users."""
    model = load_model(model_key)
    row = build_input_row(*patient_values)
    processed = transform(row, encoder, scaler)
    prediction = int(model.predict(processed)[0])
    classes = list(model.classes_)
    positive_index = classes.index(1)
    probability = float(model.predict_proba(processed)[0, positive_index])
    return prediction, probability


def patient_input_form(prefix: str):
    st.subheader("Patient details")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=1.0, max_value=100.0, value=45.0, step=1.0, key=f"{prefix}_age")
        gender = st.selectbox("Gender", GENDER_OPTIONS, key=f"{prefix}_gender")
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1, key=f"{prefix}_bmi")
        smoking = st.selectbox("Smoking history", SMOKING_OPTIONS, key=f"{prefix}_smoking")
    with right:
        hba1c = st.number_input("HbA1c level (%)", min_value=3.0, max_value=9.0, value=5.5, step=0.1, key=f"{prefix}_hba1c")
        glucose = st.number_input("Blood glucose (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0, key=f"{prefix}_glucose")
        hypertension = st.checkbox("Hypertension", key=f"{prefix}_hypertension")
        heart_disease = st.checkbox("Heart disease", key=f"{prefix}_heart_disease")
    return (age, gender, bmi, hba1c, glucose, hypertension, heart_disease, smoking)


def show_prediction(prediction: int, probability: float, model_name: str, tuned: bool):
    st.divider()
    left, right = st.columns(2)
    left.metric("Model prediction", "Diabetic" if prediction else "Non-diabetic")
    right.metric("Model-estimated positive probability", f"{probability:.2%}")
    st.progress(max(0.0, min(probability, 1.0)))
    st.caption(f"{model_name} · {'Tuned' if tuned else 'Original'} · Not a clinically validated risk estimate")


def prediction_tab(encoder, scaler):
    st.subheader("Diabetes prediction")
    with st.form("prediction_form"):
        inputs = patient_input_form("predict")
        model_name = st.selectbox("Select model", list(MODEL_FILES), key="predict_model")
        tuned = st.toggle("Use hyperparameter-tuned model", value=False, key="predict_tuned")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        model_key = get_model_key(model_name, tuned)
        try:
            with st.spinner("Running prediction..."):
                pred, prob = predict_one(model_key, inputs, encoder, scaler)
            # Persist only within the visitor's session, rather than the global data cache.
            st.session_state["last_prediction"] = (pred, prob, model_name, tuned)
        except FileNotFoundError:
            st.error(f"Missing model: {model_key}.pkl. Export it before deploying.")
        except (ValueError, KeyError) as error:
            st.error(f"Could not process this input: {error}")

    if "last_prediction" in st.session_state:
        show_prediction(*st.session_state["last_prediction"])
    st.caption("Educational course project only. Not for diagnosis or medical decisions.")


def all_models_tab(encoder, scaler):
    st.subheader("Compare all four models")
    with st.form("all_models_form"):
        inputs = patient_input_form("all")
        tuned = st.toggle("Use tuned models", value=False, key="all_tuned")
        submitted = st.form_submit_button("Compare models", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Comparing models..."):
                # Transform once, then reuse for all four models.
                row = build_input_row(*inputs)
                processed = transform(row, encoder, scaler)
                results = []
                for model_name in MODEL_FILES:
                    key = get_model_key(model_name, tuned)
                    model = load_model(key)
                    prediction = int(model.predict(processed)[0])
                    positive_index = list(model.classes_).index(1)
                    probability = float(model.predict_proba(processed)[0, positive_index])
                    results.append({
                        "Model": model_name,
                        "Version": "Tuned" if tuned else "Original",
                        "Prediction": "Diabetic" if prediction else "Non-diabetic",
                        "Probability": probability,
                    })
            st.session_state["last_comparison"] = pd.DataFrame(results)
        except FileNotFoundError as error:
            st.error(f"Missing model artifact: {error}")
        except (ValueError, KeyError) as error:
            st.error(f"Could not process this input: {error}")

    if "last_comparison" in st.session_state:
        table = st.session_state["last_comparison"]
        st.dataframe(
            table.style.format({"Probability": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(table.set_index("Model")["Probability"])
        diabetic_votes = int((table["Prediction"] == "Diabetic").sum())
        st.info(f"{diabetic_votes}/{len(table)} models predict diabetic.")
    st.caption("Educational course project only. Not for diagnosis or medical decisions.")


def comparison_tab():
    st.subheader("Model evaluation metrics")
    try:
        metrics = load_metrics()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.warning(f"Could not load metrics.json: {error}")
        return
    if metrics.empty:
        st.info("No saved evaluation metrics found. Predictions still work without metrics.json.")
        return
    if "Model" not in metrics.columns:
        st.warning("metrics.json must include model names for comparison.")
        st.dataframe(metrics, use_container_width=True)
        return
    st.dataframe(metrics.set_index("Model"), use_container_width=True)
    chart_cols = [c for c in ["F1 Score", "ROC-AUC", "Recall", "Accuracy"] if c in metrics.columns]
    if chart_cols:
        st.bar_chart(metrics.set_index("Model")[chart_cols])


def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")
    st.title("Diabetes Risk Predictor")
    st.write("Compare original and hyperparameter-tuned classifiers using patient measurements.")

    pred_tab, all_tab, metrics_tab = st.tabs(["Prediction", "All Models", "Model Comparison"])
    with pred_tab:
        try:
            encoder, scaler = load_encoder_scaler()
        except (FileNotFoundError, OSError) as error:
            st.error(f"Missing preprocessing artifacts: {error}. Run training and export artifacts first.")
            st.stop()
        prediction_tab(encoder, scaler)
    with all_tab:
        all_models_tab(encoder, scaler)
    with metrics_tab:
        comparison_tab()


if __name__ == "__main__":
    main()
