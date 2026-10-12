"""Streamlit frontend for the Diabetes Risk Prediction project.

Run:
    streamlit run app.py
"""
import json

import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURES, MODEL_DIR, transform

MODEL_FILES = {
    "Logistic Regression": "logistic",
    "K-Nearest Neighbours": "knn",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
}

GENDER_OPTIONS = ["Female", "Male"]
SMOKING_OPTIONS = ["No Info", "never", "current", "former", "not current"]


@st.cache_resource
def load_encoder_scaler():
    encoder = joblib.load(f"{MODEL_DIR}/encoder.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return encoder, scaler


@st.cache_resource
def load_model(key: str):
    return joblib.load(f"{MODEL_DIR}/{key}.pkl")


@st.cache_data
def load_metrics():
    with open(f"{MODEL_DIR}/metrics.json") as f:
        return pd.DataFrame(json.load(f))


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


def risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Low risk"
    if prob < 0.66:
        return "Moderate risk"
    return "High risk"


def patient_input_form(prefix: str):
    st.subheader("Patient details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=100.0, value=45.0, step=1.0, key=f"{prefix}_age")
        gender = st.selectbox("Gender", GENDER_OPTIONS, key=f"{prefix}_gender")
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1, key=f"{prefix}_bmi")
        smoking = st.selectbox("Smoking history", SMOKING_OPTIONS, key=f"{prefix}_smoking")
    with col2:
        hba1c = st.number_input("HbA1c level (%)", min_value=3.0, max_value=9.0, value=5.5, step=0.1, key=f"{prefix}_hba1c")
        glucose = st.number_input(
            "Blood glucose level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0, key=f"{prefix}_glucose"
        )
        hypertension = st.checkbox("Hypertension", key=f"{prefix}_hypertension")
        heart_disease = st.checkbox("Heart disease", key=f"{prefix}_heart_disease")

    return age, gender, bmi, hba1c, glucose, hypertension, heart_disease, smoking


def prediction_tab(encoder, scaler):
    inputs = patient_input_form("predict")

    model_name = st.selectbox("Model", list(MODEL_FILES.keys()), key="predict_model")

    if st.button("Predict", type="primary", key="predict_button"):
        model = load_model(MODEL_FILES[model_name])
        row = build_input_row(*inputs)
        X = transform(row, encoder, scaler)
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(model.predict(X)[0])

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Prediction", "Diabetic" if pred == 1 else "Non-diabetic")
        c2.metric("Risk probability", f"{prob * 100:.1f}%")
        st.progress(min(prob, 1.0))
        st.caption(f"{model_name} - {risk_band(prob)}")

    st.caption("Course project only - not a medical diagnostic tool.")


def all_models_tab(encoder, scaler):
    inputs = patient_input_form("all")

    if st.button("Simulate all models", type="primary", key="all_button"):
        row = build_input_row(*inputs)
        X = transform(row, encoder, scaler)

        results = []
        for model_name, key in MODEL_FILES.items():
            model = load_model(key)
            prob = float(model.predict_proba(X)[:, 1][0])
            pred = int(model.predict(X)[0])
            results.append(
                {
                    "Model": model_name,
                    "Prediction": "Diabetic" if pred == 1 else "Non-diabetic",
                    "Probability": prob,
                    "Risk band": risk_band(prob),
                }
            )

        table = pd.DataFrame(results)
        diabetic_votes = int((table["Probability"] >= 0.5).sum())

        st.divider()
        st.dataframe(
            table.assign(Probability=table["Probability"].map("{:.1%}".format)),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(table.set_index("Model")["Probability"])
        st.caption(f"{diabetic_votes}/{len(table)} models predict diabetic.")

    st.caption("Course project only - not a medical diagnostic tool.")


def comparison_tab(metrics):
    st.subheader("Model comparison")
    if metrics is None or metrics.empty:
        st.info("No metrics found. Run `python train.py` first.")
        return
    st.dataframe(metrics.set_index("Model"), use_container_width=True)

    chart_cols = [c for c in ["F1 Score", "ROC-AUC", "Recall", "Accuracy"] if c in metrics.columns]
    st.bar_chart(metrics.set_index("Model")[chart_cols])


def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
    st.title("Diabetes Risk Predictor")

    try:
        encoder, scaler = load_encoder_scaler()
        metrics = load_metrics()
        for key in MODEL_FILES.values():
            load_model(key)
    except FileNotFoundError:
        st.error("Model artifacts not found. Run `python train.py` first.")
        st.stop()

    pred_tab, all_tab, comp_tab = st.tabs(["Prediction", "All Models", "Model comparison"])
    with pred_tab:
        prediction_tab(encoder, scaler)
    with all_tab:
        all_models_tab(encoder, scaler)
    with comp_tab:
        comparison_tab(metrics)


if __name__ == "__main__":
    main()
