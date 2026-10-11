"""Train and export the diabetes prediction models for the Streamlit frontend.

Run once (or whenever you want to refresh the artifacts):

    python train.py

Outputs land in models/:
    logistic.pkl, knn.pkl, decision_tree.pkl, random_forest.pkl, metrics.json
"""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from preprocessing import MODEL_DIR, load_split


def build_models() -> dict:
    return {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_split=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate(name, model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }


def main():
    X_train, X_test, y_train, y_test = load_split()
    print(f"train: {X_train.shape} | test: {X_test.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []
    for key, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"{MODEL_DIR}/{key}.pkl")
        scores = evaluate(key, model, X_test, y_test)
        results.append(scores)
        print(f"saved {key}: F1={scores['F1 Score']:.4f} ROC-AUC={scores['ROC-AUC']:.4f}")

    metrics = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics.round(4).to_dict(orient="records"), f, indent=2)

    print("\n", metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
