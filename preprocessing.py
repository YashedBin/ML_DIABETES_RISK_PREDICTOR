"""Shared preprocessing for the Diabetes Risk Prediction project."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# PATHS and Directories for Global access 
DATA_PATH = "diabetes_prediction_dataset.csv"
MODEL_DIR = "models"
SPLIT_DIR = "data"

# pre-defined COLS and TARGET we have because the dataset is already view-ed
TARGET = "diabetes"
NUM_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CAT_COLS = ["gender", "smoking_history"]
BIN_COLS = ["hypertension", "heart_disease"]
FEATURES = CAT_COLS + NUM_COLS + BIN_COLS


# just a abstracted load_data function with directory as path
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


"""
clean_data function 
1. Drop the duplicates 
2. Others are taken
3. we drop Nulls 
4. we ofc add sanity checks with Age (10,  80) ( whotf is over age 80 and alive :D)
5. Sanity
"""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = df[df["gender"].isin(["Male", "Female"])]          # 'Other' is ~18 rows
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[(df["age"] > 0) & (df["bmi"].between(10, 80))]  # sanity bounds
    df["smoking_history"] = df["smoking_history"].replace("ever", "former")  # merge vague level
    return df.reset_index(drop=True)


"""
split_data function
1. Statify ( ofc ) the shit in 80:20 ratio and seed 42 ( f this ) 
use global variables
"""
def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified 80:20 split (raw features, before encoding/scaling)."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

"""
fit_preprocessors function 
1. Encodes with OneHot ( instead of flat ) 
2. we Standard Scale it into our limit
3. Save encoder and scaler into Pickle file
"""
def fit_preprocessors(X_train: pd.DataFrame, save: bool = True):
    """Fit encoder + scaler on TRAIN only (no leakage)."""
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    encoder.fit(X_train[CAT_COLS])
    scaler.fit(X_train[NUM_COLS])
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(encoder, f"{MODEL_DIR}/encoder.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    return encoder, scaler


"""
transform function
1. Transforms the raw features into model-ready shit
2. we also take data from user. 

"""
def transform(X: pd.DataFrame, encoder, scaler) -> pd.DataFrame:
    """Raw features -> model-ready matrix. Also used by app.py for user input."""
    cat = pd.DataFrame(
        encoder.transform(X[CAT_COLS]),
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=X.index,
    )
    num = pd.DataFrame(scaler.transform(X[NUM_COLS]), columns=NUM_COLS, index=X.index)
    return pd.concat([num, cat, X[BIN_COLS]], axis=1)


"""
save_split function
1. we make a Split directory on go (non-runtime)
2. saves into csv files with Target test and Train datasets
"""
def save_split(X_train, X_test, y_train, y_test):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(f"{SPLIT_DIR}/test.csv", index=False)

"""
load_split function
as all functions are run-go
1. read splitted Datasets
2. read also the Pickle files for encoding and standarization
3. returns a TRANSFORM-ed data
"""
def load_split():
    """Load saved split and return model-ready X/y using saved encoder+scaler."""
    tr = pd.read_csv(f"{SPLIT_DIR}/train.csv")
    te = pd.read_csv(f"{SPLIT_DIR}/test.csv")
    encoder = joblib.load(f"{MODEL_DIR}/encoder.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    return (
        transform(tr[FEATURES], encoder, scaler), transform(te[FEATURES], encoder, scaler),
        tr[TARGET], te[TARGET],
    )


"""
just a name body : D
"""
if __name__ == "__main__":
    df = clean_data(load_data())
    X_tr, X_te, y_tr, y_te = split_data(df)
    fit_preprocessors(X_tr)
    save_split(X_tr, X_te, y_tr, y_te)
    print(f"clean: {df.shape} | train: {X_tr.shape} | test: {X_te.shape}")
    print("positive rate train/test:", round(y_tr.mean(), 4), round(y_te.mean(), 4))
