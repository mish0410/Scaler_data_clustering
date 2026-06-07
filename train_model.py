import os
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("scaler_hashed_for_students.csv")
MODEL_FILE = Path("scaler_cluster_model.pkl")
CURRENT_YEAR = 2026


def clean_text(text):
    if isinstance(text, str):
        return re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Place scaler_hashed_for_students.csv in the repository root."
        )
    df = pd.read_csv(csv_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Unnamed: 0"], axis=1, inplace=True, errors="ignore")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_text)

    # Ensure numeric columns are numeric
    for col in ["ctc", "orgyear", "ctc_updated_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill categorical missing values so numeric computation continues safely
    for col in ["company_hash", "job_position", "email_hash"]:
        if col in df.columns:
            df[col] = df[col].fillna("NA")

    if "orgyear" in df.columns:
        df["orgyear"] = df["orgyear"].clip(lower=1970, upper=CURRENT_YEAR)
        df["Years of Experience"] = CURRENT_YEAR - df["orgyear"]
        df["Years of Experience"] = df["Years of Experience"].clip(lower=0)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def build_pipeline(feature_columns):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=5, random_state=42, n_init=10)),
        ]
    )


def train_and_save_model():
    print("Loading dataset...")
    df = load_data(DATA_FILE)

    print("Preparing data...")
    df = prepare_data(df)

    features = [col for col in ["ctc", "orgyear", "ctc_updated_year", "Years of Experience"] if col in df.columns]
    if len(features) < 2:
        raise ValueError(
            "Not enough numeric features found to train clusters. Check the dataset columns."
        )

    X = df[features]
    print(f"Training KMeans on features: {features}")

    pipeline = build_pipeline(features)
    pipeline.fit(X)

    model_data = {
        "pipeline": pipeline,
        "feature_columns": features,
        "year_reference": CURRENT_YEAR,
    }

    with MODEL_FILE.open("wb") as f:
        pickle.dump(model_data, f)

    print(f"Saved clustering model to {MODEL_FILE}")


if __name__ == "__main__":
    train_and_save_model()
