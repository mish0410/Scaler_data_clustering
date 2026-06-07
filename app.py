import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_FILE = Path("scaler_cluster_model.pkl")


def load_model():
    if not MODEL_FILE.exists():
        st.error(
            f"Model file not found: {MODEL_FILE}. Run train_model.py first to create the pickle file."
        )
        st.stop()

    with MODEL_FILE.open("rb") as f:
        return pickle.load(f)


def prepare_input(feature_columns, year_reference, ctc, orgyear, ctc_updated_year):
    data = {
        "ctc": [ctc],
        "orgyear": [orgyear],
        "ctc_updated_year": [ctc_updated_year],
    }
    df = pd.DataFrame(data)

    if "Years of Experience" in feature_columns:
        df["Years of Experience"] = year_reference - df["orgyear"]
        df["Years of Experience"] = df["Years of Experience"].clip(lower=0)

    return df[feature_columns]


def main():
    st.title("Scaler Clustering Prediction")
    st.write(
        "Enter numeric profile values and get a predicted cluster number from the trained model."
    )

    model_data = load_model()
    feature_columns = model_data["feature_columns"]
    year_reference = model_data.get("year_reference", 2026)
    pipeline = model_data["pipeline"]

    ctc = st.number_input("CTC", min_value=0.0, value=500.0, step=100.0)
    orgyear = st.number_input("Organization Year", min_value=1970, max_value=2026, value=2020, step=1)
    if "ctc_updated_year" in feature_columns:
        ctc_updated_year = st.number_input(
            "CTC Updated Year", min_value=1970, max_value=2026, value=2023, step=1
        )
    else:
        ctc_updated_year = 0.0

    if st.button("Predict Cluster"):
        input_df = prepare_input(feature_columns, year_reference, ctc, orgyear, ctc_updated_year)
        cluster_label = int(pipeline.predict(input_df)[0])
        st.success(f"Predicted cluster number: {cluster_label}")
        st.write(
            "Cluster labels are zero-based from the trained KMeans model. If you want 1-based labels, add 1 to this number."
        )
        st.write("Input features used for prediction:")
        st.dataframe(input_df)


if __name__ == "__main__":
    main()
