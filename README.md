# Scaler Data Clustering

This project demonstrates how to build an unsupervised clustering solution for salary and experience-related profile data. It uses a scikit-learn pipeline with StandardScaler and KMeans to group records into clusters based on numeric features such as CTC, organization year, and a derived experience feature.

## What this project does

- Loads and cleans the provided dataset from the project root
- Creates a derived feature for years of experience
- Trains a KMeans clustering model and saves it as a pickle file
- Provides a Streamlit app for predicting the cluster of new input values
- Includes an exploratory notebook for analysis and experimentation

## Project files

- [train_model.py](train_model.py): trains the clustering model and saves the serialized model
- [app.py](app.py): interactive Streamlit interface for cluster prediction
- [Scaler_clustering.ipynb](Scaler_clustering.ipynb): notebook for exploration and model development
- [requirements.txt](requirements.txt): Python dependencies for the project

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

1. Place the dataset file named `scaler_hashed_for_students.csv` in the project root.
2. Train the model:

```bash
python train_model.py
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The trained model uses 5 clusters and zero-based cluster labels.
- The app expects the generated model file to be present before prediction starts.
- The notebook can be used to inspect the data, experiment with features, and refine the clustering approach.
