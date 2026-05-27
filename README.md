# ANN Classification

This project trains and serves an Artificial Neural Network (ANN) for bank customer churn prediction. It includes:

- model training and preprocessing experiments in Jupyter notebooks
- saved preprocessing artifacts for inference
- a Streamlit app for interactive churn prediction

## Project Contents

- `data-set/Churn_Modelling.csv` - source dataset
- `experiments.ipynb` - preprocessing, feature engineering, ANN training
- `prediction.ipynb` - notebook-based inference flow
- `app.py` - Streamlit prediction app
- `models/model.h5` - trained ANN model
- `pickles/labelencoder.pkl` - saved gender label encoder
- `pickles/onehotencoder.pkl` - saved geography one-hot encoder
- `pickles/scaler.pkl` - saved feature scaler

## Problem Statement

The model predicts whether a bank customer is likely to churn based on profile and account features such as:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

Target column:

- `Exited`

## Preprocessing Pipeline

The training notebook applies the following steps:

1. Load `Churn_Modelling.csv`
2. Drop non-predictive columns:
   - `RowNumber`
   - `CustomerId`
   - `Surname`
3. Encode categorical features:
   - `Gender` with `LabelEncoder`
   - `Geography` with `OneHotEncoder`
4. Split the data into training and test sets
5. Scale numerical and encoded features using `StandardScaler`
6. Save the fitted encoders and scaler as pickle artifacts

## Model Architecture

The ANN in `experiments.ipynb` uses TensorFlow/Keras with this structure:

- Input layer with the processed feature dimension
- Dense layer: `64` units, `relu`
- Dense layer: `32` units, `relu`
- Output layer: `1` unit, `sigmoid`

Training setup:

- Optimizer: `Adam(learning_rate=0.001)`
- Loss: `BinaryCrossentropy`
- Metric: `accuracy`
- Callback: `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt` and include TensorFlow, scikit-learn, pandas, Streamlit, Jupyter, and TensorBoard.

## Run the Experiments

Open the notebooks to review preprocessing, training, and prediction steps:

```bash
jupyter notebook
```

Primary notebooks:

- `experiments.ipynb`
- `prediction.ipynb`

## Run the Streamlit App

Launch the prediction UI:

```bash
streamlit run app.py
```

The app:

- loads the trained ANN model
- loads the saved encoder and scaler artifacts
- collects user input from the UI
- applies the same preprocessing used in training
- outputs churn probability and a binary churn indication

## Prediction Workflow

Inference follows the same feature pipeline used during training:

1. Capture raw customer inputs
2. Label-encode `Gender`
3. One-hot encode `Geography`
4. Merge encoded features into the full input row
5. Scale the final feature vector using the saved scaler
6. Run `model.predict(...)`
7. Display churn probability

## Notes

- The Streamlit app depends on the existing files in `models/` and `pickles/`.
- If you retrain the model, regenerate and save the updated encoder, scaler, and model artifacts before running the app.
- The project is currently set up for binary classification on customer churn.
