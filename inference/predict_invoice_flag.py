import os
import joblib
import pandas as pd

# Base directory (inference folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct absolute path
model_path = os.path.abspath(os.path.join(
    BASE_DIR,
    "..",
    "Invoice_flagging",
    "models",
    "predict_flag_invoice.pkl"
))

print("Invoice Model Path:", model_path)  # Debug


def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler():
    scaler_path = os.path.abspath(os.path.join(
        BASE_DIR,
        "..",
        "Invoice_flagging",
        "models",
        "scaler.pkl"
    ))
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    
    # Ensure columns match training
    input_features = input_df.copy()
    if "freight" in input_features.columns:
        input_features = input_features.rename(columns={"freight": "Freight"})
    
    expected_cols = ["invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"]
    input_features = input_features[expected_cols]

    # Scale the features
    scaled_data = scaler.transform(input_features)

    input_df["Predicted_Flag"] = model.predict(scaled_data).round()

    return input_df