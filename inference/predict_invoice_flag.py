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


def predict_invoice_flag(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df).round()

    return input_df