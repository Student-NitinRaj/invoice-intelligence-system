from pathlib import Path
import joblib
import pandas as pd

# Dynamic correct path
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../Invoice_flagging/models/predict_flag_invoice.pkl"
    )
)


def load_model(model_path: Path = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice_flag(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df).round()

    return input_df


if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [10, 50],
        "invoice_dollars": [1000, 5000],
        "Freight": [50, 200],
        "total_item_quantity": [12, 55],
        "total_item_dollars": [950, 4800]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)