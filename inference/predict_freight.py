import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "freight_cost_prediction",
    "models",
    "predict_freight_model.pkl"
)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    
    return model


def predict_freight_cost(input_data):
    model = load_model()

    # Convert to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()

    # ✅ HANDLE BOTH CASES (important fix)
    input_df = input_df.rename(columns={
        "quantity": "Quantity",
        "invoice_dollars": "Dollars"
    })

    # ✅ FINAL REQUIRED FORMAT
    input_df = input_df[["Quantity", "Dollars"]]

    prediction = model.predict(input_df)

    input_df["Predicted_Freight"] = prediction

    return input_df