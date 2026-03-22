import os
import joblib
import pandas as pd

# =========================================================
# Model Path Setup (Robust for Streamlit Cloud)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "freight_cost_prediction",
    "models",
    "predict_freight_model.pkl"
)

# =========================================================
# Load Model
# =========================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    
    return model


# =========================================================
# Prediction Function
# =========================================================
def predict_freight_cost(input_data):
    model = load_model()

    # Convert input to DataFrame safely
    if isinstance(input_data, dict):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()

    # ✅ CRITICAL FIX: match training feature names
    input_df = input_df.rename(columns={
        "Quantity": "quantity",
        "Dollars": "invoice_dollars"
    })

    # ✅ Ensure correct order
    input_df = input_df[["quantity", "invoice_dollars"]]

    # Prediction
    predictions = model.predict(input_df)

    # Output
    input_df["Predicted_Freight"] = predictions

    return input_df