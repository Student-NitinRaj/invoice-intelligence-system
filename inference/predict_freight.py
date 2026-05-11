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

    # The model was trained with a single feature "Dollars"
    # We map the incoming data to match what the model expects
    model_input = pd.DataFrame()
    if "invoice_dollars" in input_df.columns:
        model_input["Dollars"] = input_df["invoice_dollars"]
    elif "Dollars" in input_df.columns:
        model_input["Dollars"] = input_df["Dollars"]
    else:
        model_input["Dollars"] = input_df.iloc[:, 0]

    prediction = model.predict(model_input)

    input_df["Predicted_Freight"] = prediction

    return input_df