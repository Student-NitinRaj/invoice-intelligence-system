import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "freight_cost_prediction",
        "models",
        "predict_freight_model.pkl"
    )
)

print("FINAL PATH:", model_path)  # DEBUG


def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df).round(2)

    return input_df