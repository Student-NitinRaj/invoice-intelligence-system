import joblib
import pandas as pd

# MODEL_PATH = "models/predict_freight_model.pkl"
MODEL_PATH = r"C:\Users\nitin\Programming\Project\Untitled Folder 1\freight_cost_prediction\models\predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    """
    model = load_model()

    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df).round(2)

    return input_df


# ✅ FIXED main block
if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)