from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib
import os

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

def main():

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Debug check (optional)
    print("Columns:", df.columns)

    # Prepare data
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    # Scale features
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, 'models/scaler.pkl'
    )

    # Train model
    grid_search = train_random_forest(x_train_scaled, y_train)

    # Evaluate model
    evaluate_classifier(
        grid_search.best_estimator_,
        x_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save model
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')

    print("✅ Model training complete & saved!")

if __name__ == "__main__":
    main()