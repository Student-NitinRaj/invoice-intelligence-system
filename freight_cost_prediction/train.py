import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data 
from model_evaluation import (
    train_linear_regression,
    train_decision_tree, 
    train_random_forest, 
    evaluate_model
)

def main():
    db_path="../Data/inventory.db"
    model_dir = Path("models") 
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(db_path)

    # Prepare data
    x, y = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Train models
    lr_model = train_linear_regression(x_train, y_train) 
    dt_model = train_decision_tree(x_train, y_train)   
    rf_model = train_random_forest(x_train, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model(lr_model, x_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, x_test, y_test, "Decision Tree Regression")) 
    results.append(evaluate_model(rf_model, x_test, y_test, "Random Forest Regression"))

    # Select best model
    best_model_info = min(results, key=lambda x: x["mae"]) 
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}") 
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()