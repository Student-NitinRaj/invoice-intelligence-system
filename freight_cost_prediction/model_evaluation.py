from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_decision_tree(x_train, y_train, max_depth=5):
    model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, max_depth=6):
    model = RandomForestRegressor(
        max_depth=max_depth, random_state=42
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    """
    Evaluate regression model and return metrics.
    """

    preds = model.predict(x_test)
    import numpy as np

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds) * 100

    print(f"\n{model_name} Performance:")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2 : {r2:.2f}")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }