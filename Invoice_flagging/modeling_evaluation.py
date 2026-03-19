from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score
)


# ------------------ TRAIN MODEL ------------------ #
def train_random_forest(x_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score, average='binary')

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)

    print("\n✅ Best Parameters:", grid_search.best_params_)

    return grid_search


# ------------------ EVALUATE MODEL ------------------ #
def evaluate_classifier(model, x_test, y_test, model_name):

    preds = model.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n📊 {model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)