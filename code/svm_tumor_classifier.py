"""
Breast Cancer Tumor Classification using SVM

This script:
1. Loads the Breast Cancer dataset from sklearn
2. Splits data into train/test sets (80/20)
3. Applies feature scaling with StandardScaler
4. Trains SVM models with linear and RBF kernels
5. Performs hyperparameter tuning using GridSearchCV
6. Evaluates the best model with standard classification metrics
7. Plots the confusion matrix
8. Saves the trained model with joblib
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
MODEL_OUTPUT_PATH = "best_svm_breast_cancer_model.joblib"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load breast cancer dataset and return features and target as pandas objects."""
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}\n")
    return X, y


def split_and_scale(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray, StandardScaler]:
    """Split data and scale features using StandardScaler."""
    print("Performing train-test split (80-20) and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}\n")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_basic_svm_models(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
) -> None:
    """Train and evaluate SVM models with linear and RBF kernels."""
    print("Training basic SVM models with different kernels...")
    kernels = ["linear", "rbf"]

    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Kernel: {kernel:>6} | Accuracy: {accuracy:.4f}")

    print()


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Perform GridSearchCV to find best SVM hyperparameters."""
    print("Running GridSearchCV for hyperparameter tuning...")

    # Pipeline ensures scaling is done inside each CV fold.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "svm__kernel": ["linear", "rbf"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}\n")

    return grid_search


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Evaluate model and print classification metrics."""
    print("Evaluating best model on test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    return cm


def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot confusion matrix using matplotlib."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    class_names = ["malignant", "benign"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def save_model(
    model: Pipeline,
    feature_names: list[str],
    best_params: dict,
    path: str = MODEL_OUTPUT_PATH,
) -> None:
    """Save trained model artifact (model + metadata) using joblib."""
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "best_params": best_params,
    }
    joblib.dump(artifact, path)
    print(f"Model artifact saved to: {path}")
    print("Saved keys: model, feature_names, best_params\n")


def predict_custom_input(
    model: Pipeline, feature_values: list[float], feature_names: list[str]
) -> str:
    """
    Predict tumor type for a custom input.

    Args:
        model: Trained model pipeline.
        feature_values: List of 30 feature values in Breast Cancer dataset order.
        feature_names: Feature names used while training.

    Returns:
        Predicted label as string: 'malignant' or 'benign'.
    """
    if len(feature_values) != 30:
        raise ValueError("Expected 30 feature values for prediction.")

    custom_df = pd.DataFrame([feature_values], columns=feature_names)
    prediction = model.predict(custom_df)[0]
    return "benign" if prediction == 1 else "malignant"


def main() -> None:
    """Run the complete workflow."""
    X, y = load_data()

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, _ = split_and_scale(X, y)

    # Baseline models on manually scaled data.
    train_basic_svm_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Use raw train/test DataFrames for pipeline-based tuning and evaluation.
    grid_search = tune_hyperparameters(X_train, y_train)
    best_model = grid_search.best_estimator_

    cm = evaluate_model(best_model, X_test, y_test)
    plot_confusion_matrix(cm)
    save_model(
        best_model,
        feature_names=X.columns.tolist(),
        best_params=grid_search.best_params_,
    )

    # Optional custom prediction example.
    sample_input = X.iloc[0].tolist()
    predicted_label = predict_custom_input(best_model, sample_input, X.columns.tolist())
    print(f"Optional custom prediction for first sample: {predicted_label}")


if __name__ == "__main__":
    main()
