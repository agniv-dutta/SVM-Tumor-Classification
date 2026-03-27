# Breast Cancer Tumor Classification (SVM)

A complete machine learning project to classify tumors as malignant or benign using Support Vector Machine (SVM) on the sklearn Breast Cancer dataset.

## Features
- Uses `pandas` and `scikit-learn`
- Loads dataset from `sklearn.datasets.load_breast_cancer`
- Train-test split: 80-20
- Feature scaling with `StandardScaler`
- Baseline SVM training with `linear` and `rbf` kernels
- Hyperparameter tuning using `GridSearchCV` over `C`, `gamma`, and `kernel`
- Evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Classification report
- Confusion matrix plot via `matplotlib`
- Saves trained model artifact with metadata using `joblib`
- Optional custom input prediction function

## Project Files
- `svm_tumor_classifier.py`: Main runnable script
- `requirements.txt`: Python dependencies
- `best_svm_breast_cancer_model.joblib`: Saved model artifact (generated after run)

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
python svm_tumor_classifier.py
```

## Expected Console Outputs
The script prints:
- Dataset shape and class distribution
- Baseline kernel accuracies (`linear`, `rbf`)
- Best `GridSearchCV` parameters and CV score
- Test metrics: Accuracy, Precision, Recall, F1-score
- Full classification report
- Confusion matrix values
- Optional custom prediction result

## Figure Output
- A confusion matrix plot window appears during execution.
- If your environment saves plotted figures automatically, you may also see files like `Outputs/Figure_1.png`.

## Saved Model Artifact Structure
The joblib file stores a dictionary with:
- `model`: Best trained sklearn pipeline (`StandardScaler` + `SVC`)
- `feature_names`: List of training feature names
- `best_params`: Best parameters found by `GridSearchCV`

Example load snippet:
```python
import joblib

artifact = joblib.load("best_svm_breast_cancer_model.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]
best_params = artifact["best_params"]
```
