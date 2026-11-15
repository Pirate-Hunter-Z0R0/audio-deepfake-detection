"""
train_baseline.py
------------------
Baseline Machine Learning Classifier Training Pipeline for Audio Deepfake Detection.

Purpose:
    Trains and compares three classical machine learning algorithms—Logistic Regression, 
    Support Vector Machine (SVM), and Random Forest—on extracted spectral features 
    (MFCC, LFCC, CQCC). Evaluates classification metrics (Accuracy, Precision, Recall, F1)
    and persists the best-performing model to disk.

Viva / Academic Explanation Guide:
    1. Logistic Regression:
       - Conceptual Mechanism: Fits a linear decision boundary using the sigmoid logistic function:
         P(y=1|x) = 1 / (1 + exp(-(w^T * x + b))).
       - Viva Explanation: Provides a baseline linear benchmark. It determines whether real and 
         synthetic speech spectral features are linearly separable without complex kernel transformations.

    2. Support Vector Machine (SVM with RBF Kernel):
       - Conceptual Mechanism: Maps 78-dimensional spectral feature vectors into an implicit 
         higher-dimensional space using Radial Basis Function (RBF) kernel K(x, x') = exp(-gamma * ||x - x'||^2)
         and finds the optimal hyperplane that maximizes the margin between Real and Fake classes.
       - Viva Explanation: SVM excels in high-dimensional feature spaces by establishing a robust,
         maximum-margin boundary that resists overfitting on small dataset splits.

    3. Random Forest Classifier:
       - Conceptual Mechanism: An ensemble method that constructs a forest of 100 decorrelated 
         decision trees using bootstrap aggregation (bagging) and random feature subsets.
       - Viva Explanation: Captures non-linear feature interactions between MFCC, LFCC, and CQCC
         statistics without requiring strict feature scaling assumptions, offering strong resistance
         to noise and outliers.

Imports:
    pandas, numpy, scikit-learn, joblib, pathlib, os, argparse
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_feature_dataset(csv_path: str = "data/features/features.csv") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads pre-extracted spectral feature dataset from CSV file using pandas.

    Parameters:
        csv_path (str): Path to features.csv.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: (Feature matrix X, Target labels y, Feature column names).
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset file '{path}' does not exist. Please run src/feature_extraction.py first.")

    df = pd.read_csv(path)
    
    if "label" not in df.columns:
        raise ValueError("CSV file missing 'label' column.")

    X = df.drop(columns=["label"]).values
    y = df["label"].values.astype(int)
    feature_names = list(df.drop(columns=["label"]).columns)

    return X, y, feature_names


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Trains a Logistic Regression baseline classifier.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0, kernel: str = "rbf") -> SVC:
    """
    Trains a Support Vector Machine (SVM) classifier with probability estimation.
    """
    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100) -> RandomForestClassifier:
    """
    Trains a Random Forest ensemble classifier.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on both training and test sets.

    Returns dictionary containing:
        - Model Name
        - Train Accuracy
        - Test Accuracy
        - Precision
        - Recall
        - F1-Score
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    return {
        "Model": model_name,
        "Train Accuracy": round(float(train_acc), 4),
        "Test Accuracy": round(float(test_acc), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "model_object": model
    }


def save_model_artifact(model: Any, filepath: str) -> None:
    """
    Serializes and saves a model artifact using joblib.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def run_baseline_training_pipeline(
    csv_path: str = "data/features/features.csv",
    models_dir: str = "models",
    results_dir: str = "results"
) -> pd.DataFrame:
    """
    Executes full baseline model training pipeline:
    1. Loads dataset
    2. Performs 80/20 stratified split
    3. Normalizes feature matrix with StandardScaler
    4. Trains Logistic Regression, SVM, and Random Forest
    5. Compares metrics and saves baseline_comparison.csv
    6. Persists best model to models/baseline_model.pkl
    """
    print("="*65)
    print("       BASELINE MACHINE LEARNING TRAINING PIPELINE       ")
    print("="*65)

    # 1. Load Dataset
    X, y, feature_names = load_feature_dataset(csv_path)
    print(f"Loaded Feature Matrix Shape : {X.shape} ({X.shape[1]} features)")
    print(f"Class Balance               : {np.sum(y == 0)} Real (0), {np.sum(y == 1)} Fake (1)")

    # 2. Stratified Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training Samples            : {X_train.shape[0]}")
    print(f"Testing Samples             : {X_test.shape[0]}")

    # Save test set split for standalone evaluation scripts (src/evaluate.py)
    test_split_path = Path("data/features/test_split.csv")
    test_split_path.parent.mkdir(parents=True, exist_ok=True)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["label"] = y_test
    test_df.to_csv(test_split_path, index=False)
    print(f"Saved Test Set Split to     : {test_split_path}")

    # 3. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature scaler artifact for Flask web app inference
    save_model_artifact(scaler, os.path.join(models_dir, "scaler.pkl"))
    print(f"Saved Feature Scaler to     : {os.path.join(models_dir, 'scaler.pkl')}")
    print("="*65)

    # 4. Train Classifiers
    print("\n[START] Training Classical ML Classifiers...")

    print("  -> Training Logistic Regression...")
    logreg_model = train_logistic_regression(X_train_scaled, y_train)

    print("  -> Training Support Vector Machine (SVM)...")
    svm_model = train_svm(X_train_scaled, y_train)

    print("  -> Training Random Forest...")
    rf_model = train_random_forest(X_train_scaled, y_train)

    # 5. Evaluate Classifiers
    eval_results = []
    models_to_evaluate = [
        ("Logistic Regression", logreg_model),
        ("Support Vector Machine (SVM)", svm_model),
        ("Random Forest", rf_model)
    ]

    for name, model_obj in models_to_evaluate:
        res = evaluate_model(model_obj, X_train_scaled, y_train, X_test_scaled, y_test, name)
        eval_results.append(res)

    # 6. Format Comparison Table
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "model_object"} for r in eval_results
    ])

    print("\n" + "="*65)
    print("               MODEL COMPARISON SUMMARY               ")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)

    # 7. Identify Best Model (Highest Test Accuracy / F1-Score)
    best_eval = max(eval_results, key=lambda x: (x["Test Accuracy"], x["F1-Score"]))
    best_model_name = best_eval["Model"]
    best_model_obj = best_eval["model_object"]

    print(f"\n[BEST MODEL] '{best_model_name}' achieved highest Test Accuracy ({best_eval['Test Accuracy']*100:.2f}%)!")

    # 8. Save Best Model and Comparison Results
    best_model_path = os.path.join(models_dir, "baseline_model.pkl")
    save_model_artifact(best_model_obj, best_model_path)
    print(f"Persisted Best Model to      : {best_model_path}")

    csv_results_path = os.path.join(results_dir, "baseline_comparison.csv")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(csv_results_path, index=False)
    print(f"Exported Comparison CSV to   : {csv_results_path}")
    print("="*65)

    return results_df


def main() -> None:
    """
    Main execution pipeline.
    """
    parser = argparse.ArgumentParser(description="Baseline Machine Learning Model Training")
    parser.add_argument("--csv-path", type=str, default="data/features/features.csv", help="Path to features.csv")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to export baseline_comparison.csv")

    args = parser.parse_args()

    run_baseline_training_pipeline(
        csv_path=args.csv_path,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
