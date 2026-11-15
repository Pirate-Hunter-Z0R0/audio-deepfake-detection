"""
evaluate.py
-----------
Evaluation Module for Audio Deepfake Detection System.

Purpose:
    Loads trained model artifacts (supports both baseline ML models like SVM/baseline_model.pkl 
    and deep learning models like ResNeXt/deep_model.h5), evaluates classification performance 
    on the held-out test split, computes core metrics (Accuracy, Precision, Recall, F1-Score, EER), 
    renders a confusion matrix, prints a summary report, and saves results to results/metrics.txt.

Viva / Academic Explanation Guide:
    1. Equal Error Rate (EER):
       - What it means in plain English: EER is the operating point where the detector's 
         False Acceptance Rate (FAR - falsely trusting a deepfake voice as genuine) EQUALS its 
         False Rejection Rate (FRR - falsely flagging a genuine human voice as deepfake).
       - Significance: Lower EER values indicate a more reliable biometric voice verification system. 
         An EER of 0.0% represents error-free classification regardless of decision threshold.

    2. Accuracy vs Precision vs Recall vs F1-Score:
       - Accuracy: Overall percentage of correctly classified audio clips.
       - Precision: Out of all clips flagged as "Fake", what fraction were truly deepfakes?
       - Recall: Out of all actual deepfakes in the dataset, what fraction did we catch?
       - F1-Score: Harmonic mean of Precision and Recall, balancing false alarms against missed detections.

Imports:
    os, sys, argparse, pathlib, typing, json, numpy, pandas, matplotlib.pyplot, scikit-learn, joblib
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


def calculate_eer(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the Equal Error Rate (EER) and its corresponding decision threshold.

    ---------------------------------------------------------------------------
    Equal Error Rate (EER) Explanation (in Plain English for Viva):
    Equal Error Rate (EER) is the threshold point where the False Acceptance Rate 
    (FAR / FPR - falsely passing an AI deepfake voice as genuine human speech) 
    EQUALS the False Rejection Rate (FRR / FNR - falsely rejecting a genuine human 
    voice as an AI deepfake).
    In plain English: EER measures the operating point where the detector makes equal 
    errors of both types. A lower EER indicates a better, more balanced detector.
    ---------------------------------------------------------------------------

    Parameters:
        y_true (np.ndarray): Ground truth binary labels (0=real, 1=fake).
        y_probs (np.ndarray): Prediction probabilities for synthetic class (class 1).

    Returns:
        Tuple[float, float]: (EER value as a percentage [0..100], threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    fnr = 1.0 - tpr  # False Rejection Rate = 1 - True Positive Rate

    # Find the index where FPR (FAR) and FNR (FRR) are closest to each other
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_val = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return float(eer_val * 100.0), float(eer_threshold)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, F1-Score, AUC, and EER metrics.

    Parameters:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Hard predicted class labels (0 or 1).
        y_probs (Optional[np.ndarray]): Probability scores for the fake class.

    Returns:
        Dict[str, float]: Dictionary containing computed evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "Accuracy": round(float(acc * 100.0), 2),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4)
    }

    if y_probs is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
            auc_val = auc(fpr, tpr)
            eer_pct, eer_thresh = calculate_eer(y_true, y_probs)
            metrics["AUC"] = round(float(auc_val), 4)
            metrics["EER_Percentage"] = round(float(eer_pct), 2)
            metrics["EER_Threshold"] = round(float(eer_thresh), 4)
        except Exception as e:
            print(f"[WARNING] Could not compute EER/AUC: {e}")

    return metrics


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
    model_name: str = "Model"
) -> None:
    """
    Generates and saves a confusion matrix visualization plot.

    Parameters:
        y_true (np.ndarray): True ground truth labels.
        y_pred (np.ndarray): Predicted class labels.
        save_path (str): Target output PNG file path.
        model_name (str): Model name tag for the plot title.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Real (0)", "Fake (1)"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")

    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Confusion Matrix plot to: {save_path}")


def load_test_dataset(test_csv: str = "data/features/test_split.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the test set split from disk. Fallbacks to 80/20 train/test split if test_split.csv is missing.

    Parameters:
        test_csv (str): Path to test split CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Test feature matrix X_test, Test labels y_test).
    """
    test_path = Path(test_csv).resolve()

    if test_path.exists():
        print(f"Loading Test Set Split from : {test_path}")
        df_test = pd.read_csv(test_path)
    else:
        full_csv = Path("data/features/features.csv").resolve()
        if not full_csv.exists():
            raise FileNotFoundError(f"Neither '{test_csv}' nor '{full_csv}' exists. Run feature extraction first.")

        print(f"[INFO] '{test_csv}' not found. Re-creating test split from {full_csv}...")
        df_full = pd.read_csv(full_csv)
        from sklearn.model_selection import train_test_split
        _, df_test = train_test_split(df_full, test_size=0.20, stratify=df_full["label"], random_state=42)

    feature_cols = [c for c in df_test.columns if c != "label"]
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values.astype(int)

    return X_test, y_test


def evaluate(
    model_path: str = "models/baseline_model.pkl",
    model_type: str = "auto",
    scaler_path: str = "models/scaler.pkl",
    test_csv: str = "data/features/test_split.csv",
    output_dir: str = "results"
) -> Dict[str, float]:
    """
    Evaluates a trained baseline or deep learning model on the test dataset.

    Parameters:
        model_path (str): Path to saved model file (.pkl or .h5).
        model_type (str): 'baseline', 'deep', or 'auto' (auto-detect by extension).
        scaler_path (str): Path to saved StandardScaler (.pkl).
        test_csv (str): Path to test split CSV file.
        output_dir (str): Directory where confusion_matrix.png and metrics.txt will be saved.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    # Auto-detect model type if requested
    if model_type == "auto":
        if model_file.suffix.lower() in [".h5", ".keras"]:
            model_type = "deep"
        else:
            model_type = "baseline"

    print("=" * 65)
    print(f"      MODEL EVALUATION PIPELINE ({model_type.upper()} MODEL)      ")
    print("=" * 65)
    print(f"Target Model File   : {model_file}")
    print(f"Model Architecture  : {'ResNeXt Deep Learning (Keras)' if model_type == 'deep' else 'SVM Baseline (scikit-learn)'}")
    print(f"Test Split Dataset  : {test_csv}")
    print("=" * 65)

    # 1. Load Test Dataset
    X_test, y_test = load_test_dataset(test_csv)
    print(f"Loaded Test Samples : {X_test.shape[0]} samples ({X_test.shape[1]} features)")

    # 2. Load Scaler
    scaler = None
    scaler_file = Path(scaler_path).resolve()
    if scaler_file.exists():
        print(f"Loading Scaler      : {scaler_file}")
        scaler = joblib.load(scaler_file)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("[WARNING] Scaler file not found. Using unscaled feature vectors.")
        X_test_scaled = X_test

    # 3. Load Model & Perform Inference
    print(f"\n[STEP] Loading model and running test set inference...")
    if model_type == "deep":
        # Import tensorflow lazily to avoid heavy import overhead if evaluating baseline
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_file))

        # Reshape (N, 78) to (N, 6, 13, 1) for Conv2D input
        N_samples = X_test_scaled.shape[0]
        X_test_2d = X_test_scaled.reshape(N_samples, 6, 13, 1)

        y_probs = model.predict(X_test_2d, verbose=0).flatten()
        y_pred = (y_probs >= 0.5).astype(int)
        model_display_name = "ResNeXt Deep Model"

    else:
        # Classical scikit-learn model (SVM / Random Forest / Logistic Regression)
        model = joblib.load(str(model_file))
        y_pred = model.predict(X_test_scaled)

        y_probs = None
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_probs = model.decision_function(X_test_scaled)

        model_display_name = "SVM Baseline Classifier"

    # 4. Compute Evaluation Metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_probs)

    # 5. Print Summary Report to Console
    print("\n" + "=" * 65)
    print(f"         EVALUATION SUMMARY REPORT ({model_display_name})         ")
    print("=" * 65)
    print(f"Total Test Samples : {len(y_test)}")
    print(f"Accuracy           : {metrics['Accuracy']:.2f}%")
    print(f"Precision          : {metrics['Precision']:.4f}")
    print(f"Recall             : {metrics['Recall']:.4f}")
    print(f"F1-Score           : {metrics['F1_Score']:.4f}")
    if "AUC" in metrics:
        print(f"AUC (ROC Area)     : {metrics['AUC']:.4f}")
    if "EER_Percentage" in metrics:
        print(f"Equal Error Rate   : {metrics['EER_Percentage']:.2f}% (Threshold: {metrics['EER_Threshold']:.4f})")
    print("=" * 65)

    # 6. Save Confusion Matrix Plot to results/confusion_matrix.png
    cm_path = str(out_dir / "confusion_matrix.png")
    plot_and_save_confusion_matrix(y_test, y_pred, save_path=cm_path, model_name=model_display_name)

    # 7. Save Summary Report to results/metrics.txt
    txt_path = out_dir / "metrics.txt"
    with open(txt_path, "w") as f:
        f.write("=================================================================\n")
        f.write(f"     AUDIO DEEPFAKE DETECTION SYSTEM — EVALUATION REPORT         \n")
        f.write("=================================================================\n\n")
        f.write(f"Model Evaluated     : {model_display_name}\n")
        f.write(f"Model Path          : {model_file}\n")
        f.write(f"Test Set File       : {test_csv}\n")
        f.write(f"Total Test Samples  : {len(y_test)}\n\n")
        f.write("-----------------------------------------------------------------\n")
        f.write("EVALUATION METRICS:\n")
        f.write("-----------------------------------------------------------------\n")
        f.write(f"Accuracy            : {metrics['Accuracy']:.2f}%\n")
        f.write(f"Precision           : {metrics['Precision']:.4f}\n")
        f.write(f"Recall              : {metrics['Recall']:.4f}\n")
        f.write(f"F1-Score            : {metrics['F1_Score']:.4f}\n")
        if "AUC" in metrics:
            f.write(f"AUC (ROC Area)      : {metrics['AUC']:.4f}\n")
        if "EER_Percentage" in metrics:
            f.write(f"Equal Error Rate    : {metrics['EER_Percentage']:.2f}%\n")
            f.write(f"EER Decision Thresh : {metrics['EER_Threshold']:.4f}\n")
        f.write("-----------------------------------------------------------------\n\n")
        f.write("EQUAL ERROR RATE (EER) EXPLANATION FOR VIVA:\n")
        f.write("-----------------------------------------------------------------\n")
        f.write("Equal Error Rate (EER) is the operating point on the ROC curve where\n")
        f.write("False Acceptance Rate (FAR - falsely passing a deepfake voice as real)\n")
        f.write("equals False Rejection Rate (FRR - falsely rejecting a real voice as fake).\n")
        f.write("A lower EER value indicates a superior and more reliable detector.\n")
        f.write("=================================================================\n")

    print(f"[OK] Summary report saved to: {txt_path}")

    # Also save evaluation_metrics.json for programmatic access
    json_path = out_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] JSON metrics saved to   : {json_path}")
    print("=" * 65)

    return metrics


def main() -> None:
    """
    Main entry point for command line execution.
    """
    parser = argparse.ArgumentParser(description="Evaluate Trained Model for Audio Deepfake Detection")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.pkl", help="Path to saved model (.pkl or .h5)")
    parser.add_argument("--model-type", type=str, choices=["baseline", "deep", "auto"], default="auto", help="Model type ('baseline', 'deep', or 'auto')")
    parser.add_argument("--scaler-path", type=str, default="models/scaler.pkl", help="Path to saved StandardScaler")
    parser.add_argument("--test-csv", type=str, default="data/features/test_split.csv", help="Path to test set CSV")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation report and plots")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        model_type=args.model_type,
        scaler_path=args.scaler_path,
        test_csv=args.test_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
