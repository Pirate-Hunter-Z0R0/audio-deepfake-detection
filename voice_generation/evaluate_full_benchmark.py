"""
voice_generation/evaluate_full_benchmark.py
-------------------------------------------
Comprehensive evaluation of Audio Deepfake Detectors (Baseline SVM and Deep ResNeXt CNN)
across both genuine human recordings (Real, label=0) and Applio RVC synthetic clones (Fake, label=1).

Computes:
    - Sample-by-sample classifications and prediction probabilities
    - Confusion Matrix (TP, TN, FP, FN) for both models
    - Detection Metrics (Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, FPR)
    - Per-Speaker Performance Breakdown
    - Exports detailed CSV to results/full_benchmark_evaluation.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.test_cloned_samples import preprocess_and_extract


def parse_speaker_id(filename: str) -> str:
    """Extracts speaker_id from filename (e.g. teammate01_rvc_0001.wav -> teammate01)."""
    name = Path(filename).stem
    for prefix in ["teammate01", "teammate02", "teammate03", "teammate04", "teammate05"]:
        if prefix in name:
            return prefix
    return name.split("_")[0]


def print_confusion_matrix(title: str, tp: int, tn: int, fp: int, fn: int):
    print(f"\n{title}")
    print("-" * 55)
    print(f"{'':22} | {'Predicted Real (0)':18} | {'Predicted Fake (1)':18}")
    print("-" * 65)
    print(f"{'Actual Real (0)':22} | {tn:^18} | {fp:^18}")
    print(f"{'Actual Fake (1)':22} | {fn:^18} | {tp:^18}")
    print("-" * 65)


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "accuracy_%": round(accuracy * 100, 2),
        "precision_%": round(precision * 100, 2),
        "recall_tpr_%": round(recall * 100, 2),
        "specificity_tnr_%": round(specificity * 100, 2),
        "fpr_%": round(fpr * 100, 2),
        "fnr_%": round(fnr * 100, 2),
        "f1_score": round(f1, 4)
    }


def evaluate_full_benchmark(
    real_dir: str = "data/real_recordings",
    synthetic_dir: str = "data/synthetic_rvc",
    scaler_path: str = "models/scaler.pkl",
    baseline_path: str = "models/baseline_model.pkl",
    deep_path: str = "models/deep_model.h5",
    results_dir: str = "results",
    output_csv: str = "full_benchmark_evaluation.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    real_path = Path(real_dir).resolve()
    synth_path = Path(synthetic_dir).resolve()

    real_files = sorted(list(real_path.glob("*.wav")) + list(real_path.glob("*.mp3")) + list(real_path.glob("*.m4a"))) if real_path.is_dir() else []
    synth_files = sorted(list(synth_path.glob("*.wav")) + list(synth_path.glob("*.mp3"))) if synth_path.is_dir() else []

    if not real_files and not synth_files:
        raise FileNotFoundError(f"No audio files found in either {real_path} or {synth_path}")

    print("=" * 80)
    print("      AUDIO DEEPFAKE DETECTION: FULL BENCHMARK EVALUATION (REAL + FAKE)      ")
    print("=" * 80)
    print(f"Real Human Recordings   : {len(real_files)} samples ({real_path})")
    print(f"Synthetic RVC Clones    : {len(synth_files)} samples ({synth_path})")
    print(f"Total Evaluation Cohort : {len(real_files) + len(synth_files)} audio files")
    print("=" * 80)

    # 1. Load Scaler
    scaler_file = Path(scaler_path).resolve()
    if not scaler_file.is_file():
        raise FileNotFoundError(f"Scaler not found at: {scaler_file}")
    scaler = joblib.load(scaler_file)
    print(f"[OK] Scaler loaded: {scaler_file.name}")

    # 2. Load Baseline SVM
    baseline_file = Path(baseline_path).resolve()
    baseline_model = None
    if baseline_file.is_file():
        baseline_model = joblib.load(baseline_file)
        print(f"[OK] Baseline SVM loaded: {baseline_file.name}")

    # 3. Load ResNeXt Deep CNN
    deep_file = Path(deep_path).resolve()
    deep_model = None
    if deep_file.is_file():
        try:
            from src.train_deep_model import build_deepfake_detector_model
            deep_model = build_deepfake_detector_model()
            deep_model.load_weights(str(deep_file))
            print(f"[OK] Deep ResNeXt CNN loaded: {deep_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not load ResNeXt: {e}")

    # Combine audio evaluation targets
    all_targets = []
    for f in real_files:
        all_targets.append((f, 0, "Real"))
    for f in synth_files:
        all_targets.append((f, 1, "Fake"))

    records = []
    
    # Tracking for confusion matrix
    svm_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    deep_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    print("\n--- Evaluating Samples ---")

    for file_path, gt_label, gt_str in all_targets:
        speaker_id = parse_speaker_id(file_path.name)
        feat = preprocess_and_extract(str(file_path)).reshape(1, -1)
        feat_scaled = scaler.transform(feat)

        record = {
            "file": file_path.name,
            "speaker_id": speaker_id,
            "ground_truth_label": gt_label,
            "ground_truth": gt_str
        }

        # Baseline SVM
        if baseline_model is not None:
            svm_pred = int(baseline_model.predict(feat_scaled)[0])
            if hasattr(baseline_model, "predict_proba"):
                svm_prob_fake = float(baseline_model.predict_proba(feat_scaled)[0][1])
            else:
                decision = float(baseline_model.decision_function(feat_scaled)[0])
                svm_prob_fake = float(1 / (1 + np.exp(-decision)))

            record["svm_pred"] = "Fake" if svm_pred == 1 else "Real"
            record["svm_conf_fake_%"] = round(svm_prob_fake * 100, 2)

            if gt_label == 1:
                if svm_pred == 1:
                    svm_status = "TP (Detected Fake)"
                    svm_counts["tp"] += 1
                else:
                    svm_status = "FN (Evaded Fake)"
                    svm_counts["fn"] += 1
            else:
                if svm_pred == 0:
                    svm_status = "TN (Correct Real)"
                    svm_counts["tn"] += 1
                else:
                    svm_status = "FP (False Alarm)"
                    svm_counts["fp"] += 1

            record["svm_status"] = svm_status

        # Deep ResNeXt CNN
        if deep_model is not None:
            X_2d = feat_scaled.reshape(1, 6, 13, 1)
            deep_prob_fake = float(deep_model.predict(X_2d, verbose=0)[0][0])
            deep_pred = int(deep_prob_fake >= 0.5)

            record["resnext_pred"] = "Fake" if deep_pred == 1 else "Real"
            record["resnext_conf_fake_%"] = round(deep_prob_fake * 100, 2)

            if gt_label == 1:
                if deep_pred == 1:
                    deep_status = "TP (Detected Fake)"
                    deep_counts["tp"] += 1
                else:
                    deep_status = "FN (Evaded Fake)"
                    deep_counts["fn"] += 1
            else:
                if deep_pred == 0:
                    deep_status = "TN (Correct Real)"
                    deep_counts["tn"] += 1
                else:
                    deep_status = "FP (False Alarm)"
                    deep_counts["fp"] += 1

            record["resnext_status"] = deep_status

        records.append(record)

    df_results = pd.DataFrame(records)

    # Display Results Table
    print("\n" + "=" * 105)
    print(f"{'File':<25} | {'Speaker':<10} | {'True Label':<10} | {'SVM Pred':<8} | {'SVM Fake%':<10} | {'ResNeXt Pred':<12} | {'ResNeXt Fake%':<12}")
    print("=" * 105)
    for _, r in df_results.iterrows():
        print(f"{r['file']:<25} | {r['speaker_id']:<10} | {r['ground_truth']:<10} | {r.get('svm_pred','-'):<8} | {r.get('svm_conf_fake_%', 0.0):>8.2f}% | {r.get('resnext_pred','-'):<12} | {r.get('resnext_conf_fake_%', 0.0):>10.2f}%")
    print("=" * 105)

    # Print Confusion Matrices
    print("\n" + "=" * 65)
    print("                   CONFUSION MATRICES                   ")
    print("=" * 65)
    print_confusion_matrix("BASELINE SVM CONFUSION MATRIX", **svm_counts)
    print_confusion_matrix("DEEP RESNEXT CNN CONFUSION MATRIX", **deep_counts)

    # Quantitative Summary Metrics
    svm_metrics = calculate_metrics(**svm_counts)
    deep_metrics = calculate_metrics(**deep_counts)

    print("\n" + "=" * 65)
    print("                   PERFORMANCE METRICS SUMMARY                   ")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Baseline SVM':<14} | {'Deep ResNeXt':<14}")
    print("-" * 65)
    print(f"{'Total Evaluated':<30} | {svm_metrics['total']:<14} | {deep_metrics['total']:<14}")
    print(f"{'Accuracy':<30} | {svm_metrics['accuracy_%']:>12.2f}% | {deep_metrics['accuracy_%']:>12.2f}%")
    print(f"{'True Negative Rate (Specificity)':<30} | {svm_metrics['specificity_tnr_%']:>12.2f}% | {deep_metrics['specificity_tnr_%']:>12.2f}%")
    print(f"{'True Positive Rate (Recall / TPR)':<30} | {svm_metrics['recall_tpr_%']:>12.2f}% | {deep_metrics['recall_tpr_%']:>12.2f}%")
    print(f"{'False Positive Rate (FPR)':<30} | {svm_metrics['fpr_%']:>12.2f}% | {deep_metrics['fpr_%']:>12.2f}%")
    print(f"{'False Negative Rate (FNR)':<30} | {svm_metrics['fnr_%']:>12.2f}% | {deep_metrics['fnr_%']:>12.2f}%")
    print(f"{'Precision':<30} | {svm_metrics['precision_%']:>12.2f}% | {deep_metrics['precision_%']:>12.2f}%")
    print(f"{'F1-Score':<30} | {svm_metrics['f1_score']:>14.4f} | {deep_metrics['f1_score']:>14.4f}")
    print("=" * 65)

    # Per-Speaker Breakdown
    print("\n--- Per-Speaker Accuracy Breakdown ---")
    speakers = sorted(df_results["speaker_id"].unique())
    for spk in speakers:
        df_spk = df_results[df_results["speaker_id"] == spk]
        spk_real = df_spk[df_spk["ground_truth_label"] == 0]
        spk_fake = df_spk[df_spk["ground_truth_label"] == 1]

        real_acc_svm = (spk_real["svm_pred"] == "Real").sum() / len(spk_real) * 100 if len(spk_real) > 0 else None
        fake_det_svm = (spk_fake["svm_pred"] == "Fake").sum() / len(spk_fake) * 100 if len(spk_fake) > 0 else None

        real_acc_dn = (spk_real["resnext_pred"] == "Real").sum() / len(spk_real) * 100 if len(spk_real) > 0 else None
        fake_det_dn = (spk_fake["resnext_pred"] == "Fake").sum() / len(spk_fake) * 100 if len(spk_fake) > 0 else None

        real_str = f"Real Spec={real_acc_svm:.1f}% ({len(spk_real)} clips)" if real_acc_svm is not None else "Real clips: N/A"
        fake_str = f"Fake TPR={fake_det_svm:.1f}% ({len(spk_fake)} clips)" if fake_det_svm is not None else "Fake clips: N/A"
        print(f"  [{spk}]: SVM -> {real_str} | {fake_str}")

    # Save to CSV
    out_dir = Path(results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / output_csv
    df_results.to_csv(out_csv, index=False)
    print(f"\n[OK] Full benchmark evaluation saved to: {out_csv}")

    return df_results, {"svm": svm_metrics, "resnext": deep_metrics}


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake detectors on real and synthetic benchmark.")
    parser.add_argument("--real-dir", type=str, default="data/real_recordings")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic_rvc")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl")
    parser.add_argument("--baseline", type=str, default="models/baseline_model.pkl")
    parser.add_argument("--deep", type=str, default="models/deep_model.h5")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-csv", type=str, default="full_benchmark_evaluation.csv")
    args = parser.parse_args()

    evaluate_full_benchmark(
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        scaler_path=args.scaler,
        baseline_path=args.baseline,
        deep_path=args.deep,
        results_dir=args.results_dir,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
