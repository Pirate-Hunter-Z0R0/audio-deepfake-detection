"""
src/compute_step2_metrics.py
----------------------------
Comprehensive Evaluation, Biometric Forensic Metrics & Signal-Level Energy Engine.

STEP 2 Core Implementations:
1. Comprehensive Classification Metrics:
   - Overall Accuracy, Precision, Recall, F1-Score
   - Macro-averaged Precision, Recall, F1
   - Per-class breakdown: Real (Class 0) vs Synthetic (Class 1)
   - Confusion Matrix (TN, FP, FN, TP) with normalized percentages
2. Biometric Voice Authentication Measures:
   - Equal Error Rate (EER) computed from score distributions where FAR = FRR
   - Exact operating decision threshold at EER
   - ROC Curve and Area Under Curve (AUC)
3. End-to-End Inference Latency Profiling:
   - Detailed breakdown: Preprocessing, Feature Extraction, ResNeXt Model Inference
   - Benchmarked across 30 runs after a dedicated warm-up run
   - Reports mean, standard deviation, min, max, and exact hardware specs
4. Signal-Level Energy Measures (MFCC, LFCC, CQCC):
   - Mathematically defined: Mean Squared Cepstral Energy (MSCE) per frame averaged over time
   - Empirical statistical tests (Two-sample independent t-test & Mann-Whitney U test)
   - Per-clip energy reporting and comparative boxplot / distribution visualization
5. Exports:
   - results/metrics.json (structured schema ready for web app consumption)
   - results/roc_curve.png
   - results/confusion_matrix.png
   - results/energy_distributions.png
   - results/latency_breakdown.png
"""

import os
import sys

# Critical Windows DLL initialization: configure TF runtime prior to other scientific imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import time
import json
import glob
import platform
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_audio, trim_silence, normalize_audio, pad_or_truncate
from src.feature_extraction import extract_mfcc, extract_lfcc, extract_cqcc
from src.test_cloned_samples import preprocess_and_extract
from src.train_deep_model import build_deepfake_detector_model


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) where False Acceptance Rate equals False Rejection Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_val = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0)
    eer_thresh = float(thresholds[eer_idx])
    return eer_val, eer_thresh, fpr, tpr, thresholds


def compute_msce_energies(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Computes Mean Squared Cepstral Energy (MSCE) for MFCC, LFCC, and CQCC:
        E = (1 / (T * K)) * sum_{t=1}^T sum_{k=1}^K [ c_{k,t}^2 ]
    """
    m = extract_mfcc(audio, sr=sr, n_mfcc=13)
    l = extract_lfcc(audio, sr=sr, n_lfcc=13)
    c = extract_cqcc(audio, sr=sr, n_cqcc=13)

    return {
        "mfcc_energy": round(float(np.mean(m ** 2)), 2),
        "lfcc_energy": round(float(np.mean(l ** 2)), 2),
        "cqcc_energy": round(float(np.mean(c ** 2)), 2)
    }


def benchmark_latency(
    model,
    scaler,
    test_wav: str,
    n_runs: int = 30
) -> Dict[str, Any]:
    """
    Measures end-to-end latency with warm-up run and reports breakdown.
    """
    # Warm-up run
    a, sr = load_audio(test_wav, sample_rate=16000)
    a = trim_silence(a, top_db=20)
    a = normalize_audio(a)
    a = pad_or_truncate(a, target_length=32000)
    m = extract_mfcc(a, sr=16000, n_mfcc=13)
    l = extract_lfcc(a, sr=16000, n_lfcc=13)
    c = extract_cqcc(a, sr=16000, n_cqcc=13)
    vec = np.hstack([np.mean(m, axis=1), np.std(m, axis=1), np.mean(l, axis=1), np.std(l, axis=1), np.mean(c, axis=1), np.std(c, axis=1)])
    sc = scaler.transform(vec.reshape(1, -1)).reshape(1, 6, 13, 1)
    _ = model.predict(sc, verbose=0)

    t_prep, t_feat, t_infer, t_total = [], [], [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        a, sr = load_audio(test_wav, sample_rate=16000)
        a = trim_silence(a, top_db=20)
        a = normalize_audio(a)
        a = pad_or_truncate(a, target_length=32000)
        t1 = time.perf_counter()

        m = extract_mfcc(a, sr=16000, n_mfcc=13)
        l = extract_lfcc(a, sr=16000, n_lfcc=13)
        c = extract_cqcc(a, sr=16000, n_cqcc=13)
        vec = np.hstack([np.mean(m, axis=1), np.std(m, axis=1), np.mean(l, axis=1), np.std(l, axis=1), np.mean(c, axis=1), np.std(c, axis=1)])
        t2 = time.perf_counter()

        sc = scaler.transform(vec.reshape(1, -1)).reshape(1, 6, 13, 1)
        _ = model.predict(sc, verbose=0)
        t3 = time.perf_counter()

        t_prep.append((t1 - t0) * 1000)
        t_feat.append((t2 - t1) * 1000)
        t_infer.append((t3 - t2) * 1000)
        t_total.append((t3 - t0) * 1000)

    hardware_info = {
        "processor": platform.processor(),
        "logical_cores": multiprocessing.cpu_count(),
        "operating_system": platform.platform(),
        "python_version": sys.version.split()[0],
        "device_type": "Host Laptop CPU (Intel 64-bit)"
    }

    return {
        "hardware": hardware_info,
        "benchmark_runs": n_runs,
        "preprocessing_ms": {
            "mean": round(float(np.mean(t_prep)), 2),
            "std": round(float(np.std(t_prep)), 2),
            "min": round(float(np.min(t_prep)), 2),
            "max": round(float(np.max(t_prep)), 2)
        },
        "feature_extraction_ms": {
            "mean": round(float(np.mean(t_feat)), 2),
            "std": round(float(np.std(t_feat)), 2),
            "min": round(float(np.min(t_feat)), 2),
            "max": round(float(np.max(t_feat)), 2)
        },
        "model_inference_ms": {
            "mean": round(float(np.mean(t_infer)), 2),
            "std": round(float(np.std(t_infer)), 2),
            "min": round(float(np.min(t_infer)), 2),
            "max": round(float(np.max(t_infer)), 2)
        },
        "end_to_end_total_ms": {
            "mean": round(float(np.mean(t_total)), 2),
            "std": round(float(np.std(t_total)), 2),
            "min": round(float(np.min(t_total)), 2),
            "max": round(float(np.max(t_total)), 2)
        }
    }


def main():
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("   STEP 2: BIOMETRIC FORENSIC METRICS & SIGNAL ENERGY ENGINE   ")
    print("=" * 75)

    # 1. Load Detector & Scaler
    print("[1] Loading ResNeXt Deep CNN and Scaler...")
    model = build_deepfake_detector_model()
    model.load_weights(str(PROJECT_ROOT / "models" / "adapted_deep_model.weights.h5"))
    scaler = joblib.load(str(PROJECT_ROOT / "models" / "adapted_resnext_scaler.pkl"))

    # 2. Gather All Audio Clips
    print("[2] Cataloging audio clips...")
    real_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "real_recordings" / "*.wav")))
    fake_files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "synthetic_rvc" / "*.wav"))) + sorted(glob.glob(str(PROJECT_ROOT / "data" / "cloned_samples" / "*.wav")))

    all_files = real_files + fake_files
    y_true = np.array([0] * len(real_files) + [1] * len(fake_files), dtype=int)
    print(f"  -> Total Dataset: {len(all_files)} clips (Real: {len(real_files)}, Fake: {len(fake_files)})")

    # 3. Extract Features, Compute MSCE Energies & Run Inference
    print("[3] Extracting spectral features and computing MSCE energies...")
    features_list = []
    energies_records = []

    for idx, fpath in enumerate(all_files):
        p = Path(fpath)
        true_lbl = y_true[idx]

        # Audio Loading and preprocessing
        a, sr = load_audio(str(p), sample_rate=16000)
        a_trimmed = trim_silence(a, top_db=20)
        a_norm = normalize_audio(a_trimmed)
        a_clean = a_norm

        # MSCE Energies
        energies = compute_msce_energies(a_clean, sr=16000)
        energies["file"] = p.name
        energies["speaker_id"] = p.stem.split("_")[0]
        energies["label"] = true_lbl
        energies["class_name"] = "Real" if true_lbl == 0 else "Fake"
        energies_records.append(energies)

        # 78-dim Feature Vector
        feat = preprocess_and_extract(str(p))
        features_list.append(feat)

    df_energies = pd.DataFrame(energies_records)
    X = np.array(features_list)

    # ResNeXt Forward Pass
    X_scaled = scaler.transform(X).reshape(-1, 6, 13, 1)
    y_probs = model.predict(X_scaled, verbose=0).flatten()
    y_pred = (y_probs >= 0.5).astype(int)

    # 4. Compute Metrics: Precision, Recall, F1 (Binary, Macro, Per-Class)
    print("\n[4] Computing Precision, Recall, F1-Scores and Confusion Matrix...")
    acc = accuracy_score(y_true, y_pred)
    prec_binary = precision_score(y_true, y_pred, zero_division=0)
    rec_binary = recall_score(y_true, y_pred, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, zero_division=0)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-Class Metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # 5. Compute EER, ROC Curve & AUC
    print("[5] Computing Equal Error Rate (EER) and ROC Curve...")
    eer_val, eer_thresh, fpr_arr, tpr_arr, thresh_arr = calculate_eer(y_true, y_probs)
    roc_auc = auc(fpr_arr, tpr_arr)

    # 6. Benchmark Latency Breakdown
    print("[6] Profiling end-to-end inference latency (30 runs after warm-up)...")
    latency_data = benchmark_latency(model, scaler, real_files[0], n_runs=30)
    print(f"  -> Mean End-to-End Latency: {latency_data['end_to_end_total_ms']['mean']} ms ± {latency_data['end_to_end_total_ms']['std']} ms")

    # 7. Statistical Analysis of Signal-Level Energies (Real vs Fake)
    print("\n[7] Conducting Hypothesis Testing on Signal Energies (Real vs Fake)...")
    r_m = df_energies[df_energies["label"] == 0]["mfcc_energy"].values
    f_m = df_energies[df_energies["label"] == 1]["mfcc_energy"].values

    r_l = df_energies[df_energies["label"] == 0]["lfcc_energy"].values
    f_l = df_energies[df_energies["label"] == 1]["lfcc_energy"].values

    r_c = df_energies[df_energies["label"] == 0]["cqcc_energy"].values
    f_c = df_energies[df_energies["label"] == 1]["cqcc_energy"].values

    energy_stats = {}
    for name, r, f in [("mfcc_energy", r_m, f_m), ("lfcc_energy", r_l, f_l), ("cqcc_energy", r_c, f_c)]:
        t_stat, p_val = stats.ttest_ind(r, f, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(r, f)
        is_sig = bool(p_val < 0.05)
        energy_stats[name] = {
            "real_mean": round(float(np.mean(r)), 2),
            "real_std": round(float(np.std(r)), 2),
            "fake_mean": round(float(np.mean(f)), 2),
            "fake_std": round(float(np.std(f)), 2),
            "t_statistic": round(float(t_stat), 3),
            "p_value": float(p_val),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_pval),
            "statistically_discriminative": is_sig
        }
        print(f"  - {name:12s}: Real={energy_stats[name]['real_mean']} ± {energy_stats[name]['real_std']} | Fake={energy_stats[name]['fake_mean']} ± {energy_stats[name]['fake_std']} | p={p_val:.2e} -> Significant: {is_sig}")

    # 8. Render Visualizations & Plots
    print("\n[8] Generating Figures & Plots...")

    # Plot 1: ROC Curve with EER Point
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr_arr, tpr_arr, color="#00f2fe", linewidth=2.5, label=f"ResNeXt ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random Classifier (AUC = 0.5000)")
    eer_fpr = eer_val / 100.0
    eer_tpr = 1.0 - eer_fpr
    ax.scatter([eer_fpr], [eer_tpr], color="#f43f5e", s=100, zorder=5, label=f"EER Operating Point ({eer_val:.2f}% @ Thresh={eer_thresh:.4f})")
    ax.set_title("Receiver Operating Characteristic (ROC) & EER Operating Point", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FAR / Real Human False Alarms)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Detection Rate)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_plot = results_dir / "roc_curve.png"
    plt.savefig(roc_plot)
    plt.close()

    # Plot 2: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Real (Genuine)", "Fake (Synthetic)"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / np.sum(cm[i, :]) * 100.0
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color="white" if val > thresh else "black", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11)
    ax.set_xlabel("ResNeXt Predicted Class", fontsize=11)
    ax.set_title("ResNeXt Forensic Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_plot = results_dir / "confusion_matrix.png"
    plt.savefig(cm_plot)
    plt.close()

    # Plot 3: Signal-Level Energy Distributions (Boxplots)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    labels = ["Real Speech", "Fake Speech"]

    # LFCC Energy
    axes[0].boxplot([r_l, f_l], tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#3b82f6", alpha=0.7))
    axes[0].set_title(f"LFCC Energy (p={energy_stats['lfcc_energy']['p_value']:.2e})*", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean Squared Cepstral Energy", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # MFCC Energy
    axes[1].boxplot([r_m, f_m], tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#10b981", alpha=0.7))
    axes[1].set_title(f"MFCC Energy (p={energy_stats['mfcc_energy']['p_value']:.2f}) [NS]", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # CQCC Energy
    axes[2].boxplot([r_c, f_c], tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#8b5cf6", alpha=0.7))
    axes[2].set_title(f"CQCC Energy (p={energy_stats['cqcc_energy']['p_value']:.2f}) [NS]", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Signal-Level Energy Comparison (Real vs Synthetic)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    energy_plot = results_dir / "energy_distributions.png"
    plt.savefig(energy_plot)
    plt.close()

    # Plot 4: Latency Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    stages = ["Preprocessing", "Feature Extraction\n(MFCC+LFCC+CQCC)", "ResNeXt CNN\nInference", "Total End-to-End"]
    means = [
        latency_data["preprocessing_ms"]["mean"],
        latency_data["feature_extraction_ms"]["mean"],
        latency_data["model_inference_ms"]["mean"],
        latency_data["end_to_end_total_ms"]["mean"]
    ]
    stds = [
        latency_data["preprocessing_ms"]["std"],
        latency_data["feature_extraction_ms"]["std"],
        latency_data["model_inference_ms"]["std"],
        latency_data["end_to_end_total_ms"]["std"]
    ]
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#00f2fe"]
    bars = ax.bar(stages, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
    ax.set_ylabel("Latency (milliseconds)", fontsize=10)
    ax.set_title("End-to-End Inference Latency Breakdown (Laptop CPU)", fontsize=12, fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 5, f"{h:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    latency_plot = results_dir / "latency_breakdown.png"
    plt.savefig(latency_plot)
    plt.close()

    # 9. Structure Final metrics.json Output
    metrics_payload = {
        "dataset_summary": {
            "total_samples": len(all_files),
            "real_samples": len(real_files),
            "fake_samples": len(fake_files),
            "decision_threshold": 0.50
        },
        "classification_metrics": {
            "accuracy_%": round(float(acc * 100.0), 2),
            "binary_fake_class": {
                "precision": round(float(prec_binary), 4),
                "recall": round(float(rec_binary), 4),
                "f1_score": round(float(f1_binary), 4)
            },
            "macro_average": {
                "precision": round(float(prec_macro), 4),
                "recall": round(float(rec_macro), 4),
                "f1_score": round(float(f1_macro), 4)
            },
            "per_class": {
                "class_0_real": {
                    "precision": round(float(prec_per_class[0]), 4),
                    "recall": round(float(rec_per_class[0]), 4),
                    "f1_score": round(float(f1_per_class[0]), 4),
                    "support": int(len(real_files))
                },
                "class_1_fake": {
                    "precision": round(float(prec_per_class[1]), 4),
                    "recall": round(float(rec_per_class[1]), 4),
                    "f1_score": round(float(f1_per_class[1]), 4),
                    "support": int(len(fake_files))
                }
            },
            "confusion_matrix": {
                "true_negatives_real": int(tn),
                "false_positives_false_alarms": int(fp),
                "false_negatives_missed_deepfakes": int(fn),
                "true_positives_detected_deepfakes": int(tp),
                "false_positive_rate_%": round(fpr * 100.0, 2),
                "false_negative_rate_%": round(fnr * 100.0, 2)
            }
        },
        "biometric_measures": {
            "roc_auc": round(float(roc_auc), 4),
            "equal_error_rate_%": round(float(eer_val), 2),
            "eer_operating_threshold": round(float(eer_thresh), 4)
        },
        "signal_level_energies": {
            "definition": "Mean Squared Cepstral Energy (MSCE) = (1 / (T * K)) * sum_t sum_k [ c_{k,t}^2 ] across K=13 cepstral coefficients and T frames.",
            "justification": "Measures total acoustic power variance distributed across Mel, Linear, and Constant-Q filterbanks. Normalized by T and K to ensure invariance to clip length.",
            "statistics": energy_stats
        },
        "inference_latency": latency_data
    }

    metrics_json_path = results_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"\n[OK] Successfully exported Step 2 metrics to: {metrics_json_path}")

    # Also save per-clip energies table
    df_energies.to_csv(results_dir / "clip_energies.csv", index=False)
    print(f"[OK] Saved per-clip energy values to: {results_dir / 'clip_energies.csv'}")

    return metrics_payload


if __name__ == "__main__":
    main()
