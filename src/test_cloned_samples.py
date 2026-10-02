"""
test_cloned_samples.py
-----------------------
Integration Test Script for Audio Deepfake Detection System.

Purpose:
    Evaluates our existing, trained detectors (Baseline SVM & Deep ResNeXt CNN)
    against our newly generated voice-cloned test samples (created via Coqui XTTS-v2).

Design Principles:
    - Imports and reuses existing preprocessing and feature extraction modules directly.
    - Does NOT modify or rebuild any training/model code.
    - Reports exact model outputs without fabricating or assuming 100% accuracy.
    - Saves evaluation results to results/cloned_samples_evaluation.csv.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_audio, trim_silence, normalize_audio, pad_or_truncate
from src.feature_extraction import extract_mfcc, extract_lfcc, extract_cqcc


def preprocess_and_extract(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Applies the standardized project pipeline:
    Audio -> Resample (16kHz) -> Trim Silence -> Peak Normalize -> Pad/Truncate (32k)
          -> MFCC (26) + LFCC (26) + CQCC (26) -> 78-dim vector.
    """
    audio, sample_rate = load_audio(file_path, sample_rate=sr)
    audio = trim_silence(audio, top_db=20)
    audio = normalize_audio(audio)
    audio = pad_or_truncate(audio, target_length=32000)

    mfcc_mat = extract_mfcc(audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc_mat, axis=1)
    mfcc_std  = np.std(mfcc_mat, axis=1)

    lfcc_mat = extract_lfcc(audio, sr=sample_rate, n_lfcc=13)
    lfcc_mean = np.mean(lfcc_mat, axis=1)
    lfcc_std  = np.std(lfcc_mat, axis=1)

    cqcc_mat = extract_cqcc(audio, sr=sample_rate, n_cqcc=13)
    cqcc_mean = np.mean(cqcc_mat, axis=1)
    cqcc_std  = np.std(cqcc_mat, axis=1)

    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        lfcc_mean, lfcc_std,
        cqcc_mean, cqcc_std
    ])
    return feature_vector


def run_integration_test(
    audio_dir_or_file: str,
    scaler_path: str = "models/scaler.pkl",
    baseline_path: str = "models/baseline_model.pkl",
    deep_path: str = "models/deep_model.h5",
    results_dir: str = "results"
) -> pd.DataFrame:
    """
    Evaluates trained models on the provided cloned audio file(s).
    """
    target = Path(audio_dir_or_file).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target path not found: {target}")

    if target.is_file():
        audio_files = [target]
    else:
        audio_files = sorted(list(target.glob("*.wav")) + list(target.glob("*.mp3")))

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in directory: {target}")

    print("=" * 70)
    print("   AUDIO DEEPFAKE DETECTOR — CLONED SAMPLES INTEGRATION TEST   ")
    print("=" * 70)
    print(f"Target Samples Path : {target}")
    print(f"Total Samples Found : {len(audio_files)}")
    print("Ground Truth Label  : FAKE / DEEPFAKE (1)")
    print("=" * 70)

    # 1. Load Scaler
    scaler_file = Path(scaler_path).resolve()
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_file}")
    scaler = joblib.load(scaler_file)
    print(f"[OK] Loaded Scaler from: {scaler_file}")

    # 2. Load Baseline Model (SVM)
    baseline_file = Path(baseline_path).resolve()
    baseline_model = None
    if baseline_file.exists():
        baseline_model = joblib.load(baseline_file)
        print(f"[OK] Loaded Baseline SVM from: {baseline_file}")

    # 3. Load Deep Model (ResNeXt)
    deep_file = Path(deep_path).resolve()
    deep_model = None
    if deep_file.exists():
        try:
            import tensorflow as tf
            deep_model = tf.keras.models.load_model(str(deep_file))
            print(f"[OK] Loaded ResNeXt Deep Model from: {deep_file}")
        except Exception as e:
            print(f"[WARNING] Could not load deep model: {e}")

    print("\n[START] Running inference pipeline across samples...\n")

    results_data = []

    for file_path in audio_files:
        try:
            # Step 1: Preprocess & Extract 78-dim features
            feat = preprocess_and_extract(str(file_path)).reshape(1, -1)
            feat_scaled = scaler.transform(feat)

            record = {
                "Filename": file_path.name,
                "Ground_Truth": "Fake (1)",
            }

            # Step 2: Evaluate Baseline SVM
            if baseline_model is not None:
                svm_pred = int(baseline_model.predict(feat_scaled)[0])
                if hasattr(baseline_model, "predict_proba"):
                    svm_prob_fake = float(baseline_model.predict_proba(feat_scaled)[0][1])
                else:
                    decision = float(baseline_model.decision_function(feat_scaled)[0])
                    svm_prob_fake = float(1 / (1 + np.exp(-decision)))

                record["SVM_Prediction"] = "Fake" if svm_pred == 1 else "Real"
                record["SVM_Confidence_Fake_%"] = round(svm_prob_fake * 100, 2)
                record["SVM_Result"] = "CORRECT (Detected)" if svm_pred == 1 else "MISCLASSIFIED (Evaded)"

            # Step 3: Evaluate Deep ResNeXt
            if deep_model is not None:
                X_2d = feat_scaled.reshape(1, 6, 13, 1)
                deep_prob_fake = float(deep_model.predict(X_2d, verbose=0)[0][0])
                deep_pred = int(deep_prob_fake >= 0.5)

                record["ResNeXt_Prediction"] = "Fake" if deep_pred == 1 else "Real"
                record["ResNeXt_Confidence_Fake_%"] = round(deep_prob_fake * 100, 2)
                record["ResNeXt_Result"] = "CORRECT (Detected)" if deep_pred == 1 else "MISCLASSIFIED (Evaded)"

            results_data.append(record)

        except Exception as e:
            print(f"[ERROR] Processing '{file_path.name}': {e}")

    df_results = pd.DataFrame(results_data)

    # Print summary table
    print("=" * 70)
    print("                    INTEGRATION TEST RESULTS                   ")
    print("=" * 70)
    cols_to_show = ["Filename"]
    if "SVM_Prediction" in df_results.columns:
        cols_to_show.extend(["SVM_Prediction", "SVM_Confidence_Fake_%", "SVM_Result"])
    if "ResNeXt_Prediction" in df_results.columns:
        cols_to_show.extend(["ResNeXt_Prediction", "ResNeXt_Confidence_Fake_%", "ResNeXt_Result"])

    print(df_results[cols_to_show].to_string(index=False))
    print("=" * 70)

    # Calculate detection rates
    if "SVM_Prediction" in df_results.columns:
        svm_detected = (df_results["SVM_Prediction"] == "Fake").sum()
        print(f"SVM Detection Rate     : {svm_detected}/{len(df_results)} ({svm_detected/len(df_results)*100:.1f}%)")
    if "ResNeXt_Prediction" in df_results.columns:
        resnext_detected = (df_results["ResNeXt_Prediction"] == "Fake").sum()
        print(f"ResNeXt Detection Rate : {resnext_detected}/{len(df_results)} ({resnext_detected/len(df_results)*100:.1f}%)")
    print("=" * 70)

    # Save to results
    out_dir = Path(results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cloned_samples_evaluation.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"[OK] Saved detailed results to: {out_csv}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Test trained deepfake detectors on custom cloned audio.")
    parser.add_argument("--audio-path", type=str, default="data/cloned_samples", help="Path to cloned WAV file or folder")
    parser.add_argument("--scaler-path", type=str, default="models/scaler.pkl", help="Path to scaler.pkl")
    parser.add_argument("--baseline-path", type=str, default="models/baseline_model.pkl", help="Path to baseline_model.pkl")
    parser.add_argument("--deep-path", type=str, default="models/deep_model.h5", help="Path to deep_model.h5")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save CSV report")

    args = parser.parse_args()

    run_integration_test(
        audio_dir_or_file=args.audio_path,
        scaler_path=args.scaler_path,
        baseline_path=args.baseline_path,
        deep_path=args.deep_path,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
