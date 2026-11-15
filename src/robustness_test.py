"""
robustness_test.py
-------------------
Robustness Testing Module for Audio Deepfake Detection.

Purpose:
    Evaluates detector performance drop under degraded audio conditions (background noise, 
    low bit-rate lossy compression artifacts, and time/pitch perturbations).

Reference Context:
    Inspired by Rabhi et al. (2024), "Audio-Deepfake Detection: Adversarial Attacks and Countermeasures".
    
    ---------------------------------------------------------------------------
    Academic / Viva Explanation Note on Robustness Methodology:
    This script implements a simplified, non-adversarial-optimization benchmark of 
    detector robustness under degraded audio conditions. Unlike full iterative gradient-based 
    adversarial attack optimization (such as FGSM or PGD attacks on neural networks), 
    this approach applies realistic physical signal degradations (SNR noise, lossy 
    compression filtering, and pitch/time jitter). This provides an accessible, intuitive 
    evaluation of how well spectral features (MFCC, LFCC, CQCC) generalize under real-world 
    degraded conditions, suitable for a beginner-level academic project.
    ---------------------------------------------------------------------------

Imports:
    os, sys, argparse, pathlib, typing, numpy, pandas, librosa, matplotlib.pyplot, scikit-learn, joblib
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path to allow importing src modules if needed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import spectral feature extraction functions from feature_extraction module
from src.feature_extraction import extract_mfcc, extract_lfcc, extract_cqcc


# ===========================================================================
# 1. AUDIO DEGRADATION FUNCTIONS
# ===========================================================================

def add_white_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """
    (a) Adds background additive Gaussian white noise at a specified Signal-to-Noise Ratio (SNR).

    Parameters:
        audio (np.ndarray): Clean 1D floating-point audio array [-1.0, 1.0].
        snr_db (float): Target SNR in decibels (dB). Lower dB = noisier signal.

    Returns:
        np.ndarray: Noisy audio waveform clipped to [-1.0, 1.0].
    """
    if audio is None or len(audio) == 0:
        return audio

    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    # Calculate noise power for target SNR: SNR_dB = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))

    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)


def apply_mp3_compression_artifacts(audio: np.ndarray, sr: int = 16000, bit_depth: int = 8, downsample_sr: int = 8000) -> np.ndarray:
    """
    (b) Simulates light MP3-style lossy compression artifacts by downsampling to a lower bitrate/frequency 
    and quantizing peak resolution before resampling back to 16 kHz.

    Parameters:
        audio (np.ndarray): Input audio array.
        sr (int): Native sampling rate (16000 Hz).
        bit_depth (int): Simulated bit depth quantization (8-bit speech codec style).
        downsample_sr (int): Target low bandwidth sampling rate (8000 Hz).

    Returns:
        np.ndarray: Compressed/degraded audio waveform.
    """
    if audio is None or len(audio) == 0:
        return audio

    try:
        # Step 1: Low-pass filter / downsample to simulate lossy codec bandwidth limitation (e.g. 8 kHz MP3/VoIP)
        audio_down = librosa.resample(audio, orig_sr=sr, target_sr=downsample_sr)

        # Step 2: Simulate bit depth quantization artifacts
        levels = 2 ** bit_depth
        audio_quantized = np.round((audio_down + 1.0) * (levels / 2.0)) / (levels / 2.0) - 1.0

        # Step 3: Resample back to standard 16 kHz
        audio_reconstructed = librosa.resample(audio_quantized, orig_sr=downsample_sr, target_sr=sr)

        # Ensure exact length match
        if len(audio_reconstructed) > len(audio):
            audio_reconstructed = audio_reconstructed[:len(audio)]
        elif len(audio_reconstructed) < len(audio):
            pad = len(audio) - len(audio_reconstructed)
            audio_reconstructed = np.pad(audio_reconstructed, (0, pad), mode="constant")

        return np.clip(audio_reconstructed, -1.0, 1.0)
    except Exception:
        return audio


def apply_pitch_time_perturbation(
    audio: np.ndarray,
    sr: int = 16000,
    pitch_shift_semitones: float = 1.5,
    max_time_shift_samples: int = 1600
) -> np.ndarray:
    """
    (c) Applies small random pitch-shift and time-shift perturbations to simulate acoustic jitter.

    Parameters:
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate (16000 Hz).
        pitch_shift_semitones (float): Pitch shift step in semitones (e.g. +1.5 semitones).
        max_time_shift_samples (int): Max sample count for random time shift offset.

    Returns:
        np.ndarray: Perturbed audio signal.
    """
    if audio is None or len(audio) == 0:
        return audio

    try:
        # 1. Pitch shift using librosa
        shifted_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_shift_semitones)

        # 2. Time shift (circular roll or offset padding)
        shift_offset = np.random.randint(-max_time_shift_samples, max_time_shift_samples)
        shifted_audio = np.roll(shifted_audio, shift_offset)

        return np.clip(shifted_audio, -1.0, 1.0)
    except Exception:
        return audio


# ===========================================================================
# 2. FEATURE EXTRACTION FOR DEGRADED AUDIO
# ===========================================================================

def extract_features_from_audio_waveform(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts 78-dimensional combined MFCC + LFCC + CQCC feature statistics directly 
    from an in-memory 1D audio waveform array.

    Returns:
        np.ndarray: 1D feature vector of length 78.
    """
    mfcc_mat = extract_mfcc(audio, sr=sr, n_mfcc=13)
    lfcc_mat = extract_lfcc(audio, sr=sr, n_lfcc=13)
    cqcc_mat = extract_cqcc(audio, sr=sr, n_cqcc=13)

    mfcc_mean = np.mean(mfcc_mat, axis=1)
    mfcc_std  = np.std(mfcc_mat, axis=1)

    lfcc_mean = np.mean(lfcc_mat, axis=1)
    lfcc_std  = np.std(lfcc_mat, axis=1)

    cqcc_mean = np.mean(cqcc_mat, axis=1)
    cqcc_std  = np.std(cqcc_mat, axis=1)

    return np.hstack([mfcc_mean, mfcc_std, lfcc_mean, lfcc_std, cqcc_mean, cqcc_std])


# ===========================================================================
# 3. ROBUSTNESS TESTING PIPELINE
# ===========================================================================

def run_robustness_test(
    model_path: str = "models/baseline_model.pkl",
    model_type: str = "auto",
    scaler_path: str = "models/scaler.pkl",
    processed_dir: str = "data/processed",
    results_dir: str = "results",
    samples_per_class: int = 100
) -> pd.DataFrame:
    """
    Runs the complete robustness test suite:
    1. Loads clean audio samples from real/ and fake/ subdirectories.
    2. Evaluates model under 4 conditions:
       - Clean Baseline
       - Added Background Noise (10 dB SNR)
       - Light MP3 Compression Artifacts (8 kHz resampling / 8-bit quantization)
       - Pitch/Time Perturbations (+1.5 semitones / time jitter)
    3. Computes Accuracy, Precision, Recall, F1-Score, and Accuracy Drop %.
    4. Saves results table to results/robustness_results.csv and bar chart to results/robustness_chart.png.

    Parameters:
        model_path (str): Path to trained model (.pkl or .h5).
        model_type (str): 'baseline', 'deep', or 'auto'.
        scaler_path (str): Path to trained StandardScaler.
        processed_dir (str): Path to data/processed folder.
        results_dir (str): Path to results output directory.
        samples_per_class (int): Number of audio files to test per class (default: 100).

    Returns:
        pd.DataFrame: Summary table of robustness results across conditions.
    """
    out_dir = Path(results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    if model_type == "auto":
        model_type = "deep" if model_file.suffix.lower() in [".h5", ".keras"] else "baseline"

    print("=" * 68)
    print(f"      AUDIO DEEPFAKE DETECTOR ROBUSTNESS SUITE ({model_type.upper()})      ")
    print("=" * 68)
    print(f"Model Path           : {model_file}")
    print(f"Audio Source Dir     : {processed_dir}")
    print(f"Samples per Class    : {samples_per_class} (Total {samples_per_class * 2} clips)")
    print("=" * 68)

    # Load Scaler
    scaler = None
    scaler_file = Path(scaler_path).resolve()
    if scaler_file.exists():
        scaler = joblib.load(scaler_file)

    # Load Model
    if model_type == "deep":
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_file))
    else:
        model = joblib.load(str(model_file))

    # Collect sample audio files from real/ and fake/
    proc_base = Path(processed_dir).resolve()
    real_dir = proc_base / "real"
    fake_dir = proc_base / "fake"

    real_files = sorted(list(real_dir.glob("*.wav")))[:samples_per_class]
    fake_files = sorted(list(fake_dir.glob("*.wav")))[:samples_per_class]

    if len(real_files) == 0 or len(fake_files) == 0:
        raise FileNotFoundError(f"No .wav files found in {real_dir} or {fake_dir}. Run preprocessing first.")

    eval_audio_samples: List[Tuple[Path, int]] = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]

    # Define test degradation conditions
    conditions = [
        ("Clean Baseline", lambda a: a),
        ("Background White Noise (10 dB SNR)", lambda a: add_white_noise(a, snr_db=10.0)),
        ("Light MP3 Compression Artifacts", lambda a: apply_mp3_compression_artifacts(a, sr=16000, bit_depth=8, downsample_sr=8000)),
        ("Pitch & Time Perturbations", lambda a: apply_pitch_time_perturbation(a, sr=16000, pitch_shift_semitones=1.5))
    ]

    results_data = []
    baseline_acc = 0.0

    for condition_name, transform_fn in conditions:
        print(f"\n[TESTING] Evaluating condition: '{condition_name}'...")

        y_true = []
        y_pred = []

        for file_path, label in eval_audio_samples:
            try:
                # Load audio waveform at 16 kHz
                audio, sr = librosa.load(str(file_path), sr=16000, mono=True)

                # Apply degradation
                degraded_audio = transform_fn(audio)

                # Extract 78-dim feature vector
                feat_vec = extract_features_from_audio_waveform(degraded_audio, sr=sr).reshape(1, -1)

                # Scale features
                if scaler is not None:
                    feat_vec = scaler.transform(feat_vec)

                # Run inference
                if model_type == "deep":
                    X_2d = feat_vec.reshape(1, 6, 13, 1)
                    prob = model.predict(X_2d, verbose=0)[0][0]
                    pred = int(prob >= 0.5)
                else:
                    pred = int(model.predict(feat_vec)[0])

                y_true.append(label)
                y_pred.append(pred)

            except Exception as e:
                print(f"  [ERROR] Processing {file_path.name} failed: {e}")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if condition_name == "Clean Baseline":
            baseline_acc = acc
            acc_drop = 0.0
        else:
            acc_drop = max(0.0, (baseline_acc - acc) * 100.0)

        results_data.append({
            "Condition": condition_name,
            "Accuracy": round(float(acc), 4),
            "Accuracy %": round(float(acc * 100.0), 2),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "Accuracy Drop %": round(float(acc_drop), 2)
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results_data)

    print("\n" + "=" * 68)
    print("             ROBUSTNESS TEST COMPARISON TABLE             ")
    print("=" * 68)
    print(df_results[["Condition", "Accuracy %", "Precision", "Recall", "F1-Score", "Accuracy Drop %"]].to_string(index=False))
    print("=" * 68)

    # Save CSV comparison table
    csv_path = out_dir / "robustness_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"[OK] Saved robustness results table to: {csv_path}")

    # Plot Bar Chart (save to both robustness_chart.png and robustness_bar_chart.png)
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    bars = plt.barh(df_results["Condition"], df_results["Accuracy %"], color=colors[:len(df_results)])

    plt.xlim(0, 110)
    plt.xlabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title(f"Model Accuracy Under Degraded Audio Conditions ({model_type.upper()})", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{width:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    plt.tight_layout()

    chart_path1 = out_dir / "robustness_chart.png"
    chart_path2 = out_dir / "robustness_bar_chart.png"
    plt.savefig(chart_path1, bbox_inches="tight")
    plt.savefig(chart_path2, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved robustness bar chart to   : {chart_path1}")
    print("=" * 68)

    return df_results


def main() -> None:
    """
    Main entry point for command line execution.
    """
    parser = argparse.ArgumentParser(description="Audio Deepfake Detector Robustness Benchmark")
    parser.add_argument("--model-path", type=str, default="models/baseline_model.pkl", help="Path to saved model (.pkl or .h5)")
    parser.add_argument("--model-type", type=str, choices=["baseline", "deep", "auto"], default="auto", help="Model type ('baseline', 'deep', or 'auto')")
    parser.add_argument("--scaler-path", type=str, default="models/scaler.pkl", help="Path to saved StandardScaler")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to preprocessed audio folder")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save robustness table and chart")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Number of audio clips to evaluate per class (default: 100)")

    args = parser.parse_args()

    run_robustness_test(
        model_path=args.model_path,
        model_type=args.model_type,
        scaler_path=args.scaler_path,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        samples_per_class=args.samples_per_class
    )


if __name__ == "__main__":
    main()
