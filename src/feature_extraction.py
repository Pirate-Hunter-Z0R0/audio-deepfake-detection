"""
feature_extraction.py
----------------------
Spectral Feature Extraction Module for AI-Based Audio Deepfake Detection System.

Purpose:
    Extracts acoustic feature representations—Mel-Frequency Cepstral Coefficients (MFCC),
    Linear-Frequency Cepstral Coefficients (LFCC), and Constant-Q Cepstral Coefficients (CQCC)—
    from preprocessed audio waveforms and exports structured feature matrices/CSVs.

Reference Context:
    Directly guided by Tahaoglu et al. (2025), "Deepfake Audio Detection with 
    Spectral Features and ResNeXt-Based Architecture".

Viva / Academic Explanation Guide:
    1. MFCC (Mel-Frequency Cepstral Coefficients):
       - What it models: The overall filter shape of the human vocal tract (throat, mouth, tongue).
       - Frequency Scale: Uses a logarithmic Mel-scale filterbank, which mimics human ear perception 
         (higher resolution at low frequencies <1 kHz, lower resolution at high frequencies).
       - Why used: Standard baseline feature for speech recognition and speaker identification.
    
    2. LFCC (Linear-Frequency Cepstral Coefficients):
       - What it models: Equal-resolution power spectral distribution across all frequency bands.
       - Frequency Scale: Uses a linear filterbank with constant bandwidth filters from 0 Hz to Nyquist (8 kHz).
       - Why crucial for Deepfake Detection: Synthetic AI voice generators (neural vocoders like HiFi-GAN, WaveGLOW)
         often leave high-frequency phase and spectral artifacts (>4 kHz). Human perception (Mel-scale) compresses
         these high frequencies away, but linear filterbanks (LFCC) preserve equal high-frequency resolution,
         making LFCC superior for detecting deepfakes (ASVspoof benchmark standard).

    3. CQCC (Constant-Q Cepstral Coefficients):
       - What it models: Geometrically spaced pitch and octave relationships using Constant-Q Transform (CQT).
       - Frequency Scale: Constant ratio of center frequency to bandwidth (Q = f / delta_f), matching musical scales.
       - Why used: Effective at capturing subtle fundamental pitch (f0) glitches and harmonic discontinuities 
         introduced during TTS voice cloning.

Imports:
    librosa, numpy, pandas, scipy.fftpack, pathlib, os, argparse
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import librosa
from scipy.fftpack import dct


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCC).

    Parameters:
        audio (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate (16000 Hz).
        n_mfcc (int): Number of MFCC coefficients to retain (default: 13).
        n_fft (int): FFT window size.
        hop_length (int): Frame hop size.

    Returns:
        np.ndarray: MFCC feature matrix of shape (n_mfcc, time_steps).
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Audio waveform is empty.")

    # Compute MFCCs using librosa's Mel-filterbank and STFT
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc


def _linear_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """
    Constructs a linear triangular filterbank matrix across [0, sr/2] Hz.
    """
    num_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, num_fft_bins)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)

    filterbank = np.zeros((n_filters, num_fft_bins))

    for i in range(n_filters):
        f_low = filter_freqs[i]
        f_center = filter_freqs[i + 1]
        f_high = filter_freqs[i + 2]

        # Up-slope
        up_mask = (fft_freqs >= f_low) & (fft_freqs <= f_center)
        if f_center > f_low:
            filterbank[i, up_mask] = (fft_freqs[up_mask] - f_low) / (f_center - f_low)

        # Down-slope
        down_mask = (fft_freqs >= f_center) & (fft_freqs <= f_high)
        if f_high > f_center:
            filterbank[i, down_mask] = (f_high - fft_freqs[down_mask]) / (f_high - f_center)

    return filterbank


def extract_lfcc(audio: np.ndarray, sr: int = 16000, n_lfcc: int = 13, n_filters: int = 40, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extracts Linear-Frequency Cepstral Coefficients (LFCC) using linear triangular filterbank and DCT.

    Parameters:
        audio (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate (16000 Hz).
        n_lfcc (int): Number of LFCC coefficients to retain (default: 13).
        n_filters (int): Number of linear filterbanks (default: 40).
        n_fft (int): FFT window size.
        hop_length (int): Frame hop size.

    Returns:
        np.ndarray: LFCC feature matrix of shape (n_lfcc, time_steps).
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Audio waveform is empty.")

    # 1. Short-Time Fourier Transform (STFT) magnitude spectrum
    stft_spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

    # 2. Linear filterbank energy calculation
    filterbank = _linear_filterbank(n_filters=n_filters, n_fft=n_fft, sr=sr)
    linear_energy = np.dot(filterbank, stft_spec)

    # 3. Log power spectrum (with offset to avoid log(0))
    log_energy = np.log(linear_energy + 1e-10)

    # 4. Discrete Cosine Transform (DCT Type II) along frequency axis
    lfcc = dct(log_energy, type=2, axis=0, norm='ortho')[:n_lfcc]
    return lfcc


def extract_cqcc(audio: np.ndarray, sr: int = 16000, n_cqcc: int = 13, bins_per_octave: int = 24, hop_length: int = 512) -> np.ndarray:
    """
    Extracts Constant-Q Cepstral Coefficients (CQCC-like) using Constant-Q Transform and DCT.

    Parameters:
        audio (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate (16000 Hz).
        n_cqcc (int): Number of CQCC coefficients to retain (default: 13).
        bins_per_octave (int): Frequency resolution per octave (default: 24).
        hop_length (int): Frame hop size.

    Returns:
        np.ndarray: CQCC feature matrix of shape (n_cqcc, time_steps).
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Audio waveform is empty.")

    try:
        # Constant-Q Transform magnitude spectrum
        cqt_spec = np.abs(librosa.cqt(audio, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=bins_per_octave))
    except Exception:
        # Fallback to STFT log-chroma if CQT fails on short signals
        cqt_spec = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length))[:84]

    # Convert to log power (dB)
    log_cqt = librosa.amplitude_to_db(cqt_spec, ref=np.max)

    # Apply DCT Type II along frequency axis to obtain CQCCs
    cqcc = dct(log_cqt, type=2, axis=0, norm='ortho')[:n_cqcc]
    return cqcc


def extract_features(file_path: str, sr: int = 16000, n_mfcc: int = 13, n_lfcc: int = 13, n_cqcc: int = 13) -> np.ndarray:
    """
    Loads a preprocessed audio file and extracts a combined feature vector consisting of 
    mean and standard deviation statistics for MFCC, LFCC, and CQCC.

    Parameters:
        file_path (str): Path to preprocessed audio WAV file.
        sr (int): Target sampling rate (16000 Hz).
        n_mfcc (int): Number of MFCC coefficients.
        n_lfcc (int): Number of LFCC coefficients.
        n_cqcc (int): Number of CQCC coefficients.

    Returns:
        np.ndarray: 1D concatenated feature vector of length (n_mfcc*2 + n_lfcc*2 + n_cqcc*2) = 78 dimensions.
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)

    # 1. Extract feature matrices across time frames
    mfcc_mat = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    lfcc_mat = extract_lfcc(audio, sr=sr, n_lfcc=n_lfcc)
    cqcc_mat = extract_cqcc(audio, sr=sr, n_cqcc=n_cqcc)

    # 2. Compute mean and standard deviation across time frames for each coefficient
    mfcc_mean = np.mean(mfcc_mat, axis=1)
    mfcc_std  = np.std(mfcc_mat, axis=1)

    lfcc_mean = np.mean(lfcc_mat, axis=1)
    lfcc_std  = np.std(lfcc_mat, axis=1)

    cqcc_mean = np.mean(cqcc_mat, axis=1)
    cqcc_std  = np.std(cqcc_mat, axis=1)

    # 3. Concatenate all statistical vectors into a single 1D feature array
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        lfcc_mean, lfcc_std,
        cqcc_mean, cqcc_std
    ])

    return feature_vector


def get_feature_column_names(n_mfcc: int = 13, n_lfcc: int = 13, n_cqcc: int = 13) -> List[str]:
    """
    Generates descriptive column headers for the extracted features dataset CSV.
    """
    cols = []
    
    # Column order must match extract_features() concatenation order:
    # [mfcc_mean_all, mfcc_std_all, lfcc_mean_all, lfcc_std_all, cqcc_mean_all, cqcc_std_all]
    
    # MFCC mean values (indices 0..12)
    for i in range(n_mfcc):
        cols.append(f"mfcc_{i}_mean")
    # MFCC std values (indices 13..25)
    for i in range(n_mfcc):
        cols.append(f"mfcc_{i}_std")

    # LFCC mean values (indices 26..38)
    for i in range(n_lfcc):
        cols.append(f"lfcc_{i}_mean")
    # LFCC std values (indices 39..51)
    for i in range(n_lfcc):
        cols.append(f"lfcc_{i}_std")

    # CQCC mean values (indices 52..64)
    for i in range(n_cqcc):
        cols.append(f"cqcc_{i}_mean")
    # CQCC std values (indices 65..77)
    for i in range(n_cqcc):
        cols.append(f"cqcc_{i}_std")

    return cols


def extract_features_dataset(
    processed_dir: str = "data/processed",
    features_dir: str = "data/features",
    max_files_per_class: Optional[int] = None
) -> str:
    """
    Loops through data/processed/real/ and data/processed/fake/, extracts MFCC+LFCC+CQCC features,
    and exports a single features.csv file in data/features/ with binary labels (0=real, 1=fake).

    Parameters:
        processed_dir (str): Folder containing preprocessed real/ and fake/ subdirectories.
        features_dir (str): Folder where features.csv will be saved.
        max_files_per_class (Optional[int]): Optional cap on files per category for fast testing.

    Returns:
        str: Absolute path to exported features.csv file.
    """
    base_processed = Path(processed_dir).resolve()
    base_features = Path(features_dir).resolve()
    base_features.mkdir(parents=True, exist_ok=True)

    csv_output_path = base_features / "features.csv"
    npy_output_path = base_features / "features.npy"

    print("="*60)
    print("      SPECTRAL FEATURE EXTRACTION (MFCC + LFCC + CQCC)      ")
    print("="*60)
    print(f"Processed Audio Input : {base_processed}")
    print(f"Features CSV Output   : {csv_output_path}")
    print("="*60)

    categories = [("real", 0), ("fake", 1)]
    feature_rows = []
    column_names = get_feature_column_names() + ["label"]

    total_extracted = 0
    total_failed = 0

    for category_name, label in categories:
        cat_dir = base_processed / category_name
        if not cat_dir.exists():
            print(f"\n[WARNING] Directory '{cat_dir}' not found. Skipping '{category_name}'.")
            continue

        audio_files = sorted(list(cat_dir.glob("*.wav")))
        total_files = len(audio_files)

        if max_files_per_class is not None and max_files_per_class < total_files:
            audio_files = audio_files[:max_files_per_class]
            print(f"\n[INFO] Capping '{category_name}' category to first {max_files_per_class} files (out of {total_files}).")
            total_files = len(audio_files)

        print(f"\n[START] Extracting features for '{category_name}' clips ({total_files} files)...")

        for idx, file_path in enumerate(audio_files, 1):
            try:
                feat_vec = extract_features(str(file_path))
                row_data = list(feat_vec) + [label]
                feature_rows.append(row_data)
                total_extracted += 1

            except Exception as e:
                print(f"\n[ERROR] Failed extracting '{file_path.name}': {e}")
                total_failed += 1

            if idx % 50 == 0 or idx == total_files:
                pct = (idx / total_files) * 100
                print(f"  [{category_name.upper()}] Feature Extraction: {idx}/{total_files} ({pct:.1f}%) complete...")

    # Create pandas DataFrame and export CSV
    df = pd.DataFrame(feature_rows, columns=column_names)
    df.to_csv(csv_output_path, index=False)

    # Save numpy array version for high-speed ML loading
    X_y_array = np.array(feature_rows)
    np.save(npy_output_path, X_y_array)

    print("\n" + "="*60)
    print("         FEATURE EXTRACTION PIPELINE COMPLETE           ")
    print("="*60)
    print(f"Total Files Extracted        : {total_extracted}")
    print(f"Feature Vector Dimension     : {len(column_names) - 1} features + 1 label")
    print(f"Features CSV Exported To     : {csv_output_path}")
    print(f"Features NPY Exported To     : {npy_output_path}")
    print("="*60)

    return str(csv_output_path)


def main() -> None:
    """
    Main entry point for command line execution.
    """
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline for Audio Deepfake Detection")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed audio directory (default: data/processed)")
    parser.add_argument("--features-dir", type=str, default="data/features", help="Path to features output directory (default: data/features)")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files per category (e.g. 500 for fast run)")

    args = parser.parse_args()

    extract_features_dataset(
        processed_dir=args.processed_dir,
        features_dir=args.features_dir,
        max_files_per_class=args.max_files
    )


if __name__ == "__main__":
    main()
