"""
preprocessing.py
----------------
Audio Preprocessing Module for AI-Based Audio Deepfake Detection System.

Purpose:
    Converts raw audio clips (.wav, .flac, .mp3) into cleaned, standardized audio signals.
    Standardization steps include resampling to a fixed sampling frequency (16 kHz), 
    trimming leading/trailing unvoiced silence, peak amplitude normalization, and 
    uniform sequence length padding/truncation.

Viva / Academic Explanation Guide:
    1. Resampling (16 kHz):
       - Human speech frequencies primarily lie between 300 Hz and 3,400 Hz.
       - By Nyquist-Shannon Theorem, a 16 kHz sampling rate captures frequencies up to 8 kHz,
         which covers all essential vocal formants and pitch characteristics while dramatically
         reducing computational complexity and memory usage compared to 44.1 kHz CD audio.
    2. Silence Trimming:
       - Leading and trailing silence contains no vocal energy or acoustic cues.
       - Trimming silence prevents the model from wasting capacity learning background zero-noise 
         patterns and ensures spectral features (MFCC, LFCC, CQCC) capture active speech.
    3. Peak Amplitude Normalization:
       - Audio recordings vary in recording volume due to microphone gain and distance.
       - Peak normalization scales waveform amplitudes into a uniform [-1.0, 1.0] range.
         This prevents loudness variations from corrupting power spectrum estimations.
    4. Fixed Length Padding / Truncating:
       - Neural networks (CNNs, ResNeXt) require uniform tensor shapes as input.
       - Standardizing audio duration (e.g., 2.0 seconds = 32,000 samples at 16 kHz) guarantees
         consistent matrix dimensions across all dataset batches.

Imports:
    librosa, soundfile, numpy, pathlib, os, argparse
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import librosa
import soundfile as sf


def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and resamples it to a fixed target sampling rate using librosa.

    Why We Resample:
        Recording hardware and speech synthesis models use different native sampling rates
        (e.g., 22.05 kHz, 44.1 kHz, 48 kHz). Standardizing to 16,000 Hz ensures uniform 
        time-frequency spacing across all spectrogram representations.

    Parameters:
        file_path (str): Path to the input audio file (.wav, .flac, .mp3).
        sample_rate (int): Desired target sampling frequency in Hz (default: 16000 Hz).

    Returns:
        Tuple[np.ndarray, int]: (1D floating-point audio array, sampling rate).
    """
    try:
        # librosa.load automatically converts multi-channel audio to mono (mono=True)
        # and resamples the signal to target sample_rate using high-quality scipy interpolation.
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio, sr
    except Exception as e:
        raise IOError(f"Failed to load audio file '{file_path}': {e}")


def load_and_resample_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Alias wrapper for load_audio for backward compatibility with project modules.
    """
    return load_audio(file_path, sample_rate=target_sr)


def trim_silence(audio: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Trims leading and trailing unvoiced silence from an audio waveform.

    Why We Trim Silence:
        Silent intervals contain zero speech information. Including long quiet margins 
        skews the calculated mean/variance of spectral feature matrices (MFCCs/LFCCs) 
        and adds uninformative zero-padded regions to spectrogram images.

    Parameters:
        audio (np.ndarray): 1D audio signal array.
        top_db (int): The threshold in decibels below top peak value to consider as silence.
                      Signals softer than (peak_db - top_db) are trimmed off.

    Returns:
        np.ndarray: Trimmed 1D audio signal.
    """
    if audio is None or len(audio) == 0:
        return audio

    # librosa.effects.trim returns trimmed audio array and interval index (start, end)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    
    # Safeguard: if trimming removed all samples (silent audio), return original
    if len(audio_trimmed) == 0:
        return audio

    return audio_trimmed


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes peak amplitude of the audio signal to scale within [-1.0, 1.0].

    Why We Normalize Amplitude:
        Audio recording gain varies wildly between different recording sessions and TTS synthesizers.
        Peak normalization ensures maximum signal amplitude equals 1.0, protecting feature extractions
        from loudness bias and preventing numerical instability during neural network backpropagation.

    Parameters:
        audio (np.ndarray): Input 1D audio waveform array.

    Returns:
        np.ndarray: Amplitude-normalized audio waveform.
    """
    if audio is None or len(audio) == 0:
        return audio

    max_peak = np.max(np.abs(audio))
    
    # Avoid division by zero if audio is completely silent
    if max_peak > 0:
        return audio / max_peak
    
    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int = 32000) -> np.ndarray:
    """
    Pads short audio clips with trailing zeros or truncates longer audio clips 
    to achieve a fixed sample length.

    Why We Pad or Truncate:
        Deep learning models (e.g. ResNeXt CNNs) require input matrices of identical shape.
        32,000 samples corresponds exactly to 2.0 seconds of audio at 16,000 Hz.

    Parameters:
        audio (np.ndarray): 1D audio signal array.
        target_length (int): Fixed number of samples (default: 32000 = 2 seconds at 16kHz).

    Returns:
        np.ndarray: Audio array of exact shape (target_length,).
    """
    current_length = len(audio)

    if current_length == target_length:
        return audio
    elif current_length > target_length:
        # Truncate to first target_length samples
        return audio[:target_length]
    else:
        # Pad trailing end with zeros
        pad_width = target_length - current_length
        return np.pad(audio, (0, pad_width), mode='constant', constant_values=0)


def preprocess_file(file_path: str, output_path: str, sample_rate: int = 16000, top_db: int = 20, target_length: Optional[int] = 32000) -> bool:
    """
    Chains audio loading, silence trimming, peak normalization, padding/truncating,
    and saves the cleaned audio to data/processed/.

    Steps:
        1. Load & Resample -> 16 kHz
        2. Trim Silence -> Remove quiet margins
        3. Normalize -> Scale amplitude to [-1.0, 1.0]
        4. Pad / Truncate -> Fixed 32,000 samples (2 seconds)
        5. Write -> Save WAV file to output_path using soundfile

    Parameters:
        file_path (str): Input raw audio file path.
        output_path (str): Output destination path in data/processed/.
        sample_rate (int): Target sampling rate (default: 16000 Hz).
        top_db (int): Silence threshold in dB (default: 20 dB).
        target_length (Optional[int]): Fixed length in samples (default: 32000).

    Returns:
        bool: True if preprocessing succeeded, False if error occurred.
    """
    try:
        # 1. Load audio and resample to target sampling rate
        audio, sr = load_audio(file_path, sample_rate=sample_rate)

        # 2. Remove leading and trailing silence
        audio_trimmed = trim_silence(audio, top_db=top_db)

        # 3. Peak amplitude normalization
        audio_normalized = normalize_audio(audio_trimmed)

        # 4. Standardize duration (pad or truncate to 32,000 samples / 2 sec)
        if target_length is not None:
            final_audio = pad_or_truncate(audio_normalized, target_length=target_length)
        else:
            final_audio = audio_normalized

        # 5. Ensure parent output directory exists and save cleaned audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, final_audio, sr, subtype='PCM_16')
        
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to preprocess '{file_path}': {e}")
        return False


def preprocess_dataset(
    raw_dir: str = "data/raw", 
    processed_dir: str = "data/processed", 
    sample_rate: int = 16000, 
    max_files_per_class: Optional[int] = None
) -> None:
    """
    Loops through data/raw/real/ and data/raw/fake/, preprocesses all audio clips,
    and saves results into data/processed/real/ and data/processed/fake/, preserving structure.

    Parameters:
        raw_dir (str): Path to raw data folder containing real/ and fake/ subdirectories.
        processed_dir (str): Path to destination folder for cleaned audio files.
        sample_rate (int): Target sampling rate (16000 Hz).
        max_files_per_class (Optional[int]): Optional cap on number of files to process per class
                                             (useful for fast prototyping).
    """
    base_raw = Path(raw_dir).resolve()
    base_processed = Path(processed_dir).resolve()

    categories = ["real", "fake"]
    
    print("="*60)
    print("       AUDIO PREPROCESSING & CLEANING PIPELINE       ")
    print("="*60)
    print(f"Input Raw Directory      : {base_raw}")
    print(f"Output Processed Directory: {base_processed}")
    print(f"Target Sampling Rate     : {sample_rate} Hz")
    print("="*60)

    total_processed = 0
    total_failed = 0

    for category in categories:
        src_category_dir = base_raw / category
        dst_category_dir = base_processed / category
        dst_category_dir.mkdir(parents=True, exist_ok=True)

        if not src_category_dir.exists():
            print(f"\n[WARNING] Directory '{src_category_dir}' not found. Skipping category '{category}'.")
            continue

        # Find all audio files
        audio_files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            audio_files.extend(list(src_category_dir.glob(ext)))

        audio_files = sorted(audio_files)
        total_in_category = len(audio_files)

        if max_files_per_class is not None and max_files_per_class < total_in_category:
            audio_files = audio_files[:max_files_per_class]
            print(f"\n[INFO] Capping '{category}' category to first {max_files_per_class} files (out of {total_in_category}).")
            total_in_category = len(audio_files)

        print(f"\n[START] Processing '{category}' speech files ({total_in_category} files)...")

        cat_success = 0
        for idx, src_file in enumerate(audio_files, 1):
            dst_file = dst_category_dir / src_file.name

            success = preprocess_file(
                file_path=str(src_file), 
                output_path=str(dst_file), 
                sample_rate=sample_rate
            )

            if success:
                cat_success += 1
                total_processed += 1
            else:
                total_failed += 1

            # Print progress bar / counter every 50 files or at completion
            if idx % 50 == 0 or idx == total_in_category:
                pct = (idx / total_in_category) * 100
                print(f"  [{category.upper()}] Progress: {idx}/{total_in_category} files ({pct:.1f}%) complete...")

    print("\n" + "="*60)
    print("          PREPROCESSING PIPELINE COMPLETE           ")
    print("="*60)
    print(f"Total Successfully Processed : {total_processed} files")
    print(f"Total Failed Files           : {total_failed} files")
    print(f"Processed Files Saved To     : {base_processed}")
    print("="*60)


def main() -> None:
    """
    Main entry point for command line execution.
    Allows specifying sample rate and optional file processing limits.
    """
    parser = argparse.ArgumentParser(description="Audio Preprocessing Pipeline for Deepfake Detection")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw audio directory (default: data/raw)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed output directory (default: data/processed)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate in Hz (default: 16000)")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files to process per class (e.g. 500 for fast test run)")

    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        sample_rate=args.sr,
        max_files_per_class=args.max_files
    )


if __name__ == "__main__":
    main()
