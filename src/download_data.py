r"""
download_data.py
----------------
Dataset Download, Extraction, Organization, and Inspection Script.

Handles Fake-or-Real (FoR) datasets (e.g. for-2sec.zip, for-norm.zip, for-original.zip)
or generates synthetic demo audio clips for testing.
"""

import os
import sys
import glob
import shutil
import zipfile
import wave
import math
import struct
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

# Define target raw data directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
REAL_DIR = RAW_DATA_DIR / "real"
FAKE_DIR = RAW_DATA_DIR / "fake"


def extract_zip_if_present() -> None:
    """
    Searches for any .zip files inside data/raw/ and extracts them.
    """
    print("\nStep 1: Checking for ZIP archives in data/raw/...")
    zip_files = list(RAW_DATA_DIR.glob("*.zip"))
    
    if not zip_files:
        print("No .zip files found to extract.")
        return

    for zip_path in zip_files:
        print(f"Found archive '{zip_path.name}'. Extracting to '{RAW_DATA_DIR}'...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            print(f"Successfully extracted '{zip_path.name}'.")
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")


def clean_demo_samples_if_real_dataset_found() -> None:
    """
    Removes synthetic demo samples (e.g. real_sample_001.wav, fake_sample_001.wav)
    if real benchmark files are being processed.
    """
    demo_real = list(REAL_DIR.glob("real_sample_*.wav"))
    demo_fake = list(FAKE_DIR.glob("fake_sample_*.wav"))

    if demo_real or demo_fake:
        print("Removing temporary demo synthetic samples to make room for benchmark dataset...")
        for f in demo_real + demo_fake:
            try:
                f.unlink()
            except Exception:
                pass


def organize_dataset() -> Tuple[int, int]:
    """
    Scans data/raw/ for extracted audio files and reorganizes them into
    data/raw/real/ and data/raw/fake/ subdirectories.
    """
    print("\nStep 2: Organizing extracted files into data/raw/real/ and data/raw/fake/...")
    
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.ogg")
    
    found_audio_files = []
    for ext in AUDIO_EXTENSIONS:
        found_audio_files.extend(RAW_DATA_DIR.rglob(ext))

    # Check if there are benchmark files outside data/raw/real and data/raw/fake
    external_files = [
        f for f in found_audio_files 
        if f.parent != REAL_DIR and f.parent != FAKE_DIR
    ]

    if external_files:
        clean_demo_samples_if_real_dataset_found()

    real_count = 0
    fake_count = 0

    for file_path in found_audio_files:
        # If file is already directly inside data/raw/real or data/raw/fake
        if file_path.parent == REAL_DIR:
            real_count += 1
            continue
        elif file_path.parent == FAKE_DIR:
            fake_count += 1
            continue

        path_parts = [p.lower() for p in file_path.parts]
        
        # Check folder structure for real vs fake
        is_real = "real" in path_parts or "bonafide" in path_parts or "human" in path_parts
        is_fake = "fake" in path_parts or "spoof" in path_parts or "synthetic" in path_parts

        dest_dir = None
        if is_real and not is_fake:
            dest_dir = REAL_DIR
            real_count += 1
        elif is_fake and not is_real:
            dest_dir = FAKE_DIR
            fake_count += 1

        if dest_dir:
            target_path = dest_dir / file_path.name
            # Resolve name collisions across train/test/val splits
            if target_path.exists():
                split_prefix = "_".join([p for p in file_path.parts[-3:-1] if p.lower() not in ["real", "fake"]])
                target_path = dest_dir / f"{split_prefix}_{file_path.name}" if split_prefix else dest_dir / f"{file_path.parent.name}_{file_path.name}"
            
            shutil.move(str(file_path), str(target_path))

    print(f"Organization Complete: {real_count} real files, {fake_count} fake files.")
    return real_count, fake_count


def remove_empty_extracted_folders() -> None:
    """
    Cleans up empty extracted nested folders under data/raw/.
    """
    for item in RAW_DATA_DIR.iterdir():
        if item.is_dir() and item not in (REAL_DIR, FAKE_DIR):
            try:
                shutil.rmtree(item)
            except Exception:
                pass


def inspect_audio_metadata(file_paths: List[Path], max_inspect: int = 500) -> Tuple[float, List[int]]:
    """
    Reads sample rate and duration from audio headers.
    Inspects up to max_inspect samples for fast processing.
    """
    total_duration = 0.0
    sample_rates = []
    valid_files_count = 0

    sample_subset = file_paths[:max_inspect] if len(file_paths) > max_inspect else file_paths

    for path in sample_subset:
        try:
            if path.suffix.lower() == ".wav":
                with wave.open(str(path), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    total_duration += duration
                    sample_rates.append(rate)
                    valid_files_count += 1
            else:
                import soundfile as sf
                info = sf.info(str(path))
                total_duration += info.duration
                sample_rates.append(info.samplerate)
                valid_files_count += 1
        except Exception:
            continue

    avg_duration = (total_duration / valid_files_count) if valid_files_count > 0 else 0.0
    return avg_duration, sample_rates


def print_dataset_summary() -> None:
    """
    Computes and displays dataset statistics.
    """
    print("\n" + "="*50)
    print("            DATASET SUMMARY REPORT            ")
    print("="*50)

    real_files = list(REAL_DIR.glob("*.*"))
    fake_files = list(FAKE_DIR.glob("*.*"))

    total_real = len(real_files)
    total_fake = len(fake_files)
    total_files = total_real + total_fake

    print(f"Real Speech Clips : {total_real} files")
    print(f"Fake Speech Clips : {total_fake} files")
    print(f"Total Dataset Size: {total_files} files")

    if total_files == 0:
        print("\n[NOTE] No files found in data/raw/real/ or data/raw/fake/.")
        print("="*50)
        return

    all_files = real_files + fake_files
    avg_duration, sample_rates = inspect_audio_metadata(all_files)
    unique_sample_rates = sorted(list(set(sample_rates))) if sample_rates else ["Unknown"]

    print(f"Average Duration  : {avg_duration:.2f} seconds")
    print(f"Sample Rates (Hz) : {unique_sample_rates}")
    print("="*50)


def main() -> None:
    print("=== AI-Based Audio Deepfake Detection: Data Preparation ===")
    extract_zip_if_present()
    organize_dataset()
    remove_empty_extracted_folders()
    print_dataset_summary()


if __name__ == "__main__":
    main()
