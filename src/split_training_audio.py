"""
src/split_training_audio.py
----------------------------
Automated Speech Chunking & Dataset Preparation Utility.

Extracts genuine human speech clips from long raw training recordings (~5-6 mins each)
located in `data/raw_training_audio/`, filters out silent/low-energy segments,
normalizes amplitudes, and exports clean 5-second PCM WAV chunks into `data/real_recordings/`.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def split_audio_into_chunks(
    input_file: Path,
    output_dir: Path,
    speaker_id: str,
    target_sr: int = 16000,
    chunk_duration_sec: float = 5.0,
    min_energy_threshold: float = 0.005,
    max_chunks: int = 30
) -> List[Path]:
    """
    Splits a long audio recording into clean, standardized non-silent chunks.
    """
    print(f"\n[Processing] {input_file.name} for speaker '{speaker_id}'...")
    
    # Load and resample to 16kHz mono
    y, sr = librosa.load(str(input_file), sr=target_sr, mono=True)
    
    # Non-silent intervals
    non_silent_intervals = librosa.effects.split(y, top_db=25)
    
    # Concatenate active speech to eliminate long pauses
    if len(non_silent_intervals) > 0:
        y_speech = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    else:
        y_speech = y

    chunk_samples = int(chunk_duration_sec * target_sr)
    total_samples = len(y_speech)
    
    num_possible = total_samples // chunk_samples
    print(f"  -> Total active speech: {total_samples / target_sr:.1f}s ({num_possible} potential {chunk_duration_sec}s chunks)")

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    count = 0
    for i in range(num_possible):
        if count >= max_chunks:
            break

        start = i * chunk_samples
        end = start + chunk_samples
        chunk = y_speech[start:end]

        # Check energy to discard silent or breathing-only chunks
        rms = np.sqrt(np.mean(chunk**2))
        if rms < min_energy_threshold:
            continue

        # Peak normalization (-0.95 to 0.95)
        peak = np.max(np.abs(chunk))
        if peak > 0:
            chunk = chunk / peak * 0.95

        count += 1
        out_filename = f"{speaker_id}_real_{count:04d}.wav"
        out_path = output_dir / out_filename

        sf.write(str(out_path), chunk, target_sr, subtype="PCM_16")
        created_files.append(out_path)

    print(f"  -> Successfully generated {len(created_files)} chunks for '{speaker_id}'")
    return created_files


def run_splitting_pipeline(
    raw_dir: str = "data/raw_training_audio",
    out_dir: str = "data/real_recordings",
    max_chunks_per_speaker: int = 30,
    chunk_sec: float = 5.0
) -> Dict[str, int]:
    raw_path = (PROJECT_ROOT / raw_dir).resolve()
    out_path = (PROJECT_ROOT / out_dir).resolve()

    if not raw_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {raw_path}")

    raw_files = sorted(list(raw_path.glob("*.wav")) + list(raw_path.glob("*.mp3")))
    if not raw_files:
        raise FileNotFoundError(f"No audio files found in: {raw_path}")

    print("=" * 70)
    print("      AUTOMATED TRAINING AUDIO CHUNKER (SPEECH SEGMENTATION)      ")
    print("=" * 70)
    print(f"Source Directory : {raw_path}")
    print(f"Output Directory : {out_path}")
    print(f"Found Recordings : {len(raw_files)}")
    print(f"Chunk Duration   : {chunk_sec} seconds")
    print("=" * 70)

    summary = {}

    for audio_file in raw_files:
        # Determine speaker_id from filename (e.g. teammate01_training.wav -> teammate01)
        name = audio_file.stem
        speaker_id = name.replace("_training", "").replace("_train", "").replace("_raw", "")

        chunks = split_audio_into_chunks(
            input_file=audio_file,
            output_dir=out_path,
            speaker_id=speaker_id,
            chunk_duration_sec=chunk_sec,
            max_chunks=max_chunks_per_speaker
        )
        summary[speaker_id] = len(chunks)

    print("\n" + "=" * 70)
    print("                     EXTRACTION SUMMARY                     ")
    print("=" * 70)
    total_clips = 0
    for spk, count in summary.items():
        print(f"  Speaker: {spk:<15} | Real Chunks Created: {count:>3} clips")
        total_clips += count
    print("-" * 70)
    print(f"  TOTAL REAL HUMAN RECORDINGS GENERATED: {total_clips} clips")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_splitting_pipeline()
