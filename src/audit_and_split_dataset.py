"""
src/audit_and_split_dataset.py
------------------------------
Step 5 - Dataset File Auditor, Unified Metadata Generation, and Speaker-Disjoint Splitter.

Purpose:
    1. Audits all genuine and synthetic audio clips (duration, sample rate, channels, clipping, SNR estimate).
    2. Builds/updates the unified master dataset catalog at `data/metadata.csv`.
    3. Syncs `data/synthetic_rvc/metadata.csv` to include all teammate voices.
    4. Implements strict SPEAKER-DISJOINT splitting (Train / Validation / Test) to eliminate
       speaker identity memorization and ensure unbiased anti-spoofing evaluation.

Splitting Strategy:
    - Train (3 speakers): teammate01, teammate02, teammate03 (genuine speech + RVC + XTTS deepfakes)
    - Validation (1 speaker): teammate04 (genuine speech + RVC deepfakes)
    - Test (1 unseen held-out speaker): teammate05 (genuine speech + RVC deepfakes) + live mic test
    * Zero speaker leakage: Neither genuine voice nor voice clones of teammate05 exist in train or val!
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Standard benchmark test sentences
TEST_SENTENCES_FILE = PROJECT_ROOT / "data" / "test_sentences.txt"
if TEST_SENTENCES_FILE.is_file():
    with open(TEST_SENTENCES_FILE, "r", encoding="utf-8") as f:
        SENTENCES = [l.strip() for l in f if l.strip() and not l.startswith("#")]
else:
    SENTENCES = [
        "This is a security verification message confirming access to the restricted terminal.",
        "The machine learning algorithm is currently analyzing the acoustic frequency spectrum.",
        "Please state your full name and authorization code for biometric verification.",
        "Our project investigates how voice conversion systems can challenge audio deepfake detectors.",
        "Synthetic speech generation requires evaluating both spectral features and temporal artifacts.",
        "Artificial intelligence models must be thoroughly tested against adversarial vocal attacks."
    ]


def audit_audio_clip(file_path: Path) -> Dict[str, Any]:
    """
    Audits an audio file for physical acoustic integrity.
    Measures duration, sample rate, channels, digital clipping, and estimated SNR.
    """
    try:
        info = sf.info(str(file_path))
        native_sr = info.samplerate
        channels = info.channels
        duration = info.duration

        # Load for signal-level metrics
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
        if len(y) == 0:
            return {
                "duration_sec": 0.0,
                "sample_rate": native_sr,
                "channels": channels,
                "clipping_ratio": 1.0,
                "snr_db": -99.0,
                "status": "REJECT_EMPTY"
            }

        # 1. Digital clipping check (samples near ±1.0)
        clipping_ratio = float(np.mean(np.abs(y) >= 0.99))

        # 2. SNR Estimation (Speech vs Background Noise Floor)
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        if len(y) > frame_len:
            frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
            frame_rms = np.sqrt(np.mean(frames**2, axis=0))
            sorted_rms = np.sort(frame_rms)
            speech_rms = np.mean(sorted_rms[int(len(sorted_rms) * 0.5):])
            noise_rms = np.mean(sorted_rms[:max(1, int(len(sorted_rms) * 0.1))])
            snr_db = float(20 * np.log10((speech_rms + 1e-6) / (noise_rms + 1e-6)))
        else:
            snr_db = 15.0

        # Quality verdict
        status = "PASS"
        if duration < 0.8:
            status = "FLAG_SHORT"
        elif clipping_ratio > 0.02:
            status = "FLAG_CLIPPED"
        elif snr_db < 8.0:
            status = "FLAG_NOISY"

        return {
            "duration_sec": round(duration, 2),
            "sample_rate": native_sr,
            "channels": channels,
            "clipping_ratio": round(clipping_ratio * 100, 3),
            "snr_db": round(snr_db, 1),
            "status": status
        }
    except Exception as e:
        return {
            "duration_sec": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "clipping_ratio": 0.0,
            "snr_db": -99.0,
            "status": f"ERROR: {str(e)[:30]}"
        }


def assign_speaker_disjoint_split(speaker_id: str) -> str:
    """
    Enforces strict speaker-disjoint splitting.
    A speaker's real and synthetic audio always reside strictly in the same split.
    """
    if speaker_id in ["teammate01", "teammate02", "teammate03"]:
        return "train"
    elif speaker_id in ["teammate04"]:
        return "val"
    else:  # teammate05 (held-out female speaker) + test_voice_1 (live mic)
        return "test"


def build_unified_metadata() -> pd.DataFrame:
    """
    Audits all audio directories and compiles the unified master metadata table.
    """
    records = []

    # 1. Real Recordings (data/real_recordings/)
    real_dir = PROJECT_ROOT / "data" / "real_recordings"
    if real_dir.is_dir():
        for f in sorted(real_dir.glob("*.wav")):
            audit = audit_audio_clip(f)
            fname = f.name
            
            # Parse speaker
            if fname.startswith("teammate"):
                speaker = fname.split("_")[0]
                cond = "quiet_room_16k"
            elif "test_voice" in fname or "live" in fname:
                speaker = "test_voice_1"
                cond = "live_microphone_browser"
            else:
                speaker = fname.split("_")[0]
                cond = "standard_speech"

            split = assign_speaker_disjoint_split(speaker)
            
            records.append({
                "file": str(Path("data/real_recordings") / fname).replace("\\", "/"),
                "filename": fname,
                "speaker_id": speaker,
                "label": 0,
                "class_name": "Real",
                "source": "data/real_recordings",
                "generation_method": "genuine_human",
                "model_used": "human_vocal_tract",
                "text": "genuine natural spoken sentence",
                "recording_condition": cond,
                "split": split,
                **audit
            })

    # 2. Synthetic RVC Clones (data/synthetic_rvc/)
    rvc_dir = PROJECT_ROOT / "data" / "synthetic_rvc"
    if rvc_dir.is_dir():
        for f in sorted(rvc_dir.glob("*.wav")):
            audit = audit_audio_clip(f)
            fname = f.name
            speaker = fname.split("_")[0]
            
            # Sentence mapping from 0001..0006
            try:
                num = int(fname.split("_")[-1].replace(".wav", ""))
                idx = (num - 1) % len(SENTENCES)
                sent_text = SENTENCES[idx]
            except Exception:
                sent_text = "Standard benchmark evaluation sentence"

            split = assign_speaker_disjoint_split(speaker)

            records.append({
                "file": str(Path("data/synthetic_rvc") / fname).replace("\\", "/"),
                "filename": fname,
                "speaker_id": speaker,
                "label": 1,
                "class_name": "Fake",
                "source": "data/synthetic_rvc",
                "generation_method": "applio_rvc_v2",
                "model_used": f"{speaker}_rvc_100e",
                "text": sent_text,
                "recording_condition": "synthetic_rvc_neural_vocoder",
                "split": split,
                **audit
            })

    # 3. Synthetic XTTS-v2 Clones (data/cloned_samples/)
    xtts_dir = PROJECT_ROOT / "data" / "cloned_samples"
    if xtts_dir.is_dir():
        for f in sorted(xtts_dir.glob("*.wav")):
            audit = audit_audio_clip(f)
            fname = f.name
            speaker = "teammate01"  # XTTS clones generated for teammate01
            split = assign_speaker_disjoint_split(speaker)

            records.append({
                "file": str(Path("data/cloned_samples") / fname).replace("\\", "/"),
                "filename": fname,
                "speaker_id": speaker,
                "label": 1,
                "class_name": "Fake",
                "source": "data/cloned_samples",
                "generation_method": "coqui_xtts_v2",
                "model_used": "xtts_v2_zero_shot",
                "text": "XTTS zero-shot voice clone sentence",
                "recording_condition": "synthetic_xtts_autoregressive",
                "split": split,
                **audit
            })

    df = pd.DataFrame(records)
    return df


def update_synthetic_rvc_metadata(df_master: pd.DataFrame) -> None:
    """Updates data/synthetic_rvc/metadata.csv so all 30 RVC clips are recorded."""
    rvc_df = df_master[df_master["source"] == "data/synthetic_rvc"].copy()
    out_path = PROJECT_ROOT / "data" / "synthetic_rvc" / "metadata.csv"
    
    export_df = pd.DataFrame({
        "file": rvc_df["filename"],
        "speaker_id": rvc_df["speaker_id"],
        "label": 1,
        "text": rvc_df["text"],
        "model_used": rvc_df["model_used"],
        "pipeline": "tts_rvc",
        "source_audio": "tts_" + rvc_df["filename"],
        "generated_at": "2026-11-11T12:00:00+00:00"
    })
    export_df.to_csv(out_path, index=False)
    print(f"[OK] Updated {out_path} with {len(export_df)} RVC entries.")


def main():
    print("=" * 75)
    print("   DFADS - STEP 5: DATASET EXPANSION, AUDIT & SPEAKER-DISJOINT SPLIT   ")
    print("=" * 75)

    df_master = build_unified_metadata()
    master_path = PROJECT_ROOT / "data" / "metadata.csv"
    df_master.to_csv(master_path, index=False)
    print(f"\n[OK] Master metadata catalog written to: {master_path}")
    print(f"     Total Audited Audio Clips: {len(df_master)}")

    # Update synthetic_rvc metadata
    update_synthetic_rvc_metadata(df_master)

    # 1. Audit Summary
    print("\n" + "-" * 40)
    print("1. AUDIO QUALITY AUDIT SUMMARY")
    print("-" * 40)
    status_counts = df_master["status"].value_counts().to_dict()
    for status, count in status_counts.items():
        print(f"  - {status:<15}: {count} clips ({count/len(df_master)*100:.1f}%)")

    # 2. Per-Speaker Counts
    print("\n" + "-" * 40)
    print("2. PER-SPEAKER DISTRIBUTION")
    print("-" * 40)
    speaker_table = pd.crosstab(df_master["speaker_id"], df_master["class_name"], margins=True)
    print(speaker_table)

    # 3. Split Distribution (Speaker-Disjoint)
    print("\n" + "-" * 40)
    print("3. SPEAKER-DISJOINT SPLIT DISTRIBUTION")
    print("-" * 40)
    split_table = pd.crosstab(df_master["split"], df_master["class_name"], margins=True)
    print(split_table)

    print("\n" + "-" * 40)
    print("4. SPEAKER PARTITIONING VERIFICATION (LEAKAGE AUDIT)")
    print("-" * 40)
    train_speakers = set(df_master[df_master["split"] == "train"]["speaker_id"])
    val_speakers = set(df_master[df_master["split"] == "val"]["speaker_id"])
    test_speakers = set(df_master[df_master["split"] == "test"]["speaker_id"])
    
    print(f"  Train Speakers     : {sorted(list(train_speakers))}")
    print(f"  Validation Speakers: {sorted(list(val_speakers))}")
    print(f"  Test Speakers      : {sorted(list(test_speakers))}")

    leakage_train_test = train_speakers.intersection(test_speakers)
    leakage_train_val  = train_speakers.intersection(val_speakers)
    leakage_val_test   = val_speakers.intersection(test_speakers)

    if not leakage_train_test and not leakage_train_val and not leakage_val_test:
        print("\n  [PASS] ZERO SPEAKER LEAKAGE VERIFIED: Speakers are 100% disjoint.")
    else:
        print(f"\n  [FAIL] LEAKAGE DETECTED: {leakage_train_test | leakage_train_val | leakage_val_test}")

    # 4. Diversity breakdown
    print("\n" + "-" * 40)
    print("5. ACOUSTIC DIVERSITY BREAKDOWN")
    print("-" * 40)
    cond_counts = df_master["recording_condition"].value_counts().to_dict()
    for cond, c in cond_counts.items():
        print(f"  - {cond:<32}: {c} clips")


if __name__ == "__main__":
    main()
