"""
voice_generation/generate_dataset.py
------------------------------------
Batch generation script for synthetic deepfake speech using the Two-Stage
(Edge-TTS + Applio RVC) pipeline.

Generates audio samples for each (sentence, model) pair, assigns consistent
filenames, and logs all metadata to metadata.csv for downstream detector evaluation.

Usage:
    python voice_generation/generate_dataset.py --sentences-file data/test_sentences.txt --limit 2
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_generation.applio_wrapper import ApplioWrapper


def load_sentences(filepath: Path) -> List[str]:
    """Loads sentences from a text file (one sentence per line)."""
    if not filepath.is_file():
        raise FileNotFoundError(f"Sentences file not found at: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not lines:
        raise ValueError(f"Sentences file is empty: {filepath}")
    
    return lines


def generate_dataset(
    sentences_file: str,
    models: List[str],
    limit: int = 0,
    config_path: str = "config.yaml",
    output_dir: str = "",
    metadata_path: str = "",
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Executes batch two-stage voice generation across specified models and sentences.

    Args:
        sentences_file (str): Path to input text file containing sentences.
        models (List[str]): List of model identifiers to generate for (e.g. ['teammate01', 'teammate02']).
        limit (int): Max number of sentences per model (0 = all sentences).
        config_path (str): Path to config.yaml.
        output_dir (str): Destination folder for final synthetic WAVs.
        metadata_path (str): Path to save metadata.csv.
        overwrite (bool): Whether to regenerate existing WAV files.

    Returns:
        pd.DataFrame: Completed metadata table.
    """
    wrapper = ApplioWrapper(config_path=config_path)
    
    # Dynamic Model Resolution: support 'auto' or 'all' to pick up newly trained models
    if not models or (len(models) == 1 and models[0].lower() in ["auto", "all"]):
        models = wrapper.get_available_models()
        print(f"[INFO] Dynamically auto-discovered {len(models)} model(s): {models}")

    if not models:
        raise ValueError(f"No voice models found or specified in {wrapper.models_dir}!")

    # Resolve directories
    out_dir = Path(output_dir) if output_dir else wrapper.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_file = Path(metadata_path) if metadata_path else out_dir / "metadata.csv"
    sentences = load_sentences(Path(sentences_file))

    if limit > 0:
        sentences = sentences[:limit]
        print(f"[INFO] Applied limit: generating {len(sentences)} sentence(s) per model.")

    # Validate all requested models before starting
    print("\n--- Validating models ---")
    for m in models:
        pth, idx = wrapper.validate_model(m)
        print(f"[OK] Verified {m}: {pth.name}")

    # Load existing metadata if present
    existing_meta: Dict[str, Dict[str, Any]] = {}
    if meta_file.is_file():
        try:
            df_old = pd.read_csv(meta_file)
            for _, row in df_old.iterrows():
                existing_meta[row["file"]] = row.to_dict()
            print(f"[INFO] Loaded {len(existing_meta)} existing record(s) from {meta_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not read existing {meta_file.name}: {e}. Starting fresh.")

    records = list(existing_meta.values())
    total_planned = len(models) * len(sentences)
    current_count = 0

    print(f"\n--- Starting generation ({total_planned} total samples planned) ---")

    for m_idx, model_name in enumerate(models, start=1):
        for s_idx, sentence in enumerate(sentences, start=1):
            current_count += 1
            # Consistent filename pattern: e.g. teammate01_rvc_0001.wav
            filename = f"{model_name}_rvc_{s_idx:04d}.wav"
            final_wav_path = out_dir / filename

            # Skip if already exists and not overwrite
            if final_wav_path.is_file() and final_wav_path.stat().st_size > 0 and not overwrite:
                print(f"[{current_count}/{total_planned}] Skipping existing: {filename}")
                continue

            print(f"\n[{current_count}/{total_planned}] Generating ({model_name}): \"{sentence[:45]}...\"")
            
            # Execute two-stage pipeline
            res = wrapper.text_to_rvc(
                text=sentence,
                model_name=model_name,
                out_path=str(final_wav_path)
            )

            # Record metadata
            record = {
                "file": filename,
                "speaker_id": model_name,
                "label": 1,  # 1 = synthetic / deepfake
                "text": sentence,
                "model_used": model_name,
                "pipeline": "tts_rvc",
                "source_audio": Path(res["intermediate_tts_path"]).name,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Update or append
            # Remove any existing record for this filename
            records = [r for r in records if r.get("file") != filename]
            records.append(record)

            # Save metadata incrementally after each generation for crash resilience
            df_meta = pd.DataFrame(records)
            df_meta.to_csv(meta_file, index=False)
            print(f"      -> Saved {filename} & updated {meta_file.name}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(meta_file, index=False)
    
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Dataset generation complete! Total records: {len(df_final)}")
    print(f"Output Directory: {out_dir}")
    print(f"Metadata File   : {meta_file}")
    print("=" * 70)
    
    return df_final


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic deepfake dataset using Applio RVC.")
    parser.add_argument("--sentences-file", type=str, default="data/test_sentences.txt",
                        help="Path to text file containing sentences.")
    parser.add_argument("--models", nargs="+", default=["auto"],
                        help="List of models to generate audio for, or 'auto' to automatically detect all trained models in models_dir.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of sentences per model (e.g. 2 for initial test).")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml.")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for synthetic audio.")
    parser.add_argument("--metadata-path", type=str, default="",
                        help="Path to save metadata.csv.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing WAV files.")

    args = parser.parse_args()

    generate_dataset(
        sentences_file=args.sentences_file,
        models=args.models,
        limit=args.limit,
        config_path=args.config,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
