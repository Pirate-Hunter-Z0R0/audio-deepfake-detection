"""
src/train_speaker_disjoint_resnext.py
-------------------------------------
Step 5 - Retraining ResNeXt with Speaker-Disjoint Partitioning.

Enforces zero speaker leakage:
  - Train: teammate01, teammate02, teammate03 (blended with FoR benchmark)
  - Validation: teammate04
  - Test: teammate05 (Held-out unseen speaker) + live mic test

Usage:
  python src/train_speaker_disjoint_resnext.py
"""

import os
import sys

# Critical Windows DLL initialization: configure TF runtime prior to other scientific imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.test_cloned_samples import preprocess_and_extract
from src.train_deep_model import build_deepfake_detector_model


def extract_features_for_files(file_list: list) -> np.ndarray:
    """Extracts 78-dim acoustic vectors for a list of file paths."""
    features = []
    for f in file_list:
        full_path = PROJECT_ROOT / f if not Path(f).is_absolute() else Path(f)
        vec = preprocess_and_extract(str(full_path))
        features.append(vec)
    return np.array(features)


def main():
    print("=" * 75)
    print("   DFADS - STEP 5: SPEAKER-DISJOINT RETRAINING (RESNEXT CNN)   ")
    print("=" * 75)

    metadata_path = PROJECT_ROOT / "data" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("Run src/audit_and_split_dataset.py first to generate data/metadata.csv")

    df_meta = pd.read_csv(metadata_path)

    # 1. Split Partitioning from metadata.csv
    train_meta = df_meta[df_meta["split"] == "train"]
    val_meta   = df_meta[df_meta["split"] == "val"]
    test_meta  = df_meta[df_meta["split"] == "test"]

    print(f"\n[1] Dataset Partitions:")
    print(f"  - Train Partition : {len(train_meta)} clips (Speakers: {sorted(train_meta['speaker_id'].unique())})")
    print(f"  - Val Partition   : {len(val_meta)} clips (Speakers: {sorted(val_meta['speaker_id'].unique())})")
    print(f"  - Test Partition  : {len(test_meta)} clips (Speakers: {sorted(test_meta['speaker_id'].unique())})")

    # 2. Extract Acoustic Features for Each Partition
    print("\n[2] Extracting 78-dim features for partitions...")
    X_train_teammates = extract_features_for_files(train_meta["file"].tolist())
    y_train_teammates = train_meta["label"].values.astype(int)

    X_val = extract_features_for_files(val_meta["file"].tolist())
    y_val = val_meta["label"].values.astype(int)

    X_test = extract_features_for_files(test_meta["file"].tolist())
    y_test = test_meta["label"].values.astype(int)

    # 3. Load FoR Benchmark Foundation (to preserve broad acoustic generalization)
    for_path = PROJECT_ROOT / "data" / "features" / "features.csv"
    if for_path.exists():
        print("  -> Blending with FoR benchmark foundation...")
        df_for = pd.read_csv(for_path)
        df_for_real = df_for[df_for["label"] == 0].sample(n=1500, random_state=42)
        df_for_fake = df_for[df_for["label"] == 1].sample(n=1500, random_state=42)
        X_for = np.vstack([df_for_real.drop(columns=["label"]).values, df_for_fake.drop(columns=["label"]).values])
        y_for = np.hstack([df_for_real["label"].values, df_for_fake["label"].values])
        
        # Up-sample teammate adaptation data to balance with FoR
        adapt_weight = 10
        X_train_full = np.vstack([X_for, np.repeat(X_train_teammates, adapt_weight, axis=0)])
        y_train_full = np.hstack([y_for, np.repeat(y_train_teammates, adapt_weight)])
    else:
        print("  -> Training directly on teammate partition...")
        X_train_full = X_train_teammates
        y_train_full = y_train_teammates

    # 4. Standard Scaling (Fit exclusively on Train)
    print("\n[3] Fitting StandardScaler exclusively on Train split...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Reshape to ResNeXt 2D Grouped Convolution Grid: (Batch, 6, 13, 1)
    X_train_2d = X_train_scaled.reshape(-1, 6, 13, 1)
    X_val_2d   = X_val_scaled.reshape(-1, 6, 13, 1)
    X_test_2d  = X_test_scaled.reshape(-1, 6, 13, 1)

    # 5. Build and Compile ResNeXt
    print("\n[4] Initializing Deep ResNeXt CNN...")
    model = build_deepfake_detector_model()
    base_weights = PROJECT_ROOT / "models" / "deep_model.h5"
    if base_weights.exists():
        model.load_weights(str(base_weights))
        print("  -> Initialized from pre-trained FoR weights.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print("\n[5] Training ResNeXt (Speaker-Disjoint: validating on teammate04)...")
    history = model.fit(
        X_train_2d,
        y_train_full,
        validation_data=(X_val_2d, y_val),
        epochs=25,
        batch_size=16,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluation on Unseen Held-Out Test Speaker (teammate05)
    print("\n" + "=" * 75)
    print("   EVALUATION ON UNSEEN HELD-OUT SPEAKER (teammate05)   ")
    print("=" * 75)
    test_probs = model.predict(X_test_2d, verbose=0).flatten()
    test_preds = (test_probs >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 1.0

    print(f"\nHeld-Out Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Held-Out Test ROC AUC  : {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=["Real", "Fake"]))

    # 7. Save Checkpoints
    out_weights = PROJECT_ROOT / "models" / "adapted_deep_model.weights.h5"
    out_scaler  = PROJECT_ROOT / "models" / "adapted_resnext_scaler.pkl"
    model.save_weights(str(out_weights))
    joblib.dump(scaler, str(out_scaler))
    print(f"\n[OK] Saved updated ResNeXt weights to : {out_weights}")
    print(f"[OK] Saved updated StandardScaler to   : {out_scaler}")


if __name__ == "__main__":
    main()
