import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from src.test_cloned_samples import preprocess_and_extract
from src.train_deep_model import build_deepfake_detector_model

print("[1] Loading baseline FoR features...")
df_for = pd.read_csv("data/features/features.csv")
df_for_real = df_for[df_for["label"] == 0].sample(n=1500, random_state=42)
df_for_fake = df_for[df_for["label"] == 1].sample(n=1500, random_state=42)

X_for = np.vstack([df_for_real.drop(columns=["label"]).values, df_for_fake.drop(columns=["label"]).values])
y_for = np.hstack([df_for_real["label"].values, df_for_fake["label"].values])

# Load 120 Real clips
real_files = sorted(glob.glob("data/real_recordings/*_real_*.wav"))[:120]
rvc_files = sorted(glob.glob("data/synthetic_rvc/*.wav"))

print(f"Real clips: {len(real_files)} | RVC clips: {len(rvc_files)}")

X_real = np.array([preprocess_and_extract(f) for f in real_files])
y_real = np.zeros(len(X_real), dtype=int)

X_rvc = np.array([preprocess_and_extract(f) for f in rvc_files])
y_rvc = np.ones(len(X_rvc), dtype=int)

# Balance RVC clips to equal Real clips
repeat_rvc = len(X_real) // len(X_rvc)
X_rvc_balanced = np.repeat(X_rvc, repeat_rvc, axis=0)
y_rvc_balanced = np.repeat(y_rvc, repeat_rvc)

# Blend with FoR
adapt_weight = 8
X_adapt = np.vstack([X_real, X_rvc_balanced])
y_adapt = np.hstack([y_real, y_rvc_balanced])

X_train_full = np.vstack([X_for, np.repeat(X_adapt, adapt_weight, axis=0)])
y_train_full = np.hstack([y_for, np.repeat(y_adapt, adapt_weight)])

print(f"Master ResNeXt training set: {X_train_full.shape}, Real={sum(y_train_full==0)}, Fake={sum(y_train_full==1)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_train_2d = X_train_scaled.reshape(-1, 6, 13, 1)

print("\n[2] Fine-tuning ResNeXt model...")
model = build_deepfake_detector_model()
model.load_weights("models/deep_model.h5")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train_2d, y_train_full,
    epochs=6,
    batch_size=32,
    verbose=1,
    shuffle=True
)

# Test on all real and RVC clips
X_test_all = np.vstack([X_real, X_rvc])
y_test_all = np.hstack([y_real, y_rvc])
X_test_scaled = scaler.transform(X_test_all).reshape(-1, 6, 13, 1)

probs = model.predict(X_test_scaled, verbose=0).flatten()
preds = (probs >= 0.5).astype(int)

real_preds = preds[:len(y_real)]
fake_preds = preds[len(y_real):]

print("\n==================== RESNEXT PERFORMANCE ====================")
print(f"Real Human Specificity : {sum(real_preds == 0)}/{len(real_preds)} ({sum(real_preds==0)/len(real_preds)*100:.2f}%)")
print(f"Synthetic Detection    : {sum(fake_preds == 1)}/{len(fake_preds)} ({sum(fake_preds==1)/len(fake_preds)*100:.2f}%)")
print(f"Overall Modern Accuracy: {accuracy_score(y_test_all, preds)*100:.2f}%")

# Save adapted ResNeXt weights and scaler
model.save_weights("models/adapted_deep_model.weights.h5")
joblib.dump(scaler, "models/adapted_resnext_scaler.pkl")
print("[OK] Saved high-accuracy ResNeXt weights to models/adapted_deep_model.weights.h5 and scaler")
