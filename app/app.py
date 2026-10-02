"""
app.py
------
Flask Web Application Server for AI-Based Audio Deepfake Detection System.

Purpose:
    Provides a user-friendly web interface allowing users to upload audio files (.wav, .mp3, .flac),
    triggers preprocessing and spectral feature extraction on-the-fly, runs inference using the
    trained SVM baseline model, and returns genuine vs. synthetic classification results with
    confidence probabilities.

Architecture Overview:
    1. On startup  -> Load baseline_model.pkl (SVM) and scaler.pkl into memory
    2. GET  /      -> Render the main upload UI (index.html)
    3. POST /predict -> Receive file, preprocess audio, extract 78-dim features, classify, return JSON
    4. GET  /health -> Simple health-check endpoint confirming model is loaded

Imports:
    flask, numpy, os, sys, pathlib, joblib, librosa, soundfile
"""

import os
import sys
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Path Setup: add the project root to sys.path so we can import src/ modules
# ---------------------------------------------------------------------------
# app.py lives at: audio-deepfake-detection/app/app.py
# project root is: audio-deepfake-detection/
APP_DIR = Path(__file__).resolve().parent          # .../app/
PROJECT_ROOT = APP_DIR.parent                       # .../audio-deepfake-detection/
sys.path.insert(0, str(PROJECT_ROOT))              # Allow "import src.xxx"

# Now import our custom feature extraction and preprocessing functions
from src.feature_extraction import extract_mfcc, extract_lfcc, extract_cqcc
from src.preprocessing import load_audio, trim_silence, normalize_audio, pad_or_truncate

# ---------------------------------------------------------------------------
# Flask Application Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = APP_DIR / "static" / "uploads"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

# ---------------------------------------------------------------------------
# Model Cache: Holds loaded model artifacts to avoid re-loading on every request
# ---------------------------------------------------------------------------
MODEL_CACHE: Dict[str, Any] = {}


def allowed_file(filename: str) -> bool:
    """
    Checks whether the uploaded filename has an allowed audio extension.

    Parameters:
        filename (str): The original uploaded filename.

    Returns:
        bool: True if extension is in ALLOWED_EXTENSIONS, False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_prediction_models() -> None:
    """
    Loads the pre-trained SVM baseline model and StandardScaler from disk
    into the global MODEL_CACHE dictionary at application startup.

    Why load at startup:
        joblib.load() reads the pickled sklearn objects from disk — this takes
        a fraction of a second. Pre-loading avoids adding this delay to every
        HTTP prediction request, keeping responses fast.

    Model paths:
        - models/baseline_model.pkl  -> Trained SVM classifier (sklearn Pipeline)
        - models/scaler.pkl          -> StandardScaler fitted on training features
    """
    baseline_path = PROJECT_ROOT / "models" / "baseline_model.pkl"
    scaler_path   = PROJECT_ROOT / "models" / "scaler.pkl"

    if baseline_path.exists():
        MODEL_CACHE["model"] = joblib.load(str(baseline_path))
        print(f"[OK] Loaded baseline model from: {baseline_path}")
    else:
        print(f"[WARNING] Baseline model not found at: {baseline_path}")

    if scaler_path.exists():
        MODEL_CACHE["scaler"] = joblib.load(str(scaler_path))
        print(f"[OK] Loaded scaler from: {scaler_path}")
    else:
        print(f"[WARNING] Scaler not found at: {scaler_path}")


def preprocess_and_extract_features(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Runs the complete audio preprocessing and feature extraction pipeline on an uploaded file.

    Pipeline Steps:
        1. Load & Resample -> Convert to mono 16 kHz waveform
        2. Trim Silence    -> Remove leading/trailing quiet regions
        3. Normalize       -> Scale amplitude to [-1.0, 1.0]
        4. Pad/Truncate    -> Fix length to 32,000 samples (2.0 seconds)
        5. Extract MFCC    -> 13 coefficients x time frames -> mean + std (26 values)
        6. Extract LFCC    -> 13 coefficients x time frames -> mean + std (26 values)
        7. Extract CQCC    -> 13 coefficients x time frames -> mean + std (26 values)
        8. Concatenate     -> 78-dimensional feature vector

    Parameters:
        file_path (str): Path to the uploaded audio file.
        sr (int): Target sampling rate in Hz (default: 16000).

    Returns:
        np.ndarray: 1D float array of shape (78,) representing the feature vector.
    """
    # Step 1: Load audio and resample to 16 kHz
    audio, sample_rate = load_audio(file_path, sample_rate=sr)

    # Step 2: Remove silent regions from beginning and end
    audio = trim_silence(audio, top_db=20)

    # Step 3: Scale amplitude to [-1.0, 1.0] to remove loudness variation
    audio = normalize_audio(audio)

    # Step 4: Standardize duration to exactly 2 seconds (32,000 samples at 16 kHz)
    audio = pad_or_truncate(audio, target_length=32000)

    # Step 5: Extract MFCC features (Mel-Frequency Cepstral Coefficients)
    # Shape: (13, time_frames) -> summarized to mean + std = 26 values
    mfcc_mat = extract_mfcc(audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc_mat, axis=1)
    mfcc_std  = np.std(mfcc_mat, axis=1)

    # Step 6: Extract LFCC features (Linear-Frequency Cepstral Coefficients)
    # Superior for detecting high-frequency artifacts in AI-synthesized audio
    lfcc_mat = extract_lfcc(audio, sr=sample_rate, n_lfcc=13)
    lfcc_mean = np.mean(lfcc_mat, axis=1)
    lfcc_std  = np.std(lfcc_mat, axis=1)

    # Step 7: Extract CQCC features (Constant-Q Cepstral Coefficients)
    # Captures pitch glitches and harmonic anomalies introduced by TTS models
    cqcc_mat = extract_cqcc(audio, sr=sample_rate, n_cqcc=13)
    cqcc_mean = np.mean(cqcc_mat, axis=1)
    cqcc_std  = np.std(cqcc_mat, axis=1)

    # Step 8: Concatenate all statistics into one 78-dimensional feature vector
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,   # 26 values
        lfcc_mean, lfcc_std,   # 26 values
        cqcc_mean, cqcc_std    # 26 values
    ])                          # Total: 78 features

    return feature_vector


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """
    Renders the main web interface homepage.
    Passes model status info to the template for display.
    """
    model_loaded = "model" in MODEL_CACHE and MODEL_CACHE["model"] is not None
    return render_template("index.html", model_loaded=model_loaded)


@app.route("/health", methods=["GET"])
def health():
    """
    Health-check endpoint. Returns JSON indicating whether models are loaded.
    Useful for verifying deployment status without uploading a file.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": "model" in MODEL_CACHE,
        "scaler_loaded": "scaler" in MODEL_CACHE
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Workflow:
        1. Validate that an audio file was included in the POST request
        2. Validate the file extension (.wav, .mp3, .flac, etc.)
        3. Save file temporarily to app/static/uploads/ with a UUID filename
        4. Run full preprocessing + feature extraction pipeline
        5. Apply StandardScaler normalization (same scaler used during training)
        6. Run SVM classifier inference
        7. Return JSON result with prediction label and confidence scores
        8. Clean up the temporary uploaded file

    Returns:
        JSON: {
            "filename": str,
            "prediction": "Real" | "Fake",
            "label": 0 | 1,
            "confidence_real": float,   # Probability [0.0, 1.0]
            "confidence_fake": float,   # Probability [0.0, 1.0]
            "confidence_pct": float,    # Confidence % for the predicted class
            "error": null | str
        }
    """
    # --- Step 1: Validate request contains a file ---
    if "audio" not in request.files:
        return jsonify({"error": "No audio file found in request. Use field name 'audio'."}), 400

    file = request.files["audio"]

    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No file selected."}), 400

    # --- Step 2: Validate file extension ---
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # --- Step 3: Check that models are actually loaded ---
    if "model" not in MODEL_CACHE or "scaler" not in MODEL_CACHE:
        return jsonify({"error": "Prediction models are not loaded. Please restart the server."}), 503

    # --- Step 4: Save uploaded file with a unique name to avoid collisions ---
    original_name = file.filename
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    temp_path = None
    try:
        file.save(save_path)
        temp_path = save_path

        # --- Step 5: Preprocess audio + extract 78-dim feature vector ---
        feature_vector = preprocess_and_extract_features(save_path)

        # Reshape from (78,) to (1, 78) — sklearn expects 2D input (samples x features)
        X = feature_vector.reshape(1, -1)

        # --- Step 6: Apply the same StandardScaler used during training ---
        # This ensures feature values are on the same scale the SVM was trained on
        X_scaled = MODEL_CACHE["scaler"].transform(X)

        # --- Step 7: Run SVM prediction ---
        model = MODEL_CACHE["model"]

        # Get predicted class label: 0 = Real, 1 = Fake
        predicted_label = int(model.predict(X_scaled)[0])

        # Get probability scores for both classes (requires probability=True on SVM)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]  # shape: (2,) -> [P(real), P(fake)]
            confidence_real = float(proba[0])
            confidence_fake = float(proba[1])
        else:
            # Fallback using decision function if predict_proba unavailable
            decision = float(model.decision_function(X_scaled)[0])
            # Sigmoid approximation to convert to probability-like score
            confidence_fake = float(1 / (1 + np.exp(-decision)))
            confidence_real = 1.0 - confidence_fake

        prediction_label = "Fake" if predicted_label == 1 else "Real"
        confidence_pct = round((confidence_fake if predicted_label == 1 else confidence_real) * 100, 2)

        # --- Step 8: Return JSON prediction result ---
        return jsonify({
            "filename": original_name,
            "prediction": prediction_label,
            "label": predicted_label,
            "confidence_real": round(confidence_real * 100, 2),
            "confidence_fake": round(confidence_fake * 100, 2),
            "confidence_pct": confidence_pct,
            "error": None
        })

    except Exception as e:
        # Log the full traceback server-side, but return a clean error message to the client
        print(f"[ERROR] Prediction failed for '{original_name}':")
        traceback.print_exc()
        return jsonify({
            "filename": original_name,
            "prediction": None,
            "error": f"Feature extraction failed: {str(e)}"
        }), 500

    finally:
        # Always clean up the temporary uploaded file after processing
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Starts the Flask development server.

    Steps:
        1. Create the uploads folder if it doesn't exist
        2. Load model artifacts into memory
        3. Start Flask dev server on http://localhost:5000
    """
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    print("=" * 60)
    print("   AI-Based Audio Deepfake Detection System - Web App   ")
    print("=" * 60)
    load_prediction_models()
    print(f"\n[OK] Server starting at http://localhost:5000")
    print("[OK] Press Ctrl+C to stop the server.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
