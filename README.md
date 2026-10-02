# AI-Based Audio Deepfake Detection System

The **AI-Based Audio Deepfake Detection System** (AudioShield) is an academic project that classifies audio clips as either **genuine human speech** or **AI-generated synthetic (deepfake) speech**. The system extracts spectral features — Mel-Frequency Cepstral Coefficients (MFCC), Linear-Frequency Cepstral Coefficients (LFCC), and Constant-Q Cepstral Coefficients (CQCC) — to train both classical machine learning baselines (SVM, Random Forest, Logistic Regression) and a lightweight ResNeXt-inspired CNN architecture. Robustness testing evaluates detector resilience against environmental audio degradations (noise, compression, pitch shifts), drawing from adversarial attack research. A Flask web application provides an intuitive interface for real-time audio deepfake detection.

---

## 📁 Project Folder Structure

```text
audio-deepfake-detection/
├── data/
│   ├── raw/                    # Original FoR-2sec dataset (.wav files)
│   │   ├── real/               # 17,870 genuine speech clips
│   │   └── fake/               # 17,870 synthetic/deepfake clips
│   ├── processed/              # Preprocessed audio (resampled, trimmed, normalized)
│   │   ├── real/
│   │   └── fake/
│   └── features/               # Extracted spectral feature arrays
│       ├── features.csv        # 78-dim feature vectors + labels (35,740 rows)
│       ├── features.npy        # NumPy binary format of the same data
│       └── test_split.csv      # Held-out 20% test split for reproducible evaluation
├── src/
│   ├── download_data.py        # Downloads and organizes the FoR-2sec dataset
│   ├── preprocessing.py        # Audio loading, resampling, silence trimming, normalization
│   ├── feature_extraction.py   # Spectral feature extraction (MFCC, LFCC, CQCC)
│   ├── train_baseline.py       # Classical ML training (Logistic Regression, SVM, Random Forest)
│   ├── train_deep_model.py     # ResNeXt-inspired CNN training (TensorFlow/Keras)
│   ├── evaluate.py             # Metrics (Accuracy, Precision, Recall, F1, EER) & visualizations
│   └── robustness_test.py      # Noise/compression/pitch-shift degradation testing
├── models/
│   ├── baseline_model.pkl      # Best classical ML model (SVM with RBF kernel)
│   ├── scaler.pkl              # StandardScaler fitted on training features
│   └── deep_model.h5           # Trained ResNeXt-inspired CNN model
├── app/
│   ├── app.py                  # Flask server and prediction API
│   ├── templates/
│   │   └── index.html          # Web interface (drag-and-drop upload UI)
│   └── static/                 # Static assets and uploaded audio files
├── results/
│   ├── baseline_comparison.csv # Accuracy/Precision/Recall/F1 for all baseline models
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   ├── roc_curve.png           # ROC curve with AUC score
│   ├── training_curves.png     # Deep model training/validation accuracy & loss curves
│   ├── robustness_results.csv  # Performance under degraded audio conditions
│   ├── robustness_chart.png    # Bar chart of robustness results
│   ├── metrics.txt             # Full evaluation report
│   └── evaluation_metrics.json # Machine-readable metrics
├── notebooks/                  # Jupyter notebooks for EDA and experiments
├── docs/                       # Reference paper notes and documentation
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
```

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.10 – 3.12** (TensorFlow does not yet support Python 3.13+)
- **pip** package manager
- ~2 GB disk space for the FoR-2sec dataset

### 1. Clone or Navigate to the Project

```bash
cd audio-deepfake-detection
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Full Pipeline

Execute these commands **from the project root directory** (`audio-deepfake-detection/`):

### Step 1: Download the Dataset

```bash
python src/download_data.py
```
Downloads the **FoR-2sec** dataset (Fake-or-Real, 2-second clips — 35,740 total WAV files) and organizes it into `data/raw/real/` and `data/raw/fake/`.

### Step 2: Preprocess Audio

```bash
python src/preprocessing.py
```
Resamples all audio to **16 kHz**, trims silence, normalizes amplitude, and pads/truncates to a uniform **32,000 samples** (2 seconds). Saves to `data/processed/`.

### Step 3: Extract Features

```bash
python src/feature_extraction.py
```
Extracts **78-dimensional feature vectors** (MFCC + LFCC + CQCC means and standard deviations) from each preprocessed audio file. Saves to `data/features/features.csv`.

### Step 4: Train Baseline Models

```bash
python src/train_baseline.py
```
Trains **Logistic Regression**, **SVM (RBF kernel)**, and **Random Forest** classifiers. Saves the best model to `models/baseline_model.pkl` and comparison metrics to `results/baseline_comparison.csv`.

### Step 5: Train Deep Learning Model

```bash
python src/train_deep_model.py
```
Trains a **ResNeXt-inspired lightweight CNN** using TensorFlow/Keras. Reshapes 78-dim vectors into 6×13 2D grids for convolutional processing. Saves to `models/deep_model.h5` with training curves to `results/training_curves.png`.

### Step 6: Evaluate Models

```bash
# Evaluate the baseline SVM model
python src/evaluate.py --model-path models/baseline_model.pkl --model-type baseline

# Evaluate the deep learning model
python src/evaluate.py --model-path models/deep_model.h5 --model-type deep
```
Computes **Accuracy, Precision, Recall, F1-Score, AUC, and Equal Error Rate (EER)**. Saves confusion matrix to `results/confusion_matrix.png` and full report to `results/metrics.txt`.

### Step 7: Run Robustness Tests

```bash
# Test baseline model robustness
python src/robustness_test.py --model-path models/baseline_model.pkl --model-type baseline

# Test deep model robustness
python src/robustness_test.py --model-path models/deep_model.h5 --model-type deep
```
Applies three degradation types — **white noise** (10 dB SNR), **MP3-like compression** artifacts, and **pitch/time shift** perturbations — and reports how accuracy/F1 changes. Results saved to `results/robustness_results.csv` and `results/robustness_chart.png`.

### Step 8: Launch the Web Application

```bash
python app/app.py
```
Starts the Flask server at **http://localhost:5000**. Upload any audio file (.wav, .mp3, .flac) through the web interface to get a real-time genuine vs. deepfake classification with confidence scores.

<!--
---

## 📊 Results Summary

### Baseline Model Performance

| Model               | Accuracy       | Precision      | Recall         | F1-Score       |
|----------------------|----------------|----------------|----------------|----------------|
| Logistic Regression  | [ACCURACY]%    | [PRECISION]    | [RECALL]       | [F1]           |
| SVM (RBF kernel)     | [ACCURACY]%    | [PRECISION]    | [RECALL]       | [F1]           |
| Random Forest        | [ACCURACY]%    | [PRECISION]    | [RECALL]       | [F1]           |

### Deep Learning Model Performance

| Metric    | Value          |
|-----------|----------------|
| Accuracy  | [ACCURACY]%    |
| Precision | [PRECISION]    |
| Recall    | [RECALL]       |
| F1-Score  | [F1]           |
| AUC       | [AUC]          |
| EER       | [EER]%         |

### Robustness Test Findings

| Condition             | Baseline Accuracy | Deep Model Accuracy |
|-----------------------|-------------------|---------------------|
| Clean (no degradation)| [ACCURACY]%       | [ACCURACY]%         |
| White Noise (10 dB)   | [ACCURACY]%       | [ACCURACY]%         |
| MP3 Compression       | [ACCURACY]%       | [ACCURACY]%         |
| Pitch/Time Shift      | [ACCURACY]%       | [ACCURACY]%         |

> **Key Finding:** Both models show significant accuracy drops under audio degradation, especially white noise. This aligns with findings from Rabhi et al. (2024) that standard spectral-feature-based detectors are vulnerable to signal perturbations.

---
-->

## 🔬 Technical Details

### Feature Extraction (78 dimensions)

| Feature | Description | Dimensions |
|---------|-------------|------------|
| **MFCC** | Mel-Frequency Cepstral Coefficients — captures speech spectral envelope on a perceptually-motivated Mel scale | 13 mean + 13 std = 26 |
| **LFCC** | Linear-Frequency Cepstral Coefficients — uses linearly-spaced filterbanks, capturing fine spectral artifacts that Mel-scale may miss | 13 mean + 13 std = 26 |
| **CQCC** | Constant-Q Cepstral Coefficients — uses constant-Q transform with log-spaced frequency resolution, good for detecting pitch-related synthesis artifacts | 13 mean + 13 std = 26 |

### Model Architectures

- **SVM (RBF kernel):** Finds the optimal decision boundary in high-dimensional feature space. The RBF kernel maps features into infinite-dimensional space to find non-linear separation between real and fake audio patterns.
- **Random Forest:** Ensemble of decision trees that each learn different feature splits. Majority voting across trees provides robust classification less prone to overfitting.
- **ResNeXt CNN:** Simplified version of ResNeXt architecture using grouped convolutions with cardinality=4, residual skip connections, and GlobalAveragePooling2D. Processes 78-dim vectors reshaped into 6×13 2D grids.

### Dataset

- **FoR-2sec** (Fake-or-Real): 35,740 two-second audio clips (17,870 real + 17,870 fake)
- Sample rate: 16 kHz | Duration: 2 seconds | Format: WAV

---

## 📚 References

1. **Tahaoglu, G., Akbal, E., & Dogan, S. (2025).** *"Deepfake Audio Detection with Spectral Features and ResNeXt-Based Architecture."* — Proposes using MFCC/LFCC/CQCC spectral features with a ResNeXt50 CNN for audio deepfake detection.

2. **Rabhi, B., Ma, H., Bakiras, S., & Ben Ayed, R. (2024).** *"Audio-Deepfake Detection: Adversarial Attacks and Countermeasures."* — Demonstrates that audio deepfake detectors are vulnerable to adversarial perturbations and evaluates countermeasures for improved robustness.

---

*Built with Python, librosa, scikit-learn, TensorFlow/Keras, and Flask.*
