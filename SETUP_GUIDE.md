# 🚀 Quick Setup & Run Guide for Collaborators

> **Welcome!** This guide walks you through setting up and running the **AI-Based Audio Deepfake Detection System** on your local machine in under 5 minutes.
> 
> 💡 **Good News:** All trained models (`models/baseline_model.pkl`, `models/scaler.pkl`, and `models/deep_model.h5`) are **already pre-trained and included**. You do **not** need to download 5 GB of raw audio or train for hours just to run the system!

---

## 📋 System Prerequisites

* **Operating System:** Windows 10/11, macOS, or Linux
* **Python Version:** **Python 3.10, 3.11, or 3.12**  
  *(⚠️ **Important:** Do NOT use Python 3.13, as TensorFlow does not yet have official wheels for 3.13).*
* **Git:** (If cloning via command line)

---

## ⚡ 3-Minute Quick Start

Open your terminal or PowerShell inside the project folder (`audio-deepfake-detection/`) and follow these 4 steps:

### Step 1: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
```

---

### Step 2: Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If you see an execution policy error on Windows, run this one command first: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again).*

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

*(You will know it's active when you see `(.venv)` at the beginning of your terminal prompt).*

---

### Step 3: Install Required Libraries

Upgrade pip and install the exact project dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(This installs PyTorch/TensorFlow, Librosa, Scikit-learn, Flask, and audio processing tools).*

---

### Step 4: Launch the Web Application! 🎉

Once installation completes, launch the Flask web application:
```bash
python app/app.py
```

* Open your browser and go to: **`http://localhost:5000`**
* You will see the **AudioShield Web Interface**.
* **Drag and drop any speech audio file** (`.wav`, `.mp3`, `.flac`) into the drop zone.
* The system will automatically preprocess the audio, extract 78-dimensional spectral features (MFCC, LFCC, CQCC), and display:
  * **Classification:** `Real` or `Fake`
  * **Confidence Percentage:** e.g., `99.2% Fake`
  * **Acoustic Waveform Player**

To stop the web server, press `Ctrl + C` in your terminal.

---

## 🧪 Running Other Parts of the Project

The virtual environment must be active `(.venv)` when running these commands:

### 1. Test Cloned Audio Samples (Integration Test)
To evaluate the detector against custom voice-cloned samples:
```bash
python src/test_cloned_samples.py --audio-path data/cloned_samples
```
* It will test both the **Baseline SVM** and **Deep ResNeXt CNN** on the audio clips and save a side-by-side comparison report to `results/cloned_samples_evaluation.csv`.

---

### 2. Run Comprehensive Model Evaluation & EER
To calculate Accuracy, Precision, Recall, F1, and **Equal Error Rate (EER)** on the 20% test split:

```bash
# Evaluate the Baseline SVM model:
python src/evaluate.py --model baseline

# Evaluate the Deep ResNeXt CNN model:
python src/evaluate.py --model deep
```
* Generates confusion matrices and metric reports in the `results/` folder.

---

### 3. Run Robustness Testing (Adversarial Stress Test)
To test how well the detector holds up against real-world degraded audio (Gaussian white noise, MP3 compression artifacts, and pitch shifts):
```bash
python src/robustness_test.py --model baseline --num-samples 50
```
* Generates `results/robustness_results.csv` and `results/robustness_chart.png`.

---

## 🔄 Optional: Re-Training from Scratch (Full Pipeline)

If you have downloaded the full FoR-2sec audio dataset and wish to retrain all models from scratch:

```bash
# 1. Download the full dataset (5+ GB)
python src/download_data.py

# 2. Preprocess audio (16 kHz, silence trim, normalize to 32k samples)
python src/preprocessing.py

# 3. Extract 78-dim spectral features (MFCC + LFCC + CQCC)
python src/feature_extraction.py

# 4. Train classical ML baseline models (SVM, Random Forest, Logistic Regression)
python src/train_baseline.py

# 5. Train the deep ResNeXt-inspired CNN architecture
python src/train_deep_model.py
```

---

## ❓ Troubleshooting & FAQs

### Q1: PowerShell says `Execution of scripts is disabled on this system`
* **Fix:** Run this command in PowerShell before activating:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

### Q2: Port 5000 is already in use
* **Fix:** Open `app/app.py`, scroll to the bottom line:
  ```python
  app.run(host="0.0.0.0", port=5001, debug=False)
  ```
  Change `5000` to `5001`, save, and run `python app/app.py`. Then visit `http://localhost:5001`.

### Q3: TensorFlow displays `This TensorFlow binary is optimized...` or CPU warnings
* **Fix:** This is completely normal! It simply indicates that TensorFlow is running on your CPU using AVX/SSE hardware vector instructions. It is not an error.

### Q4: Which Python version is installed?
* **Check:** Run `python --version`. If it prints `Python 3.13.x`, please install Python 3.11 or 3.12 and recreate the virtual environment using `py -3.11 -m venv .venv`.

---

## 📁 Repository Directory Reference

```text
audio-deepfake-detection/
├── app/                  # Flask web application and UI
├── data/
│   ├── cloned_samples/   # Test samples generated via voice cloning
│   └── features/         # Extracted 78-dim feature arrays (features.csv, test_split.csv)
├── docs/                 # Project reports, architecture diagrams & viva guides
├── models/               # Pre-trained model weights (SVM, ResNeXt, Scaler)
├── notebooks/            # Google Colab notebooks (voice cloning, experiments)
├── results/              # Evaluation metrics, confusion matrices, robustness charts
├── src/                  # Pipeline source code (preprocessing, extraction, training, eval)
├── requirements.txt      # Dependency list
├── README.md             # Project overview
└── SETUP_GUIDE.md        # This guide!
```

Have fun exploring the project! If you run into any issues, check the troubleshooting section above.
