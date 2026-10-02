**Synopsis**

**AI-Based Audio Deepfake Detection System using ResNeXt**

*Department:  **Information Science & Engineering***

***The National Institute of Engineering**,  **Mysuru**  —*  Academic Year: **2026–27**

|**Batch No:  C4**|||||
| :- | :- | :- | :- | :- | :- |
|**USN**|**NAME**|**EMAIL ID**|**MOBILE NUMBER**||
|-|-|-|-||
|-|-|-|-||
|-|- |-|-||
|-|-|-|-||
|<p>**Project Title: AI-Based Audio Deepfake Detection System Using ResNeXt**</p><p>**Broad Area:** Artificial Intelligence / Audio Signal Processing<br>**Type of Project:** Research & Development / Application</p>|||||
|**Abstract:**<br>The rapid evolution of generative artificial intelligence and neural speech synthesis has made it increasingly feasible to generate highly realistic synthetic human voices. These audio deepfakes, produced via modern Text-to-Speech (TTS) and Voice Conversion (VC) techniques, present significant security risks to biometric voice authentication systems, digital media verification, and forensic investigations. The proposed project aims to develop an AI-based Audio Deepfake Detection System capable of classifying input audio samples as genuine human speech or synthetic deepfake speech. The system implements an end-to-end processing pipeline beginning with signal preprocessing—standardizing sampling rates (16 kHz), trimming unvoiced silence, and normalizing waveform amplitudes. To expose acoustic anomalies left by neural synthesis models, the system extracts complementary spectral representations: Mel-Frequency Cepstral Coefficients (MFCC) to capture vocal-tract spectral envelopes, Linear-Frequency Cepstral Coefficients (LFCC) to preserve high-frequency artifacts, and Constant-Q Cepstral Coefficients (CQCC) to capture fundamental pitch and harmonic irregularities. These spectral features are processed and structured into multi-dimensional representations for classification using a ResNeXt-based deep learning architecture, which employs grouped convolutions and residual connections to capture discriminative acoustic cues. Classical machine learning models (such as Support Vector Machines and Random Forest) serve as baseline classifiers for comparative evaluation. The system is designed and evaluated on a controlled, manageable subset of an established audio deepfake benchmark dataset under available computational resources, using standard performance metrics including Accuracy, Precision, Recall, F1-score, and Equal Error Rate (EER).|||||
|<p>**Objectives:<br>1\. Signal Preprocessing & Standardization:** Implement an audio signal preprocessing pipeline to standardize input recordings via resampling to a uniform sampling frequency (16 kHz), trimming unvoiced leading/trailing silence, and normalizing peak amplitudes to eliminate loudness bias.</p><p>**2. Multi-Feature Spectral Extraction:** Extract complementary acoustic and spectral feature representations—specifically Mel-Frequency Cepstral Coefficients (MFCC), Linear-Frequency Cepstral Coefficients (LFCC), and Constant-Q Cepstral Coefficients (CQCC)—to capture vocal-tract, high-frequency, and pitch/harmonic synthesis artifacts.</p><p>**3. ResNeXt Deep Learning Classification:** Design, train, and validate a ResNeXt-based deep learning model utilizing grouped convolutions (cardinality) and residual skip connections to effectively learn discriminative representations for binary genuine-versus-deepfake classification.</p><p>**4. Classical Baseline Benchmarking:** Train and evaluate traditional machine learning baseline classifiers (such as Support Vector Machines with RBF kernel and Random Forest) on the extracted feature vectors to provide rigorous performance comparisons against the deep learning model.</p><p>**5. Quantitative Metric Evaluation:** Systematically evaluate the detection framework on a held-out test split using standard performance metrics, including Accuracy, Precision, Recall, F1-score, and Equal Error Rate (EER).</p><p>**6. Feasibility & Robustness Analysis:** Investigate system performance, practical computational feasibility, and detector behavior under controlled dataset constraints and degraded audio conditions.</p>|||||
|**Software Requirements:**|||||
|<p>**Operating System                :**  Windows 10/11, Linux (Ubuntu)</p><p>**Development Platforms      :**  VS Code, Jupyter Notebook, Google Colab</p><p>**Python Environment           :**  Python 3.10–3.12</p><p>**Libraries & Frameworks    :**  PyTorch / TensorFlow, librosa, NumPy, pandas, scikit-learn, matplotlib, Flask</p>|||||
|**Hardware Requirements:**|||||
|<p>**Processor                      :**  Intel Core i3/i5/i7 or AMD Ryzen equivalent</p><p>**Memory                        :**  Minimum 8 GB RAM (16 GB recommended)</p><p>**GPU / Computational Resource :**  Google Colab / Kaggle Cloud GPU (NVIDIA T4) or local CPU/GPU</p>|||||
|**Guide:**|||||
|**Name:**|` `**-**||||
|**Designation:**|` `**-**||||
|**Contact No.**|` `**-**||||
