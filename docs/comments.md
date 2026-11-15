# Project Concepts & Explanations Reference

> **Purpose:** This document consolidates every "why" explanation from across the project codebase into a single, readable reference. It is written in plain English so a team member can read any section aloud and explain the concept to an examiner without needing to look at the code.

---

## 1. Preprocessing

### Why resample everything to 16 kHz?

Human speech frequencies primarily lie between 300 Hz and 3,400 Hz. According to the Nyquist–Shannon Sampling Theorem, a sampling rate of 16,000 Hz can faithfully capture frequencies up to 8,000 Hz, which comfortably covers all essential vocal formants and pitch characteristics. Meanwhile, many raw audio files arrive at higher rates like 44.1 kHz (CD quality) or 48 kHz (studio quality), which wastes computational time and memory. Different recording hardware and TTS synthesizers also output at different native rates. Resampling everything to a single, fixed rate of 16 kHz ensures that every spectrogram and feature matrix has uniform time-frequency spacing, making all samples directly comparable.

### Why trim silence from the beginning and end?

Leading and trailing silence in an audio clip contains no vocal energy or acoustic cues — it is essentially dead air. If left in place, these silent margins skew the mean and variance of spectral feature matrices like MFCCs and LFCCs, because the feature extraction algorithms will compute statistics over frames that contain no speech information. Trimming silence ensures that every computed feature reflects actual speech content, preventing the model from wasting capacity learning that "silence equals real" or "silence equals fake."

### Why normalize peak amplitude to [−1.0, 1.0]?

Audio recordings vary wildly in loudness depending on microphone gain, recording distance, and synthesis settings. A quiet recording of real speech and a loud recording of fake speech would produce different power spectra even if the underlying vocal content were identical. Peak normalization scales the waveform so that its maximum absolute amplitude equals 1.0, removing this loudness bias. It also prevents numerical instability during neural network training — extremely large or small input values can cause gradient explosions or vanishing gradients during backpropagation.

### Why pad or truncate to exactly 32,000 samples?

Deep learning models like CNNs require input tensors of identical shape within each batch. Since different audio clips have different durations after silence trimming, we standardize every clip to exactly 32,000 samples, which corresponds to 2.0 seconds of audio at 16 kHz. Clips shorter than 2 seconds are zero-padded at the end; clips longer than 2 seconds are truncated. This guarantees consistent matrix dimensions across all samples in the dataset.

---

## 2. Feature Extraction (MFCC / LFCC / CQCC)

### What are MFCCs and why do they matter?

Mel-Frequency Cepstral Coefficients model the overall filter shape of the human vocal tract — the throat, mouth, and tongue configuration that shapes speech sounds. They use a logarithmic Mel-scale filterbank, which mimics how the human ear perceives frequency: higher resolution at low frequencies (below ~1 kHz) and lower resolution at high frequencies. MFCCs are the standard baseline feature for speech recognition and speaker identification tasks because they compactly capture the spectral envelope of speech.

### What are LFCCs and why are they better for deepfake detection?

Linear-Frequency Cepstral Coefficients use a linearly-spaced filterbank with equal bandwidth across the entire frequency range from 0 Hz up to the Nyquist frequency (8 kHz at 16 kHz sampling rate). Unlike the Mel scale, which compresses high frequencies, a linear filterbank preserves equal resolution across all frequency bands. This matters for deepfake detection because AI voice generators — neural vocoders like HiFi-GAN and WaveGLOW — often leave subtle spectral artifacts in the high-frequency range above 4 kHz. Human hearing (and the Mel scale) naturally compresses these frequencies, hiding the artifacts. LFCCs preserve full high-frequency detail, making them superior for catching synthetic speech. LFCC is a standard feature in the ASVspoof anti-spoofing benchmark.

### What are CQCCs and what do they detect?

Constant-Q Cepstral Coefficients use the Constant-Q Transform, where the ratio of center frequency to bandwidth remains constant (Q = f / Δf). This creates geometrically-spaced frequency bins that match musical octave relationships. CQCCs are particularly effective at detecting subtle fundamental pitch (f₀) glitches and harmonic discontinuities that text-to-speech voice cloning systems introduce. While MFCC captures the vocal tract shape and LFCC catches high-frequency artifacts, CQCC focuses on pitch and harmonic structure anomalies.

### Why use mean and standard deviation as summary statistics?

Each feature extractor (MFCC, LFCC, CQCC) produces a 2D matrix of shape (13 coefficients × T time frames) for each audio clip. Since different clips may have different numbers of time frames, we cannot feed raw matrices directly into a fixed-size classifier. Instead, we summarize each coefficient across the time dimension by computing its mean (average value over time) and standard deviation (how much it varies over time). This produces a fixed-length 78-dimensional vector: 13 coefficients × 2 statistics × 3 feature types = 78 values per audio clip.

### Why combine all three feature types?

Each feature type captures a different aspect of the audio signal. MFCCs model the vocal tract shape, LFCCs capture high-frequency synthesis artifacts, and CQCCs detect pitch and harmonic anomalies. By concatenating all three, the classifier gets a more comprehensive view of the audio. A deepfake that might pass inspection under one feature type might be caught by another, improving overall detection robustness.

---

## 3. Baseline Models (SVM / Random Forest / Logistic Regression)

### Why start with Logistic Regression?

Logistic Regression fits a linear decision boundary using the sigmoid function. It serves as a baseline benchmark to test whether the spectral features of real and fake speech are linearly separable — that is, whether a simple straight line (or hyperplane) in 78-dimensional space can separate the two classes. If Logistic Regression performs well, it means the extracted features already carry strong discriminative power. If it performs poorly, it tells us we need non-linear models.

### Why use SVM with an RBF kernel?

A Support Vector Machine finds the optimal decision boundary that maximizes the margin (distance) between the closest data points of each class. The Radial Basis Function (RBF) kernel maps the 78-dimensional feature vectors into an implicit, higher-dimensional space where non-linear patterns become linearly separable. This allows the SVM to learn complex, curved decision boundaries between real and fake audio patterns. SVMs are particularly effective in high-dimensional spaces and resist overfitting when the number of features is comparable to the number of samples.

### Why include Random Forest?

A Random Forest builds an ensemble of 100 decision trees, each trained on a different bootstrap sample of the data with a random subset of features. The final prediction is made by majority voting across all trees. This approach captures non-linear feature interactions between MFCC, LFCC, and CQCC statistics without requiring strict assumptions about feature distributions or scaling. Random Forests are naturally resistant to noise and outliers, and they provide a useful comparison point against the SVM.

### Why scale features with StandardScaler?

StandardScaler transforms each feature to have zero mean and unit variance. This is critical for SVM and Logistic Regression, which are sensitive to feature scales — a feature with values in the thousands would dominate one with values near zero, even if the smaller-valued feature is more informative. Random Forest does not strictly require scaling (it uses threshold-based splits), but applying the same scaler across all models ensures a fair comparison and consistent preprocessing for the Flask web app.

### Why use a stratified 80/20 train-test split?

Stratification ensures that both the training set and test set maintain the same proportion of real vs. fake samples as the full dataset. Without stratification, random chance could create a training set that has more fake samples and a test set with more real samples (or vice versa), biasing the evaluation metrics. The fixed random seed (42) ensures the split is reproducible — running the script multiple times always produces the same train and test sets.

---

## 4. Deep Learning Model (CNN / ResNeXt-Inspired)

### Why reshape the 78-dimensional vector into a 6×13 2D grid?

Standard 1D feature vectors lose spatial and relational context. By reshaping the 78 features into a 2D grid with 6 rows and 13 columns, we create a structure that 2D convolutional filters can exploit. Each row represents a different feature statistic group (MFCC mean, MFCC std, LFCC mean, LFCC std, CQCC mean, CQCC std), and each column represents one of the 13 cepstral coefficient indices. This arrangement allows convolutional filters to learn local patterns both across coefficient indices (horizontal patterns) and across different feature modalities (vertical patterns — such as "MFCC mean is high but LFCC std is low").

### Why use ResNeXt architecture instead of a plain CNN?

ResNeXt introduces the concept of "cardinality" — the number of parallel, grouped convolution paths within each block. Instead of making the network deeper (more layers, which risks vanishing gradients) or wider (more channels per layer, which increases memory), ResNeXt splits the feature maps into multiple smaller groups, processes each group through its own convolution filter, and then aggregates the results. This significantly increases the network's representation power and feature selectivity while keeping the total parameter count and computational cost low. Our implementation is a simplified, lightweight version of the full ResNeXt50 described in Tahaoglu et al. (2025), optimized to train on a laptop CPU.

### What does the 1×1 bottleneck convolution do?

The 1×1 convolution at the start of each ResNeXt block reduces the channel depth (number of feature maps) before the expensive 3×3 convolution. This compression step dramatically reduces the number of multiplication operations without losing important information, because the 1×1 convolution learns to combine channels intelligently. After the 3×3 grouped convolution, another 1×1 convolution expands the channels back to the target depth.

### What is a residual skip connection and why does it help?

A residual (identity) skip connection adds the original input of a block directly to the block's output. This means the block only needs to learn the "residual" — the difference between the input and the desired output — rather than learning the entire transformation from scratch. Skip connections solve two critical problems: they prevent vanishing gradients (the gradient can flow directly through the shortcut during backpropagation) and they enable smooth optimization (the network can easily learn to pass information through unchanged when needed). If the input and output shapes differ, a 1×1 projection convolution adjusts the shortcut dimensions.

### Why use GlobalAveragePooling2D instead of Flatten?

After the convolutional blocks produce 2D feature maps, we need to convert them into a 1D vector for the classification layers. The naive approach (Flatten) would create a very large vector and require many Dense layer parameters, leading to overfitting. GlobalAveragePooling2D instead computes the average value of each feature map, producing one number per channel. This reduces the parameter count dramatically and acts as a structural regularizer — the model cannot memorize spatial positions, forcing it to learn translation-invariant features.

### Why use sigmoid activation for the output?

Binary classification (real vs. fake) requires the final layer to output a single probability between 0.0 and 1.0. The sigmoid function maps any real number to this range. A value above 0.5 indicates "fake" (class 1), and below 0.5 indicates "real" (class 0). This is paired with binary cross-entropy loss, which measures how well the predicted probabilities match the true binary labels.

### Why use early stopping?

Early stopping monitors the validation loss during training and automatically stops if it stops improving for a specified number of epochs (patience=5). Without early stopping, the model might continue training past the point where it starts memorizing the training data (overfitting), leading to poor performance on unseen test data. Restoring the best weights ensures we keep the model from the epoch with the lowest validation loss.

### Why use Dropout regularization?

Dropout randomly deactivates 30% of neurons during each training step, forcing the remaining neurons to learn more robust, independent features rather than co-adapting. At inference time, all neurons are active but their outputs are scaled down. This prevents overfitting by effectively training an ensemble of sub-networks within a single model.

---

## 5. Evaluation Metrics

### What does Accuracy tell us?

Accuracy is the overall percentage of audio clips that were correctly classified — both real clips correctly identified as real and fake clips correctly identified as fake. While intuitive, accuracy can be misleading if the dataset is imbalanced. In our case, the dataset is balanced (50/50 real/fake), so accuracy is a reliable metric.

### What is Precision and why does it matter?

Precision answers the question: "Out of all clips the model flagged as fake, what fraction were actually fake?" A low precision means the model produces many false alarms — flagging genuine speech as deepfake. In real-world applications, low precision could mean innocent people are wrongly accused of using synthetic voices.

### What is Recall and why does it matter?

Recall answers the question: "Out of all actual deepfakes in the dataset, what fraction did we catch?" A low recall means the model misses many deepfakes, letting synthetic speech pass as genuine. In security-critical applications, low recall is dangerous because it means deepfakes slip through undetected.

### What is F1-Score and when is it useful?

The F1-Score is the harmonic mean of Precision and Recall, providing a single number that balances both concerns. Unlike a simple average, the harmonic mean punishes extreme imbalances — if either Precision or Recall is very low, the F1-Score will also be low. It is the preferred metric when you care equally about catching deepfakes (Recall) and not raising false alarms (Precision).

### What is Equal Error Rate (EER) and why is it important?

Equal Error Rate is the operating point on the detector's ROC curve where the False Acceptance Rate (FAR — falsely trusting a deepfake voice as genuine) equals the False Rejection Rate (FRR — falsely flagging a genuine human voice as deepfake). In plain English, it is the threshold where the detector makes equal numbers of both types of mistakes. A lower EER indicates a more reliable and balanced detector. EER is the standard evaluation metric in biometric voice verification and anti-spoofing benchmarks like ASVspoof, and it is threshold-independent — unlike accuracy, which depends on a fixed decision boundary of 0.5.

### What is AUC (Area Under the ROC Curve)?

AUC measures the overall ability of the model to distinguish between real and fake across all possible decision thresholds. An AUC of 1.0 means perfect separation; 0.5 means random guessing. AUC is useful because it evaluates the model's ranking ability — whether fake clips consistently receive higher probability scores than real clips — regardless of what threshold you ultimately choose for classification.

---

## 6. Robustness Testing

### Why test robustness at all?

A model that achieves 99% accuracy on clean test data might fail catastrophically when faced with real-world audio conditions — background noise in a phone call, compression from messaging apps, or slight pitch shifts from different playback speeds. Robustness testing reveals whether the model has learned genuine deepfake detection features or has merely memorized surface-level patterns in the clean training data. This evaluation is inspired by Rabhi et al. (2024), which demonstrated that standard audio deepfake detectors are vulnerable to adversarial perturbations.

### Why add white noise at 10 dB SNR?

White noise is a simple, universal model of environmental background noise — think of air conditioning hum, traffic, or crowd noise. Adding noise at a Signal-to-Noise Ratio (SNR) of 10 dB means the speech signal is 10 times louder than the noise in power terms, which represents a moderately noisy environment. If the model's accuracy drops sharply under this condition, it indicates that the spectral features the model relies on are easily corrupted by ambient noise, suggesting the model may struggle in real-world deployments where audio is rarely pristine.

### Why simulate MP3 compression artifacts?

In practice, audio is rarely transmitted or stored in raw WAV format. Messaging apps, social media platforms, and VoIP calls apply lossy compression that removes high-frequency detail and introduces quantization noise. Our simulation downsamples the audio to 8 kHz (telephone bandwidth) and quantizes to 8-bit depth, then resamples back to 16 kHz. This tests whether the model's discriminative features survive the kind of information loss that happens when audio travels through real communication channels.

### Why test with pitch shift and time jitter?

Pitch shifting and time displacement simulate small acoustic perturbations that could occur through playback speed variations, resampling errors, or intentional manipulation by an adversary trying to evade detection. A pitch shift of +1.5 semitones and random time offset tests whether the model has learned robust spectral patterns or is fragile to small signal transformations. This is a simplified, non-adversarial version of the gradient-based attacks described in Rabhi et al. (2024).

### What does accuracy drop tell us?

The accuracy drop percentage quantifies how much worse the model performs under each degradation compared to its clean baseline. A small drop suggests the model's features are robust and generalize well; a large drop (e.g., from 99% to 50%) indicates the model is essentially guessing and that the features it relies on have been destroyed by the degradation. This helps identify which real-world conditions pose the greatest threat to deployment reliability.

---

## 7. Web Interface

### Why pre-load models at server startup?

Loading a trained model from disk (deserializing the pickle file) takes a fraction of a second, but adding this delay to every HTTP request would degrade user experience. By loading the SVM model and StandardScaler into memory once at startup, every subsequent prediction request can skip the disk I/O and run inference immediately, keeping response times fast.

### Why does the prediction endpoint replicate the full preprocessing pipeline?

The model was trained on features extracted from preprocessed audio — resampled, silence-trimmed, normalized, and padded. If we fed raw uploaded audio directly into the feature extractor without these preprocessing steps, the resulting feature values would be on a completely different scale and distribution than what the model learned during training. The prediction endpoint must apply the exact same preprocessing chain to ensure consistency between training and inference.

### Why scale features with the same saved scaler?

The StandardScaler fitted during training knows the mean and standard deviation of each feature across the training set. At inference time, we must apply the same transformation — subtracting the training mean and dividing by the training standard deviation — to the new audio's features. Using a freshly computed scaler on a single sample would produce meaningless normalized values. This is why the scaler is saved as a separate artifact (scaler.pkl) and loaded alongside the model.

### Why use a UUID for temporary uploaded files?

When multiple users upload files simultaneously, using the original filename could cause collisions — two users uploading "audio.wav" would overwrite each other. Generating a unique UUID-based filename for each upload prevents this. The temporary file is also deleted after processing to avoid accumulating uploaded audio on the server.

### Why use decision function fallback for confidence scores?

The SVM model is trained with `probability=True`, which enables Platt scaling to produce calibrated probability estimates. However, if for any reason probability estimation is unavailable, the prediction endpoint falls back to the SVM's raw decision function value and applies a sigmoid transformation to approximate a probability-like score. This ensures the web interface always returns meaningful confidence percentages to the user.

---

## Quick Answers (Likely Viva Questions)

### Q1: Why did you choose MFCC, LFCC, and CQCC specifically? Why not just use MFCCs alone?

MFCCs are the standard baseline for speech processing and capture the overall vocal tract shape, but they use a Mel-scale filterbank that compresses high frequencies — exactly where AI voice synthesizers leave detectable artifacts. LFCCs use a linear filterbank that preserves equal resolution at high frequencies, making them better at catching the spectral footprints of neural vocoders like HiFi-GAN. CQCCs use the Constant-Q Transform with geometrically-spaced frequency bins, which is particularly sensitive to pitch glitches and harmonic discontinuities that text-to-speech systems introduce. By combining all three, we cover different aspects of the audio signal: vocal tract shape (MFCC), high-frequency synthesis artifacts (LFCC), and pitch/harmonic anomalies (CQCC). This multi-feature approach follows the methodology of Tahaoglu et al. (2025), who showed that combining these features with a ResNeXt architecture achieves strong detection performance.

### Q2: What is the ResNeXt architecture and how does it differ from a standard CNN?

A standard CNN stacks convolutional layers sequentially — each layer applies filters across all input channels simultaneously. ResNeXt introduces "cardinality" as a new dimension alongside depth and width: instead of one large convolution, each ResNeXt block splits the input channels into multiple parallel groups (in our case, 4 groups), processes each group through its own smaller convolution, and then merges the results. This grouped convolution approach dramatically increases the diversity of learned features while keeping the parameter count manageable. ResNeXt blocks also include residual skip connections, where the input is added directly to the output, preventing vanishing gradients and enabling the network to learn incrementally. Our implementation is a simplified, lightweight version of the full ResNeXt50 — just two blocks with 32 and 64 filters — designed to train on a laptop CPU in minutes rather than requiring GPU clusters.

### Q3: What is Equal Error Rate (EER) and why do you use it instead of just accuracy?

Equal Error Rate is the point where the detector's false acceptance rate (letting deepfakes through) equals its false rejection rate (blocking genuine speech). Unlike accuracy, which depends on a fixed threshold of 0.5, EER evaluates the detector across all possible thresholds and finds the one where both types of errors are balanced. This makes EER a more robust metric for comparing detectors — a model with lower EER is better at separating real from fake regardless of what threshold you choose. EER is the standard metric in biometric voice verification and anti-spoofing competitions like ASVspoof, which is why both of our reference papers use it.

### Q4: Why does your model's accuracy drop so much under noisy conditions?

Our model relies on spectral features — statistical summaries of frequency patterns over time. When noise is added, these frequency patterns get corrupted: the noise adds energy across all frequency bins, shifting the mean values and inflating the standard deviations of MFCC, LFCC, and CQCC features. At 10 dB SNR, the noise is substantial enough to distort the feature distributions away from what the model saw during training on clean audio. This finding aligns with Rabhi et al. (2024), who showed that even simple signal perturbations can degrade detector accuracy dramatically. In a production system, we would need to either train on noise-augmented data (data augmentation), use noise-robust features, or apply denoising as a preprocessing step.

### Q5: Why preprocess audio before feature extraction? Can't you just extract features from raw audio directly?

Raw audio files come in different sampling rates (8 kHz to 48 kHz), different loudness levels, different durations, and often have silence at the start and end. If you extract MFCCs from a 44.1 kHz file and a 16 kHz file, the frequency bin spacing will be different, making the features incomparable. If one file is much louder than another, the power spectrum values will differ even for identical speech content. If files have different durations, you cannot create uniform-shaped tensors for batch processing in neural networks. Preprocessing standardizes all these variables so that every feature vector reflects only the acoustic content of the speech — not the recording conditions.

### Q6: How does the Flask web application use the trained model to make predictions?

When a user uploads an audio file through the web interface, the Flask server runs the exact same pipeline as training: it loads the audio, resamples to 16 kHz, trims silence, normalizes amplitude, pads to 32,000 samples, and then extracts the 78-dimensional MFCC+LFCC+CQCC feature vector. This vector is then scaled using the saved StandardScaler (which applies the same mean and variance transformation learned from the training data), and finally passed to the saved SVM model for classification. The SVM outputs a probability score for each class, and the result — "Real" or "Fake" along with confidence percentages — is returned to the user as a JSON response and displayed in the web interface.
