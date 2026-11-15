# Professor Demonstration Walkthrough: Two-Stage Voice Cloning & Applio

**Project Title:** AI-Based Audio Deepfake Detection System Using ResNeXt  
**Module:** Two-Stage Synthetic Speech Generation (Edge-TTS + Applio RVC)  
**Document Purpose:** Complete oral script, live demonstration steps, and panel defense strategy for presenting to project guides and external evaluators.  
**Target Duration:** 4 to 6 minutes.

---

## 1. Executive Strategy: The "Why" Behind This Demo

When presenting to your professor, **frame this module correctly from second one**:

> ⚠️ **Key Framing:** Do NOT present this as just *"we cloned a voice."*  
> **Present it as:** *"To rigorously evaluate our Audio Deepfake Detector, we developed an adversarial test harness that generates realistic, custom voice clones of our own team members using a decoupled two-stage pipeline."*

This immediately shows the professor that you understand research methodology: you built a generator specifically to stress-test your detector.

---

## 2. Pre-Demo Checklist (Prepare 15 Minutes Before)

A live demo can fail due to poor college Wi-Fi or Colab session timeouts. Always prepare **both Plan A (Live)** and **Plan B (Offline Backup)**.

### Plan A: Live Colab / Applio Setup
- [ ] Colab session running with **T4 GPU**.
- [ ] Run the `Start Applio` cell so the public Gradio URL is open in a browser tab.
- [ ] Verify `teammate01_rvc.pth` and its `.index` file appear in the **Inference** tab dropdown.
- [ ] Keep a clean 5-second test sentence ready to convert.
- [ ] Test your laptop speakers / audio volume beforehand!

### Plan B: Offline Backup on Your Laptop (Never Fails)
If college Wi-Fi disconnects or Colab disconnects, do not panic. Have these 3 WAV files saved in a folder on your desktop:
1. `01_real_reference.wav` — The real team member's original recording.
2. `02_stage1_base_tts.wav` — The generic Indian English Edge-TTS audio.
3. `03_stage2_rvc_deepfake.wav` — The final converted deepfake audio.

You can play them directly via VLC or Windows Media Player and run your local detector script offline!

---

## 3. Step-by-Step Demonstration Script (5 Minutes)

```
   ┌───────────────────────────────────────────────────────────────┐
   │                     5-MINUTE DEMO ROADMAP                     │
   ├───────────┬───────────┬───────────┬───────────┬───────────────┤
   │  STEP 1   │  STEP 2   │  STEP 3   │  STEP 4   │    STEP 5     │
   │  Ethical  │ Two-Stage │ Live RVC  │   3-Way   │ Bridge to our │
   │  Consent  │ Concept   │ In Applio │ Playback  │ Deepfake Det. │
   │   (30s)   │   (45s)   │  (1.5m)   │   (1m)    │     (1m)      │
   └───────────┴───────────┴───────────┴───────────┴───────────────┘
```

---

### Step 1: Ethical Consent & Project Context (30 Seconds)

**What to do:** Stand beside the screen. Keep your posture confident.  
**What to say:**
> *"Respected Professor, before demonstrating the synthetic speech generation, we want to highlight our strict ethical consent protocol. Under our project guidelines, we did not scrape or use any public figure or celebrity voice. The target voice model was trained strictly on a 3.5-minute reference recording provided by our own team member with their explicit consent.*
> 
> *Our objective here is not just voice cloning, but creating realistic test stimuli to evaluate our ResNeXt deepfake detector."*

---

### Step 2: Explain the Two-Stage Architecture (45 Seconds)

**What to do:** Show the architecture flowchart ([`docs/rvc_architecture_pipeline.png`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/docs/rvc_architecture_pipeline.png)) or draw it on the whiteboard:

$$\text{Text Prompt} \xrightarrow[\text{Edge-TTS (en-IN)}]{\text{Stage 1}} \text{Intermediate Speech} \xrightarrow[\text{Applio RVC (100 Epochs)}]{\text{Stage 2}} \text{Final Deepfake Audio}$$

**What to say:**
> *"When we initially tested standard zero-shot cloning tools like XTTS-v2, we observed a major flaw called **'Accent Leakage'**—the models forced an American accent onto Indian English words because their foundational language models were trained on Western data.*
> 
> *To solve this, we designed a decoupled two-stage architecture:*
> * *In **Stage 1**, we use Microsoft Edge-TTS with an Indian voice (`en-IN-PrabhatNeural`) to guarantee authentic pronunciation, cadence, and syllable timing.*
> * *In **Stage 2**, we pass that speech through **Applio RVC**, which strips the generic voice and reconstructs the audio using our team member's vocal tract timbre."*

---

### Step 3: Live Conversion in Applio UI (90 Seconds)

**What to do:** Switch to the **Applio Web UI** browser tab. Click on the **`Inference`** tab.

1. **Point to Model:** Show the model dropdown:
   > *"Here in Applio, you can see our trained model checkpoint: `teammate01_rvc.pth`, which we trained for 100 epochs (2,100 gradient steps) on an NVIDIA Tesla T4 GPU."*
2. **Point to Index:** Show the index file selected:
   > *"Notice the FAISS index file is linked. RVC stands for Retrieval-based Voice Conversion. The index performs nearest-neighbor search against our team member's feature vectors to ensure acoustic stability."*
3. **Show Parameters:**
   * **Index Rate:** Point to `0.80` (*"Controls how strongly our team member's timbre is retrieved"*).
   * **Pitch Estimator:** Point to `rmvpe` (*"We use RMVPE for frame-accurate pitch tracking"*).
4. **Input Audio:** Select `stage1_base_speech.wav`.
5. **Click `Convert`:** It will convert in ~2 to 3 seconds on the GPU.

---

### Step 4: The 3-Way Audio Comparison (The "Hero" Moment) (60 Seconds)

**What to do:** Play the three audio clips sequentially through your laptop speakers.

#### 1. Play `Real Reference Audio` (Our Team Member)
> *"First, listen to our team member's actual voice recording."*  
*(Play 3 seconds — let the professor register their tone and pitch).*

#### 2. Play `Stage 1 Base TTS Audio` (Edge-TTS)
> *"Second, listen to the Stage 1 base TTS. Notice that the Indian English pronunciation and words are completely clear, but the voice sounds like a generic assistant."*  
*(Play 3 seconds).*

#### 3. Play `Stage 2 RVC Converted Audio` (The Deepfake)
> *"Now, listen to the final Stage 2 output after RVC conversion."*  
*(Play the converted file).*  
> *"Notice what happened: the pronunciation and rhythm remained identical to Stage 1, but the acoustic timbre, formant resonances, and pitch profile shifted completely into our team member's voice identity."*

---

### Step 5: Connecting Back to the Deepfake Detector (The Grand Finale) (60 Seconds)

**What to do:** Switch to your local VS Code terminal and run the test script:
```bash
python src/test_cloned_samples.py --audio-path data/cloned_samples
```
Or open your local web app at `http://localhost:5000` ([`app/app.py`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/app/app.py)) and drop the converted file into the browser!

**What to say:**
> *"Now comes the core of our minor project: While this generated voice clone is highly convincing to human ears, can our AI system detect it?*
> 
> *We take this exact RVC audio file and pass it into our detection pipeline:*
> 1. *It resamples the audio to 16 kHz and normalizes it.*
> 2. *It extracts our 78-dimensional multi-spectral vector—specifically **LFCC** to capture the high-frequency neural vocoder artifacts above 4 kHz, and **CQCC** to detect harmonic pitch transitions.*
> 3. *Our **ResNeXt CNN** and **SVM** classifiers evaluate the feature map.*
> 
> *As you can see on the screen, our detector successfully flagged this audio as **FAKE / DEEPFAKE** with high confidence, proving that acoustic vocoder footprints remain detectable even when perceptual voice cloning succeeds."*

---

## 4. Technical Settings Cheat Sheet (For Live Demo)

Keep these settings selected in the Applio **Inference** tab:

| Parameter | Setting | Why to Explain to Professor |
| :--- | :--- | :--- |
| **Model** | `teammate01_rvc.pth` | Checkpoint trained on our consenting team member (100 epochs, 2,100 steps). |
| **Index File** | `added_IVF256...index` | FAISS database containing the target speaker's ContentVec feature clusters. |
| **Index Rate** | `0.75` – `0.85` | Balance between target speaker similarity and phonetic clarity. |
| **Pitch Algorithm** | `rmvpe` | Robust Model for Vocal Pitch Estimation (avoids voice cracking). |
| **Pitch Shift** | `0` | Keeps the base speech's original key (no octave shift needed). |

---

## 5. Potential Professor Questions & Model Answers

### Q1: "Why did you use Applio instead of building an RVC model from scratch?"
> *"Building RVC from scratch would mean re-implementing VITS GAN generators, ContentVec foundation models, and FAISS indexing libraries—which is beyond the scope of a minor project and irrelevant to our core objective. Our core research contribution is the **Audio Deepfake Detection System Using ResNeXt**. Applio provided a production-grade, fairseq-free RVC v2 implementation that allowed us to train custom voice weights on our own hardware and generate authentic adversarial test stimuli."*

### Q2: "How much data was used to train this voice model?"
> *"We used approximately 3.5 minutes of clean, 44.1 kHz mono WAV audio recorded in a quiet room. In RVC v2, because the model leverages pretrained ContentVec representations and fine-tunes a pretrained 40k VITS generator, 3 to 5 minutes of clean speech is the optimal sweet spot to learn vocal tract resonances without overfitting."*

### Q3: "Why did you use RMVPE instead of Crepe for pitch tracking?"
> *"TorchCrepe is an accurate neural pitch tracker, but it requires significant GPU memory and increases training time by over 300%. RMVPE (Robust Model for Vocal Pitch Estimation) is a modern convolutional-recurrent model designed specifically for noisy and polyphonic speech. It gave us identical pitch accuracy while allowing our 100-epoch training to finish in under 17 minutes on a Tesla T4 GPU."*

### Q4: "What is the difference between MFCC, LFCC, and CQCC in catching this RVC audio?"
> *"MFCC models human ear perception and the vocal tract envelope, which RVC mimics well. However, RVC uses a **HiFi-GAN vocoder** composed of transposed convolutions, which creates subtle mathematical phase and power anomalies in the linear high frequencies above 4 kHz. **LFCC (Linear-Frequency Cepstral Coefficients)** preserves linear resolution across high frequencies and catches these vocoder artifacts. Simultaneously, **CQCC (Constant-Q)** catches harmonic micro-glitches from the RMVPE pitch transitions."*

### Q5: "Can your system run in real-time?"
> *"Yes. The inference time for Stage 1 Edge-TTS is under 1 second, Stage 2 RVC conversion takes approximately 2 seconds on GPU, and our local ResNeXt detector processes the 2-second audio feature extraction and classification in approximately 80 milliseconds on CPU."*

---

*Print or review this walkthrough before your demonstration.*
