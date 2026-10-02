# ResNeXt: Split-Transform-Merge Architecture Reference

> **Reference Context:** Inspired by Xie et al. (CVPR 2017), *"Aggregated Residual Transformations for Deep Neural Networks"* and adapted for spectral audio deepfake detection following Tahaoglu et al. (2025).

---

## 1. Architecture Diagram Files

The technical diagram illustrating the **Split-Transform-Merge** paradigm and its implementation in this project is available in two formats:
* **High-Resolution Image (300 DPI PNG):** [`docs/resnext_split_transform_merge.png`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/docs/resnext_split_transform_merge.png)
* **Scalable Vector Graphic (SVG):** [`docs/resnext_split_transform_merge.svg`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/docs/resnext_split_transform_merge.svg)

---

## 2. Core Paradigm: Split-Transform-Merge

In traditional deep convolutional networks, designers expand capacity by:
1. **Going Deeper:** Adding more layers (e.g., VGG, ResNet-101) — increases risk of vanishing gradients and latency.
2. **Going Wider:** Increasing channel depth per layer (e.g., Wide ResNet) — drastically inflates parameter count quadratically ($O(C^2)$).

**ResNeXt** introduces a third dimension: **Cardinality ($C$)**, which represents the number of independent, parallel transformation paths within a block.

```
                         Input Feature Map (x)
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │ (Identity Shortcut)     ▼                         │
         │                      [ SPLIT ]                    │
         │             Project into C parallel paths         │
         │                         │                         │
         │     ┌──────────┬────────┴─────────┬──────────┐    │
         │     ▼          ▼                  ▼          ▼    │
         │   Path 1     Path 2    ···      Path 3     Path C │
         │   [ 𝒯₁ ]     [ 𝒯₂ ]             [ 𝒯₃ ]     [ 𝒯𝚌 ] │  <-- [ TRANSFORM ]
         │     │          │                  │          │    │
         │     └──────────┴────────┬─────────┴──────────┘    │
         │                         ▼                         │
         │                     [ MERGE ]                     │
         │               Element-wise Addition (∑)           │
         │                         │                         │
         └────────────────────────►(+)◄──────────────────────┘
                                   │
                                 [ReLU]
                                   │
                                   ▼
                         Output Feature Map (y)
```

### The Three Phases Explained:

1. **SPLIT:**
   * The incoming multi-channel feature representation $x$ is partitioned into $C$ lower-dimensional subspaces using a $1 \times 1$ bottleneck convolution.
   * Rather than processing all channels together, each subspace handles an isolated slice of the acoustic representation.

2. **TRANSFORM:**
   * Each path applies a non-linear transformation $\mathcal{T}_i(x)$ with identical topology (in our model, a $3 \times 3$ convolution followed by Batch Normalization and ReLU).
   * Because all branches share the same structure (**homogeneous multi-branch**), hyperparameter tuning is straightforward compared to Inception-style architectures that require manual tuning of different filter sizes ($1 \times 1, 3 \times 3, 5 \times 5$).

3. **MERGE:**
   * The outputs of all $C$ paths are aggregated via element-wise addition:
     $$\sum_{i=1}^{C} \mathcal{T}_i(x)$$
   * This is followed by an expansion $1 \times 1$ convolution that projects features to the target output channel depth.

4. **RESIDUAL SHORTCUT:**
   * The original identity input $x$ is added directly to the aggregated transformation:
     $$y = \mathrm{ReLU}\left(x + \sum_{i=1}^{C} \mathcal{T}_i(x)\right)$$
   * This preserves gradient highways during backpropagation, eliminating vanishing gradients.

---

## 3. Concrete Numerical Example (Our Audio Deepfake Detector)

In our project ([`src/train_deep_model.py`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/src/train_deep_model.py)), ResNeXt processes a **$6 \times 13 \times 1$ spectral grid** with **Cardinality $C = 4$**.

Here is the step-by-step tensor transformation through **ResNeXt Stage 1**:

| Step | Operation | Input Shape | Output Shape | Parameters / Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Input from Stem** | $(N, 6, 13, 1)$ | $(N, 6, 13, 16)$ | Stem $3\times3$ Conv (16 filters) extracts initial spatial features |
| **1** | **Split & Bottleneck** | $(N, 6, 13, 16)$ | $(N, 6, 13, 16)$ | $1 \times 1$ Conv compresses feature depth before spatial convolution |
| **2** | **Transform (Cardinality $C=4$)** | $(N, 6, 13, 16)$ | $(N, 6, 13, 16)$ | Grouped $3 \times 3$ Conv with `groups=4`. Each group processes $16 / 4 = 4$ channels |
| **3** | **Expand** | $(N, 6, 13, 16)$ | $(N, 6, 13, 32)$ | $1 \times 1$ Conv expands channels to 32 target depth |
| **4** | **Shortcut Projection** | $(N, 6, 13, 16)$ | $(N, 6, 13, 32)$ | $1 \times 1$ Conv projects shortcut from 16 to 32 channels to match addition shape |
| **5** | **Merge & Residual Add** | $(N, 6, 13, 32)$ | $(N, 6, 13, 32)$ | Element-wise sum $(x + \mathcal{T}(x))$ followed by ReLU |

---

## 4. Hardware Optimization: Grouped Convolutions

While the **conceptual** view of ResNeXt uses $C$ parallel branches, running $C$ separate convolution layers in software would incur substantial memory overhead and kernel-launch latency.

In practice, Xie et al. proved that **$C$ separate branches are mathematically identical to a single Grouped Convolution** with `groups = C`:

```python
# Conceptual multi-branch (slow to execute):
branches = [Conv2D(4, (3,3))(x[:, :, :, i*4:(i+1)*4]) for i in range(4)]
merged = tf.concat(branches, axis=-1)

# Equivalent Grouped Convolution (fast, 1 GPU/CPU kernel launch):
merged = layers.Conv2D(16, kernel_size=(3, 3), padding="same", groups=4)(x)
```

Our implementation in [`src/train_deep_model.py`](file:///d:/Projects/Deepfake%20Detection/audio-deepfake-detection/src/train_deep_model.py#L117) uses `groups=4` directly, allowing the entire model to train on a standard laptop CPU in under 3 minutes.

---

## 5. Why ResNeXt Excels at Audio Deepfake Detection

1. **Multi-Modal Feature Diversity:**
   * The input 2D grid contains three distinct acoustic feature modalities: **MFCC** (vocal tract envelope), **LFCC** (high-frequency vocoder artifacts), and **CQCC** (pitch glitches).
   * The 4 parallel paths can specialize: Path 1 can specialize in detecting MFCC formant patterns, Path 2 in LFCC high-frequency artifacts, and Paths 3 & 4 in cross-feature harmonic correlations.

2. **Parameter Efficiency:**
   * A standard dense convolution with 64 filters requires $64 \times 64 \times 3 \times 3 = 36,864$ weights per layer.
   * Grouped convolution with $C=4$ splits the channels into 4 groups of 16, requiring only $4 \times (16 \times 16 \times 3 \times 3) = 9,216$ weights — **a 75% reduction in parameters** with no loss in representation capacity.

3. **Resilience to Overfitting:**
   * Because academic audio deepfake datasets have structured spectral representations, large monolithic CNNs quickly memorize training speaker identities. The bottlenecking and grouped sparsity of ResNeXt act as structural regularizers, forcing the network to learn generalized synthesis artifacts.

---

## 6. Viva & Project Review Defense Cheatsheet

* **Q: What does Cardinality mean in ResNeXt?**
  > *"Cardinality is the size of the set of transformations — the number of parallel, grouped paths within each block. In our project, Cardinality $C = 4$, meaning features are split into 4 parallel paths, transformed independently, and aggregated."*

* **Q: How does ResNeXt differ from standard ResNet?**
  > *"Standard ResNet passes all channels through a single monolithic $3 \times 3$ convolution path. ResNeXt splits channels across $C$ parallel grouped convolutions. ResNeXt proves that increasing cardinality is more effective at reducing error than increasing depth or width."*

* **Q: How does ResNeXt differ from Google Inception?**
  > *"Inception uses heterogeneous branches with different filter sizes ($1 \times 1, 3 \times 3, 5 \times 5$, max pooling) that require delicate manual hyperparameter tuning. ResNeXt uses homogeneous branches where all paths share the exact same topology, making it simpler and more robust."*

* **Q: How are you implementing ResNeXt on CPU without extreme memory usage?**
  > *"We implement the Split-Transform-Merge paradigm using Grouped Convolutions (`groups=4`) in Keras/TensorFlow. This executes all 4 paths in a single hardware-optimized matrix operation, keeping parameter count down to ~80k and training time to under 3 minutes on a laptop CPU."*
