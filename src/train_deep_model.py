"""
train_deep_model.py
--------------------
Deep Learning Training Pipeline for Audio Deepfake Detection.

Purpose:
    Constructs and trains a ResNeXt-inspired Convolutional Neural Network (CNN) architecture 
    to classify audio clips into genuine speech (0) vs. AI-generated deepfake speech (1)
    using 2D spectral representations (MFCC, LFCC, CQCC).

Reference Context:
    Inspired by Tahaoglu et al. (2025), "Deepfake Audio Detection with Spectral Features 
    and ResNeXt-Based Architecture". This script implements a simplified, lightweight version 
    of ResNeXt optimized for laptop CPU execution while retaining core grouped convolution 
    and residual connection concepts.

Viva / Academic Explanation Guide:
    1. Why ResNeXt?
       - ResNeXt introduces the concept of "Cardinality" (the number of parallel, grouped 
         convolution paths within a block).
       - Instead of making the network deeper (more layers) or wider (more channels per layer), 
         ResNeXt splits feature maps into multiple smaller paths, processes them in parallel, 
         and aggregates (adds) their outputs.
       - This significantly increases representation power and feature selectivity while 
         keeping total parameter count and FLOPS low.

    2. Why Reshape 1D Vector (78) into 2D Grid (6 x 13 x 1)?
       - Standard 1D feature vectors lose 2D spatial-frequency context.
       - Arranging 78 features into 6 rows x 13 columns allows 2D convolutional filters 
         to learn local patterns across coefficient indices (columns: 1..13) and across 
         acoustic feature modalities (rows: MFCC mean, MFCC std, LFCC mean, LFCC std, CQCC mean, CQCC std).

    3. Key Architectural Elements:
       - Bottleneck 1x1 Conv: Reduces channel dimension before expensive 3x3 convolutions.
       - Grouped 3x3 Conv: Performs parallel feature transformations across cardinality paths.
       - Residual Skip Connection (Identity Shortcut): Adds input directly to block output 
         (shortcut + x), preventing vanishing gradients and aiding smooth optimization.
       - GlobalAveragePooling2D: Replaces heavy Flatten + Dense layers, reducing parameter 
         count dramatically and preventing overfitting.
       - Sigmoid Activation: Outputs a continuous probability [0.0, 1.0] for binary classification.

Imports:
    os, argparse, typing, numpy, pandas, matplotlib, sklearn, tensorflow
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


def reshape_features_to_2d(X: np.ndarray) -> np.ndarray:
    """
    Reshapes a 1D feature vector array of size 78 into a 2D grid of shape (6, 13, 1).

    Parameters:
        X (np.ndarray): 2D array of shape (N_samples, 78).

    Returns:
        np.ndarray: 4D tensor array of shape (N_samples, 6, 13, 1) suitable for Conv2D.
        
    Explanation for Viva:
        - 6 rows correspond to feature statistic groups:
          Row 0: MFCC Mean (13)
          Row 1: MFCC Std  (13)
          Row 2: LFCC Mean (13)
          Row 3: LFCC Std  (13)
          Row 4: CQCC Mean (13)
          Row 5: CQCC Std  (13)
        - 13 columns correspond to the cepstral coefficient index (1 to 13).
        - 1 channel corresponds to single-channel (grayscale) 2D input.
    """
    if X.shape[1] != 78:
        raise ValueError(f"Expected feature vector dimension of 78, but got {X.shape[1]}")

    N_samples = X.shape[0]
    return X.reshape(N_samples, 6, 13, 1)


def build_resnext_block(inputs: tf.Tensor, filters: int = 32, cardinality: int = 4, stride: int = 1) -> tf.Tensor:
    """
    Constructs a simplified grouped-convolution ResNeXt bottleneck block with a residual skip connection.

    Parameters:
        inputs (tf.Tensor): Input feature map tensor.
        filters (int): Target output channel depth (default: 32).
        cardinality (int): Number of grouped convolution paths (default: 4).
        stride (int): Convolution stride length for spatial downsampling (default: 1).

    Returns:
        tf.Tensor: Transformed tensor after grouped convolution and residual shortcut addition.

    Viva Explanation:
        - Step 1 (1x1 Conv Bottleneck): Reduces feature depth to filters/2 to compress computation.
        - Step 2 (3x3 Grouped Conv): Splits channels into `cardinality` groups, processing sub-features in parallel.
        - Step 3 (1x1 Conv Expansion): Expands channels back to target `filters`.
        - Step 4 (Residual Connection): Adds `shortcut + output`. If shortcut shape differs, a 1x1 conv projects it.
    """
    shortcut = inputs
    bottleneck_width = max(filters // 2, cardinality)

    # 1. Bottleneck 1x1 Convolution
    x = layers.Conv2D(bottleneck_width, kernel_size=(1, 1), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 2. Grouped 3x3 Convolution (Cardinality paths)
    x = layers.Conv2D(bottleneck_width, kernel_size=(3, 3), strides=stride, padding="same", groups=cardinality)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3. Expansion 1x1 Convolution
    x = layers.Conv2D(filters, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # 4. Adjust shortcut dimensions if necessary (e.g. channel size mismatch or stride > 1)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 5. Residual Addition (Element-wise sum of identity shortcut and transformed features)
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x


def build_deepfake_detector_model(
    input_shape: Tuple[int, int, int] = (6, 13, 1),
    cardinality: int = 4
) -> tf.keras.Model:
    """
    Builds the complete lightweight ResNeXt-inspired CNN architecture for binary audio deepfake detection.

    Parameters:
        input_shape (Tuple[int, int, int]): Shape of input 2D feature grid (6, 13, 1).
        cardinality (int): Number of grouped convolution paths in ResNeXt blocks (default: 4).

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.

    Viva / Architectural Layer Breakdown:
        - Input Layer: Accepts (6, 13, 1) spectral feature grids.
        - Initial Conv Stem: 3x3 Conv (16 filters) + BatchNorm + ReLU to extract basic spatial features.
        - ResNeXt Stage 1: ResNeXt block with 32 filters and cardinality=4.
        - ResNeXt Stage 2: ResNeXt block with 64 filters and cardinality=4.
        - Global Average Pooling: Summarizes 2D feature maps into a single 1D vector by averaging across space.
          This reduces parameters dramatically compared to Flatten, preventing overfitting.
        - Dense Classification Head: 32 units + ReLU + Dropout(0.3) -> 1 unit + Sigmoid activation.
    """
    inputs = layers.Input(shape=input_shape, name="input_spectral_grid")

    # Initial Convolutional Stem
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Stage 1: First ResNeXt Block (32 filters)
    x = build_resnext_block(x, filters=32, cardinality=cardinality, stride=1)

    # Stage 2: Second ResNeXt Block (64 filters)
    x = build_resnext_block(x, filters=64, cardinality=cardinality, stride=1)

    # Global Average Pooling (Spatial Reduction)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Dense Classification Head
    x = layers.Dense(32, activation="relu", name="dense_fc1")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="binary_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNeXt_Audio_Deepfake_Detector")

    # Compile model with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


def plot_and_save_training_curves(
    history: tf.keras.callbacks.History,
    save_path: str = "results/training_curves.png"
) -> None:
    """
    Generates and saves clean side-by-side Matplotlib plots of Training vs Validation Loss and Accuracy.

    Parameters:
        history (tf.keras.callbacks.History): Keras training history object.
        save_path (str): File path where the plot PNG image will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5), dpi=300)

    # Plot 1: Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, "ro-", label="Validation Accuracy", linewidth=2)
    plt.title("ResNeXt Model: Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Plot 2: Loss Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, "ro-", label="Validation Loss", linewidth=2)
    plt.title("ResNeXt Model: Loss vs. Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss (Binary Crossentropy)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Training curves saved to: {save_path}")


def train_deep_model(
    features_csv: str = "data/features/features.csv",
    model_save_path: str = "models/deep_model.h5",
    curves_save_path: str = "results/training_curves.png",
    epochs: int = 30,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[tf.keras.Model, Dict[str, float]]:
    """
    Executes the end-to-end deep learning training pipeline:
    1. Loads dataset from features.csv
    2. Standardizes features with StandardScaler
    3. Reshapes 78-dim vectors into (6, 13, 1) 2D grids
    4. Performs stratified 80/20 train/test split
    5. Builds ResNeXt-inspired CNN architecture
    6. Trains model with EarlyStopping and ModelCheckpoint
    7. Evaluates test set metrics, saves training curve plots and final model.h5 file.

    Parameters:
        features_csv (str): Path to features CSV file.
        model_save_path (str): Output destination for trained Keras model (.h5).
        curves_save_path (str): Output destination for loss/accuracy curve plot PNG.
        epochs (int): Maximum training epochs (default: 30).
        batch_size (int): Mini-batch size (default: 32).
        test_size (float): Proportion of dataset held out for testing (default: 0.2).
        random_state (int): Seed for reproducible train/test split.

    Returns:
        Tuple[tf.keras.Model, Dict[str, float]]: (Trained model, test metrics dictionary).
    """
    print("=" * 65)
    print("   RESNEXT DEEP LEARNING MODEL TRAINING (TENSORFLOW/KERAS)   ")
    print("=" * 65)
    print(f"Features Dataset Path  : {features_csv}")
    print(f"Model Target Path      : {model_save_path}")
    print(f"Curves Plot Path       : {curves_save_path}")
    print("=" * 65)

    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features file not found at: {features_csv}. Run feature_extraction.py first.")

    # 1. Load CSV data
    print("\n[STEP 1] Loading dataset...")
    df = pd.read_csv(features_csv)
    print(f"  Loaded dataset shape: {df.shape} ({df.shape[0]} samples, {df.shape[1]} columns)")

    # Separate features X and binary labels y (0=Real, 1=Fake)
    feature_cols = [c for c in df.columns if c != "label"]
    X_raw = df[feature_cols].values
    y = df["label"].values.astype(np.float32)

    # 2. Standardize feature values using StandardScaler
    print("\n[STEP 2] Standardizing feature values...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Reshape 1D (78) feature vectors to 2D (6, 13, 1) grids
    print("\n[STEP 3] Reshaping 78-dim vectors into (6, 13, 1) 2D spectral grids...")
    X_2d = reshape_features_to_2d(X_scaled)
    print(f"  Reshaped tensor batch shape: {X_2d.shape}")

    # 4. Stratified Train/Test Split (80% train, 20% test)
    print(f"\n[STEP 4] Splitting dataset into {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test...")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )
    test_split_path = Path("data/features/test_split.csv")
    test_split_path.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(test_split_path, index=False)
    print(f"  Saved test split to : {test_split_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Training set size   : {X_train.shape[0]} samples")
    print(f"  Testing set size    : {X_test.shape[0]} samples")

    # 5. Build ResNeXt-inspired CNN Model
    print("\n[STEP 5] Constructing ResNeXt-inspired CNN Architecture...")
    model = build_deepfake_detector_model(input_shape=(6, 13, 1), cardinality=4)
    model.summary()

    # 6. Configure Callbacks (Early Stopping + Model Checkpoint)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # 7. Train Model
    print(f"\n[STEP 6] Training model for up to {epochs} epochs (Batch Size: {batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,  # 15% of training set for validation monitoring
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early_stop, cb_checkpoint],
        verbose=1
    )

    # 8. Evaluate on Test Set
    print("\n[STEP 7] Evaluating trained ResNeXt model on held-out test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print("=" * 65)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test AUC      : {test_auc:.4f}")
    print("=" * 65)

    # 9. Plot and save accuracy & loss curves
    print("\n[STEP 8] Generating training performance curves plot...")
    plot_and_save_training_curves(history, save_path=curves_save_path)

    # 10. Ensure model is saved to model_save_path
    model.save(model_save_path)
    print(f"[OK] Trained model successfully saved to: {model_save_path}")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc)
    }

    return model, metrics


def main() -> None:
    """
    Main entry point for command line execution of train_deep_model.py.
    """
    parser = argparse.ArgumentParser(description="Train ResNeXt Deep Learning Model for Audio Deepfake Detection")
    parser.add_argument("--features-csv", type=str, default="data/features/features.csv", help="Path to features CSV")
    parser.add_argument("--model-path", type=str, default="models/deep_model.h5", help="Path to save trained model")
    parser.add_argument("--curves-path", type=str, default="results/training_curves.png", help="Path to save training curves plot")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")

    args = parser.parse_args()

    train_deep_model(
        features_csv=args.features_csv,
        model_save_path=args.model_path,
        curves_save_path=args.curves_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
