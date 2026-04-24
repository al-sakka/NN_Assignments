"""
Assignment 2 - Problem 1: MLP Digit Classification on ReducedMNIST
==================================================================
Train Multilayer Perceptrons with 1, 3, and 4 hidden layers using
three different feature representations (DCT, PCA, Autoencoder)
on the ReducedMNIST dataset.

Requirements: torch, numpy, scipy, scikit-learn, matplotlib, Pillow
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(__file__).resolve().parent / "ReducedMNIST_kaggle"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP hyper-parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Feature parameters
DCT_BLOCK = 15          # keep top-left 15x15 DCT coefficients -> 225 features
PCA_VARIANCE = 0.95     # keep components explaining 95% of variance

# Autoencoder parameters
AE_BOTTLENECK = 64      # size of the compressed representation
AE_EPOCHS = 20
AE_LR = 1e-3

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_split(split_dir: Path):
    """Load images and labels from a directory with digit sub-folders (0-9).

    Each sub-folder contains .jpg/.png images of that digit.
    Returns:
        images : ndarray of shape (N, 784), float64, values in [0, 1]
        labels : ndarray of shape (N,),     int32
    """
    images, labels = [], []
    for digit in range(10):
        class_dir = split_dir / str(digit)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        files = sorted(
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        for fpath in files:
            img = np.array(Image.open(fpath).convert("L"), dtype=np.float64)
            images.append(img.ravel() / 255.0)
            labels.append(digit)

    return np.vstack(images), np.array(labels, dtype=np.int32)


def load_dataset(root: Path):
    """Auto-detect train/test folders under *root* and load both splits."""
    train_dir = test_dir = None
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        # Must contain sub-folders 0..9
        if not all((d / str(i)).is_dir() for i in range(10)):
            continue
        name = d.name.lower()
        if train_dir is None and ("train" in name or "trainging" in name):
            train_dir = d
        if test_dir is None and "test" in name:
            test_dir = d

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(f"Cannot find train/test folders under {root}")

    print(f"Train dir: {train_dir}")
    print(f"Test  dir: {test_dir}")

    X_train, y_train = load_split(train_dir)
    X_test, y_test = load_split(test_dir)
    print(f"Loaded {X_train.shape[0]} train, {X_test.shape[0]} test samples.\n")
    return X_train, y_train, X_test, y_test


# ===========================================================================
# 2. FEATURE EXTRACTION
# ===========================================================================

def extract_dct(images: np.ndarray, block_size: int = DCT_BLOCK) -> np.ndarray:
    """Apply 2-D DCT and keep the top-left block as features.

    The top-left DCT coefficients capture the lowest-frequency (most
    important) patterns in each image, acting as a compact descriptor.
    """
    n = images.shape[0]
    features = np.zeros((n, block_size * block_size), dtype=np.float64)
    for i in range(n):
        dct_img = dctn(images[i].reshape(28, 28), norm="ortho")
        features[i] = dct_img[:block_size, :block_size].ravel()
    print(f"  DCT: {n} images -> {block_size*block_size} features")
    return features


def extract_pca(X_train: np.ndarray, X_test: np.ndarray,
                var_threshold: float = PCA_VARIANCE):
    """Fit PCA on training data, then transform both splits.

    We keep enough components to explain *var_threshold* of the total
    variance.  This removes redundant/noisy dimensions.
    """
    pca = PCA(n_components=var_threshold, svd_solver="full")
    train_feat = pca.fit_transform(X_train)
    test_feat = pca.transform(X_test)
    n_comp = pca.n_components_
    explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA: keeping {n_comp} components ({explained:.1f}% variance)")
    return train_feat, test_feat, n_comp


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder for feature extraction.

    Architecture: 784 -> 256 -> bottleneck -> 256 -> 784
    After training, only the encoder part is used to produce features.
    """

    def __init__(self, input_dim: int = 784, bottleneck: int = AE_BOTTLENECK):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),       # output in [0, 1] to match input range
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_train: np.ndarray):
    """Train the autoencoder and return the model.

    The autoencoder learns a compressed representation by being forced to
    reconstruct its input through a narrow bottleneck layer.
    """
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    ae = Autoencoder(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=AE_LR)

    print(f"  Training Autoencoder ({AE_EPOCHS} epochs, bottleneck={AE_BOTTLENECK})...")
    for epoch in range(AE_EPOCHS):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            output = ae(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"    Epoch {epoch+1}/{AE_EPOCHS}  Loss: {avg:.6f}")
    return ae


def extract_autoencoder(ae: Autoencoder, X_train: np.ndarray, X_test: np.ndarray):
    """Pass data through the encoder to get bottleneck features."""
    ae.eval()
    with torch.no_grad():
        train_feat = ae.encode(
            torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
        test_feat = ae.encode(
            torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
    print(f"  Autoencoder: {X_train.shape[0]} train / {X_test.shape[0]} test "
          f"-> {train_feat.shape[1]} features")
    return train_feat, test_feat


# ===========================================================================
# 3. MLP MODEL
# ===========================================================================

def build_mlp(input_dim: int, hidden_layers: list[int], num_classes: int = 10):
    """Construct an MLP with the specified hidden layer sizes.

    Each hidden layer is followed by ReLU activation.
    The output layer has *num_classes* neurons (no softmax here because
    CrossEntropyLoss in PyTorch applies log-softmax internally).

    Example:
        build_mlp(225, [256, 128, 64])  # 3 hidden layers
        -> Linear(225,256) -> ReLU -> Linear(256,128) -> ReLU
           -> Linear(128,64) -> ReLU -> Linear(64,10)
    """
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


# Hidden layer configurations to test
MLP_CONFIGS = {
    "1-Hidden": [128],
    "3-Hidden": [256, 128, 64],
    "4-Hidden": [256, 128, 64, 32],
}


# ===========================================================================
# 4. TRAINING & EVALUATION
# ===========================================================================

def train_and_evaluate(model, train_feat, train_labels, test_feat, test_labels):
    """Train the MLP and return (test_accuracy%, training_time_ms).

    Training loop:
      - Uses Adam optimizer (adaptive learning rate).
      - CrossEntropyLoss = Softmax + Negative-Log-Likelihood.
      - Mini-batch gradient descent for efficiency.
    """
    # Standardize features (zero-mean, unit-variance) for stable training
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat)
    test_feat = scaler.transform(test_feat)

    # Create PyTorch data loaders
    train_ds = TensorDataset(
        torch.tensor(train_feat, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(test_feat, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training ---
    model.train()
    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_time_ms = (time.perf_counter() - t_start) * 1000

    # --- Evaluation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = 100.0 * correct / total
    return accuracy, train_time_ms


# ===========================================================================
# 5. MAIN — RUN ALL EXPERIMENTS
# ===========================================================================

def main():
    import platform, os
    print("=" * 70)
    print("Problem 1: MLP Classification on ReducedMNIST")
    print(f"Device: {DEVICE}")
    print(f"Processor: {platform.processor()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}, PyTorch: {torch.__version__}")
    print("=" * 70)

    # ---- Load data ----
    X_train, y_train, X_test, y_test = load_dataset(DATASET_ROOT)

    # ---- Extract features ----
    print("Extracting features...")

    dct_train = extract_dct(X_train, DCT_BLOCK)
    dct_test = extract_dct(X_test, DCT_BLOCK)

    pca_train, pca_test, n_pca = extract_pca(X_train, X_test, PCA_VARIANCE)

    ae_model = train_autoencoder(X_train)
    ae_train, ae_test = extract_autoencoder(ae_model, X_train, X_test)

    # Bundle features: (name, train_feat, test_feat)
    feature_sets = [
        ("DCT", dct_train, dct_test),
        ("PCA", pca_train, pca_test),
        ("AutoEncoder", ae_train, ae_test),
    ]

    # ---- Train & evaluate MLPs ----
    # Results table:  results[feat_name][config_name] = (accuracy, time_ms)
    results = {}

    for feat_name, feat_train, feat_test in feature_sets:
        results[feat_name] = {}
        input_dim = feat_train.shape[1]
        print(f"\n--- Feature: {feat_name} (dim={input_dim}) ---")

        for config_name, hidden_sizes in MLP_CONFIGS.items():
            print(f"  MLP {config_name}: {hidden_sizes} ... ", end="", flush=True)
            model = build_mlp(input_dim, hidden_sizes)
            acc, t_ms = train_and_evaluate(
                model, feat_train, y_train, feat_test, y_test
            )
            results[feat_name][config_name] = (acc, t_ms)
            print(f"Acc={acc:.1f}%  Time={t_ms:.1f} ms")

    # ---- Print summary table ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    header = f"{'Configuration':<16}"
    for feat_name in ["DCT", "PCA", "AutoEncoder"]:
        header += f"| {'Acc (%)':>8} {'Time (ms)':>10} "
    print(header)
    print("-" * len(header))

    for config_name in MLP_CONFIGS:
        row = f"{config_name:<16}"
        for feat_name in ["DCT", "PCA", "AutoEncoder"]:
            acc, t = results[feat_name][config_name]
            row += f"| {acc:>8.1f} {t:>10.1f} "
        print(row)

    print()

    # ---- Plot results ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    config_names = list(MLP_CONFIGS.keys())
    x = np.arange(len(config_names))
    width = 0.25

    for i, feat_name in enumerate(["DCT", "PCA", "AutoEncoder"]):
        accs = [results[feat_name][c][0] for c in config_names]
        times = [results[feat_name][c][1] for c in config_names]
        axes[0].bar(x + i * width, accs, width, label=feat_name)
        axes[1].bar(x + i * width, times, width, label=feat_name)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy by MLP Depth & Feature Type")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(config_names)
    axes[0].legend()
    axes[0].set_ylim(80, 100)

    axes[1].set_ylabel("Processing Time (ms)")
    axes[1].set_title("Processing Time by MLP Depth & Feature Type")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(config_names)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "mlp_results.png", dpi=150)
    print("Plot saved to mlp_results.png")


if __name__ == "__main__":
    main()
