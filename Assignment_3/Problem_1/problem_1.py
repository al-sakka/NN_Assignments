"""
Problem 1: Synthetic Data Generation using Conditional VAE
===========================================================
Pipeline:
  1. Load ReducedMNIST (only 350 samples per class for training)
  2. Apply data augmentation (15x) to boost CVAE training data
  3. Train a Conditional VAE on the augmented data
  4. Train a LeNet-5 classifier on the original 350/class data
  5. Generate 5 runs × 1000 samples/class from the CVAE
  6. Filter generated samples by LeNet-5 confidence into datasets A, B, C
  7. Retrain LeNet-5 with real + synthetic data and compare accuracies
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# ============================================================
# 0. REPRODUCIBILITY & DEVICE
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# 1. DATA LOADING — ReducedMNIST from folder structure
# ============================================================
# The dataset is stored as images in folders named 0-9.
# We use only 350 images per class for training (limited data scenario).

TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "ReducedMNIST_kaggle",
                         "Reduced MNIST Data", "Reduced Training data")
TEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "ReducedMNIST_kaggle",
                         "Reduced MNIST Data", "Reduced Testing data")

SAMPLES_PER_CLASS_TRAIN = 350   # limited real data
SAMPLES_PER_CLASS_1000  = 1000  # upper-bound baseline


def load_images_from_folder(root_dir, max_per_class=None):
    """Load images from class-subfolder structure. Returns tensors."""
    images, labels = [], []
    for class_idx in range(10):
        class_dir = os.path.join(root_dir, str(class_idx))
        if not os.path.isdir(class_dir):
            continue
        filenames = sorted(os.listdir(class_dir))
        if max_per_class is not None:
            filenames = filenames[:max_per_class]
        for fname in filenames:
            path = os.path.join(class_dir, fname)
            img = Image.open(path).convert("L")          # grayscale
            img = img.resize((28, 28))                    # ensure 28×28
            arr = np.array(img, dtype=np.float32) / 255.0 # normalise to [0,1]
            images.append(arr)
            labels.append(class_idx)
    images = np.array(images)[:, np.newaxis, :, :]  # (N, 1, 28, 28)
    labels = np.array(labels)
    return torch.tensor(images), torch.tensor(labels, dtype=torch.long)


print("Loading data...")
X_train_350, y_train_350 = load_images_from_folder(TRAIN_DIR, SAMPLES_PER_CLASS_TRAIN)
X_train_1000, y_train_1000 = load_images_from_folder(TRAIN_DIR, SAMPLES_PER_CLASS_1000)
X_test, y_test = load_images_from_folder(TEST_DIR)
print(f"  Train-350 : {X_train_350.shape}")
print(f"  Train-1000: {X_train_1000.shape}")
print(f"  Test      : {X_test.shape}")

# ============================================================
# 2. DATA AUGMENTATION (15×)
# ============================================================
# We augment the 350/class training data so the CVAE has enough variety.
# Augmentations: small rotation, translation, scaling — realistic for digits.

augmentation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),   # back to [0,1] tensor
])

AUG_FACTOR = 15  # produce 15 augmented copies per original image


def augment_dataset(X, y, factor=AUG_FACTOR):
    """Create `factor` augmented copies of each image."""
    aug_images, aug_labels = [], []
    for i in range(len(X)):
        for _ in range(factor):
            augmented = augmentation_transform(X[i])  # (1, 28, 28)
            aug_images.append(augmented)
            aug_labels.append(y[i].item())
    aug_images = torch.stack(aug_images)
    aug_labels = torch.tensor(aug_labels, dtype=torch.long)
    return aug_images, aug_labels


print("Augmenting data (15×) for CVAE training...")
X_aug, y_aug = augment_dataset(X_train_350, y_train_350, AUG_FACTOR)
# Combine original + augmented for CVAE training
X_cvae = torch.cat([X_train_350, X_aug], dim=0)
y_cvae = torch.cat([y_train_350, y_aug], dim=0)
print(f"  CVAE training set: {X_cvae.shape} (original + {AUG_FACTOR}× augmented)")

# ============================================================
# 3. CONDITIONAL VAE ARCHITECTURE
# ============================================================
# The CVAE conditions on the class label so we can generate specific digits.
# Architecture: Conv encoder → latent (μ, log σ²) → Conv decoder.
# Label is embedded and concatenated with the image / latent vector.

NUM_CLASSES = 10
LATENT_DIM = 64


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder with convolutional layers.
    - Encoder: image (1,28,28) + label embedding → μ, logvar
    - Decoder: z + label embedding → reconstructed image
    """

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Label embedding: map class → a learned 28×28 channel
        self.label_emb = nn.Embedding(num_classes, 28 * 28)

        # --- Encoder ---
        # Input: 2 channels (image + label map)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),   # (32, 14, 14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 7, 7)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # (128, 4, 4)
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),                                # 128*4*4 = 2048
        )
        self.fc_mu     = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        # --- Decoder ---
        # Input: z (latent_dim) + label embedding (latent_dim)
        self.label_emb_dec = nn.Embedding(num_classes, latent_dim)
        self.fc_decode = nn.Linear(latent_dim * 2, 2048)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # (64, 7, 7)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # (32, 14, 14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # (1, 28, 28)
            nn.Sigmoid(),  # pixel values in [0,1]
        )

    def encode(self, x, y):
        """Encode image x conditioned on label y → (μ, logvar)."""
        # Create a label channel: embed label → reshape to (1, 28, 28)
        label_map = self.label_emb(y).view(-1, 1, 28, 28)
        # Concatenate image and label map along channel dim → (2, 28, 28)
        x_cond = torch.cat([x, label_map], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z from N(μ, σ²) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        """Decode latent z conditioned on label y → reconstructed image."""
        label_emb = self.label_emb_dec(y)
        z_cond = torch.cat([z, label_emb], dim=1)  # (B, latent_dim*2)
        h = self.fc_decode(z_cond)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    """
    VAE loss = Reconstruction (BCE) + KL divergence.
    BCE measures how well the decoder reconstructs the input.
    KL pushes the latent distribution toward a standard normal N(0,1).
    """
    bce = F.binary_cross_entropy(recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


# ============================================================
# 4. LeNet-5 CLASSIFIER
# ============================================================
# Standard LeNet-5 adapted for 28×28 grayscale input.
# Used both for confidence filtering and for final evaluation.

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),    # (6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (6, 14, 14)
            nn.Conv2d(6, 16, 5),               # (16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2),                   # (16, 5, 5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_classifier(model, X, y, epochs=15, lr=1e-3, batch_size=64):
    """Train a classifier on given data."""
    model.to(DEVICE)
    model.train()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
    return model


def evaluate_classifier(model, X, y, batch_size=256):
    """Evaluate classifier accuracy."""
    model.eval()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ============================================================
# 5. TRAINING
# ============================================================

# --- 5a. Train LeNet-5 on 350 real samples (for confidence filtering) ---
print("\n=== Training LeNet-5 on 350 real samples (for filtering) ===")
lenet_filter = LeNet5()
lenet_filter = train_classifier(lenet_filter, X_train_350, y_train_350, epochs=20)
acc_350 = evaluate_classifier(lenet_filter, X_test, y_test)
print(f"  LeNet-5 (350 real) test accuracy: {acc_350*100:.2f}%")

# --- 5b. Train CVAE ---
print("\n=== Training Conditional VAE ===")
cvae = ConditionalVAE().to(DEVICE)
cvae_optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
cvae_dataset = TensorDataset(X_cvae, y_cvae)
cvae_loader = DataLoader(cvae_dataset, batch_size=128, shuffle=True)

CVAE_EPOCHS = 50
for epoch in range(1, CVAE_EPOCHS + 1):
    cvae.train()
    epoch_loss = 0
    for xb, yb in cvae_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        cvae_optimizer.zero_grad()
        recon, mu, logvar = cvae(xb, yb)
        loss = vae_loss(recon, xb, mu, logvar)
        loss.backward()
        cvae_optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}/{CVAE_EPOCHS}  Loss: {epoch_loss/len(cvae_dataset):.4f}")

# ============================================================
# 6. GENERATE SYNTHETIC DATA — 5 runs × 1000 per class
# ============================================================

def generate_samples(model, num_per_class=1000, num_classes=10):
    """Generate `num_per_class` samples for each class using the CVAE."""
    model.eval()
    all_images, all_labels = [], []
    with torch.no_grad():
        for c in range(num_classes):
            z = torch.randn(num_per_class, model.latent_dim).to(DEVICE)
            labels = torch.full((num_per_class,), c, dtype=torch.long).to(DEVICE)
            imgs = model.decode(z, labels).cpu()
            all_images.append(imgs)
            all_labels.append(labels.cpu())
    return torch.cat(all_images), torch.cat(all_labels)


print("\n=== Generating synthetic data (5 runs × 1000/class) ===")
all_gen_images, all_gen_labels = [], []
for run in range(5):
    imgs, lbls = generate_samples(cvae, num_per_class=1000)
    all_gen_images.append(imgs)
    all_gen_labels.append(lbls)
    print(f"  Run {run+1}: generated {imgs.shape[0]} samples")

# Pool all 5 runs together → 50,000 total
X_gen = torch.cat(all_gen_images)
y_gen = torch.cat(all_gen_labels)
print(f"  Total generated: {X_gen.shape[0]}")

# ============================================================
# 7. CONFIDENCE FILTERING using LeNet-5
# ============================================================
# Dataset A: all generated samples
# Dataset B: confidence ≥ 0.9  (high-quality)
# Dataset C: 0.6 ≤ confidence ≤ 0.9  (medium-quality)

def compute_confidences(model, X, batch_size=256):
    """Return max softmax probability for each sample."""
    model.eval()
    confs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(DEVICE)
            probs = F.softmax(model(xb), dim=1)
            max_conf, _ = probs.max(dim=1)
            confs.append(max_conf.cpu())
    return torch.cat(confs)


print("\n=== Filtering generated samples by confidence ===")
confidences = compute_confidences(lenet_filter, X_gen)

# Dataset A: all
mask_A = torch.ones(len(X_gen), dtype=torch.bool)
# Dataset B: confidence ≥ 0.9
mask_B = confidences >= 0.9
# Dataset C: 0.6 ≤ confidence < 0.9
mask_C = (confidences >= 0.6) & (confidences < 0.9)

datasets = {
    "A (all)":       (X_gen[mask_A], y_gen[mask_A]),
    "B (conf≥0.9)":  (X_gen[mask_B], y_gen[mask_B]),
    "C (0.6≤conf<0.9)": (X_gen[mask_C], y_gen[mask_C]),
}

for name, (X_d, y_d) in datasets.items():
    print(f"  Dataset {name}: {len(X_d)} samples")

# ============================================================
# 8. RETRAIN & EVALUATE — real + synthetic
# ============================================================
# For each synthetic dataset, we combine 350 real + synthetic, retrain LeNet-5,
# and measure test accuracy. Compare with baselines.

print("\n=== Evaluation ===")
results = {}

# Baseline: 350 real
results["350 real (baseline)"] = acc_350

# Baseline: 1000 real
lenet_1000 = LeNet5()
lenet_1000 = train_classifier(lenet_1000, X_train_1000, y_train_1000, epochs=20)
acc_1000 = evaluate_classifier(lenet_1000, X_test, y_test)
results["1000 real (baseline)"] = acc_1000

# Real 350 + each synthetic dataset
for name, (X_syn, y_syn) in datasets.items():
    X_combined = torch.cat([X_train_350, X_syn])
    y_combined = torch.cat([y_train_350, y_syn])
    model = LeNet5()
    model = train_classifier(model, X_combined, y_combined, epochs=20)
    acc = evaluate_classifier(model, X_test, y_test)
    results[f"350 real + VAE {name}"] = acc

# ============================================================
# 9. PRINT RESULTS TABLE
# ============================================================
print("\n" + "=" * 60)
print("PROBLEM 1 — CONDITIONAL VAE RESULTS")
print("=" * 60)
print(f"{'Training Data':<40} {'Accuracy (%)':<12}")
print("-" * 52)
for desc, acc in results.items():
    print(f"{desc:<40} {acc*100:>8.2f}%")
print("=" * 60)

# Save results to file for the report
results_file = os.path.join(os.path.dirname(__file__), "vae_results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    f.write("Training Data,Accuracy\n")
    for desc, acc in results.items():
        f.write(f"{desc},{acc*100:.2f}\n")
print(f"\nResults saved to {results_file}")
