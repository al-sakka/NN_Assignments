"""
Problem 2: Synthetic Data Generation using Conditional DCGAN
=============================================================
Pipeline (identical to Problem 1 but using a GAN instead of VAE):
  1. Load ReducedMNIST (only 350 samples per class for training)
  2. Apply data augmentation (15×) to boost GAN training data
  3. Train a Conditional DCGAN (Deep Convolutional GAN)
  4. Train a LeNet-5 classifier on the original 350/class data
  5. Generate 5 runs × 1000 samples/class from the cDCGAN
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image

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
# 1. DATA LOADING
# ============================================================
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "ReducedMNIST_kaggle",
                         "Reduced MNIST Data", "Reduced Training data")
TEST_DIR  = os.path.join(os.path.dirname(__file__), "..", "ReducedMNIST_kaggle",
                         "Reduced MNIST Data", "Reduced Testing data")

SAMPLES_PER_CLASS_TRAIN = 350


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
            img = Image.open(path).convert("L")
            img = img.resize((28, 28))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_idx)
    images = np.array(images)[:, np.newaxis, :, :]
    labels = np.array(labels)
    return torch.tensor(images), torch.tensor(labels, dtype=torch.long)


print("Loading data...")
X_train_350, y_train_350 = load_images_from_folder(TRAIN_DIR, SAMPLES_PER_CLASS_TRAIN)
X_train_1000, y_train_1000 = load_images_from_folder(TRAIN_DIR, 1000)
X_test, y_test = load_images_from_folder(TEST_DIR)
print(f"  Train-350 : {X_train_350.shape}")
print(f"  Train-1000: {X_train_1000.shape}")
print(f"  Test      : {X_test.shape}")

# ============================================================
# 2. DATA AUGMENTATION (15×)
# ============================================================
augmentation_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

AUG_FACTOR = 15


def augment_dataset(X, y, factor=AUG_FACTOR):
    aug_images, aug_labels = [], []
    for i in range(len(X)):
        for _ in range(factor):
            augmented = augmentation_transform(X[i])
            aug_images.append(augmented)
            aug_labels.append(y[i].item())
    return torch.stack(aug_images), torch.tensor(aug_labels, dtype=torch.long)


print("Augmenting data (15×) for GAN training...")
X_aug, y_aug = augment_dataset(X_train_350, y_train_350, AUG_FACTOR)
X_gan_train = torch.cat([X_train_350, X_aug], dim=0)
y_gan_train = torch.cat([y_train_350, y_aug], dim=0)
print(f"  GAN training set: {X_gan_train.shape}")

# ============================================================
# 3. CONDITIONAL DCGAN ARCHITECTURE
# ============================================================
# The Generator takes noise z + label → fake image.
# The Discriminator takes image + label → real/fake score.

NUM_CLASSES = 10
NOISE_DIM = 100  # dimension of random noise vector z


class Generator(nn.Module):
    """
    Conditional Generator.
    Input: noise z (100) + label embedding (10) → concatenated → upsample to (1,28,28).
    Uses transposed convolutions (fractionally-strided convolutions).
    """

    def __init__(self, noise_dim=NOISE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.noise_dim = noise_dim
        # Label embedding: class → dense vector
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Project noise+label to a spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        # Upsample: (256, 7, 7) → (1, 28, 28)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (128, 14, 14)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # (64, 28, 28)
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),                        # (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, z, y):
        label_emb = self.label_emb(y)
        x = torch.cat([z, label_emb], dim=1)
        x = self.fc(x).view(-1, 256, 7, 7)
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Conditional Discriminator.
    Input: image (1,28,28) + label map (1,28,28) → real/fake probability.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 28 * 28)

        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),   # (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        label_map = self.label_emb(y).view(-1, 1, 28, 28)
        x_cond = torch.cat([x, label_map], dim=1)  # (2, 28, 28)
        return self.model(x_cond).squeeze(1)


# ============================================================
# 4. LeNet-5 CLASSIFIER (same as Problem 1)
# ============================================================

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
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
    model.to(DEVICE).train()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model


def evaluate_classifier(model, X, y, batch_size=256):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


# ============================================================
# 5. TRAINING
# ============================================================

# --- 5a. Train LeNet-5 on 350 real samples ---
print("\n=== Training LeNet-5 on 350 real samples (for filtering) ===")
lenet_filter = LeNet5()
lenet_filter = train_classifier(lenet_filter, X_train_350, y_train_350, epochs=20)
acc_350 = evaluate_classifier(lenet_filter, X_test, y_test)
print(f"  LeNet-5 (350 real) test accuracy: {acc_350*100:.2f}%")

# --- 5b. Train Conditional DCGAN ---
print("\n=== Training Conditional DCGAN ===")
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)

# Use separate Adam optimizers with GAN-specific learning rates
optG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion_gan = nn.BCELoss()

gan_loader = DataLoader(TensorDataset(X_gan_train, y_gan_train),
                        batch_size=128, shuffle=True, drop_last=True)

GAN_EPOCHS = 100
for epoch in range(1, GAN_EPOCHS + 1):
    for real_imgs, real_labels in gan_loader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(DEVICE)
        real_labels = real_labels.to(DEVICE)

        # Labels for real/fake with label smoothing
        real_target = torch.full((batch_size,), 0.9, device=DEVICE)
        fake_target = torch.zeros(batch_size, device=DEVICE)

        # --- Train Discriminator ---
        # Real
        out_real = netD(real_imgs, real_labels)
        loss_real = criterion_gan(out_real, real_target)

        # Fake
        z = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        fake_imgs = netG(z, fake_labels)
        out_fake = netD(fake_imgs.detach(), fake_labels)
        loss_fake = criterion_gan(out_fake, fake_target)

        loss_D = loss_real + loss_fake
        optD.zero_grad()
        loss_D.backward()
        optD.step()

        # --- Train Generator ---
        # Generator wants discriminator to think fakes are real
        out_fake2 = netD(fake_imgs, fake_labels)
        loss_G = criterion_gan(out_fake2, torch.ones(batch_size, device=DEVICE))
        optG.zero_grad()
        loss_G.backward()
        optG.step()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch}/{GAN_EPOCHS}  D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}")

# ============================================================
# 6. GENERATE SYNTHETIC DATA — 5 runs × 1000 per class
# ============================================================

def generate_gan_samples(generator, num_per_class=1000, num_classes=10):
    generator.eval()
    all_images, all_labels = [], []
    with torch.no_grad():
        for c in range(num_classes):
            z = torch.randn(num_per_class, NOISE_DIM, device=DEVICE)
            labels = torch.full((num_per_class,), c, dtype=torch.long, device=DEVICE)
            imgs = generator(z, labels).cpu()
            all_images.append(imgs)
            all_labels.append(labels.cpu())
    return torch.cat(all_images), torch.cat(all_labels)


print("\n=== Generating synthetic data (5 runs × 1000/class) ===")
all_gen_images, all_gen_labels = [], []
for run in range(5):
    imgs, lbls = generate_gan_samples(netG, num_per_class=1000)
    all_gen_images.append(imgs)
    all_gen_labels.append(lbls)
    print(f"  Run {run+1}: generated {imgs.shape[0]} samples")

X_gen = torch.cat(all_gen_images)
y_gen = torch.cat(all_gen_labels)
print(f"  Total generated: {X_gen.shape[0]}")

# ============================================================
# 7. CONFIDENCE FILTERING
# ============================================================

def compute_confidences(model, X, batch_size=256):
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

mask_A = torch.ones(len(X_gen), dtype=torch.bool)
mask_B = confidences >= 0.9
mask_C = (confidences >= 0.6) & (confidences < 0.9)

datasets = {
    "A (all)":            (X_gen[mask_A], y_gen[mask_A]),
    "B (conf≥0.9)":       (X_gen[mask_B], y_gen[mask_B]),
    "C (0.6≤conf<0.9)":   (X_gen[mask_C], y_gen[mask_C]),
}

for name, (X_d, y_d) in datasets.items():
    print(f"  Dataset {name}: {len(X_d)} samples")

# ============================================================
# 8. RETRAIN & EVALUATE
# ============================================================
print("\n=== Evaluation ===")
results = {}

results["350 real (baseline)"] = acc_350

lenet_1000 = LeNet5()
lenet_1000 = train_classifier(lenet_1000, X_train_1000, y_train_1000, epochs=20)
results["1000 real (baseline)"] = evaluate_classifier(lenet_1000, X_test, y_test)

for name, (X_syn, y_syn) in datasets.items():
    X_combined = torch.cat([X_train_350, X_syn])
    y_combined = torch.cat([y_train_350, y_syn])
    model = LeNet5()
    model = train_classifier(model, X_combined, y_combined, epochs=20)
    results[f"350 real + GAN {name}"] = evaluate_classifier(model, X_test, y_test)

# ============================================================
# 9. PRINT RESULTS TABLE
# ============================================================
print("\n" + "=" * 60)
print("PROBLEM 2 — CONDITIONAL DCGAN RESULTS")
print("=" * 60)
print(f"{'Training Data':<40} {'Accuracy (%)':<12}")
print("-" * 52)
for desc, acc in results.items():
    print(f"{desc:<40} {acc*100:>8.2f}%")
print("=" * 60)

results_file = os.path.join(os.path.dirname(__file__), "gan_results.txt")
with open(results_file, "w", encoding="utf-8") as f:
    f.write("Training Data,Accuracy\n")
    for desc, acc in results.items():
        f.write(f"{desc},{acc*100:.2f}\n")
print(f"\nResults saved to {results_file}")
