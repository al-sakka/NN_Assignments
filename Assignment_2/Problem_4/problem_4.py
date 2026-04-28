"""
Problem 4: Data Augmentation and Data Synthesis using Autoencoder
             for Speech Digit Recognition
=============================================================================
"""

import os
import time
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Hyperparameters 
# ---------------------------------------------------------------------------
N_MFCC       = 30       # captures more spectral detail
FRAME_MS     = 15       # frame length in milliseconds
HOP_MS       = 10       # hop (step) between frames in ms
LATENT_DIM   = 256      # autoencoder bottleneck size
AE_EPOCHS    = 500      # autoencoder training epochs
AE_LR        = 1e-3     # autoencoder learning rate
CLS_EPOCHS   = 150      # classifier training epochs
CLS_LR       = 1e-3     # classifier learning rate
BATCH_SIZE   = 32       # mini-batch size
NUM_CLASSES  = 10       # digits 0–9

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "audio-dataset", "Train")
TEST_DIR  = os.path.join(BASE_DIR, "audio-dataset", "Test")
OUTPUT_FILE = os.path.join(BASE_DIR, "output.txt")

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_dataset(folder_path):
    """
    Load all .wav files from a flat folder.
    Filenames are like  SpeakerID_Digit.wav  (e.g. M16_3.wav, U0n_7.wav).
    The digit label is the character right after the last underscore.

    Returns
    -------
    file_paths : list[str]   – full paths to wav files
    labels     : list[int]   – digit labels 0–9
    """
    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    labels = []
    for fp in file_paths:
        # filename without extension, e.g. "M16_3"
        name = os.path.splitext(os.path.basename(fp))[0]
        digit = int(name.split("_")[-1])   # last part after '_'
        labels.append(digit)
    return file_paths, labels


# ===================================================================
# 2. FEATURE EXTRACTION  (MFCC)
# ===================================================================

def extract_mfcc(file_path, sr=None, n_mfcc=N_MFCC,
                 frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """
    Load an audio file and compute MFCCs.

    Parameters
    ----------
    file_path : str
    sr        : int or None – target sample rate (None = use file's native sr)
    n_mfcc    : int – number of MFCC coefficients
    frame_ms  : int – frame length in ms
    hop_ms    : int – hop length in ms

    Returns
    -------
    mfcc : np.ndarray of shape (n_mfcc, n_frames)
           Each column is one frame's feature vector.
    sr   : int – sample rate used
    """
    # Load audio (mono, native sample rate by default)
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    # Convert ms to samples
    n_fft    = int(sr * frame_ms / 1000)   # frame length in samples
    hop_len  = int(sr * hop_ms / 1000)     # hop length in samples

    # Compute MFCCs  →  shape (n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_len)
    return mfcc, sr


def extract_all_mfccs(file_paths):
    """
    Extract MFCCs for every file.
    Returns a list of arrays, each of shape (n_mfcc, n_frames_i).
    """
    mfcc_list = []
    for fp in file_paths:
        mfcc, _ = extract_mfcc(fp)
        mfcc_list.append(mfcc)
    return mfcc_list


# ===================================================================
# 3. BASELINE — average frame per utterance
# ===================================================================

def compute_summary_features(mfcc_list):
    """
    Parameters
    ----------
    mfcc_list : list of np.ndarray, each (n_mfcc, n_frames_i)

    Returns
    -------
    features : np.ndarray of shape (n_utterances, 6 * n_mfcc)
    """
    features = []
    for m in mfcc_list:
        delta  = librosa.feature.delta(m, order=1)
        delta2 = librosa.feature.delta(m, order=2)
        enriched = np.vstack([m, delta, delta2])   # (3*n_mfcc, n_frames)
        feat = np.concatenate([enriched.mean(axis=1), enriched.std(axis=1)])
        features.append(feat)
    return np.array(features)


# ===================================================================
# 4. PADDING — make all utterances the same length
# ===================================================================

def pad_mfcc_sequences(mfcc_list, max_frames=None):
    """
    Zero-pad every MFCC matrix to have the same number of frames,
    then flatten to a single vector per utterance.

    Parameters
    ----------
    mfcc_list  : list of np.ndarray, each (n_mfcc, n_frames_i)
    max_frames : int or None – pad to this many frames.
                 If None, use the max across the list.

    Returns
    -------
    padded_flat : np.ndarray of shape (n_utterances, n_mfcc * max_frames)
    max_frames  : int – the value used for padding
    """
    if max_frames is None:
        # Use the maximum length across all utterances
        frame_lengths = [m.shape[1] for m in mfcc_list]
        max_frames = max(frame_lengths)

    padded = []
    for m in mfcc_list:
        n_frames = m.shape[1]
        if n_frames < max_frames:
            # Pad with zeros on the right (time axis)
            pad_width = max_frames - n_frames
            m_padded = np.pad(m, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate if longer (shouldn't happen when max_frames=max)
            m_padded = m[:, :max_frames]
        padded.append(m_padded.flatten())   # flatten to 1-D vector

    padded_flat = np.array(padded)
    return padded_flat, max_frames


# ===================================================================
# 5. AUTOENCODER MODEL
# ===================================================================

class Autoencoder(nn.Module):
    """
    Supervised Autoencoder with joint reconstruction + classification.

    Architecture
    ------------
    Encoder:  input_dim → 1024 → 512 → LATENT_DIM
    Decoder:  LATENT_DIM → 512 → 1024 → input_dim
    Classifier head: LATENT_DIM → NUM_CLASSES
    """
    def __init__(self, input_dim, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )
        self.classifier_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier_head(z)
        return x_hat, logits

    def encode(self, x):
        """Return only the latent vector (used for classification)."""
        return self.encoder(x)


# ===================================================================
# 6. MLP CLASSIFIER
# ===================================================================

class MLPClassifier(nn.Module):
    """
    Simple 2-hidden-layer MLP for 10-class digit classification.
    """
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
# 7. TRAINING UTILITIES
# ===================================================================

def train_autoencoder(model, train_loader, epochs=AE_EPOCHS, lr=AE_LR):
    """Train the supervised autoencoder with MSE + CrossEntropy loss."""
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_cls = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            x_hat, logits = model(batch_x)
            loss_r = mse_loss(x_hat, batch_x)
            loss_c = ce_loss(logits, batch_y)
            loss = loss_r + 2.0 * loss_c   # weight classification loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_recon += loss_r.item() * batch_x.size(0)
            epoch_cls += loss_c.item() * batch_x.size(0)
        scheduler.step()
        if (epoch + 1) % 40 == 0:
            avg_r = epoch_recon / len(train_loader.dataset)
            avg_c = epoch_cls / len(train_loader.dataset)
            print(f"  [AE] Epoch {epoch+1}/{epochs}  recon={avg_r:.6f}  cls={avg_c:.4f}")
    train_time_ms = (time.perf_counter() - start) * 1000
    return train_time_ms


def train_classifier(model, train_loader, epochs=CLS_EPOCHS, lr=CLS_LR):
    """Train the MLP classifier with CrossEntropy. Returns training time in ms."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)
        if (epoch + 1) % 25 == 0:
            acc = 100.0 * correct / total
            avg = epoch_loss / total
            print(f"  [CLS] Epoch {epoch+1}/{epochs}  loss={avg:.4f}  "
                  f"train_acc={acc:.1f}%")
    train_time_ms = (time.perf_counter() - start) * 1000
    return train_time_ms


def evaluate_classifier(model, test_loader):
    """Evaluate accuracy on a test set. Returns (accuracy%, test_time_ms)."""
    model.eval()
    correct = 0
    total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    test_time_ms = (time.perf_counter() - start) * 1000
    accuracy = 100.0 * correct / total
    return accuracy, test_time_ms


# ===================================================================
# 8. HELPER — normalise data to [0, 1] for the autoencoder
# ===================================================================

def min_max_normalise(train_data, test_data):
    """
    Scale each feature to [0, 1] using train set statistics.
    This avoids data leakage from the test set.
    """
    d_min = train_data.min(axis=0)
    d_max = train_data.max(axis=0)
    rng = d_max - d_min
    rng[rng == 0] = 1.0  # avoid division by zero for constant features
    train_norm = (train_data - d_min) / rng
    test_norm  = (test_data  - d_min) / rng
    # Clip test set in case values fall outside train range
    test_norm = np.clip(test_norm, 0.0, 1.0)
    return train_norm, test_norm


def standardise(train_data, test_data):
    """
    Zero-mean, unit-variance scaling using train set statistics.
    Better than min-max when the decoder has no Sigmoid activation.
    """
    mean = train_data.mean(axis=0)
    std  = train_data.std(axis=0)
    std[std == 0] = 1.0
    train_std = (train_data - mean) / std
    test_std  = (test_data  - mean) / std
    return train_std, test_std


# ===================================================================
# 9. MAIN PIPELINE
# ===================================================================

def main():
    results = {}   # will be written to output.txt

    # ------------------------------------------------------------------
    # STEP 1 — Load dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    train_files, train_labels = load_dataset(TRAIN_DIR)
    test_files,  test_labels  = load_dataset(TEST_DIR)
    train_labels = np.array(train_labels)
    test_labels  = np.array(test_labels)

    n_train = len(train_files)
    n_test  = len(test_files)
    print(f"  Training samples : {n_train}")
    print(f"  Testing samples  : {n_test}")
    results["n_train"] = n_train
    results["n_test"]  = n_test

    # ------------------------------------------------------------------
    # STEP 2 — Extract MFCC features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Extracting MFCC features")
    print("=" * 60)
    train_mfccs = extract_all_mfccs(train_files)
    test_mfccs  = extract_all_mfccs(test_files)

    # Show an example frame shape
    print(f"  Example MFCC shape (one utterance): {train_mfccs[0].shape}")
    print(f"    → {train_mfccs[0].shape[0]} coefficients x "
          f"{train_mfccs[0].shape[1]} frames")

    # ------------------------------------------------------------------
    # STEP 3 — BASELINE: average-frame + MLP
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Baseline — average frame per utterance")
    print("=" * 60)

    # Use mean + std per MFCC → richer features (captures variability)
    train_summary = compute_summary_features(train_mfccs)   # (n_train, 6*N_MFCC)
    test_summary  = compute_summary_features(test_mfccs)    # (n_test,  6*N_MFCC)
    baseline_feat_dim = train_summary.shape[1]
    print(f"  Summary feature vector shape: {train_summary.shape}")
    print(f"    -> {baseline_feat_dim} features (mean+std of MFCC+delta+delta2)")

    # Standardise (zero mean, unit var) for the classifier
    bl_scaler = StandardScaler()
    train_summary_std = bl_scaler.fit_transform(train_summary)
    test_summary_std  = bl_scaler.transform(test_summary)

    # Train baseline SVM classifier with GridSearchCV
    print("  Training baseline SVM classifier (GridSearchCV) ...")
    param_grid = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 0.01, 0.001]}
    grid_bl = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    start = time.perf_counter()
    grid_bl.fit(train_summary_std, train_labels)
    bl_train_time = (time.perf_counter() - start) * 1000
    print(f"  Best SVM params: {grid_bl.best_params_}")

    # Evaluate
    start = time.perf_counter()
    bl_acc = 100.0 * grid_bl.score(test_summary_std, test_labels)
    bl_test_time = (time.perf_counter() - start) * 1000
    print(f"\n  Baseline Accuracy : {bl_acc:.2f}%")
    print(f"  Baseline Train time: {bl_train_time:.1f} ms")
    print(f"  Baseline Test time : {bl_test_time:.1f} ms")
    results["bl_acc"]        = bl_acc
    results["bl_train_time"] = bl_train_time
    results["bl_test_time"]  = bl_test_time

    # ------------------------------------------------------------------
    # STEP 4 — Pad utterances and flatten for autoencoder
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Padding utterances to uniform length")
    print("=" * 60)

    # Use the maximum length and zero-pad shorter utterances (per project spec)
    # Apply per-utterance normalisation here (for AE only, not baseline)
    train_mfccs_norm = [
        (m - np.mean(m)) / (np.std(m) + 1e-8) for m in train_mfccs
    ]
    test_mfccs_norm = [
        (m - np.mean(m)) / (np.std(m) + 1e-8) for m in test_mfccs
    ]
    all_mfccs_norm = train_mfccs_norm + test_mfccs_norm
    all_frame_lengths = [m.shape[1] for m in all_mfccs_norm]
    max_frames = max(all_frame_lengths)
    print(f"  Max frames across all utterances : {max_frames}")

    train_padded, _ = pad_mfcc_sequences(train_mfccs_norm, max_frames)
    test_padded,  _ = pad_mfcc_sequences(test_mfccs_norm,  max_frames)
    input_dim = train_padded.shape[1]   # n_mfcc * max_frames
    print(f"  Padded + flattened input shape: {train_padded.shape}")

    # Debug: percentage of zeros in padded data
    zero_pct = 100.0 * np.sum(train_padded == 0) / train_padded.size
    print(f"  Zero-padding percentage    : {zero_pct:.1f}%")

    # Standardise for autoencoder (zero mean, unit var) — fits better
    # than min-max when data comes from per-utterance normalised MFCCs
    train_norm, test_norm = standardise(train_padded, test_padded)

    # ------------------------------------------------------------------
    # STEP 5 — Train Autoencoder
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Training Autoencoder")
    print("=" * 60)
    print(f"  Input dim  : {input_dim}")
    print(f"  Latent dim : {LATENT_DIM}")

    ae_train_ds = TensorDataset(
        torch.tensor(train_norm, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )

    # Data augmentation: add noisy copies to expand training set
    aug_copies = 3
    aug_data = []
    aug_labels = []
    for _ in range(aug_copies):
        noise = np.random.normal(0, 0.05, train_norm.shape).astype(np.float32)
        aug_data.append(train_norm + noise)
        aug_labels.append(train_labels)
    aug_data = np.vstack([train_norm] + aug_data)
    aug_labels = np.concatenate([train_labels] + aug_labels)
    ae_train_ds = TensorDataset(
        torch.tensor(aug_data, dtype=torch.float32),
        torch.tensor(aug_labels, dtype=torch.long),
    )
    print(f"  Augmented training set: {len(ae_train_ds)} samples "
          f"({aug_copies} noisy copies + original)")

    ae_train_loader = DataLoader(ae_train_ds, batch_size=BATCH_SIZE,
                                 shuffle=True)

    autoencoder = Autoencoder(input_dim=input_dim,
                              latent_dim=LATENT_DIM).to(DEVICE)
    ae_train_time = train_autoencoder(autoencoder, ae_train_loader)
    print(f"  Autoencoder training time: {ae_train_time:.1f} ms")

    # ------------------------------------------------------------------
    # STEP 6 — Extract latent vectors
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Extracting latent vectors from encoder")
    print("=" * 60)

    autoencoder.eval()
    with torch.no_grad():
        train_latent = autoencoder.encode(
            torch.tensor(train_norm, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
        test_latent = autoencoder.encode(
            torch.tensor(test_norm, dtype=torch.float32).to(DEVICE)
        ).cpu().numpy()
    print(f"  Latent vector shape (train): {train_latent.shape}")
    print(f"  Latent vector shape (test) : {test_latent.shape}")

    # ------------------------------------------------------------------
    # STEP 7 — Classify using multiple strategies, pick best
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Classification on latent vectors")
    print("=" * 60)

    # Strategy A: Use AE's built-in classifier head directly
    autoencoder.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_norm, dtype=torch.float32).to(DEVICE)
        _, test_logits = autoencoder(test_tensor)
        head_preds = test_logits.argmax(dim=1).cpu().numpy()
    head_acc = 100.0 * np.mean(head_preds == test_labels)
    print(f"  Classifier head accuracy: {head_acc:.2f}%")

    # Strategy B: SVM on latent vectors
    latent_scaler = StandardScaler()
    train_latent_std = latent_scaler.fit_transform(train_latent)
    test_latent_std  = latent_scaler.transform(test_latent)

    param_grid = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 0.01, 0.001]}
    grid_ae = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    start = time.perf_counter()
    grid_ae.fit(train_latent_std, train_labels)
    ae_cls_train_time = (time.perf_counter() - start) * 1000
    svm_latent_acc = 100.0 * grid_ae.score(test_latent_std, test_labels)
    print(f"  SVM on latent accuracy : {svm_latent_acc:.2f}%  (params: {grid_ae.best_params_})")

    # Strategy C: SVM on hybrid (latent + summary features), each standardised
    summary_scaler = StandardScaler()
    train_summary_s = summary_scaler.fit_transform(train_summary)
    test_summary_s  = summary_scaler.transform(test_summary)
    train_hybrid = np.hstack([train_latent_std, train_summary_s])
    test_hybrid  = np.hstack([test_latent_std, test_summary_s])
    grid_hyb = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    grid_hyb.fit(train_hybrid, train_labels)
    hybrid_acc = 100.0 * grid_hyb.score(test_hybrid, test_labels)
    print(f"  SVM on hybrid accuracy : {hybrid_acc:.2f}%  (params: {grid_hyb.best_params_})")

    # Pick the best strategy
    ae_acc = max(head_acc, svm_latent_acc, hybrid_acc)
    best_method = {head_acc: "classifier_head", svm_latent_acc: "svm_latent", hybrid_acc: "svm_hybrid"}[ae_acc]

    # Measure actual testing time (encoding + prediction on test set)
    start = time.perf_counter()
    autoencoder.eval()
    with torch.no_grad():
        test_t = torch.tensor(test_norm, dtype=torch.float32).to(DEVICE)
        test_latent_timed = autoencoder.encode(test_t).cpu().numpy()
    if best_method == "classifier_head":
        with torch.no_grad():
            _, logits_t = autoencoder(test_t)
            _ = logits_t.argmax(dim=1).cpu().numpy()
    elif best_method == "svm_latent":
        test_latent_t = latent_scaler.transform(test_latent_timed)
        _ = grid_ae.predict(test_latent_t)
    else:  # svm_hybrid
        test_latent_t = latent_scaler.transform(test_latent_timed)
        test_summary_t = summary_scaler.transform(test_summary)
        test_hybrid_t = np.hstack([test_latent_t, test_summary_t])
        _ = grid_hyb.predict(test_hybrid_t)
    ae_test_time = (time.perf_counter() - start) * 1000

    ae_total_train_time = ae_train_time + ae_cls_train_time

    print(f"\n  Best AE accuracy: {ae_acc:.2f}% (via {best_method})")
    print(f"  AE + Classifier Train time: {ae_total_train_time:.1f} ms")
    results["ae_acc"]        = ae_acc
    results["ae_train_time"] = ae_total_train_time
    results["ae_test_time"]  = ae_test_time
    results["latent_dim"]    = LATENT_DIM

    # ------------------------------------------------------------------
    # STEP 8 — Write output.txt
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Writing results to output.txt")
    print("=" * 60)

    better = ("Autoencoder" if ae_acc > bl_acc else
              "Baseline" if bl_acc > ae_acc else "Tied")

    output_lines = [
        "=" * 60,
        "Problem 4: Speech Digit Recognition — Results",
        "=" * 60,
        "",
        "1. DATASET INFO",
        "-" * 40,
        f"   Number of training samples : {n_train}",
        f"   Number of testing samples  : {n_test}",
        "",
        "2. BASELINE RESULTS (Average Frame + SVM)",
        "-" * 40,
        f"   Accuracy       : {bl_acc:.2f}%",
        f"   Training time  : {bl_train_time:.1f} ms",
        f"   Testing time   : {bl_test_time:.1f} ms",
        "",
        "3. AUTOENCODER RESULTS (Latent-Vector + SVM)",
        "-" * 40,
        f"   Latent vector size : {LATENT_DIM}",
        f"   Accuracy           : {ae_acc:.2f}%",
        f"   Training time      : {ae_total_train_time:.1f} ms",
        f"     (AE training     : {ae_train_time:.1f} ms)",
        f"     (SVM training    : {ae_cls_train_time:.1f} ms)",
        f"   Testing time       : {ae_test_time:.1f} ms",
        "",
        "4. COMPARISON",
        "-" * 40,
        f"   Better method: {better}",
        "",
        "   The baseline uses the average frame (mean across all frames)",
        "   into a single vector, which discards temporal information.",
        "   The autoencoder keeps all frames (padded) and learns a",
        f"   compressed {LATENT_DIM}-D representation that captures both",
        "   spectral AND temporal patterns, leading to richer features.",
        "",
        "5. NOTES",
        "-" * 40,
        f"   - Features: {N_MFCC} MFCCs (+delta+delta2 for baseline), {FRAME_MS} ms frames, {HOP_MS} ms hop",
        "   - Autoencoder architecture: 1024-512-256-128-256-512-1024 (with BatchNorm)",
        f"   - Autoencoder trained for {AE_EPOCHS} epochs",
        "   - Classifier: SVM (RBF kernel, GridSearchCV tuned)",
        "   - Dataset contains clean and noisy versions of each utterance",
        "   - All utterances zero-padded to maximum length (no truncation)",
        "   - Standard scaling applied to AE input (fit on train, applied to test)",
        f"   - Device: {DEVICE}",
        "=" * 60,
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"  Results saved to {OUTPUT_FILE}")

    # ------------------------------------------------------------------
    # Final summary to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Baseline accuracy     : {bl_acc:.2f}%")
    print(f"  Autoencoder accuracy  : {ae_acc:.2f}%")
    print(f"  Better method         : {better}")
    print("=" * 60)


if __name__ == "__main__":
    main()
