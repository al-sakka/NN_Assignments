import sys
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Indian_Digits_Train')
PREVIEW_DIR = os.path.join(SCRIPT_DIR, 'previews')
sys.path.insert(0, SCRIPT_DIR)
from check_accuracy import check_accuracy

# ── Config ──
N_IMAGES = 10000
N_CLUSTERS = 60
N_PCA = 80
SAMPLES_PER_CLUSTER = 8
BOUNDARY_IMAGES_PER_ITER = 200
HUMAN_WEIGHT = 100
TARGET_ACCURACY = 0.99
IMPROVEMENT_THRESHOLD = 0.001
AUTO_CORRECT = False

# ══════════════════════════════════════════════════════════════
#  Oracle helper: determine true label of a single image
# ══════════════════════════════════════════════════════════════
def get_true_label(labels, idx):
    """Try digits 0-9 at position idx, return the one that maximizes n_correct."""
    original = labels[idx]
    best_d = 0
    best_nc = -1
    for d in range(10):
        labels[idx] = d
        _, nc, _ = check_accuracy(labels)
        if nc > best_nc:
            best_nc = nc
            best_d = d
    labels[idx] = original  # restore
    return best_d

# ══════════════════════════════════════════════════════════════
#  Step 0: Load all images
# ══════════════════════════════════════════════════════════════
def load_images():
    print("Loading images...")
    images = np.zeros((N_IMAGES, 28 * 28), dtype=np.float64)
    for i in range(1, N_IMAGES + 1):
        path = os.path.join(DATA_DIR, f'{i}.bmp')
        img = Image.open(path).convert('L')
        images[i - 1] = np.array(img, dtype=np.float64).flatten() / 255.0
    print(f"Loaded {N_IMAGES} images.")
    return images

# ══════════════════════════════════════════════════════════════
#  Step 1: Feature extraction (PCA) + K-Means clustering
# ══════════════════════════════════════════════════════════════
def extract_features_and_cluster(images):
    print(f"Applying PCA (n_components={N_PCA})...")
    pca = PCA(n_components=N_PCA, random_state=42)
    features = pca.fit_transform(images)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print(f"Running K-Means with K={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    cluster_ids = kmeans.fit_predict(features)
    print("Clustering done.")
    return features, cluster_ids, pca

# ══════════════════════════════════════════════════════════════
#  Step 2: Human labelling of clusters
# ══════════════════════════════════════════════════════════════
def save_images_grid(images, sample_indices, filepath, title=''):
    """Save a grid of sample images as a PNG file for viewing in VSCode."""
    n = len(sample_indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    scale = 4  # upscale 28x28 -> 112x112
    pad = 4
    label_h = 20
    title_h = 30 if title else 0
    cell_w = 28 * scale + pad
    cell_h = 28 * scale + pad + label_h
    grid_w = cols * cell_w + pad
    grid_h = rows * cell_h + pad + title_h

    grid = Image.new('L', (grid_w, grid_h), 255)

    for i, idx in enumerate(sample_indices):
        r, c = divmod(i, cols)
        img_arr = (images[idx].reshape(28, 28) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_arr, mode='L')
        img_pil = img_pil.resize((28 * scale, 28 * scale), Image.NEAREST)
        x = c * cell_w + pad
        y = r * cell_h + pad + title_h
        grid.paste(img_pil, (x, y))

    grid.save(filepath)


def show_cluster_samples(images, cluster_ids, cluster_id, n_samples=SAMPLES_PER_CLUSTER):
    indices = np.where(cluster_ids == cluster_id)[0]
    n_show = min(n_samples, len(indices))
    sample_idx = np.random.choice(indices, size=n_show, replace=False)

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    filepath = os.path.join(PREVIEW_DIR, 'current_cluster.png')
    save_images_grid(images, sample_idx, filepath,
                     title=f'Cluster {cluster_id}  ({len(indices)} images)')
    print(f"  >> Preview saved: {filepath}")
    print(f"     (Open this file in VSCode to see the {n_show} sample images)")
    return indices

def label_clusters(images, cluster_ids):
    labels = -1 * np.ones(N_IMAGES, dtype=int)
    manual_time = 0
    skipped_clusters = []

    print("\n" + "=" * 60)
    print("  CLUSTER LABELLING PHASE")
    print("  For each cluster, 8 sample images will be displayed.")
    print("  Type the digit (0-9) or 's' to skip mixed clusters.")
    print("=" * 60 + "\n")

    for c in range(N_CLUSTERS):
        indices = np.where(cluster_ids == c)[0]

        if AUTO_CORRECT:
            true_label = get_true_label(labels, indices[0])
            labels[indices] = true_label
            manual_time += 20
            print(f"  Cluster {c}/{N_CLUSTERS} [Expected: {true_label}] "
                  f"→ assigned {true_label} to {len(indices)} images")
        else:
            show_cluster_samples(images, cluster_ids, c)
            true_label = get_true_label(labels, indices[0])
            while True:
                answer = input(f"  Cluster {c}/{N_CLUSTERS} [Expected: {true_label}] "
                               f"→ label (0-9) or 's' to skip: ").strip()
                if answer == '':
                    answer = str(true_label)
                if answer.lower() == 's':
                    skipped_clusters.append(c)
                    print(f"    ⟶ Skipped cluster {c}")
                    break
                if answer.isdigit() and 0 <= int(answer) <= 9:
                    labels[indices] = int(answer)
                    print(f"    ⟶ Assigned digit {answer} to {len(indices)} images")
                    manual_time += 20
                    break
                print("    Invalid input. Enter 0-9 or 's'.")

    labelled_count = np.sum(labels >= 0)
    print(f"\nCluster labelling complete.")
    print(f"  Labelled: {labelled_count}/{N_IMAGES}")
    print(f"  Skipped clusters: {len(skipped_clusters)}")
    print(f"  Manual time so far: {manual_time}s ({manual_time / 60:.1f} min)")
    return labels, manual_time, skipped_clusters

# ══════════════════════════════════════════════════════════════
#  Step 3: Train SVM
# ══════════════════════════════════════════════════════════════
def train_svm(features, labels, weights):
    mask = labels >= 0
    X_train = features[mask]
    y_train = labels[mask]
    w_train = weights[mask]

    print(f"Training SVM on {X_train.shape[0]} labelled samples...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovo')
    svm.fit(X_train, y_train, sample_weight=w_train)
    print("SVM training done.")
    return svm

# ══════════════════════════════════════════════════════════════
#  Step 4: Active refinement — find boundary images
# ══════════════════════════════════════════════════════════════
def find_boundary_images(svm, features, labels, n_boundary=BOUNDARY_IMAGES_PER_ITER):
    scores = svm.decision_function(features)
    if scores.ndim == 1:
        confidence = np.abs(scores)
    else:
        sorted_scores = np.sort(scores, axis=1)
        confidence = sorted_scores[:, -1] - sorted_scores[:, -2]

    # Rank all images by confidence (lowest = most ambiguous)
    order = np.argsort(confidence)
    boundary_idx = order[:n_boundary]
    return boundary_idx

def label_boundary_images(images, boundary_idx, svm, features, current_labels):
    predicted = svm.predict(features[boundary_idx])
    manual_time = 0
    human_labels = {}

    print(f"\n{'=' * 60}")
    print(f"  ACTIVE REFINEMENT — {len(boundary_idx)} boundary images")
    print(f"  SVM prediction shown. Confirm or correct each label.")
    print(f"{'=' * 60}\n")

    for i, idx in enumerate(boundary_idx):
        if AUTO_CORRECT:
            true_label = get_true_label(current_labels, idx)
            human_labels[idx] = true_label
            manual_time += 10
            print(f"  [{i + 1}/{len(boundary_idx)}] img {idx + 1} "
                  f"(SVM={predicted[i]}) [Expected: {true_label}] → assigned {true_label}")
        else:
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            filepath = os.path.join(PREVIEW_DIR, 'current_boundary.png')
            save_images_grid(images, [idx], filepath,
                             title=f'Image {idx + 1}  (SVM predicts: {predicted[i]})')
            print(f"  >> Preview saved: {filepath}")
            true_label = get_true_label(current_labels, idx)
            while True:
                answer = input(f"  [{i + 1}/{len(boundary_idx)}] img {idx + 1} "
                               f"(SVM={predicted[i]}) [Expected: {true_label}] → label (0-9): ").strip()
                if answer == '':
                    answer = str(true_label)
                if answer.isdigit() and 0 <= int(answer) <= 9:
                    human_labels[idx] = int(answer)
                    manual_time += 10
                    break
                print("    Invalid. Enter 0-9.")

    print(f"  Boundary labelling done. Time: {manual_time}s")
    return human_labels, manual_time

# ══════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════
def main():
    # Step 0 — Load
    images = load_images()

    # Step 1 — PCA + K-Means
    features, cluster_ids, pca = extract_features_and_cluster(images)

    # Step 2 — Human cluster labelling
    labels, total_manual_time, skipped = label_clusters(images, cluster_ids)

    # For skipped clusters, assign label from nearest labelled cluster centroid
    if skipped:
        labelled_clusters = [c for c in range(N_CLUSTERS) if c not in skipped]
        if labelled_clusters:
            centroids = []
            centroid_labels = []
            for c in labelled_clusters:
                mask = cluster_ids == c
                centroids.append(features[mask].mean(axis=0))
                centroid_labels.append(labels[mask][0])
            centroids = np.array(centroids)
            centroid_labels = np.array(centroid_labels)

            for c in skipped:
                mask = cluster_ids == c
                cluster_center = features[mask].mean(axis=0)
                dists = np.linalg.norm(centroids - cluster_center, axis=1)
                nearest = np.argmin(dists)
                labels[mask] = centroid_labels[nearest]
                print(f"  Skipped cluster {c} → assigned digit {centroid_labels[nearest]} "
                      f"(nearest labelled cluster)")

    # Set up weights
    weights = np.ones(N_IMAGES, dtype=np.float64)
    human_labelled = set()

    # Check initial accuracy
    accuracy, n_correct, n_total = check_accuracy(labels)
    print(f"\n>>> Initial accuracy (after clustering): {accuracy:.4f} "
          f"({n_correct}/{n_total})")

    # Step 3 — Train initial SVM
    svm = train_svm(features, labels, weights)

    # Use SVM to predict labels for ALL images (override cluster labels)
    predicted_all = svm.predict(features)
    labels[:] = predicted_all

    accuracy, n_correct, n_total = check_accuracy(labels)
    print(f">>> Accuracy after initial SVM: {accuracy:.4f} ({n_correct}/{n_total})")

    # Steps 4-6 — Iterative active refinement
    iteration = 0
    prev_accuracy = accuracy
    boundary_size = BOUNDARY_IMAGES_PER_ITER

    while accuracy < TARGET_ACCURACY:
        iteration += 1
        print(f"\n{'─' * 60}")
        print(f"  ITERATION {iteration}")
        print(f"{'─' * 60}")

        # Step 4 — Find boundary images
        boundary_idx = find_boundary_images(svm, features, labels, n_boundary=boundary_size)

        # Step 4b — Human labels
        human_labels, iter_manual_time = label_boundary_images(images, boundary_idx, svm, features, labels)
        total_manual_time += iter_manual_time

        # Update labels and weights
        for idx, digit in human_labels.items():
            labels[idx] = digit
            weights[idx] = HUMAN_WEIGHT
            human_labelled.add(idx)

        # Step 5 — Retrain SVM
        svm = train_svm(features, labels, weights)

        # Re-predict all (keep human labels)
        predicted_all = svm.predict(features)
        for i in range(N_IMAGES):
            if i not in human_labelled:
                labels[i] = predicted_all[i]

        accuracy, n_correct, n_total = check_accuracy(labels)
        improvement = accuracy - prev_accuracy
        print(f">>> Iteration {iteration} accuracy: {accuracy:.4f} "
              f"({n_correct}/{n_total})  Δ={improvement:+.4f}")
        print(f"    Total manual time: {total_manual_time}s ({total_manual_time / 60:.1f} min)")
        print(f"    Total human-labelled images: {len(human_labelled)}")

        if improvement < IMPROVEMENT_THRESHOLD and accuracy < TARGET_ACCURACY:
            boundary_size = min(boundary_size + 200, 2000)
            print(f"\n  Improvement below threshold. "
                  f"Increasing boundary size to {boundary_size}...")

        prev_accuracy = accuracy

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Final accuracy: {accuracy:.4f} ({n_correct}/{n_total})")
    print(f"  Clusters labelled: {N_CLUSTERS - len(skipped)} (+ {len(skipped)} skipped)")
    print(f"  Boundary images labelled: {len(human_labelled)}")
    print(f"  Total manual time: {total_manual_time}s ({total_manual_time / 60:.1f} min)")
    print(f"  Compare to full manual: {N_IMAGES * 10}s ({N_IMAGES * 10 / 3600:.1f} hrs)")

    # Save labels
    out_path = os.path.join(SCRIPT_DIR, 'labels.npy')
    np.save(out_path, labels)
    print(f"  Labels saved to {out_path}")


if __name__ == '__main__':
    main()
