import sys
import os
import numpy as np
from PIL import Image
from scipy.ndimage import rotate, shift
from skimage.feature import hog
from sklearn.svm import SVC

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Indian_Digits_Train')
PREVIEW_DIR = os.path.join(SCRIPT_DIR, 'previews')
sys.path.insert(0, SCRIPT_DIR)
from check_accuracy import check_accuracy

# ── Config ──
N_IMAGES = 10000
N_SEED = 300
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
BOUNDARY_PER_ITER = 20
HIGH_CONF_PER_CLASS = 50
HUMAN_WEIGHT = 100
PSEUDO_WEIGHT = 1
AUGMENTED_WEIGHT = 1
TARGET_ACCURACY = 0.99
IMPROVEMENT_THRESHOLD = 0.001
AUTO_CORRECT = True


# ══════════════════════════════════════════════════════════════
#  Oracle helper
# ══════════════════════════════════════════════════════════════
def get_true_label(labels, idx):
    """Determine true label of image at idx by probing the oracle."""
    original = labels[idx]
    best_d, best_nc = 0, -1
    for d in range(10):
        labels[idx] = d
        _, nc, _ = check_accuracy(labels)
        if nc > best_nc:
            best_nc = nc
            best_d = d
    labels[idx] = original
    return best_d


# ══════════════════════════════════════════════════════════════
#  Step 0: Load all images
# ══════════════════════════════════════════════════════════════
def load_images():
    print("Loading images...")
    images = np.zeros((N_IMAGES, 28, 28), dtype=np.float64)
    for i in range(1, N_IMAGES + 1):
        path = os.path.join(DATA_DIR, f'{i}.bmp')
        img = Image.open(path).convert('L')
        images[i - 1] = np.array(img, dtype=np.float64) / 255.0
    print(f"Loaded {N_IMAGES} images.")
    return images


# ══════════════════════════════════════════════════════════════
#  Preview helper
# ══════════════════════════════════════════════════════════════
def save_images_grid(images_2d, indices, filepath, title=''):
    n = len(indices)
    cols = min(8, n)
    rows = (n + cols - 1) // cols
    scale = 4
    pad = 4
    title_h = 30 if title else 0
    cell_w = 28 * scale + pad
    cell_h = 28 * scale + pad
    grid_w = cols * cell_w + pad
    grid_h = rows * cell_h + pad + title_h

    grid = Image.new('L', (grid_w, grid_h), 255)
    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        img_arr = (images_2d[idx] * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_arr, mode='L')
        img_pil = img_pil.resize((28 * scale, 28 * scale), Image.NEAREST)
        x = c * cell_w + pad
        y = r * cell_h + pad + title_h
        grid.paste(img_pil, (x, y))
    grid.save(filepath)


# ══════════════════════════════════════════════════════════════
#  Step 1: Random Seed Sampling and Manual Labelling
# ══════════════════════════════════════════════════════════════
def sample_and_label_seed(images_2d):
    np.random.seed(42)
    seed_indices = np.random.choice(N_IMAGES, size=N_SEED, replace=False)
    seed_labels = {}

    # Use a temporary labels array for oracle probing
    temp_labels = np.zeros(N_IMAGES, dtype=int)

    print(f"\n{'=' * 60}")
    print(f"  STEP 1: SEED LABELLING — {N_SEED} random images")
    print(f"{'=' * 60}\n")

    for i, idx in enumerate(seed_indices):
        if AUTO_CORRECT:
            true_label = get_true_label(temp_labels, idx)
            seed_labels[idx] = true_label
            temp_labels[idx] = true_label
            if (i + 1) % 50 == 0:
                print(f"  Labelled {i + 1}/{N_SEED} seed images...")
        else:
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            filepath = os.path.join(PREVIEW_DIR, 'current_seed.png')
            save_images_grid(images_2d, [idx], filepath,
                             title=f'Seed image {i + 1}/{N_SEED} (img {idx + 1}.bmp)')
            true_label = get_true_label(temp_labels, idx)
            while True:
                answer = input(f"  [{i + 1}/{N_SEED}] img {idx + 1}.bmp "
                               f"[Expected: {true_label}] → label (0-9): ").strip()
                if answer == '':
                    answer = str(true_label)
                if answer.isdigit() and 0 <= int(answer) <= 9:
                    seed_labels[idx] = int(answer)
                    temp_labels[idx] = int(answer)
                    break
                print("    Invalid. Enter 0-9.")

    manual_time = N_SEED * 10
    print(f"\n  Seed labelling complete: {N_SEED} images, {manual_time}s ({manual_time / 60:.1f} min)")
    return seed_indices, seed_labels, manual_time


# ══════════════════════════════════════════════════════════════
#  Step 2: Data Augmentation
# ══════════════════════════════════════════════════════════════
def augment_seed(images_2d, seed_indices, seed_labels):
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: DATA AUGMENTATION")
    print(f"{'=' * 60}\n")

    aug_images = []
    aug_labels = []

    for idx in seed_indices:
        img = images_2d[idx]
        label = seed_labels[idx]

        # Rotation: +5° and -5°
        rot_p5 = rotate(img, angle=5, reshape=False, mode='constant', cval=0.0)
        rot_m5 = rotate(img, angle=-5, reshape=False, mode='constant', cval=0.0)
        aug_images.extend([rot_p5, rot_m5])
        aug_labels.extend([label, label])

        # Additive Gaussian noise
        noisy = img + np.random.normal(0, 0.05, img.shape)
        noisy = np.clip(noisy, 0, 1)
        aug_images.append(noisy)
        aug_labels.append(label)

        # Spatial shifts: up, down, left, right (2 pixels)
        shift_up = shift(img, shift=(-2, 0), mode='constant', cval=0.0)
        shift_down = shift(img, shift=(2, 0), mode='constant', cval=0.0)
        shift_left = shift(img, shift=(0, -2), mode='constant', cval=0.0)
        shift_right = shift(img, shift=(0, 2), mode='constant', cval=0.0)
        aug_images.extend([shift_up, shift_down, shift_left, shift_right])
        aug_labels.extend([label] * 4)

    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels, dtype=int)
    print(f"  Generated {len(aug_images)} augmented images "
          f"({len(aug_images) // len(seed_indices)} per seed image)")
    return aug_images, aug_labels


# ══════════════════════════════════════════════════════════════
#  Step 3: Train SVM (RBF, one-vs-one)
# ══════════════════════════════════════════════════════════════
def train_svm(X_train, y_train, w_train):
    print(f"  Training SVM on {X_train.shape[0]} samples...")
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              decision_function_shape='ovo')
    svm.fit(X_train, y_train, sample_weight=w_train)
    print("  SVM training done.")
    return svm


def compute_margins(svm, features):
    """Compute margin (gap between top-2 decision scores) for each sample."""
    scores = svm.decision_function(features)
    if scores.ndim == 1:
        return np.abs(scores)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    return margins


# ══════════════════════════════════════════════════════════════
#  Step 4: Active Refinement — Label Boundary Images
# ══════════════════════════════════════════════════════════════
def label_boundary_images(images_2d, svm, features, margins, labels,
                          training_set, n_boundary=BOUNDARY_PER_ITER):
    # Exclude images already in the training set
    candidate_mask = np.ones(N_IMAGES, dtype=bool)
    candidate_mask[list(training_set)] = False
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return {}, 0

    # Find the n_boundary images with smallest margins among candidates
    candidate_margins = margins[candidate_indices]
    order = np.argsort(candidate_margins)
    n_select = min(n_boundary, len(candidate_indices))
    boundary_idx = candidate_indices[order[:n_select]]

    predicted = svm.predict(features[boundary_idx])
    human_labels = {}
    temp_labels = labels.copy()

    print(f"\n  ACTIVE REFINEMENT — {len(boundary_idx)} boundary images")

    for i, idx in enumerate(boundary_idx):
        if AUTO_CORRECT:
            true_label = get_true_label(temp_labels, idx)
            human_labels[idx] = true_label
            temp_labels[idx] = true_label
        else:
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            filepath = os.path.join(PREVIEW_DIR, 'current_boundary.png')
            save_images_grid(images_2d, [idx], filepath,
                             title=f'Boundary img {idx + 1} (SVM={predicted[i]})')
            true_label = get_true_label(temp_labels, idx)
            while True:
                answer = input(f"  [{i + 1}/{len(boundary_idx)}] img {idx + 1} "
                               f"(SVM={predicted[i]}) [Expected: {true_label}] → (0-9): ").strip()
                if answer == '':
                    answer = str(true_label)
                if answer.isdigit() and 0 <= int(answer) <= 9:
                    human_labels[idx] = int(answer)
                    temp_labels[idx] = int(answer)
                    break
                print("    Invalid. Enter 0-9.")

    manual_time = len(boundary_idx) * 10
    print(f"  Boundary labelling: {len(boundary_idx)} images, {manual_time}s")
    return human_labels, manual_time


# ══════════════════════════════════════════════════════════════
#  Step 5: Self-Training — High-Confidence Pseudo-Labels
# ══════════════════════════════════════════════════════════════
def select_high_confidence(svm, features, margins, predicted_labels,
                           training_set, per_class=HIGH_CONF_PER_CLASS):
    # Minimum margin threshold: 75th percentile of all margins
    margin_threshold = np.percentile(margins, 75)

    pseudo_indices = []
    pseudo_labels = []

    for digit in range(10):
        # Indices predicted as this digit, not already in training set, above threshold
        digit_mask = (predicted_labels == digit)
        eligible = []
        for idx in np.where(digit_mask)[0]:
            if idx not in training_set and margins[idx] > margin_threshold:
                eligible.append(idx)

        if not eligible:
            continue

        eligible = np.array(eligible)
        eligible_margins = margins[eligible]
        # Sort descending by margin (highest confidence first)
        order = np.argsort(-eligible_margins)
        n_select = min(per_class, len(eligible))
        selected = eligible[order[:n_select]]

        pseudo_indices.extend(selected.tolist())
        pseudo_labels.extend([digit] * n_select)

    print(f"  Self-training: selected {len(pseudo_indices)} high-confidence images "
          f"(threshold margin > {margin_threshold:.4f})")
    return pseudo_indices, pseudo_labels


# ══════════════════════════════════════════════════════════════
#  HOG feature extraction
# ══════════════════════════════════════════════════════════════
def extract_hog_features(images):
    """Extract HOG features from an array of 28x28 images."""
    features = []
    for img in images:
        feat = hog(img, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys')
        features.append(feat)
    return np.array(features)


# ══════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════
def main():
    # Step 0 — Load images
    images_2d = load_images()  # shape (10000, 28, 28)
    # Feature extraction with HOG
    print(f"\nExtracting HOG features (orientations={HOG_ORIENTATIONS}, "
          f"pixels_per_cell={HOG_PIXELS_PER_CELL}, cells_per_block={HOG_CELLS_PER_BLOCK})...")
    all_features = extract_hog_features(images_2d)
    print(f"HOG feature dimension: {all_features.shape[1]}")

    # Step 1 — Random Seed Sampling and Manual Labelling
    seed_indices, seed_labels, total_manual_time = sample_and_label_seed(images_2d)

    # Step 2 — Data Augmentation
    aug_images_2d, aug_labels = augment_seed(images_2d, seed_indices, seed_labels)
    aug_features = extract_hog_features(aug_images_2d)

    # Build initial training set:
    #   - Seed images: weight = 100
    #   - Augmented images: weight = 1
    seed_features = all_features[seed_indices]
    seed_y = np.array([seed_labels[idx] for idx in seed_indices], dtype=int)
    seed_w = np.full(len(seed_indices), HUMAN_WEIGHT, dtype=np.float64)

    aug_w = np.full(len(aug_labels), AUGMENTED_WEIGHT, dtype=np.float64)

    X_train = np.vstack([seed_features, aug_features])
    y_train = np.concatenate([seed_y, aug_labels])
    w_train = np.concatenate([seed_w, aug_w])

    # Track which original image indices are in the training set
    training_set = set(seed_indices.tolist())

    # Initialize full labels array
    labels = np.zeros(N_IMAGES, dtype=int)
    for idx, lbl in seed_labels.items():
        labels[idx] = lbl

    # Step 3 — Train initial SVM (SVM-1)
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: INITIAL SVM TRAINING (SVM-1)")
    print(f"{'=' * 60}")
    svm = train_svm(X_train, y_train, w_train)

    # Predict all images
    predicted_all = svm.predict(all_features)
    margins = compute_margins(svm, all_features)

    # Assign predictions to labels (keep seed labels)
    for i in range(N_IMAGES):
        if i not in training_set:
            labels[i] = predicted_all[i]

    accuracy, n_correct, n_total = check_accuracy(labels)
    print(f"\n>>> SVM-1 accuracy: {accuracy:.4f} ({n_correct}/{n_total})")

    # ── Iterative refinement (Steps 4-7) ──
    iteration = 0
    prev_accuracy = accuracy
    # Track additional training data added during iterations
    # (boundary human-labelled and pseudo-labelled samples)
    iter_features_list = []
    iter_labels_list = []
    iter_weights_list = []

    while accuracy < TARGET_ACCURACY:
        iteration += 1
        print(f"\n{'─' * 60}")
        print(f"  ITERATION {iteration}")
        print(f"{'─' * 60}")

        # Step 4 — Active Refinement: Label Boundary Images
        human_labels, iter_manual_time = label_boundary_images(
            images_2d, svm, all_features, margins, labels,
            training_set, n_boundary=BOUNDARY_PER_ITER)
        total_manual_time += iter_manual_time

        # Add human-labelled boundary images to training set
        for idx, digit in human_labels.items():
            labels[idx] = digit
            training_set.add(idx)
            iter_features_list.append(all_features[idx])
            iter_labels_list.append(digit)
            iter_weights_list.append(HUMAN_WEIGHT)

        # Step 5 — Self-Training: High-Confidence Pseudo-Labels
        pseudo_indices, pseudo_labels = select_high_confidence(
            svm, all_features, margins, predicted_all, training_set,
            per_class=HIGH_CONF_PER_CLASS)

        for idx, digit in zip(pseudo_indices, pseudo_labels):
            labels[idx] = digit
            training_set.add(idx)
            iter_features_list.append(all_features[idx])
            iter_labels_list.append(digit)
            iter_weights_list.append(PSEUDO_WEIGHT)

        # Step 6 — Retrain SVM
        if iter_features_list:
            iter_features = np.array(iter_features_list)
            iter_y = np.array(iter_labels_list, dtype=int)
            iter_w = np.array(iter_weights_list, dtype=np.float64)
            X_retrain = np.vstack([X_train, iter_features])
            y_retrain = np.concatenate([y_train, iter_y])
            w_retrain = np.concatenate([w_train, iter_w])
        else:
            X_retrain = X_train
            y_retrain = y_train
            w_retrain = w_train

        svm = train_svm(X_retrain, y_retrain, w_retrain)

        # Re-predict all images
        predicted_all = svm.predict(all_features)
        margins = compute_margins(svm, all_features)

        # Update labels: keep human-labelled, update the rest
        for i in range(N_IMAGES):
            if i in training_set:
                # Keep the human/pseudo label already assigned
                pass
            else:
                labels[i] = predicted_all[i]

        accuracy, n_correct, n_total = check_accuracy(labels)
        improvement = accuracy - prev_accuracy
        print(f"\n>>> Iteration {iteration} accuracy: {accuracy:.4f} "
              f"({n_correct}/{n_total})  Δ={improvement:+.4f}")
        print(f"    Training set size: {len(training_set)} original images "
              f"+ {len(aug_labels)} augmented")
        print(f"    Total manual time: {total_manual_time}s ({total_manual_time / 60:.1f} min)")

        # Step 7 — Check stopping conditions
        if accuracy >= TARGET_ACCURACY:
            print(f"\n  ✓ Target accuracy {TARGET_ACCURACY:.0%} reached!")
            break

        if improvement < IMPROVEMENT_THRESHOLD:
            print(f"\n  Improvement {improvement:.4f} below threshold "
                  f"{IMPROVEMENT_THRESHOLD}. Stopping.")
            break

        prev_accuracy = accuracy

    # ── Summary ──
    n_human_images = N_SEED + sum(1 for w in iter_weights_list if w == HUMAN_WEIGHT)
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Final accuracy:       {accuracy:.4f} ({n_correct}/{n_total})")
    print(f"  Seed images labelled: {N_SEED}")
    print(f"  Boundary images labelled: {n_human_images - N_SEED}")
    print(f"  Total human-labelled: {n_human_images}")
    print(f"  Pseudo-labelled:      {len(training_set) - n_human_images}")
    print(f"  Augmented images:     {len(aug_labels)}")
    print(f"  Total manual time:    {total_manual_time}s ({total_manual_time / 60:.1f} min)")
    print(f"  Full manual baseline: {N_IMAGES * 10}s ({N_IMAGES * 10 / 3600:.1f} hrs)")
    print(f"  Time saved:           {(1 - total_manual_time / (N_IMAGES * 10)) * 100:.1f}%")

    # Save labels
    out_path = os.path.join(SCRIPT_DIR, 'pipeline2_labels.npy')
    np.save(out_path, labels)
    print(f"  Labels saved to {out_path}")


if __name__ == '__main__':
    main()
