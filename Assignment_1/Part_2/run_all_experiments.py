#================================================================
# ASSIGNMENT 1 - PART 2: COMPLETE EXPERIMENT RUNNER
# Run this file directly: python run_all_experiments.py
# ================================================================

import numpy as np
import time
from sklearn.metrics import accuracy_score

from step0_load_data import load_data
from step1_features import (
    extract_dct_features,
    extract_pca_features,
    extract_hog_features,
)
from step2_kmeans import train_kmeans_classifier, predict_kmeans
from step3_svm import train_and_test_svm

# ---- SETUP ----
np.random.seed(42)

# Load data (replace with your actual loading method)
train_images, train_labels, test_images, test_labels = load_data(
    prefer_mat=False,
    use_kagglehub=True,
)

# ---- EXTRACT FEATURES ----
print("=== Extracting all features ===")
dct_train  = extract_dct_features(train_images)
dct_test   = extract_dct_features(test_images)
pca_train, pca_test, K_pca = extract_pca_features(train_images, test_images)
hog_train  = extract_hog_features(train_images)
hog_test   = extract_hog_features(test_images)

# Bundle features for easy iteration
feature_sets = {
    'DCT': (dct_train,  dct_test),
    'PCA': (pca_train,  pca_test),
    'HOG': (hog_train,  hog_test),
}

# Storage for the results table
results = {}  # key: (classifier_name, feature_name) → (accuracy, time)
all_preds = {}  # store predictions for confusion matrices

# ================================================================
# K-MEANS EXPERIMENTS
# ================================================================
print("\n\n=== K-Means Experiments ===")

for K in [1, 4, 16, 32]:
    for fname, (tr_feat, te_feat) in feature_sets.items():
        print(f"\n-- {fname} features, K={K} per class --")

        t_start = time.time()

        # Train: compute centroids per class
        centroids, c_labels = train_kmeans_classifier(tr_feat, train_labels, K)

        # Test: classify using nearest centroid
        preds = predict_kmeans(te_feat, centroids, c_labels)

        elapsed = time.time() - t_start
        acc     = accuracy_score(test_labels, preds)

        print(f"  Accuracy: {acc*100:.2f}%  |  Time: {elapsed:.1f}s")

        key = f"KMeans_K{K}"
        results[(key, fname)] = (acc, elapsed)
        all_preds[(key, fname)] = preds

# ================================================================
# SVM EXPERIMENTS
# ================================================================
print("\n\n=== SVM Experiments ===")

for kernel in ['linear', 'rbf']:
    for fname, (tr_feat, te_feat) in feature_sets.items():
        print(f"\n-- {fname} features, SVM {kernel} kernel --")

        t_start = time.time()
        _, acc, preds, __ = train_and_test_svm(tr_feat, train_labels,
                                                  te_feat, test_labels, kernel)
        elapsed = time.time() - t_start

        key = f"SVM_{kernel}"
        results[(key, fname)] = (acc, elapsed)
        all_preds[(key, fname)] = preds

# ================================================================
# PRINT RESULTS TABLE
# ================================================================
print("\n\n" + "="*60)
print("                RESULTS SUMMARY")
print("="*60)
print(f"{'Classifier':<22} {'DCT':>8} {'PCA':>8} {'HOG':>8}")
print("-"*50)

for clf in ['KMeans_K1', 'KMeans_K4', 'KMeans_K16', 'KMeans_K32',
              'SVM_linear', 'SVM_rbf']:
    row = f"{clf:<22}"
    for ft in ['DCT', 'PCA', 'HOG']:
        acc, _ = results.get((clf, ft), (0, 0))
        row += f" {acc*100:>7.2f}%"
    print(row)