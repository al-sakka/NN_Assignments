import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import time

def train_kmeans_classifier(train_features, train_labels, K_per_class):
    """
    Train K-Means per class and collect all centroids.

    For each digit 0–9:
      - Take all training images of that digit
      - Run K-Means with K_per_class clusters
      - Store the K_per_class centroids and their class label

    Returns:
      centroids       : np.array of shape (10*K, D) — all centroids
      centroid_labels : np.array of shape (10*K,)   — class of each centroid
    """
    centroids_list       = []
    centroid_labels_list = []

    for digit in range(10):
        # Select only the rows belonging to this digit class
        mask        = (train_labels == digit)
        class_feats = train_features[mask]  # shape: (1000, D)

        print(f"  Digit {digit}: {class_feats.shape[0]} samples, K={K_per_class} clusters")

        if K_per_class == 1:
            # Special case: one centroid = the mean of all samples in this class
            class_centroids = class_feats.mean(axis=0, keepdims=True)  # (1, D)
        else:
            # Run K-Means clustering on this class's features
            # n_init=10: run 10 times with different random starts, keep the best
            # random_state=42: for reproducibility
            km = KMeans(n_clusters=K_per_class, n_init=10, random_state=42)
            km.fit(class_feats)
            class_centroids = km.cluster_centers_  # (K, D)

        centroids_list.append(class_centroids)
        centroid_labels_list.append(np.full(class_centroids.shape[0], digit))

    # Concatenate all classes into one big centroid matrix
    centroids       = np.vstack(centroids_list)       # (10*K, D)
    centroid_labels = np.concatenate(centroid_labels_list)  # (10*K,)

    return centroids, centroid_labels


def predict_kmeans(test_features, centroids, centroid_labels):
    """
    Classify test images using the nearest-centroid rule.

    For each test image:
      1. Compute its Euclidean distance to ALL centroids
      2. Find the centroid with minimum distance
      3. Assign that centroid's class label as the prediction

    This is the core of the 'minimum distance classifier.'
    """
    # scipy's cdist computes pairwise distances efficiently
    # Result shape: (N_test, 10*K) — one distance per test-centroid pair
    from scipy.spatial.distance import cdist
    distances = cdist(test_features, centroids, metric='euclidean')

    # For each test image, find which centroid is closest
    nearest_centroid_idx = np.argmin(distances, axis=1)  # (N_test,)

    # Look up that centroid's class label
    predictions = centroid_labels[nearest_centroid_idx]

    return predictions