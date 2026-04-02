"""
Assignment 1 - Part 2: K-Means Classifier
Author: Antar
Description: Train class-wise K-Means centroids and predict by nearest centroid.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def train_kmeans_classifier(train_features, train_labels, k_per_class):
    """Train K-Means inside each digit class and return all centroids."""
    centroids_list = []
    centroid_labels_list = []

    for digit in range(10):
        class_feats = train_features[train_labels == digit]
        if class_feats.shape[0] == 0:
            raise ValueError(f'No samples found for class {digit}.')

        print(f"  Digit {digit}: {class_feats.shape[0]} samples, K={k_per_class}")

        if k_per_class == 1:
            class_centroids = class_feats.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k_per_class, n_init=10, random_state=42)
            km.fit(class_feats)
            class_centroids = km.cluster_centers_

        centroids_list.append(class_centroids)
        centroid_labels_list.append(np.full(class_centroids.shape[0], digit, dtype=np.int32))

    centroids = np.vstack(centroids_list)
    centroid_labels = np.concatenate(centroid_labels_list)
    return centroids, centroid_labels


def predict_kmeans(test_features, centroids, centroid_labels):
    """Predict labels by nearest centroid distance."""
    distances = cdist(test_features, centroids, metric='euclidean')
    nearest_centroid_idx = np.argmin(distances, axis=1)
    return centroid_labels[nearest_centroid_idx]
