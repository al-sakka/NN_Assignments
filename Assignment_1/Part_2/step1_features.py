"""
Assignment 1 - Part 2: Feature Extraction
Author: Antar
Description: Extract DCT, PCA, and HOG features from MNIST images.
"""

import time

import numpy as np
from scipy.fft import dctn
from sklearn.decomposition import PCA
from skimage.feature import hog


def extract_dct_features(images, block_size=15):
    """Extract top-left DCT block from each 28x28 image."""
    n_samples = images.shape[0]
    n_features = block_size * block_size
    features = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        dct_img = dctn(images[i].reshape(28, 28), norm='ortho')
        features[i] = dct_img[:block_size, :block_size].ravel()

    print(f"DCT: {n_samples} images -> {n_features} features")
    return features


def extract_pca_features(train_images, test_images, var_threshold=0.95):
    """Fit PCA on train split and transform both splits."""
    pca_full = PCA().fit(train_images)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.argmax(cumvar >= var_threshold) + 1)

    pca = PCA(n_components=k).fit(train_images)
    train_feat = pca.transform(train_images)
    test_feat = pca.transform(test_images)

    print(f"PCA: keeping {k} components ({cumvar[k-1]*100:.1f}% variance)")
    return train_feat, test_feat, k


def extract_hog_features(images, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Extract HOG features from flattened 28x28 images."""
    feature_list = []

    for i in range(images.shape[0]):
        img = images[i].reshape(28, 28)
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True,
        )
        feature_list.append(feat)

    features = np.array(feature_list, dtype=np.float64)
    print(f"HOG: {images.shape[0]} images -> {features.shape[1]} features")
    return features


if __name__ == '__main__':
    from step0_load_data import load_data

    train_images, _, test_images, _ = load_data(prefer_mat=False, use_kagglehub=True)

    print('\n=== Feature Extraction Demo ===')

    t0 = time.time()
    extract_dct_features(train_images)
    extract_dct_features(test_images)
    print(f'DCT elapsed: {time.time() - t0:.2f}s')

    t0 = time.time()
    extract_pca_features(train_images, test_images)
    print(f'PCA elapsed: {time.time() - t0:.2f}s')

    t0 = time.time()
    extract_hog_features(train_images)
    extract_hog_features(test_images)
    print(f'HOG elapsed: {time.time() - t0:.2f}s')
