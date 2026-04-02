import numpy as np
from scipy.fft import dctn           # N-dimensional DCT
from sklearn.decomposition import PCA
from skimage.feature import hog     # Histogram of Oriented Gradients
import time

# ================================================================
# 1A. DCT FEATURES (225 dimensions = 15×15 lowest-frequency block)
# ================================================================
def extract_dct_features(images, block_size=15):
    """
    Extract 2D DCT features from a set of images.

    images     : np.array of shape (N, 784), values in [0,1]
    block_size : keep the top-left block_size x block_size DCT coefficients
                 e.g., block_size=15 → 225 features

    Returns    : np.array of shape (N, block_size*block_size)
    """
    N = images.shape[0]
    n_feats = block_size ** 2
    features = np.zeros((N, n_feats))

    for i in range(N):
        # Reshape flat 784-vector back to 28×28 image
        img = images[i].reshape(28, 28)

        # Apply 2D DCT (scipy's dctn with norm='ortho' for standard normalisation)
        dct_img = dctn(img, norm='ortho')

        # Take top-left block_size × block_size corner = low-frequency components
        block = dct_img[:block_size, :block_size]

        # Flatten and store
        features[i] = block.ravel()

    print(f"DCT: {N} images → {n_feats} features each")
    return features


# ================================================================
# 1B. PCA FEATURES (≥95% variance explained)
# ================================================================
def extract_pca_features(train_images, test_images, var_threshold=0.95):
    """
    Fit PCA on training data, project both train and test.

    IMPORTANT: We fit ONLY on training data, then apply the same
    transformation to test data. If we included test data in the fit,
    that would be 'data leakage' — the model would have secretly
    seen the test data during training.

    Returns: train_feat, test_feat, K (number of components kept)
    """
    # Fit PCA — n_components='mle' auto-selects, but we want ≥95% explained variance
    # So we first fit with all components, then find the cutoff
    pca_full = PCA().fit(train_images)

    # Cumulative variance explained by each component
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # Find minimum K such that cumvar[K-1] >= var_threshold
    K = int(np.argmax(cumvar >= var_threshold) + 1)

    print(f"PCA: keeping {K} components (explains {cumvar[K-1]*100:.1f}% variance)")

    # Refit PCA keeping only K components (faster than full PCA)
    pca = PCA(n_components=K).fit(train_images)

    # Transform both splits
    train_feat = pca.transform(train_images)  # [N_train × K]
    test_feat  = pca.transform(test_images)   # [N_test  × K]

    return train_feat, test_feat, K


# ================================================================
# 1C. HOG FEATURES
# ================================================================
def extract_hog_features(images, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """
    Extract HOG (Histogram of Oriented Gradients) features.

    images           : np.array of shape (N, 784)
    pixels_per_cell  : size of each HOG cell in pixels (4×4 is good for 28×28)
    cells_per_block  : how many cells in each normalization block

    Returns: np.array of shape (N, D) where D is determined by the HOG params
    """
    feature_list = []

    for i in range(images.shape[0]):
        img = images[i].reshape(28, 28)

        # hog() from skimage returns a flat feature vector
        # orientations=9: 9 angle bins from 0° to 180° (every 20°)
        # pixels_per_cell: each small region of 4x4 pixels forms one cell
        # cells_per_block: normalize over 2x2 groups of cells
        # feature_vector=True: return flat array (not a 3D block array)
        feat = hog(img,
                   orientations=9,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   feature_vector=True)
        feature_list.append(feat)

    features = np.array(feature_list)
    print(f"HOG: {images.shape[0]} images → {features.shape[1]} features each")
    return features


# ---- EXTRACT ALL FEATURES ----
if __name__ == '__main__':
    from step0_load_data import load_data

    train_images, train_labels, test_images, test_labels = load_data(prefer_mat=True)

    print("\n=== Extracting Features ===")

    t0 = time.time()
    dct_train = extract_dct_features(train_images)
    dct_test = extract_dct_features(test_images)
    print(f"DCT time: {time.time()-t0:.1f}s")

    t0 = time.time()
    pca_train, pca_test, K_pca = extract_pca_features(train_images, test_images)
    print(f"PCA time: {time.time()-t0:.1f}s")

    t0 = time.time()
    hog_train = extract_hog_features(train_images)
    hog_test = extract_hog_features(test_images)
    print(f"HOG time: {time.time()-t0:.1f}s")