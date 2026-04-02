# ================================================================
# STEP 0: LOAD AND PREPARE THE REDUCED MNIST DATASET
# ================================================================
# The ReducedMNIST has:
#   Training: 1000 images per digit × 10 digits = 10,000 images
#   Testing:  200  images per digit × 10 digits = 2,000  images
# Each image is 28×28 = 784 pixels (grayscale, 0-255)
# ================================================================

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


def _find_mat_file(search_root):
    """Find the most likely ReducedMNIST .mat file inside a folder tree."""
    root = Path(search_root)
    candidates = list(root.rglob('*.mat'))
    if not candidates:
        raise FileNotFoundError(f"No .mat files found in: {root}")

    preferred_names = {
        'reducedmnist.mat',
        'reduced_mnist.mat',
    }

    for c in candidates:
        if c.name.lower() in preferred_names:
            return c

    # Fallback: first .mat file found.
    return candidates[0]


def _collect_image_files(class_dir):
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        files.extend(class_dir.glob(ext))
    return sorted(files)


def _load_image_split(split_dir):
    """Load one split folder where subfolders are labels 0..9 and files are images."""
    split_dir = Path(split_dir)
    images, labels = [], []

    for digit in range(10):
        class_dir = split_dir / str(digit)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_files = _collect_image_files(class_dir)
        if not image_files:
            raise FileNotFoundError(f"No image files found in: {class_dir}")

        for img_path in image_files:
            img = plt.imread(img_path)
            if img.ndim == 3:
                img = img.mean(axis=2)
            images.append(img.reshape(-1).astype(np.float64))
            labels.append(digit)

    X = np.vstack(images)
    y = np.array(labels, dtype=np.int32)
    return X, y


def load_from_jpg_dataset(dataset_root):
    """
    Load Reduced MNIST from JPG folders.
    Expected structure contains folders similar to:
      Reduced Trainging data/0..9
      Reduced Testing data/0..9
    """
    root = Path(dataset_root)
    train_dir = None
    test_dir = None

    for d in root.rglob('*'):
        if not d.is_dir():
            continue
        name = d.name.lower()
        if train_dir is None and ('train' in name or 'trainging' in name):
            if all((d / str(i)).is_dir() for i in range(10)):
                train_dir = d
        if test_dir is None and ('test' in name or 'testing' in name):
            if all((d / str(i)).is_dir() for i in range(10)):
                test_dir = d

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not locate train/test JPG folders under: {root}"
        )

    train_images, train_labels = _load_image_split(train_dir)
    test_images, test_labels = _load_image_split(test_dir)
    return train_images, train_labels, test_images, test_labels

# ----- Option A: If your data is a .mat file -----
def load_from_mat(filepath='ReducedMNIST.mat'):
    mat_path = Path(filepath)
    if not mat_path.is_absolute():
        mat_path = Path(__file__).resolve().parent / mat_path

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset file: {mat_path}\n"
            "Place ReducedMNIST.mat in Assignment_1/Part_2, or use load_from_keras()."
        )

    data = scipy.io.loadmat(mat_path)
    # Adjust key names to match what's actually in your .mat file
    required = ['train_images', 'train_labels', 'test_images', 'test_labels']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            "MAT file is missing expected keys: " + ", ".join(missing)
        )

    train_images = data['train_images'].astype(np.float64)
    train_labels = data['train_labels'].ravel().astype(np.int32)
    test_images  = data['test_images'].astype(np.float64)
    test_labels  = data['test_labels'].ravel().astype(np.int32)
    return train_images, train_labels, test_images, test_labels

# ----- Option B: Download directly from the internet -----
def load_from_keras():
    # Uses Keras to download MNIST (requires: pip install tensorflow)
    # Then we manually reduce to 1000 train / 200 test per class
    from tensorflow.keras.datasets import mnist
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    # Flatten 28x28 → 784 and normalize to [0,1]
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_test_full  = X_test_full.reshape(-1, 784)  / 255.0

    # Sample 1000 per class for training, 200 per class for testing
    train_idx, test_idx = [], []
    for digit in range(10):
        tr = np.where(y_train_full == digit)[0][:1000]
        te = np.where(y_test_full  == digit)[0][:200]
        train_idx.extend(tr); test_idx.extend(te)

    return (X_train_full[train_idx], y_train_full[train_idx],
            X_test_full[test_idx],  y_test_full[test_idx])


def load_from_openml():
    """
    Download MNIST using scikit-learn OpenML and reduce it to
    1000 train / 200 test samples per class.
    """
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int32)

    # Standard MNIST split in OpenML: first 60k train, last 10k test.
    X_train_full, X_test_full = X[:60000], X[60000:]
    y_train_full, y_test_full = y[:60000], y[60000:]

    train_idx, test_idx = [], []
    for digit in range(10):
        tr = np.where(y_train_full == digit)[0][:1000]
        te = np.where(y_test_full == digit)[0][:200]
        train_idx.extend(tr)
        test_idx.extend(te)

    return (
        X_train_full[train_idx], y_train_full[train_idx],
        X_test_full[test_idx], y_test_full[test_idx],
    )


def load_from_kagglehub(dataset='mohamedgamal07/reduced-mnist',
                        local_dir_name='ReducedMNIST_kaggle',
                        force_download=False):
    """
    Download Reduced MNIST from KaggleHub into the run directory
    and load images from JPG folders.
    Requires `kagglehub` and valid Kaggle credentials.
    """
    import kagglehub

    run_dir = Path(__file__).resolve().parent
    dataset_dir = run_dir / local_dir_name

    has_images = dataset_dir.exists() and any(dataset_dir.rglob('*.jpg'))
    if force_download and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        has_images = False

    if not has_images:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Kaggle dataset to: {dataset_dir}")
        kagglehub.dataset_download(dataset, output_dir=str(dataset_dir))

    return load_from_jpg_dataset(dataset_dir)


def load_data(prefer_mat=True, mat_filepath='ReducedMNIST.mat',
              use_kagglehub=True, kaggle_dataset='mohamedgamal07/reduced-mnist'):
    """
    Load ReducedMNIST from .mat when available, otherwise fall back to Keras MNIST.
    Returns normalized image vectors in [0, 1].
    """
    if prefer_mat:
        try:
            train_images, train_labels, test_images, test_labels = load_from_mat(mat_filepath)
        except FileNotFoundError:
            print("ReducedMNIST.mat not found.")
            loaded = False

            if use_kagglehub:
                try:
                    print("Trying KaggleHub ReducedMNIST download...")
                    train_images, train_labels, test_images, test_labels = load_from_kagglehub(kaggle_dataset)
                    loaded = True
                except ModuleNotFoundError:
                    print("kagglehub is not installed.")
                except Exception as exc:
                    print(f"KaggleHub download failed: {exc}")

            if not loaded:
                print("Falling back to online MNIST download...")
                try:
                    train_images, train_labels, test_images, test_labels = load_from_keras()
                except ModuleNotFoundError:
                    print("TensorFlow not installed, using OpenML MNIST fallback...")
                    train_images, train_labels, test_images, test_labels = load_from_openml()
    else:
        loaded = False

        if use_kagglehub:
            try:
                print("Loading ReducedMNIST JPG dataset via KaggleHub...")
                train_images, train_labels, test_images, test_labels = load_from_kagglehub(kaggle_dataset)
                loaded = True
            except ModuleNotFoundError:
                print("kagglehub is not installed.")
            except Exception as exc:
                print(f"KaggleHub load failed: {exc}")

        if not loaded:
            try:
                train_images, train_labels, test_images, test_labels = load_from_keras()
            except ModuleNotFoundError:
                print("TensorFlow not installed, using OpenML MNIST fallback...")
                train_images, train_labels, test_images, test_labels = load_from_openml()

    if train_images.max() > 1.0:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    # Demo/preview mode when this file is run directly.
    train_images, train_labels, test_images, test_labels = load_data(prefer_mat=True)

    print(f"Train: {train_images.shape} | Test: {test_images.shape}")
    print(f"Labels range: {train_labels.min()} to {train_labels.max()}")

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for d in range(10):
        idx = np.where(train_labels == d)[0][0]
        axes[d].imshow(train_images[idx].reshape(28, 28), cmap='gray')
        axes[d].set_title(str(d)); axes[d].axis('off')
    plt.suptitle('Sample Images: Digits 0-9')
    plt.show()