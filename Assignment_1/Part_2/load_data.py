"""
Assignment 1 - Part 2: Data Loading
Author: Antar
Description: Load ReducedMNIST from MAT or JPG sources with safe fallbacks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def _collect_image_files(class_dir):
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        files.extend(class_dir.glob(ext))
    return sorted(files)


def _load_image_split(split_dir):
    images = []
    labels = []
    split_dir = Path(split_dir)

    for digit in range(10):
        class_dir = split_dir / str(digit)
        if not class_dir.exists():
            raise FileNotFoundError(f'Missing class folder: {class_dir}')

        image_files = _collect_image_files(class_dir)
        if not image_files:
            raise FileNotFoundError(f'No image files found in: {class_dir}')

        for img_path in image_files:
            img = plt.imread(img_path)
            if img.ndim == 3:
                img = img.mean(axis=2)  # Convert RGB to grayscale if needed.
            images.append(img.reshape(-1).astype(np.float64))
            labels.append(digit)

    return np.vstack(images), np.array(labels, dtype=np.int32)


def load_from_jpg_dataset(dataset_root):
    """Load train/test splits from JPG folders with digit subdirectories."""
    root = Path(dataset_root)
    train_dir = None
    test_dir = None

    for directory in root.rglob('*'):
        if not directory.is_dir():
            continue

        if not all((directory / str(i)).is_dir() for i in range(10)):
            continue

        name = directory.name.lower()
        if train_dir is None and ('train' in name or 'trainging' in name):
            train_dir = directory
        if test_dir is None and ('test' in name or 'testing' in name):
            test_dir = directory

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(f'Could not locate train/test folders under: {root}')

    train_images, train_labels = _load_image_split(train_dir)
    test_images, test_labels = _load_image_split(test_dir)
    return train_images, train_labels, test_images, test_labels


def load_from_mat(filepath='ReducedMNIST.mat'):
    """Load ReducedMNIST from a MAT file with expected key names."""
    mat_path = Path(filepath)
    if not mat_path.is_absolute():
        mat_path = Path(__file__).resolve().parent / mat_path

    if not mat_path.exists():
        raise FileNotFoundError(f'Could not find dataset file: {mat_path}')

    data = scipy.io.loadmat(mat_path)
    required = ('train_images', 'train_labels', 'test_images', 'test_labels')
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError('MAT file is missing keys: ' + ', '.join(missing))

    train_images = data['train_images'].astype(np.float64)
    train_labels = data['train_labels'].ravel().astype(np.int32)
    test_images = data['test_images'].astype(np.float64)
    test_labels = data['test_labels'].ravel().astype(np.int32)
    return train_images, train_labels, test_images, test_labels


def load_from_keras():
    """Download MNIST via Keras and reduce to 1000/200 samples per class."""
    from tensorflow.keras.datasets import mnist

    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    x_train_full = x_train_full.reshape(-1, 784) / 255.0
    x_test_full = x_test_full.reshape(-1, 784) / 255.0

    train_idx = []
    test_idx = []
    for digit in range(10):
        train_idx.extend(np.where(y_train_full == digit)[0][:1000])
        test_idx.extend(np.where(y_test_full == digit)[0][:200])

    return (
        x_train_full[train_idx],
        y_train_full[train_idx],
        x_test_full[test_idx],
        y_test_full[test_idx],
    )


def load_from_openml():
    """Download MNIST via OpenML and reduce to 1000/200 samples per class."""
    from sklearn.datasets import fetch_openml

    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    x = x.astype(np.float64) / 255.0
    y = y.astype(np.int32)

    x_train_full, x_test_full = x[:60000], x[60000:]
    y_train_full, y_test_full = y[:60000], y[60000:]

    train_idx = []
    test_idx = []
    for digit in range(10):
        train_idx.extend(np.where(y_train_full == digit)[0][:1000])
        test_idx.extend(np.where(y_test_full == digit)[0][:200])

    return (
        x_train_full[train_idx],
        y_train_full[train_idx],
        x_test_full[test_idx],
        y_test_full[test_idx],
    )


def load_from_kagglehub(
    dataset='mohamedgamal07/reduced-mnist',
    local_dir_name='ReducedMNIST_kaggle',
    force_download=False,
):
    """Download ReducedMNIST JPG dataset via KaggleHub and load it."""
    import kagglehub

    _ = local_dir_name  # Kept for backward compatibility with older calls.

    download_kwargs = {}
    if force_download:
        download_kwargs['force_download'] = True

    try:
        path = Path(kagglehub.dataset_download(dataset, **download_kwargs))
    except TypeError:
        # Older kagglehub versions may not support force_download.
        path = Path(kagglehub.dataset_download(dataset))

    training_data_path = path / 'Reduced MNIST Data' / 'Reduced Trainging data'
    test_data_path = path / 'Reduced MNIST Data' / 'Reduced Testing data'

    if training_data_path.exists() and test_data_path.exists():
        train_images, train_labels = _load_image_split(training_data_path)
        test_images, test_labels = _load_image_split(test_data_path)
        return train_images, train_labels, test_images, test_labels

    print(f'Expected Kaggle folder layout not found under: {path}')
    return load_from_jpg_dataset(path)


def load_data(
    prefer_mat=False,
    mat_filepath='ReducedMNIST.mat',
    use_kagglehub=True,
    kaggle_dataset='mohamedgamal07/reduced-mnist',
):
    """Load dataset and always return flattened normalized image vectors."""
    if prefer_mat:
        try:
            train_images, train_labels, test_images, test_labels = load_from_mat(mat_filepath)
        except FileNotFoundError:
            print('ReducedMNIST.mat not found.')
            train_images = train_labels = test_images = test_labels = None
    else:
        train_images = train_labels = test_images = test_labels = None

    if train_images is None and use_kagglehub:
        try:
            print('Loading ReducedMNIST JPG dataset via KaggleHub...')
            train_images, train_labels, test_images, test_labels = load_from_kagglehub(kaggle_dataset)
        except ModuleNotFoundError:
            print('kagglehub is not installed.')
        except Exception as exc:
            print(f'KaggleHub load failed: {exc}')

    if train_images is None:
        try:
            train_images, train_labels, test_images, test_labels = load_from_keras()
        except ModuleNotFoundError:
            print('TensorFlow not installed, using OpenML fallback...')
            train_images, train_labels, test_images, test_labels = load_from_openml()

    if train_images.max() > 1.0:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data(
        prefer_mat=False,
        use_kagglehub=True,
    )

    print(f'Train: {train_images.shape} | Test: {test_images.shape}')
    print(f'Labels range: {train_labels.min()} to {train_labels.max()}')

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for digit in range(10):
        idx = np.where(train_labels == digit)[0][0]
        axes[digit].imshow(train_images[idx].reshape(28, 28), cmap='gray')
        axes[digit].set_title(str(digit))
        axes[digit].axis('off')
    plt.suptitle('Sample Images: Digits 0-9')
    plt.show()
