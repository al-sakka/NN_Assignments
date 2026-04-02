"""
Assignment 1 - Part 2: SVM Classifier
Author: Antar
Description: Train and evaluate linear or RBF SVM models.
"""

import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_and_test_svm(
    train_feat,
    train_labels,
    test_feat,
    test_labels,
    kernel='rbf',
    verbose=True,
):
    """Train SVM, run inference, and return model outputs."""
    if verbose:
        print(f"\nTraining SVM ({kernel} kernel)...")

    scaler = StandardScaler().fit(train_feat)
    x_train = scaler.transform(train_feat)
    x_test = scaler.transform(test_feat)

    if kernel == 'linear':
        model = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
    elif kernel == 'rbf':
        model = SVC(kernel='rbf', C=10.0, gamma='scale', decision_function_shape='ovo')
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'.")

    t_start = time.time()
    model.fit(x_train, train_labels)
    predictions = model.predict(x_test)
    elapsed = time.time() - t_start

    acc = accuracy_score(test_labels, predictions)
    if verbose:
        print(f"  Accuracy: {acc*100:.2f}% | Time: {elapsed:.2f}s")

    return model, acc, predictions, scaler
