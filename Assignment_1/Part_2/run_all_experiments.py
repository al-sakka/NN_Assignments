"""
Assignment 1 - Part 2: Experiment Runner
Author: Antar
Description: Run feature extraction, K-Means, and SVM experiments.
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score

from step0_load_data import load_data
from step1_features import extract_dct_features, extract_hog_features, extract_pca_features
from step2_kmeans import predict_kmeans, train_kmeans_classifier
from step3_svm import train_and_test_svm


FEATURES = ('DCT', 'PCA', 'HOG')
CLASSIFIERS = ('KMeans_K1', 'KMeans_K4', 'KMeans_K16', 'KMeans_K32', 'SVM_linear', 'SVM_rbf')


def _print_section(title):
    line = '=' * 78
    print(f"\n{line}\n{title}\n{line}")


def _print_result_line(classifier, feature_name, acc, elapsed):
    print(f"{classifier:<12} | {feature_name:<3} | Accuracy: {acc*100:6.2f}% | Time: {elapsed:7.2f}s")


def _print_summary(results):
    _print_section('RESULTS SUMMARY (REAL ELAPSED TIME)')
    print(f"{'Classifier':<12} | {'Feature':<3} | {'Accuracy':>10} | {'Time (s)':>8}")
    print('-' * 56)

    for classifier in CLASSIFIERS:
        for feature_name in FEATURES:
            acc, elapsed = results.get((classifier, feature_name), (0.0, 0.0))
            print(f"{classifier:<12} | {feature_name:<3} | {acc*100:9.2f}% | {elapsed:8.2f}")


def run_experiments():
    np.random.seed(42)

    train_images, train_labels, test_images, test_labels = load_data(
        prefer_mat=False,
        use_kagglehub=True,
    )

    _print_section('FEATURE EXTRACTION')
    dct_train = extract_dct_features(train_images)
    dct_test = extract_dct_features(test_images)
    pca_train, pca_test, _ = extract_pca_features(train_images, test_images)
    hog_train = extract_hog_features(train_images)
    hog_test = extract_hog_features(test_images)

    feature_sets = {
        'DCT': (dct_train, dct_test),
        'PCA': (pca_train, pca_test),
        'HOG': (hog_train, hog_test),
    }

    results = {}
    all_preds = {}

    _print_section('K-MEANS EXPERIMENTS')
    for k in (1, 4, 16, 32):
        classifier_name = f'KMeans_K{k}'
        for feature_name, (tr_feat, te_feat) in feature_sets.items():
            t_start = time.time()
            centroids, c_labels = train_kmeans_classifier(tr_feat, train_labels, k)
            preds = predict_kmeans(te_feat, centroids, c_labels)
            elapsed = time.time() - t_start
            acc = accuracy_score(test_labels, preds)

            results[(classifier_name, feature_name)] = (acc, elapsed)
            all_preds[(classifier_name, feature_name)] = preds
            _print_result_line(classifier_name, feature_name, acc, elapsed)

    _print_section('SVM EXPERIMENTS')
    for kernel in ('linear', 'rbf'):
        classifier_name = f'SVM_{kernel}'
        for feature_name, (tr_feat, te_feat) in feature_sets.items():
            t_start = time.time()
            _, acc, preds, _ = train_and_test_svm(
                tr_feat,
                train_labels,
                te_feat,
                test_labels,
                kernel,
                verbose=False,
            )
            elapsed = time.time() - t_start

            results[(classifier_name, feature_name)] = (acc, elapsed)
            all_preds[(classifier_name, feature_name)] = preds
            _print_result_line(classifier_name, feature_name, acc, elapsed)

    _print_summary(results)
    return results, all_preds, test_labels


if __name__ == '__main__':
    run_experiments()
