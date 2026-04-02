"""
Assignment 1 - Part 2: Plots
Author: Antar
Description: Visualize confusion matrices and experiment metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


FEATURES = ('DCT', 'PCA', 'HOG')
CLASSIFIERS = ('KMeans_K1', 'KMeans_K4', 'KMeans_K16', 'KMeans_K32', 'SVM_linear', 'SVM_rbf')
CLASSIFIER_LABELS = ('KMeans K=1', 'KMeans K=4', 'KMeans K=16', 'KMeans K=32', 'SVM linear', 'SVM RBF')
COLORS = ('#4f8ef7', '#f7c14f', '#5cf7b0')


def _draw_confusion(ax, true_labels, pred_labels, title):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(10)))
    acc = accuracy_score(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{title}\nAccuracy: {acc*100:.2f}%")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')


def _draw_metric_bars(ax, results, metric):
    x = np.arange(len(CLASSIFIERS))
    width = 0.25

    for i, (feature_name, color) in enumerate(zip(FEATURES, COLORS)):
        if metric == 'accuracy':
            values = [results.get((clf, feature_name), (0.0, 0.0))[0] * 100 for clf in CLASSIFIERS]
            ylabel = 'Accuracy (%)'
            title = 'Accuracy by Classifier and Feature'
        else:
            values = [results.get((clf, feature_name), (0.0, 0.0))[1] for clf in CLASSIFIERS]
            ylabel = 'Time (s)'
            title = 'Elapsed Time by Classifier and Feature'

        ax.bar(x + i * width, values, width, label=feature_name, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSIFIER_LABELS, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()


def plot_all_together(test_labels, all_preds, results):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _draw_confusion(ax1, test_labels, all_preds[('KMeans_K32', 'HOG')], 'Best K-Means: HOG + K=32')
    _draw_confusion(ax2, test_labels, all_preds[('SVM_rbf', 'HOG')], 'Best SVM: HOG + RBF')
    _draw_metric_bars(ax3, results, metric='accuracy')
    _draw_metric_bars(ax4, results, metric='time')

    fig.suptitle('Assignment 1 Part 2 Dashboard', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    from run_all_experiments import run_experiments

    results, all_preds, test_labels = run_experiments()
    plot_all_together(test_labels, all_preds, results)


if __name__ == '__main__':
    main()
