import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def _draw_confusion_matrix(ax, true_labels, pred_labels, title=''):
    """
    Draw a 10x10 confusion matrix for digit classification on an axis.

    Each cell (i, j) shows how many times true digit i
    was predicted as digit j.
    Diagonal = correct. Off-diagonal = mistakes.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(10)))
    acc = accuracy_score(true_labels, pred_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(range(10)))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{title}\nAccuracy: {acc*100:.2f}%", fontsize=13)
    ax.set_xlabel('Predicted Digit')
    ax.set_ylabel('True Digit')


def _draw_summary_bars(ax, results):
    # Bonus: bar chart comparing all results.
    clf_names = ['KMeans K=1', 'KMeans K=4', 'KMeans K=16', 'KMeans K=32',
                 'SVM linear', 'SVM RBF']
    keys = ['KMeans_K1', 'KMeans_K4', 'KMeans_K16', 'KMeans_K32',
            'SVM_linear', 'SVM_rbf']
    colors = ['#4f8ef7', '#f7c14f', '#5cf7b0']  # DCT, PCA, HOG
    feat_list = ['DCT', 'PCA', 'HOG']

    x = np.arange(len(clf_names))
    width = 0.25

    for i, (ft, col) in enumerate(zip(feat_list, colors)):
        accs = [results.get((k, ft), (0, 0))[0] * 100 for k in keys]
        ax.bar(x + i * width, accs, width, label=ft, color=col, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(clf_names, rotation=20)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 102)
    ax.set_title('All Results: Features x Classifiers')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_confusion_matrix(true_labels, pred_labels, title=''):
    """Standalone confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_confusion_matrix(ax, true_labels, pred_labels, title)
    plt.tight_layout()
    plt.show()


def plot_summary_bars(results):
    """Standalone bar chart plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_summary_bars(ax, results)
    plt.tight_layout()
    plt.show()


def plot_all_together(test_labels, all_preds, results):
    """Show the two confusion matrices and summary bars in one window."""
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    _draw_confusion_matrix(
        ax1,
        test_labels,
        all_preds[('KMeans_K32', 'HOG')],
        'Best K-Means: HOG + K=32',
    )
    _draw_confusion_matrix(
        ax2,
        test_labels,
        all_preds[('SVM_rbf', 'HOG')],
        'Best SVM: HOG + RBF Kernel',
    )
    _draw_summary_bars(ax3, results)

    fig.suptitle('Assignment 1 Part 2 Results', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    # Pull experiment outputs from the runner script.
    from run_all_experiments import all_preds, results, test_labels
    plot_all_together(test_labels, all_preds, results)


if __name__ == '__main__':
    main()