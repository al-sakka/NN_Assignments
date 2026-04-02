from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

def train_and_test_svm(train_feat, train_labels, test_feat, test_labels, kernel='rbf'):
    """
    Train a multi-class SVM and evaluate on test data.

    Scikit-learn's SVC uses one-vs-one (OvO) multi-class by default:
    it trains C(10,2) = 45 binary SVMs, one for each pair of digits.
    To classify a new sample, all 45 SVMs vote, and the majority wins.

    kernel : 'linear' or 'rbf'
      - linear: straight hyperplane boundary. Fast, good baseline.
      - rbf   : Radial Basis Function (Gaussian) kernel. Curved boundary.
                Almost always better for image data. Slower to train.

    StandardScaler: normalizes features to zero mean, unit variance.
    This is VERY important for SVM — features on different scales can
    make the optimizer converge poorly or favour certain dimensions.
    """
    print(f"\nTraining SVM ({kernel} kernel)...")

    # --- Scale features (fit only on training data!) ---
    scaler = StandardScaler().fit(train_feat)
    X_train = scaler.transform(train_feat)
    X_test  = scaler.transform(test_feat)

    # --- Build SVM model ---
    if kernel == 'linear':
        # C: regularization. Larger C = try harder to fit training data (risk overfitting)
        # Smaller C = larger margin, more misclassifications allowed (risk underfitting)
        model = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')

    elif kernel == 'rbf':
        # gamma='scale': sets gamma = 1/(n_features * X.var()) automatically
        # C=10: higher C works well for digit recognition
        model = SVC(kernel='rbf', C=10.0, gamma='scale', decision_function_shape='ovo')

    # --- Train ---
    t_start = time.time()
    model.fit(X_train, train_labels)
    train_time = time.time() - t_start
    print(f"  Training time: {train_time:.1f}s")

    # --- Predict ---
    t_pred = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - t_pred

    # --- Evaluate ---
    acc = accuracy_score(test_labels, predictions)
    print(f"  Accuracy: {acc*100:.2f}% | Train: {train_time:.1f}s | Predict: {pred_time:.2f}s")

    return model, acc, predictions, scaler