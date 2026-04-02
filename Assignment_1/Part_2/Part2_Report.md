# Assignment 1 - Part 2 Report

Author: Antar

## 1. Objective

The goal of Part 2 is to build and compare a complete handwritten digit classification pipeline on the ReducedMNIST dataset using:

1. Feature extractors:
   1. DCT
   1. PCA
   1. HOG
1. Classifiers:
   1. K-means with K in {1, 4, 16, 32}
   1. SVM with linear kernel
   1. SVM with nonlinear kernel

The assignment requires reporting:

1. Accuracy and processing time for all feature-classifier combinations
1. Confusion matrices for the best-performing K-means and SVM models
1. A final comparison and conclusions

## 2. End-to-End Pipeline Flow

The implemented flow in [run_all_experiments.py](run_all_experiments.py) is:

1. Load dataset and normalize images
1. Extract one feature set (DCT, PCA, or HOG)
1. Train/evaluate K-means for each K value
1. Train/evaluate SVM (linear and nonlinear)
1. Record accuracy and elapsed time
1. Build final comparison table
1. Plot confusion matrices and summary bars

### 2.1 Data Loading

Implemented in [step0_load_data.py](step0_load_data.py).

The loader supports:

1. Local MAT file if present
1. Kaggle ReducedMNIST JPG folders (downloaded to [ReducedMNIST_kaggle](ReducedMNIST_kaggle))
1. Other fallbacks if needed

Images are converted to grayscale float arrays in [0, 1].

### 2.2 Feature Extraction

Implemented in [step1_features.py](step1_features.py).

1. DCT features: 15x15 low-frequency block, flattened to 225-D
1. PCA features: principal components preserving at least 95% variance
1. HOG features: gradient orientation histogram descriptor

### 2.3 K-means Classification

Implemented in [step2_kmeans.py](step2_kmeans.py).

For each class, a separate K-means model is trained. During prediction, the label is selected by minimum distance to class centroids. Experiments were run for K = 1, 4, 16, 32.

### 2.4 SVM Classification

Implemented in [step3_svm.py](step3_svm.py).

Two variants were evaluated:

1. Linear SVM (`kernel="linear"`)
1. Nonlinear SVM (`kernel="rbf"`)

Both were trained in multiclass mode via `sklearn.svm.SVC`.

### 2.5 Visualization

Implemented in [step5_plots.py](step5_plots.py).

A single dashboard window is generated that includes:

1. Confusion matrix of best K-means result
1. Confusion matrix of best SVM result
1. Accuracy comparison bar chart
1. Time comparison bar chart

## 3. Nonlinear SVM (Required Explanation)

The nonlinear SVM in this work is the RBF-kernel SVM:

$$
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
$$

Why nonlinear SVM helps:

1. Digit classes are not perfectly linearly separable in raw or compressed feature spaces
1. RBF maps samples into a higher-dimensional implicit space
1. This allows curved decision boundaries that better separate confusing classes

Configuration used in experiments:

1. `kernel="rbf"`
1. `C=10`
1. `gamma="scale"`

This model achieved the highest overall test accuracy in this assignment (with DCT features).

## 4. Experimental Results

All results below were measured from the actual execution of [run_all_experiments.py](run_all_experiments.py) on the current setup.

### 4.1 K-means Results (Accuracy %, Time s)

|Feature|K=1|K=4|K=16|K=32|
|---|---|---|---|---|
|DCT|87.95 (0.88s)|92.80 (0.73s)|95.50 (1.06s)|**96.15 (2.05s)**|
|PCA|88.20 (1.70s)|92.80 (1.08s)|95.45 (1.77s)|95.95 (2.90s)|
|HOG|86.80 (13.64s)|92.95 (14.13s)|95.20 (16.25s)|95.70 (20.95s)|

Best K-means configuration:

1. Feature: DCT
1. K: 32
1. Accuracy: 96.15%
1. Time: 2.05 s

### 4.2 SVM Results (Accuracy %, Time s)

|Feature|Linear SVM|Nonlinear SVM (RBF)|
|---|---|---|
|DCT|95.30 (6.15s)|**96.90 (5.05s)**|
|PCA|95.95 (8.95s)|96.60 (7.53s)|
|HOG|95.70 (63.24s)|96.40 (40.55s)|

Best SVM configuration:

1. Feature: DCT
1. Kernel: RBF
1. Accuracy: 96.90%
1. Time: 5.05 s

## 5. Final Comparison Table (Required)

|Classifier|DCT Accuracy|DCT Time|PCA Accuracy|PCA Time|HOG Accuracy|HOG Time|
|---|---|---|---|---|---|---|
|KMeans K=1|87.95|0.88|88.20|1.70|86.80|13.64|
|KMeans K=4|92.80|0.73|92.80|1.08|92.95|14.13|
|KMeans K=16|95.50|1.06|95.45|1.77|95.20|16.25|
|KMeans K=32|**96.15**|2.05|95.95|2.90|95.70|20.95|
|SVM Linear|95.30|6.15|95.95|8.95|95.70|63.24|
|SVM RBF|**96.90**|5.05|96.60|7.53|96.40|40.55|

## 6. Confusion Matrices (Best Models)

Confusion matrices are generated for:

1. Best K-means: DCT + K=32
1. Best SVM: DCT + RBF

These are displayed in the dashboard and are produced from [step5_plots.py](step5_plots.py).

## 7. Discussion

Key observations:

1. DCT performed consistently well across both classifier families
1. Increasing K in K-means improved class representation and accuracy
1. Nonlinear SVM (RBF) outperformed linear SVM on all feature types
1. HOG provided strong accuracy but at significantly higher runtime cost

Trade-off summary:

1. Best absolute accuracy: SVM RBF + DCT (96.90%)
1. Fastest high-accuracy model: K-means K=32 + DCT (96.15% at lower runtime)

## 8. Conclusion

The assignment requirements for Part 2 were completed with a full comparison of DCT, PCA, and HOG features using K-means and SVM (linear/nonlinear). The best model was nonlinear SVM (RBF) with DCT features, reaching 96.90% accuracy. K-means with DCT and K=32 achieved competitive accuracy with lower computational cost, making it attractive when speed is prioritized.
