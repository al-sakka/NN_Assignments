import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from matplotlib.patches import Patch
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.stats import chi2
import os

# ─── output directory for figures ───────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── data ──────────────────────────────────────────────────────────────────────
#               Oil  Temp(F)  Insulation
raw_data = np.array([
    [270,  40,  4],
    [362,  27,  4],
    [162,  40, 10],
    [ 45,  73,  6],
    [ 91,  65,  7],
    [233,  65, 40],
    [372,  10,  6],
    [305,   9, 10],
    [234,  24, 10],
    [122,  65,  4],
    [ 25,  66, 10],
    [210,  41,  6],
    [450,  22,  4],
    [325,  40,  4],
    [ 52,  60, 10],
])

Oil  = raw_data[:, 0]
Temp = raw_data[:, 1]
Ins  = raw_data[:, 2]

X = np.column_stack([Temp, Ins])  # shape (15, 2)

# ─── helper functions ──────────────────────────────────────────────────────────

def plot_3d_scatter(temp, ins, oil, predictions, avg_oil, title, fname):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter3D(temp, ins, oil, color="blue", label="True Values")
    ax.scatter3D(temp, ins, predictions, color="red", label="Predictions")

    # average plane
    t_range = np.linspace(temp.min(), temp.max(), 10)
    i_range = np.linspace(ins.min(), ins.max(), 10)
    tg, ig = np.meshgrid(t_range, i_range)
    ax.plot_surface(tg, ig, np.full_like(tg, avg_oil), alpha=0.25, color="green")

    ax.set_xlabel("Temperature (F)")
    ax.set_ylabel("Insulation (in)")
    ax.set_zlabel("Oil")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(color="blue",  label="True Values"),
        Patch(color="red",   label="Predictions"),
        Patch(color="green", alpha=0.3, label="Average Oil"),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.show()


def plot_2d_projection(temp, ins, oil, predictions, avg_oil, title_sfx, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(temp, oil, color="blue", label="True")
    ax1.scatter(temp, predictions, color="red", label="Predicted")
    ax1.axhline(avg_oil, color="green", ls="--", alpha=0.4, label="Avg Oil")
    ax1.set_xlabel("Temperature (F)")
    ax1.set_ylabel("Oil")
    ax1.set_title(f"Temperature vs Oil {title_sfx}")
    ax1.legend()

    ax2.scatter(ins, oil, color="blue", label="True")
    ax2.scatter(ins, predictions, color="red", label="Predicted")
    ax2.axhline(avg_oil, color="green", ls="--", alpha=0.4, label="Avg Oil")
    ax2.set_xlabel("Insulation (in)")
    ax2.set_ylabel("Oil")
    ax2.set_title(f"Insulation vs Oil {title_sfx}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.show()


def plot_3d_surface(temp, ins, oil, model, title, fname, is_pipeline=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.scatter3D(temp, ins, oil, color="blue")

    t_range = np.linspace(temp.min(), temp.max(), 30)
    i_range = np.linspace(ins.min(), ins.max(), 30)
    tg, ig = np.meshgrid(t_range, i_range)
    Xg = np.column_stack([tg.ravel(), ig.ravel()])
    og = model.predict(Xg).reshape(tg.shape)
    ax.plot_surface(tg, ig, og, alpha=0.5, color="red")

    ax.set_xlabel("Temperature (F)")
    ax.set_ylabel("Insulation (in)")
    ax.set_zlabel("Oil")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(color="blue",      label="True Values"),
        Patch(color="red", alpha=0.5, label="Model Surface"),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.show()


def detect_outliers(X, confidence=0.95):
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov_inv = inv(cov)
    dists = np.array([mahalanobis(row, mean, cov_inv) for row in X])
    threshold = np.sqrt(chi2.ppf(confidence, X.shape[1]))
    outlier_mask = dists > threshold
    return outlier_mask, dists, threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  (a) Linear fit
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("WITH ALL DATA (including potential outliers)")
print("=" * 60)

avg_oil = Oil.mean()

# --- Linear ---
lin_model = LinearRegression().fit(X, Oil)
lin_pred  = lin_model.predict(X)
r2_lin    = r2_score(Oil, lin_pred)

print(f"\nLinear model:  Oil = {lin_model.coef_[0]:.4f}·Temp + {lin_model.coef_[1]:.4f}·Ins + {lin_model.intercept_:.4f}")
print(f"  R² = {r2_lin:.6f}")

plot_3d_scatter(Temp, Ins, Oil, lin_pred, avg_oil,
                "Linear Fit – True vs Predicted", "p2_linear_3d.png")
plot_2d_projection(Temp, Ins, Oil, lin_pred, avg_oil,
                   "(Linear)", "p2_linear_2d.png")
plot_3d_surface(Temp, Ins, Oil, lin_model,
                "Linear Model Surface", "p2_linear_surface.png")

# --- Quadratic ---
quad_pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                          LinearRegression())
quad_pipe.fit(X, Oil)
quad_pred = quad_pipe.predict(X)
r2_quad   = r2_score(Oil, quad_pred)

poly_step = quad_pipe.named_steps["polynomialfeatures"]
lr_step   = quad_pipe.named_steps["linearregression"]
feat_names = poly_step.get_feature_names_out(["Temp", "Ins"])
print(f"\nQuadratic model:")
for name, coef in zip(feat_names, lr_step.coef_):
    print(f"  {coef:+.4f} · {name}")
print(f"  intercept = {lr_step.intercept_:.4f}")
print(f"  R² = {r2_quad:.6f}")

plot_3d_scatter(Temp, Ins, Oil, quad_pred, avg_oil,
                "Quadratic Fit – True vs Predicted", "p2_quad_3d.png")
plot_2d_projection(Temp, Ins, Oil, quad_pred, avg_oil,
                   "(Quadratic)", "p2_quad_2d.png")
plot_3d_surface(Temp, Ins, Oil, quad_pipe,
                "Quadratic Model Surface", "p2_quad_surface.png")

# R² comparison
print(f"\nR² comparison:  Linear = {r2_lin:.6f},  Quadratic = {r2_quad:.6f}")
best_label = "Quadratic" if r2_quad > r2_lin else "Linear"
print(f"→ Better fit: {best_label}")

# ═══════════════════════════════════════════════════════════════════════════════
#  (b) Outlier detection and removal
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("OUTLIER DETECTION (Mahalanobis distance on input space)")
print("=" * 60)

# full feature set (Temp, Ins) for outlier detection
outlier_mask, dists, threshold = detect_outliers(X)
print(f"Threshold (95% confidence, df=2): {threshold:.4f}")
for i, (d, m) in enumerate(zip(dists, outlier_mask)):
    if m:
        print(f"  Outlier at index {i}: Temp={Temp[i]}, Ins={Ins[i]}, Oil={Oil[i]}, "
              f"Mahal. dist={d:.4f}")

X_clean    = X[~outlier_mask]
Oil_clean  = Oil[~outlier_mask]
Temp_clean = Temp[~outlier_mask]
Ins_clean  = Ins[~outlier_mask]

avg_oil_clean = Oil_clean.mean()

print(f"\nRemoved {outlier_mask.sum()} outlier(s). Remaining samples: {len(Oil_clean)}")

# --- Refit linear ---
print(f"\n{'=' * 60}")
print("AFTER OUTLIER REMOVAL")
print("=" * 60)

lin_model_c = LinearRegression().fit(X_clean, Oil_clean)
lin_pred_c  = lin_model_c.predict(X_clean)
r2_lin_c    = r2_score(Oil_clean, lin_pred_c)

print(f"\nLinear model (cleaned):  Oil = {lin_model_c.coef_[0]:.4f}·Temp + {lin_model_c.coef_[1]:.4f}·Ins + {lin_model_c.intercept_:.4f}")
print(f"  R² = {r2_lin_c:.6f}")

plot_3d_scatter(Temp_clean, Ins_clean, Oil_clean, lin_pred_c, avg_oil_clean,
                "Linear Fit – Cleaned Data", "p2_linear_3d_clean.png")
plot_2d_projection(Temp_clean, Ins_clean, Oil_clean, lin_pred_c, avg_oil_clean,
                   "(Linear, cleaned)", "p2_linear_2d_clean.png")
plot_3d_surface(Temp_clean, Ins_clean, Oil_clean, lin_model_c,
                "Linear Model Surface – Cleaned", "p2_linear_surface_clean.png")

# --- Refit quadratic ---
quad_pipe_c = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                            LinearRegression())
quad_pipe_c.fit(X_clean, Oil_clean)
quad_pred_c = quad_pipe_c.predict(X_clean)
r2_quad_c   = r2_score(Oil_clean, quad_pred_c)

poly_step_c = quad_pipe_c.named_steps["polynomialfeatures"]
lr_step_c   = quad_pipe_c.named_steps["linearregression"]
feat_names_c = poly_step_c.get_feature_names_out(["Temp", "Ins"])
print(f"\nQuadratic model (cleaned):")
for name, coef in zip(feat_names_c, lr_step_c.coef_):
    print(f"  {coef:+.4f} · {name}")
print(f"  intercept = {lr_step_c.intercept_:.4f}")
print(f"  R² = {r2_quad_c:.6f}")

plot_3d_scatter(Temp_clean, Ins_clean, Oil_clean, quad_pred_c, avg_oil_clean,
                "Quadratic Fit – Cleaned Data", "p2_quad_3d_clean.png")
plot_2d_projection(Temp_clean, Ins_clean, Oil_clean, quad_pred_c, avg_oil_clean,
                   "(Quadratic, cleaned)", "p2_quad_2d_clean.png")
plot_3d_surface(Temp_clean, Ins_clean, Oil_clean, quad_pipe_c,
                "Quadratic Model Surface – Cleaned", "p2_quad_surface_clean.png")

# R² comparison (cleaned)
print(f"\nR² comparison (cleaned):  Linear = {r2_lin_c:.6f},  Quadratic = {r2_quad_c:.6f}")
best_label_c = "Quadratic" if r2_quad_c > r2_lin_c else "Linear"
print(f"→ Better fit (cleaned): {best_label_c}")

# comparison before / after
print(f"\n--- Model Comparison ---")
print(f"{'Model':<20} {'R² (with outlier)':>18} {'R² (cleaned)':>14}")
print(f"{'Linear':<20} {r2_lin:>18.6f} {r2_lin_c:>14.6f}")
print(f"{'Quadratic':<20} {r2_quad:>18.6f} {r2_quad_c:>14.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  (c) Prediction for Temp=15 F, Insulation=5
# ═══════════════════════════════════════════════════════════════════════════════
x_new = np.array([[15, 5]])

pred_lin  = lin_model_c.predict(x_new)[0]
pred_quad = quad_pipe_c.predict(x_new)[0]

print(f"\n{'=' * 60}")
print("PREDICTION for Temp = 15 F, Insulation = 5 (cleaned models)")
print("=" * 60)
print(f"  Linear model:    Oil = {pred_lin:.2f}")
print(f"  Quadratic model: Oil = {pred_quad:.2f}")
