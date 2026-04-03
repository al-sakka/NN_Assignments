import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os

# ─── create output directory for figures ───────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── data ──────────────────────────────────────────────────────────────────────
years = np.array([1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996])
y = np.array([12400, 10900, 10000, 1050, 9500, 8900, 8000, 7800, 7600, 7200])
x = (years - 1987).reshape(-1, 1)

# ─── helper functions ──────────────────────────────────────────────────────────

def fit_polynomial(x, y, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    return model, poly, y_pred, r2


def polynomial_equation_str(model, degree):
    coeffs = model.coef_
    intercept = model.intercept_
    terms = []
    for i in range(degree, 0, -1):
        terms.append(f"{coeffs[i]:+.4f}·x^{i}" if i > 1 else f"{coeffs[i]:+.4f}·x")
    terms.append(f"{intercept:+.4f}")
    return "y = " + " ".join(terms)


# ═══════════════════ (a) Scatter plot ══════════════════════════════════════════
plt.figure()
plt.scatter(x, y, color="blue", zorder=5, label="Original Data")
plt.xlabel("Years since 1987")
plt.ylabel("Number of insured persons")
plt.title("Scatter Plot of Insurance Data")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_scatter.png"), dpi=150)
plt.show()

# ═══════════════════ (a) Fit linear, quadratic, cubic ═════════════════════════
models = {}
r2_scores = {}
colors = ["green", "orange", "purple"]

print("=" * 60)
print("WITH OUTLIER")
print("=" * 60)

for deg in [1, 2, 3]:
    model, poly, y_pred, r2 = fit_polynomial(x, y, deg)
    models[deg] = (model, poly)
    r2_scores[deg] = r2
    eq = polynomial_equation_str(model, deg)
    print(f"\nDegree {deg}: {eq}")
    print(f"  R² = {r2:.6f}")

# ── individual fit plots ──
for deg in [1, 2, 3]:
    model, poly = models[deg]
    x_smooth = np.linspace(0, 10, 200).reshape(-1, 1)
    y_smooth = model.predict(poly.transform(x_smooth))
    plt.figure()
    plt.scatter(x, y, color="blue", zorder=5, label="Data")
    plt.plot(x_smooth, y_smooth, color="red",
             label=f"Degree {deg}  (R²={r2_scores[deg]:.4f})")
    plt.xlabel("Years since 1987")
    plt.ylabel("Number of insured persons")
    plt.title(f"Polynomial Fit – Degree {deg}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"p1_fit_deg{deg}.png"), dpi=150)
    plt.show()

# ── overlay all fits ──
plt.figure()
plt.scatter(x, y, color="blue", zorder=5, label="Data")
for i, deg in enumerate([1, 2, 3]):
    model, poly = models[deg]
    x_smooth = np.linspace(0, 10, 200).reshape(-1, 1)
    y_smooth = model.predict(poly.transform(x_smooth))
    plt.plot(x_smooth, y_smooth, color=colors[i],
             label=f"Degree {deg}  (R²={r2_scores[deg]:.4f})")
plt.xlabel("Years since 1987")
plt.ylabel("Number of insured persons")
plt.title("Overlay of Polynomial Fits (with outlier)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_overlay_with_outlier.png"), dpi=150)
plt.show()

# Identify best fit
best_deg = max(r2_scores, key=r2_scores.get)
print(f"\n→ Best fit with outlier: degree {best_deg}  (R² = {r2_scores[best_deg]:.6f})")

# ═══════════════════ (b) Outlier removal ══════════════════════════════════════
# The value 1050 at x=3 (year 1990) is an outlier
outlier_idx = 3
print(f"\nOutlier detected at x={outlier_idx} (year {years[outlier_idx]}): y = {y[outlier_idx]}")

x_clean = np.delete(x, outlier_idx, axis=0)
y_clean = np.delete(y, outlier_idx)

models_clean = {}
r2_clean = {}

print("\n" + "=" * 60)
print("AFTER OUTLIER REMOVAL")
print("=" * 60)

for deg in [1, 2, 3]:
    model, poly, y_pred, r2 = fit_polynomial(x_clean, y_clean, deg)
    models_clean[deg] = (model, poly)
    r2_clean[deg] = r2
    eq = polynomial_equation_str(model, deg)
    print(f"\nDegree {deg}: {eq}")
    print(f"  R² = {r2:.6f}")

best_deg_clean = max(r2_clean, key=r2_clean.get)
print(f"\n→ Best fit without outlier: degree {best_deg_clean}  (R² = {r2_clean[best_deg_clean]:.6f})")

print("\nR² comparison (cleaned data):")
for deg in [1, 2, 3]:
    gain = r2_clean[deg] - r2_clean.get(deg - 1, 0) if deg > 1 else r2_clean[deg]
    print(f"  Degree {deg}: R² = {r2_clean[deg]:.6f}" +
          (f"  (gain = {gain:.6f})" if deg > 1 else ""))

# ── overlay all fits on cleaned data ──
plt.figure()
plt.scatter(x_clean, y_clean, color="blue", zorder=5, label="Cleaned Data")
for i, deg in enumerate([1, 2, 3]):
    model, poly = models_clean[deg]
    x_smooth = np.linspace(0, 10, 200).reshape(-1, 1)
    y_smooth = model.predict(poly.transform(x_smooth))
    plt.plot(x_smooth, y_smooth, color=colors[i],
             label=f"Degree {deg}  (R²={r2_clean[deg]:.4f})")
plt.xlabel("Years since 1987")
plt.ylabel("Number of insured persons")
plt.title("Overlay of Polynomial Fits (outlier removed)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_overlay_clean.png"), dpi=150)
plt.show()

# ═══════════════════ (c) Best-fit function graph ═════════════════════════════
best_model, best_poly = models_clean[best_deg_clean]
x_smooth = np.linspace(0, 10, 200).reshape(-1, 1)
y_smooth = best_model.predict(best_poly.transform(x_smooth))

plt.figure()
plt.scatter(x_clean, y_clean, color="blue", zorder=5, label="Cleaned Data")
plt.plot(x_smooth, y_smooth, color="red",
         label=f"Best Fit: Degree {best_deg_clean}  (R²={r2_clean[best_deg_clean]:.4f})")
plt.xlabel("Years since 1987")
plt.ylabel("Number of insured persons")
plt.title("Best-Fit Function with Cleaned Data")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_best_fit.png"), dpi=150)
plt.show()

# ═══════════════════ (d) Prediction for 1997 ═════════════════════════════════
x_1997 = np.array([[10]])  # 1997 - 1987 = 10
y_pred_1997 = best_model.predict(best_poly.transform(x_1997))[0]

print(f"\n{'=' * 60}")
print(f"PREDICTION FOR 1997 (x = 10)")
print(f"{'=' * 60}")
print(f"Using best model (degree {best_deg_clean}, cleaned data):")
print(f"  Predicted number of insured persons in 1997 = {y_pred_1997:.0f}")

# ── prediction plot ──
plt.figure()
plt.scatter(x_clean, y_clean, color="blue", zorder=5, label="Cleaned Data")
plt.plot(x_smooth, y_smooth, color="red",
         label=f"Best Fit (deg {best_deg_clean})")
plt.scatter(10, y_pred_1997, color="green", marker="x", s=150, zorder=6,
            label=f"Prediction 1997: {y_pred_1997:.0f}")
plt.xlabel("Years since 1987")
plt.ylabel("Number of insured persons")
plt.title("Prediction for 1997")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "p1_prediction_1997.png"), dpi=150)
plt.show()

# predictions for all degrees
print("\nAll predictions for 1997:")
print(f"  {'Degree':<10} {'With outlier':>14} {'Cleaned':>10}")
for deg in [1, 2, 3]:
    model_o, poly_o = models[deg]
    pred_o = model_o.predict(poly_o.transform(x_1997))[0]
    model_c, poly_c = models_clean[deg]
    pred_c = model_c.predict(poly_c.transform(x_1997))[0]
    print(f"  {deg:<10} {pred_o:>14.0f} {pred_c:>10.0f}")

# ═══════════════════ Hand calculation ════════════════════════════
print(f"\n{'=' * 60}")
print("HAND CALCULATION — Linear Fit (cleaned data, 9 points)")
print("=" * 60)

x_vals = x_clean.flatten()
y_vals = y_clean

n = len(x_vals)
# Design matrix for linear: [1, x]
X_design = np.column_stack([np.ones(n), x_vals])

print(f"\nDesign matrix X (shape {X_design.shape}):")
print(X_design)

XTX = X_design.T @ X_design
print(f"\nX^T X =")
print(XTX)

XTX_inv = np.linalg.inv(XTX)
print(f"\n(X^T X)^(-1) =")
print(XTX_inv)

XTy = X_design.T @ y_vals
print(f"\nX^T y =")
print(XTy)

theta = XTX_inv @ XTy
print(f"\ntheta = (X^T X)^(-1) X^T y =")
print(f"  a0 (intercept) = {theta[0]:.4f}")
print(f"  a1 (slope)     = {theta[1]:.4f}")
print(f"\nLinear equation: y = {theta[1]:.4f} x + {theta[0]:.4f}")

# Verify R²
y_pred_hand = X_design @ theta
SS_res = np.sum((y_vals - y_pred_hand)**2)
SS_tot = np.sum((y_vals - np.mean(y_vals))**2)
R2_hand = 1 - SS_res / SS_tot
print(f"R² = 1 - SS_res/SS_tot = 1 - {SS_res:.2f}/{SS_tot:.2f} = {R2_hand:.6f}")
