"""
Analytical verification of the fitted parameters (theta, M, X), and a
visual overlay of the data against the reconstructed curve.

Why this is a stronger check than nearest-point L1 distance
-------------------------------------------------------------
fit_curve.py verifies the fit by sampling the candidate curve on a
finite t-grid and measuring the distance from each data point to its
NEAREST sampled point. That's accurate, but it's still bounded by the
resolution of the grid, and it doesn't use any structure of the
equations.

Because the rotation matrix
    [cos(theta)  -sin(theta)]
    [sin(theta)   cos(theta)]
is orthogonal, its inverse is just its transpose. So, given a fitted
(theta, X), we can invert the rotation directly for every data point
and recover the exact (t, b) pair that must have produced it, with NO
grid or search involved:

    t_recovered =  (x - X) * cos(theta) + (y - 42) * sin(theta)
    b_recovered = -(x - X) * sin(theta) + (y - 42) * cos(theta)

If (theta, M, X) are correct, b_recovered must equal
e^(M*|t_recovered|) * sin(0.3 * t_recovered) for every point, and
t_recovered must fall inside (6, 60). This script checks both
conditions directly (closed-form, no sampling error), which is a
stronger and more precise confirmation than the nearest-point method.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/xy_data.csv"

THETA_DEG = 30.0
M = 0.03
X = 55.0


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df["x"].values, df["y"].values


def analytical_recovery(data_x, data_y, theta_deg, M, X):
    theta = np.radians(theta_deg)

    dx = data_x - X
    dy = data_y - 42.0

    # Inverse rotation (transpose, since rotation matrices are orthogonal)
    t_recovered = dx * np.cos(theta) + dy * np.sin(theta)
    b_recovered = -dx * np.sin(theta) + dy * np.cos(theta)

    # What b SHOULD be, given the recovered t and the fitted M
    b_expected = np.exp(M * np.abs(t_recovered)) * np.sin(0.3 * t_recovered)

    residual = np.abs(b_recovered - b_expected)
    return t_recovered, b_recovered, b_expected, residual


def main():
    data_x, data_y = load_data()
    t_rec, b_rec, b_exp, residual = analytical_recovery(
        data_x, data_y, THETA_DEG, M, X
    )

    in_range = np.logical_and(t_rec > 6, t_rec < 60)

    print("--- Analytical (closed-form) verification ---")
    print(f"theta = {THETA_DEG} deg, M = {M}, X = {X}")
    print(f"Recovered t range      : [{t_rec.min():.4f}, {t_rec.max():.4f}]")
    print(f"Points with t in (6,60): {in_range.sum()} / {len(t_rec)}")
    print(f"Mean |b_recovered - b_expected| residual: {residual.mean():.8f}")
    print(f"Max  |b_recovered - b_expected| residual: {residual.max():.8f}")

    results = {
        "theta_deg": THETA_DEG,
        "M": M,
        "X": X,
        "t_recovered_min": float(t_rec.min()),
        "t_recovered_max": float(t_rec.max()),
        "points_in_t_range": int(in_range.sum()),
        "total_points": int(len(t_rec)),
        "mean_residual": float(residual.mean()),
        "max_residual": float(residual.max()),
    }
    with open("analytical_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved analytical_verification.json")

    # --- Overlay plot: data points vs reconstructed curve ---
    theta = np.radians(THETA_DEG)
    t_grid = np.linspace(6, 60, 2000)
    base_y = np.exp(M * np.abs(t_grid)) * np.sin(0.3 * t_grid)
    curve_x = t_grid * np.cos(theta) - base_y * np.sin(theta) + X
    curve_y = 42 + t_grid * np.sin(theta) + base_y * np.cos(theta)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_x, data_y, s=6, alpha=0.5, label="Given data points", color="tab:blue")
    plt.plot(curve_x, curve_y, color="tab:red", linewidth=1.5, label="Fitted curve (θ=30°, M=0.03, X=55)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data points vs. reconstructed curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("overlay_plot.png", dpi=150)
    print("Saved overlay_plot.png")


if __name__ == "__main__":
    main()
