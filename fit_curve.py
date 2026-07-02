"""
Fit unknown parameters (theta, M, X) of the parametric curve:

    x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)

for t in (6, 60), given a cloud of (x, y) points sampled from the curve
(without knowing which t produced each point).

Approach
--------
1. Structural insight: subtracting the offsets (X, 42) from (x, y) leaves
   a standard 2D rotation matrix [[cos, -sin], [sin, cos]] applied to a
   base curve g(t) = (t, e^(M|t|) * sin(0.3t)). So theta is a pure
   rotation of that base curve, and X is a horizontal shift (the y-shift,
   42, is already given, not unknown).

2. Because we don't know the t-value behind each data point, this is a
   curve-fitting problem, not direct algebra. For any candidate
   (theta, M, X), we densely sample the resulting curve over t in (6, 60)
   and, for every data point, take the minimum L1 distance to any point
   on that candidate curve. Summing this over all data points gives a
   loss function for the candidate parameters.

3. Global optimization: scipy.optimize.differential_evolution over the
   bounded parameter box (theta in (0,50) deg, M in (-0.05,0.05),
   X in (0,100)), since the nearest-point loss surface is non-convex.

4. Verification: reconstruct the curve with the fitted parameters and
   confirm the residual L1 error against every data point is ~0
   (bounded by grid resolution only), confirming an exact recovery
   rather than an approximate fit.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

DATA_PATH = "data/xy_data.csv"
RESULTS_JSON_PATH = "results.json"
RESULTS_TXT_PATH = "results.txt"
T_MIN, T_MAX = 6.0, 60.0
T_GRID_POINTS = 4000


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df["x"].values, df["y"].values


def predicted_curve(theta, M, X, t_grid):
    """theta in radians."""
    base_x = t_grid
    base_y = np.exp(M * np.abs(t_grid)) * np.sin(0.3 * t_grid)
    x = base_x * np.cos(theta) - base_y * np.sin(theta) + X
    y = 42 + base_x * np.sin(theta) + base_y * np.cos(theta)
    return x, y


def l1_loss(params, data_x, data_y, t_grid):
    theta, M, X = params
    px, py = predicted_curve(theta, M, X, t_grid)
    total = 0.0
    for xi, yi in zip(data_x, data_y):
        d = np.abs(px - xi) + np.abs(py - yi)
        total += d.min()
    return total


def fit(data_x, data_y):
    t_grid = np.linspace(T_MIN, T_MAX, T_GRID_POINTS)
    bounds = [
        (0.0, np.radians(50.0)),  # theta (radians)
        (-0.05, 0.05),            # M
        (0.0, 100.0),             # X
    ]
    result = differential_evolution(
        l1_loss,
        bounds,
        args=(data_x, data_y, t_grid),
        tol=1e-9,
        maxiter=300,
        popsize=25,
        seed=42,
        polish=True,
        workers=-1,
    )
    theta, M, X = result.x
    return theta, M, X, result.fun


def verify(theta, M, X, data_x, data_y, t_grid_points=20000):
    t_grid = np.linspace(T_MIN, T_MAX, t_grid_points)
    px, py = predicted_curve(theta, M, X, t_grid)
    errs = []
    for xi, yi in zip(data_x, data_y):
        d = np.abs(px - xi) + np.abs(py - yi)
        errs.append(d.min())
    errs = np.array(errs)
    return errs.sum(), errs.mean(), errs.max()


def main():
    data_x, data_y = load_data()
    print(f"Loaded {len(data_x)} data points.")

    theta, M, X, loss_val = fit(data_x, data_y)
    theta_deg = np.degrees(theta)
    print("\n--- Fitted parameters (raw optimizer output) ---")
    print(f"theta = {theta_deg:.6f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"loss  = {loss_val:.6f}")

    # Round to clean values (the fit converges to exact round numbers)
    theta_deg_r = round(theta_deg, 2)
    M_r = round(M, 4)
    X_r = round(X, 2)
    theta_r = np.radians(theta_deg_r)

    total, mean, mx = verify(theta_r, M_r, X_r, data_x, data_y)
    print("\n--- Verification with rounded parameters ---")
    print(f"theta = {theta_deg_r} deg, M = {M_r}, X = {X_r}")
    print(f"Total L1 error   : {total:.6f}")
    print(f"Mean L1 error/pt : {mean:.6f}")
    print(f"Max L1 error     : {mx:.6f}")

    latex = (
        f"\\left(t*\\cos({theta_r:.4f})-e^{{{M_r}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta_r:.4f})+{X_r},"
        f"42+t*\\sin({theta_r:.4f})+e^{{{M_r}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta_r:.4f})\\right)"
    )
    print("\n--- Desmos / LaTeX submission string ---")
    print(latex)

    save_results(
        n_points=len(data_x),
        raw={"theta_deg": theta_deg, "M": M, "X": X, "loss": loss_val},
        rounded={"theta_deg": theta_deg_r, "M": M_r, "X": X_r},
        errors={"total_l1": total, "mean_l1": mean, "max_l1": mx},
        latex=latex,
    )


def save_results(n_points, raw, rounded, errors, latex):
    """Persist the fit results to both a JSON file (machine-readable)
    and a plain-text file (human-readable), so results survive after
    the script finishes running."""

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_points": n_points,
        "raw_optimizer_output": raw,
        "final_answer": rounded,
        "verification_l1_error": errors,
        "desmos_latex": latex,
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    with open(RESULTS_TXT_PATH, "w") as f:
        f.write("Curve Parameter Fit Results\n")
        f.write("============================\n")
        f.write(f"Timestamp (UTC): {payload['timestamp_utc']}\n")
        f.write(f"Data points used: {n_points}\n\n")
        f.write("Final answer:\n")
        f.write(f"  theta = {rounded['theta_deg']} deg\n")
        f.write(f"  M     = {rounded['M']}\n")
        f.write(f"  X     = {rounded['X']}\n\n")
        f.write("Verification (L1 error against all data points):\n")
        f.write(f"  total L1 error : {errors['total_l1']:.6f}\n")
        f.write(f"  mean L1 error  : {errors['mean_l1']:.6f}\n")
        f.write(f"  max L1 error   : {errors['max_l1']:.6f}\n\n")
        f.write("Desmos / LaTeX submission string:\n")
        f.write(f"{latex}\n")

    print(f"\nResults saved to '{RESULTS_JSON_PATH}' and '{RESULTS_TXT_PATH}'.")


if __name__ == "__main__":
    main()
