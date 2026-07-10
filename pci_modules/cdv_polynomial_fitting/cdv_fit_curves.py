"""
cdv_fit_curves.py

Fits polynomial regression models to the Corrected Deduct Value (CDV)
correction curves (ASTM D6433 Fig. X3.26 for asphalt).

Key differences from the distress deduct-value curves:
  - X-axis (TDV) is LINEAR, not logarithmic → we fit against raw TDV
  - Each curve is indexed by q (integer, 1–8), not distress type + severity
  - q=1 is an exact straight line (CDV == TDV), so a degree-1 fit suffices
    for that curve; higher q curves need degree 3 or 4

Usage
-----
    from cdv_fit_curves import build_cdv_models, predict_cdv, fit_report
    models = build_cdv_models()
    cdv    = predict_cdv(models, q=3, tdv=85.0)
"""

import numpy as np
from cdv_digitized_data import CDV_POINTS

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def fit_single_cdv_curve(tdv_pts, cdv_pts, degree: int = 3):
    """
    Fit a polynomial to one (TDV, CDV) curve.

    Returns
    -------
    np.poly1d
        Polynomial that takes raw TDV (linear scale) as input.
    """
    tdv = np.asarray(tdv_pts, dtype=float)
    cdv = np.asarray(cdv_pts, dtype=float)
    coeffs = np.polyfit(tdv, cdv, degree)
    return np.poly1d(coeffs)


def r_squared(model, tdv_pts, cdv_pts) -> float:
    tdv = np.asarray(tdv_pts, dtype=float)
    cdv = np.asarray(cdv_pts, dtype=float)
    predicted = model(tdv)
    ss_res = np.sum((cdv - predicted) ** 2)
    ss_tot = np.sum((cdv - np.mean(cdv)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def build_cdv_models(degree: int = 3, verbose: bool = True) -> dict:
    """
    Build and return polynomial models for all q-curves.

    Parameters
    ----------
    degree : int
        Polynomial degree to use for q >= 2.
        q=1 always uses degree=1 (it is a straight line CDV == TDV).
    verbose : bool
        Print R² and RMSE for each curve.

    Returns
    -------
    dict  {q_int: np.poly1d}
    """
    models = {}
    for q, points in sorted(CDV_POINTS.items()):
        tdv_pts = points["tdv"]
        cdv_pts = points["cdv"]

        # q=1 is a perfect 45-degree line; force degree=1 so the fit is exact
        d = 1 if q == 1 else degree
        model = fit_single_cdv_curve(tdv_pts, cdv_pts, degree=d)
        models[q] = model

        if verbose:
            r2 = r_squared(model, tdv_pts, cdv_pts)
            preds = model(np.asarray(tdv_pts, dtype=float))
            rmse = float(np.sqrt(np.mean((np.asarray(cdv_pts) - preds) ** 2)))
            print(f"  q={q}  degree={d}  R²={r2:.5f}  RMSE={rmse:.3f}")

    return models


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_cdv(models: dict, q: int, tdv: float) -> float:
    """
    Return the corrected deduct value for a given q and total deduct value.

    Parameters
    ----------
    models : dict returned by build_cdv_models()
    q      : number of deduct values > 2 (integer, clamped to 1–max_q)
    tdv    : total deduct value (sum of individual deduct values), >= 0

    Returns
    -------
    float  CDV in [0, 100]
    """
    if not models:
        raise RuntimeError("models dict is empty — call build_cdv_models() first.")
    if tdv < 0:
        raise ValueError(f"tdv must be >= 0, got {tdv}")

    max_q = max(models.keys())
    q_clamped = max(1, min(q, max_q))

    poly = models[q_clamped]
    raw = float(poly(tdv))
    return _clamp(raw)


# ---------------------------------------------------------------------------
# Fit report
# ---------------------------------------------------------------------------


def fit_report(models: dict) -> None:
    """
    Print a dense sample table to spot-check every fitted model
    against a range of TDV values.
    """
    tdv_samples = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]

    header = "TDV   |" + "".join(f"  q={q:<3}" for q in sorted(models))
    print(header)
    print("-" * len(header))
    for tdv in tdv_samples:
        row = f"{tdv:<6}|"
        for q in sorted(models):
            cdv = predict_cdv(models, q, float(tdv))
            row += f"  {cdv:>5.1f}"
        print(row)


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------


def check_monotonicity(models: dict) -> None:
    """
    Sanity check: for a fixed TDV, CDV should decrease as q increases
    (more deducts → lower correction needed per deduct).
    Also CDV should increase as TDV increases for a fixed q.
    Prints warnings for any violations.
    """
    tdv_range = np.linspace(5, 200, 50)
    issues = []

    # Check CDV increases with TDV for each q
    for q, model in models.items():
        cdv_vals = [_clamp(float(model(t))) for t in tdv_range]
        for i in range(1, len(cdv_vals)):
            if cdv_vals[i] < cdv_vals[i - 1] - 0.5:  # allow tiny float noise
                issues.append(
                    f"  q={q}: CDV not monotone increasing at TDV≈{tdv_range[i]:.1f} "
                    f"(CDV dropped from {cdv_vals[i-1]:.2f} to {cdv_vals[i]:.2f})"
                )

    # Check CDV decreases as q increases for each TDV
    q_list = sorted(models.keys())
    for tdv in [20, 50, 100, 150]:
        cdvs = [_clamp(float(models[q](tdv))) for q in q_list]
        for i in range(1, len(cdvs)):
            if cdvs[i] > cdvs[i - 1] + 0.5:
                issues.append(
                    f"  TDV={tdv}: CDV not monotone decreasing as q increases "
                    f"(q={q_list[i-1]}→{q_list[i]}: {cdvs[i-1]:.2f}→{cdvs[i]:.2f})"
                )

    if issues:
        print(f"⚠  Monotonicity warnings ({len(issues)} found):")
        for msg in issues:
            print(msg)
        print(
            "  Consider re-digitizing affected curves or adjusting polynomial degree."
        )
    else:
        print("✓  All monotonicity checks passed.")


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Building CDV models ===")
    models = build_cdv_models(degree=3, verbose=True)

    print("\n=== Fit report (sample CDV values) ===")
    fit_report(models)

    print("\n=== Monotonicity checks ===")
    check_monotonicity(models)

    print("\n=== Sample predictions ===")
    for q in [7, 6, 5, 4, 1,2]:
        for tdv in [100.5, 96.7, 91, 85, 44.3, 20]:
            cdv = predict_cdv(models, q, float(tdv))
            print(f"  q={q}  TDV={tdv:>4}  ->  CDV={cdv:.2f}")
