"""
Fits a polynomial regression model to each digitized (density, deduct value)
curve, for each distress type and severity level.

Because the original ASTM charts use a logarithmic x-axis (density), we fit
the polynomial against log10(density) rather than raw density. This matches
the visual shape of the source curves much more closely than a raw-density
polynomial would, and avoids wild extrapolation behavior near density -> 0.

Usage:
    from fit_curves import build_all_models, predict_deduct_value
    models = build_all_models(degree=3)
    dv = predict_deduct_value(models, "alligator", "medium", density=12.5)
"""

import sys
import os

parent_folder = os.path.abspath((os.path.join(os.path.dirname(__file__), "./")))
sys.path.append(parent_folder)

import numpy as np
from digitized_data import ALL_CURVES


def fit_single_curve(density, dv, degree=3):
    """
    Fit a polynomial of given degree to one (density, deduct_value) curve.
    Returns a numpy.poly1d object that takes log10(density) as input.
    """
    density = np.asarray(density, dtype=float)
    dv = np.asarray(dv, dtype=float)

    log_density = np.log10(density)
    coeffs = np.polyfit(log_density, dv, degree)
    model = np.poly1d(coeffs)
    return model


def fit_quality(model, density, dv):
    """
    Returns R^2 for a fitted model against its source points.
    Useful for sanity-checking whether the chosen polynomial degree
    is adequate, or overfitting / underfitting the digitized points.
    """
    density = np.asarray(density, dtype=float)
    dv = np.asarray(dv, dtype=float)
    log_density = np.log10(density)

    predicted = model(log_density)
    ss_res = np.sum((dv - predicted) ** 2)
    ss_tot = np.sum((dv - np.mean(dv)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return r2


def build_all_models(degree=3, verbose=True):
    """
    Builds polynomial models for every distress type and severity level
    defined in digitized_data.ALL_CURVES.

    Returns a nested dict:
        models[distress_type][severity] = poly1d model
    """
    models = {}
    for distress_type, severities in ALL_CURVES.items():
        models[distress_type] = {}
        for sev, points in severities.items():
            model = fit_single_curve(points["density"], points["dv"], degree=degree)
            models[distress_type][sev] = model

            if verbose:
                r2 = fit_quality(model, points["density"], points["dv"])
                print(
                    f"[{distress_type:10s}] severity={sev}  "
                    f"degree={degree}  R^2={r2:.4f}"
                )
    return models


def clamp(value, low=0.0, high=100.0):
    """Deduct values are bounded 0-100 in the original charts; the fitted
    polynomial can stray outside that range at extreme densities, so we
    clamp predictions to stay physically meaningful."""
    return max(low, min(high, value))


def predict_deduct_value(models, distress_type, severity, density):
    """
    Predict a deduct value for a given distress type, severity level
    ("low", "medium", or "high"), and density percentage.

    distress_type: one of "alligator", "linear", "pothole"
    severity: "low", "medium", "high"
    density: density percentage (must be > 0; the curves are undefined at 0
             because the x-axis is logarithmic)
    """
    if distress_type not in models:
        raise ValueError(
            f"Unknown distress_type '{distress_type}'. "
            f"Choose from {list(models.keys())}"
        )
    if severity not in models[distress_type]:
        raise ValueError(f"Unknown severity '{severity}'. Choose from L, M, H")
    if density <= 0:
        raise ValueError("density must be > 0 (log scale x-axis)")

    model = models[distress_type][severity]
    log_density = np.log10(density)
    raw_value = model(log_density)
    return clamp(raw_value)


if __name__ == "__main__":
    # Quick demo / sanity check when run directly
    models = build_all_models(degree=3, verbose=True)

    print("\nSample predictions:")
    test_cases = [
        ("alligator", "low", 0.52),
        ("alligator", "medium", 25),
        ("alligator", "high", 0.56),
        ("linear", "low", 2),
        ("linear", "medium", 20),
        ("linear", "high", 50),
        ("pothole", "low", 0.04),
        ("pothole", "medium", 0.5),
        ("pothole", "high", 5),
    ]
    for distress_type, sev, density in test_cases:
        dv = predict_deduct_value(models, distress_type, sev, density)
        print(
            f"  {distress_type:10s} severity={sev}  density={density:>6}%  "
            f"-> deduct value = {dv:.2f}"
        )
