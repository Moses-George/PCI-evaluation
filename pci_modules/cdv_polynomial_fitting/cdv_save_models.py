"""
cdv_save_models.py

Run this ONCE to fit CDV polynomial models and save their coefficients
to cdv_models.json. Production code loads that file — no refitting needed.

Re-run only when:
  - You update cdv_digitized_data.py with better digitized points
  - You want to change the polynomial degree
"""

import json
import numpy as np
from cdv_fit_curves import build_cdv_models, predict_cdv, check_monotonicity


def export_cdv_models(degree: int = 3, output_path: str = "cdv_models.json"):
    print(f"Fitting CDV models (degree={degree})...")
    models = build_cdv_models(degree=degree, verbose=True)

    print("\nRunning monotonicity checks before saving...")
    check_monotonicity(models)

    # Serialize: poly1d.coeffs is a numpy array, convert to plain list
    serializable = {}
    for q, poly in models.items():
        serializable[str(q)] = {
            "q":           q,
            "degree":      int(len(poly.coeffs) - 1),
            "coefficients": poly.coeffs.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved {len(models)} CDV curves to '{output_path}'")
    print("Keys in file:", list(serializable.keys()))

    # Quick round-trip verification
    print("\nRound-trip verification (load back and spot-check):")
    with open(output_path) as f:
        loaded = json.load(f)
    for q_str, data in loaded.items():
        poly = np.poly1d(data["coefficients"])
        cdv_50  = float(np.clip(poly(50),  0, 100))
        cdv_100 = float(np.clip(poly(100), 0, 100))
        print(f"  q={q_str}  CDV(TDV=50)={cdv_50:.2f}  CDV(TDV=100)={cdv_100:.2f}")


if __name__ == "__main__":
    export_cdv_models(degree=3, output_path="cdv_models.json")
