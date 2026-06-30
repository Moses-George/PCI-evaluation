"""
save_models.py

Run this script ONCE to fit the polynomial models and save their
coefficients to a JSON file (models.json).

You only need to re-run this if:
  - You update the digitized data points in digitized_data.py
  - You want to change the polynomial degree

After running this once, production code only needs to load models.json.
"""

import json
import numpy as np
from fit_curves import build_all_models

def export_models(degree=3, output_path="models.json"):
    models = build_all_models(degree=degree, verbose=True)

    # np.poly1d stores coefficients as a numpy array;
    # we convert to plain Python list for JSON serialization
    serializable = {}
    for distress_type, severities in models.items():
        serializable[distress_type] = {}
        for sev, poly in severities.items():
            serializable[distress_type][sev] = {
                "coefficients": poly.coeffs.tolist(),
                "degree": degree,
            }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nCoefficients saved to {output_path}")
    print("You can now use DeductValueCalculator in production without refitting.")


if __name__ == "__main__":
    export_models(degree=3)


# [alligator ] severity=low  degree=3  R^2=0.9995
# [alligator ] severity=medium  degree=3  R^2=1.0000
# [alligator ] severity=high  degree=3  R^2=0.9994
# [linear    ] severity=low  degree=3  R^2=0.9998
# [linear    ] severity=medium  degree=3  R^2=0.9996
# [linear    ] severity=high  degree=3  R^2=0.9981
# [pothole   ] severity=low  degree=3  R^2=1.0000
# [pothole   ] severity=medium  degree=3  R^2=0.9999
# [pothole   ] severity=high  degree=3  R^2=0.9999