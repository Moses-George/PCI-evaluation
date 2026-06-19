"""
plot_curves.py

Plots the digitized points alongside the fitted polynomial curves for
each distress type, so you can visually sanity-check the fit against
the original ASTM chart shape (re-digitize and re-run this whenever you
replace the placeholder points in digitized_data.py with real extracted
values).
"""

import numpy as np
import matplotlib.pyplot as plt
from digitized_data import ALL_CURVES
from fit_curves import build_all_models

SEVERITY_COLORS = {"low": "tab:green", "medium": "tab:orange", "high": "tab:red"}
TITLES = {
    "alligator": "Alligator Cracking (Fig. X3.1)",
    "linear": "Longitudinal & Transverse Cracking (Fig. X3.14)",
    "pothole": "Potholes (Fig. X3.18)",
}


def plot_all(degree=3, save_path="deduct_value_curves.png"):
    models = build_all_models(degree=degree, verbose=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))

    for ax, (distress_type, severities) in zip(axes, ALL_CURVES.items()):
        for sev, points in severities.items():
            density = np.asarray(points["density"], dtype=float)
            dv = np.asarray(points["dv"], dtype=float)
            color = SEVERITY_COLORS[sev]

            # digitized points
            ax.scatter(
                density,
                dv,
                color=color,
                marker="o",
                s=35,
                label=f"{sev} (digitized)",
                zorder=3,
            )

            # fitted curve, densely sampled across the log range
            d_min, d_max = density.min(), density.max()
            smooth_density = np.logspace(np.log10(d_min), np.log10(d_max), 200)
            model = models[distress_type][sev]
            smooth_dv = [model(np.log10(d)) for d in smooth_density]
            ax.plot(
                smooth_density,
                smooth_dv,
                color=color,
                linewidth=1.8,
                label=f"{sev} (fitted)",
                zorder=2,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Density (%)")
        ax.set_ylabel("Deduct Value")
        ax.set_title(TITLES[distress_type])
        ax.set_ylim(0, 100)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Digitized points vs. fitted polynomial models ",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    plot_all()
