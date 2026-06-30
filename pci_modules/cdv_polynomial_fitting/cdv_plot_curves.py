"""
cdv_plot_curves.py

Generates four complementary plots for visual validation of the CDV
correction curve fits:

  Fig 1 — All q-curves overlaid (mirrors the layout of ASTM Fig. X3.26)
  Fig 2 — Individual subplots per q, showing digitized points vs. fit
  Fig 3 — Residuals per q-curve (how far the polynomial deviates from
           each digitized point)
  Fig 4 — CDV vs. q at fixed TDV values (cross-section across the family)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cdv_digitized_data import CDV_POINTS
from cdv_fit_curves import build_cdv_models, predict_cdv, _clamp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# One distinct colour per q-value
Q_VALUES = sorted(CDV_POINTS.keys())
PALETTE = mpl.colormaps["tab10"].resampled(len(Q_VALUES))
Q_COLORS = {q: PALETTE(i) for i, q in enumerate(Q_VALUES)}

TDV_SMOOTH_MAX = 210  # upper limit for smooth curve evaluation


def _smooth_tdv(q):
    """Dense TDV array spanning the range of digitized points for one curve."""
    tdv_pts = CDV_POINTS[q]["tdv"]
    return np.linspace(min(tdv_pts), max(tdv_pts), 300)


# ---------------------------------------------------------------------------
# Figure 1 — all curves overlaid  (mirrors ASTM Fig. X3.26)
# ---------------------------------------------------------------------------


def plot_all_curves_overlay(models, ax=None, save_path=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))

    for q in Q_VALUES:
        color = Q_COLORS[q]
        points = CDV_POINTS[q]

        # digitized points
        ax.scatter(
            points["tdv"], points["cdv"], color=color, s=30, zorder=4, marker="o"
        )

        # fitted smooth curve
        tdv_smooth = _smooth_tdv(q)
        cdv_smooth = [_clamp(float(models[q](t))) for t in tdv_smooth]
        ax.plot(
            tdv_smooth,
            cdv_smooth,
            color=color,
            linewidth=1.8,
            label=f"q = {q}",
            zorder=3,
        )

    ax.set_xlabel("Total Deduct Value (TDV)", fontsize=11)
    ax.set_ylabel("Corrected Deduct Value (CDV)", fontsize=11)
    ax.set_title(
        "CDV Correction Curves — All q values\n" "(ASTM D6433 Fig. X3.26 — Asphalt)",
        fontsize=12,
    )
    ax.set_xlim(0, TDV_SMOOTH_MAX)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="q = number of\ndeducts > 2",
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        framealpha=0.85,
    )

    # Annotate each curve with its q label at the right end
    for q in Q_VALUES:
        tdv_end = max(CDV_POINTS[q]["tdv"])
        cdv_end = _clamp(float(models[q](tdv_end)))
        ax.annotate(
            f"q={q}",
            xy=(tdv_end, cdv_end),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=Q_COLORS[q],
            va="center",
        )

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        return fig


# ---------------------------------------------------------------------------
# Figure 2 — individual subplots per q (points vs. fit)
# ---------------------------------------------------------------------------


def plot_individual_subplots(models, save_path=None):
    n = len(Q_VALUES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 3.2), squeeze=False
    )
    axes_flat = [ax for row in axes for ax in row]

    for idx, q in enumerate(Q_VALUES):
        ax = axes_flat[idx]
        color = Q_COLORS[q]
        points = CDV_POINTS[q]
        tdv_a = np.asarray(points["tdv"], dtype=float)
        cdv_a = np.asarray(points["cdv"], dtype=float)

        # digitized points
        ax.scatter(
            tdv_a, cdv_a, color=color, s=40, zorder=4, label="Digitized", marker="o"
        )

        # fitted curve
        tdv_smooth = _smooth_tdv(q)
        cdv_smooth = [_clamp(float(models[q](t))) for t in tdv_smooth]
        ax.plot(
            tdv_smooth,
            cdv_smooth,
            color=color,
            linewidth=2,
            label="Fitted poly",
            zorder=3,
        )

        # R²
        predicted = models[q](tdv_a)
        ss_res = np.sum((cdv_a - predicted) ** 2)
        ss_tot = np.sum((cdv_a - np.mean(cdv_a)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        ax.set_title(f"q = {q}   (R² = {r2:.5f})", fontsize=10)

        ax.set_xlabel("TDV", fontsize=9)
        ax.set_ylabel("CDV", fontsize=9)
        ax.set_xlim(0, max(tdv_a) + 10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Individual CDV Curve Fits — Digitized points vs. Polynomial\n",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — residuals per q-curve
# ---------------------------------------------------------------------------


def plot_residuals(models, save_path=None):
    n = len(Q_VALUES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False
    )
    axes_flat = [ax for row in axes for ax in row]

    for idx, q in enumerate(Q_VALUES):
        ax = axes_flat[idx]
        color = Q_COLORS[q]
        points = CDV_POINTS[q]
        tdv_a = np.asarray(points["tdv"], dtype=float)
        cdv_a = np.asarray(points["cdv"], dtype=float)

        residuals = cdv_a - models[q](tdv_a)
        rmse = float(np.sqrt(np.mean(residuals**2)))

        ax.bar(tdv_a, residuals, width=4, color=color, alpha=0.7, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, zorder=4)
        ax.set_title(f"q = {q}   RMSE = {rmse:.3f}", fontsize=10)
        ax.set_xlabel("TDV", fontsize=9)
        ax.set_ylabel("Residual (actual − predicted)", fontsize=8)
        ax.set_xlim(0, max(tdv_a) + 10)
        ax.grid(True, alpha=0.3, axis="y")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "CDV Fit Residuals per q-curve\n"
        "(ideal: bars close to zero with no systematic trend)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — CDV vs. q at fixed TDV values (cross-section view)
# ---------------------------------------------------------------------------


def plot_cross_sections(models, save_path=None):
    fixed_tdv_values = [20, 40, 60, 80, 100, 120, 150, 180]
    q_arr = np.array(Q_VALUES)

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = mpl.colormaps["plasma"].resampled(len(fixed_tdv_values))
    colors = [cmap(i) for i in range(len(fixed_tdv_values))]

    for color, tdv in zip(colors, fixed_tdv_values):
        cdvs = [predict_cdv(models, int(q), float(tdv)) for q in q_arr]
        ax.plot(
            q_arr,
            cdvs,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"TDV = {tdv}",
            markersize=5,
        )

    ax.set_xlabel("q  (number of deduct values > 2)", fontsize=11)
    ax.set_ylabel("Corrected Deduct Value (CDV)", fontsize=11)
    ax.set_title(
        "CDV vs. q at Fixed TDV Values\n"
        "(Decreases monotonically as q increases)",
        fontsize=12,
    )
    ax.set_xticks(Q_VALUES)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Total Deduct Value", fontsize=9, title_fontsize=9, loc="upper right"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Combined 2×2 summary figure (all four views on one canvas)
# ---------------------------------------------------------------------------


def plot_summary_grid(models, save_path="cdv_curves_summary.png"):
    fig = plt.figure(figsize=(18, 13))

    # ---- top-left: overlay ----
    ax1 = fig.add_subplot(2, 2, 1)
    plot_all_curves_overlay(models, ax=ax1)

    # ---- top-right: cross-section ----
    ax2 = fig.add_subplot(2, 2, 2)
    fixed_tdv_values = [20, 40, 60, 80, 100, 120, 150, 180]
    q_arr = np.array(Q_VALUES)
    cmap = mpl.colormaps["plasma"].resampled(len(fixed_tdv_values))
    colors = [cmap(i) for i in range(len(fixed_tdv_values))]
    for color, tdv in zip(colors, fixed_tdv_values):
        cdvs = [predict_cdv(models, int(q), float(tdv)) for q in q_arr]
        ax2.plot(
            q_arr,
            cdvs,
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"TDV={tdv}",
            markersize=4,
        )
    ax2.set_xlabel("q  (number of deduct values > 2)", fontsize=10)
    ax2.set_ylabel("CDV", fontsize=10)
    ax2.set_title("Cross-section: CDV vs. q at fixed TDV", fontsize=11)
    ax2.set_xticks(Q_VALUES)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7.5, loc="upper right", title="TDV", title_fontsize=8)

    # ---- bottom-left: residuals (mini grid within one subplot) ----
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Residuals overview (RMSE per q-curve)", fontsize=11)
    rmse_vals = []
    for q in Q_VALUES:
        points = CDV_POINTS[q]
        tdv_a = np.asarray(points["tdv"], dtype=float)
        cdv_a = np.asarray(points["cdv"], dtype=float)
        residuals = cdv_a - models[q](tdv_a)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        rmse_vals.append(rmse)
    bars = ax3.bar(
        Q_VALUES, rmse_vals, color=[Q_COLORS[q] for q in Q_VALUES], alpha=0.85
    )
    ax3.set_xlabel("q", fontsize=10)
    ax3.set_ylabel("RMSE (CDV units)", fontsize=10)
    ax3.set_xticks(Q_VALUES)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, rmse_vals):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ---- bottom-right: heatmap CDV(TDV, q) ----
    ax4 = fig.add_subplot(2, 2, 4)
    tdv_grid = np.linspace(0, 200, 80)
    q_grid = Q_VALUES
    Z = np.array(
        [[predict_cdv(models, int(q), float(tdv)) for tdv in tdv_grid] for q in q_grid]
    )
    im = ax4.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[0, 200, Q_VALUES[0] - 0.5, Q_VALUES[-1] + 0.5],
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
    )
    plt.colorbar(im, ax=ax4, label="CDV")
    ax4.set_xlabel("Total Deduct Value (TDV)", fontsize=10)
    ax4.set_ylabel("q", fontsize=10)
    ax4.set_yticks(Q_VALUES)
    ax4.set_title("CDV Heatmap over TDV × q space", fontsize=11)

    fig.suptitle(
        "CDV Correction Curve Fits — Complete Validation Summary\n"
        "(ASTM D6433 Fig. X3.26, Asphalt)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary grid: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building CDV models...")
    models = build_cdv_models(verbose=False)

    plot_all_curves_overlay(models, save_path="cdv_overlay.png")
    plot_individual_subplots(models, save_path="cdv_individual.png")
    plot_residuals(models, save_path="cdv_residuals.png")
    plot_cross_sections(models, save_path="cdv_cross_sections.png")
    plot_summary_grid(models, save_path="cdv_summary.png")

    print("\nAll plots saved.")
