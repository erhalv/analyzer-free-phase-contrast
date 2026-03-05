import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import matplotlib

full_palette = sns.color_palette("deep", n_colors=6)
palette = [full_palette[i] for i in [0, 1, 2]]
markers = ["o", "s", "^"]
linestyles = ["-", "--", "-."]

# Global font size for the entire figure (ticks, labels, legend, titles, etc.)
FONT_SIZE = 10

# Keep seaborn context but rely on rcParams for exact sizes
sns.set_context("paper", font_scale=1)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": FONT_SIZE,
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE
})

BASE_NAMES = [
    "sdnr_results_60keV_all_sizes_scaled_12cm_2D_pixel_large_samples_217_mean_only_abs.csv",
    "sdnr_results_60keV_all_sizes_scaled_12cm_2D_pixel_large_samples_217_phi.csv",
    "sdnr_results_60keV_all_sizes_scaled_12cm_2D_pixel_large_samples_217_total_phase_only_phase_v2.csv",
]

DISPLAY_NAMES = [
    "transmission",
    "differential phase",
    "integrated phase"
]

LESION_SIZES = ["SDNR_50um", "SDNR_100um", "SDNR_200um"]
Y_LIM = (0, 16)
X_TICKS = [1e4, 1e6, 1e8, 1e10]

def load_data(filename):
    filepath = os.path.join(os.getcwd(), filename)
    print(f"Loading: {filename}")
    return pd.read_csv(filepath)

data = {name: load_data(fname) for name, fname in zip(DISPLAY_NAMES, BASE_NAMES)}

fig, axes = plt.subplots(1, 3, figsize=(6, 2.2), sharey=True)
# Make epsilon plots square

for ax, lesion in zip(axes, LESION_SIZES):
    for (display_name, df), color, linestyle, marker in zip(
        data.items(), palette, linestyles, markers
    ):
        ax.plot(
            df["photons"],
            df[lesion],
            label=display_name,
            color=color,
            linestyle=linestyle,
            linewidth=1.5
        )

    #lesion_label = lesion.replace("SDNR_", "").replace("um", "\mu m")
    # ax.set_title(rf"{lesion_label}")
    ax.set_xlabel("Photons per pixel")
    ax.set_xscale("log")
    ax.set_xticks(X_TICKS)
    #ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)
    ax.axhline(5, color="red", linestyle="-", linewidth=1, alpha=0.8)
    ax.grid(True, which="both", linestyle="-", alpha=0.4)
    ax.set_box_aspect(1)
    ax.tick_params(labelleft=True)
    lesion_label = lesion.replace("SDNR_", "").replace("um", r"\,\mu m")
    ax.text(
        0.05,
        0.9,
        rf"${lesion_label}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_SIZE,
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )

axes[0].set_ylabel("SDNR")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=FONT_SIZE,
    bbox_to_anchor=(0.5, -0.15)
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.savefig("photons_vs_sdnr_three_signals_by_lesion.pdf", dpi=600, bbox_inches="tight")
