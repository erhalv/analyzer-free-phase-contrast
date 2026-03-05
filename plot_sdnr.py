import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from parameter import *
import time
import seaborn as sns
from scipy.interpolate import interp1d
from matplotlib.ticker import LogFormatter
import matplotlib

'Just a plotting function (not very important)'

palette = sns.color_palette("deep", n_colors=4)
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
linestyles = ["-", "--", "-.", ":"]
#sns.set_style("whitegrid")

# Global font size for the entire figure (ticks, labels, legend, titles, etc.)
FONT_SIZE = 10

# Keep seaborn context but rely on rcParams for exact sizes
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

def plot_sdnr_vs_photons(photons_list, sdnr_list1,file_name=None):
    """
    photons_list: sequence of photon counts
    sdnr_list: sequence of SDNR values (same length)
    sdnr2_list: optional sequence of SDNR^2 values
    file_name: optional path to save figure; if None, show interactively
    """
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    ax.semilogx(photons_list, sdnr_list1, marker='o', linestyle='-', color = 'blue' ,label='80 μm tumor cell')
    #ax.semilogx(photons_list, sdnr_list2, marker='o', linestyle='-', color = 'orange', label='40 μm tumor cell')
    #ax.semilogx(photons_list, sdnr_list3, marker='o', linestyle='-', color = 'purple', label='80 μm tumor cell')
    #ax.semilogx(photons_list, sdnr_list4, marker='*', linestyle='-', color = 'blue', label='20 μm tumor cell')
    #ax.semilogx(photons_list, sdnr_list5, marker='*', linestyle='-', color = 'orange', label='40 μm tumor cell')
    #ax.semilogx(photons_list, sdnr_list6, marker='*', linestyle='-', color = 'purple', label='80 μm tumor cell')    
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1, label='SDNR = 5')
    ax.axhline(y=3.0, color='green', linestyle='--', linewidth=1, label='SDNR = 3')
    
    ax.set_xlabel('Photons per pixel')
    ax.set_ylabel('SDNR')
    ax.set_title('SDNR vs Photon Count at 38keV comparing lesion sizes')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_ylim(0,25)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1)
    plt.tight_layout()

    plt.savefig(file_name, dpi=300, bbox_inches='tight')

def find_photon_for_threshold(photons, sdnr, threshold, interp=True):
    """
    Return photon count where sdnr first reaches `threshold`.
    photons: 1D sequence of photon counts
    sdnr: 1D sequence of SDNR values (same length)
    threshold: scalar threshold (e.g. 3 or 5)
    interp: if True, interpolate on log10(photon) between bracketing points
    Returns float photon count or None if threshold not reached.
    """
    p = np.asarray(photons, dtype=float)
    s = np.asarray(sdnr, dtype=float)

    # sort by photon (ascending)
    order = np.argsort(p)
    p = p[order]
    s = s[order]

    # find first index where s >= threshold
    idx_arr = np.where(s >= threshold)[0]
    if idx_arr.size == 0:
        return None
    idx = idx_arr[0]

    # if threshold met at first point, return that photon
    if idx == 0 or not interp:
        return float(p[0])

    # interpolate in log-photon space for semilog plots
    x0, x1 = np.log10(p[idx-1]), np.log10(p[idx])
    y0, y1 = s[idx-1], s[idx]
    if y1 == y0:
        return float(p[idx])  # avoid divide by zero
    frac = (threshold - y0) / (y1 - y0)
    logp = x0 + frac * (x1 - x0)
    return float(10**logp)

def find_crossings_in_file(file_path, threshold=5.0, save_csv=None):
    """
    Read an SDNR results CSV and find photon counts where each SDNR column
    reaches `threshold` (interpolated in log-photon space).

    file_path: path to CSV (must contain a `photons` column)
    threshold: SDNR threshold to find (default 5.0)
    save_csv: optional path to write a summary CSV with columns
              `column`, `label`, `photon_for_threshold`.

    Returns: (summary_df, results_dict)
      - summary_df: pandas.DataFrame with columns `column`, `label`, `photon_for_threshold`
      - results_dict: mapping column -> photon (float) or None
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if 'photons' not in df.columns:
        raise ValueError('CSV must contain a "photons" column')

    photons = df['photons'].to_numpy()
    rows = []
    results = {}

    for col in df.columns:
        if col == 'photons':
            continue
        sdnr = df[col].to_numpy()
        photon_val = find_photon_for_threshold(photons, sdnr, threshold, interp=True)
        # derive a human-friendly label if column looks like SDNR_<size>
        if col.startswith('SDNR_'):
            label = col[len('SDNR_'):]
        else:
            label = col
        rows.append({'column': col, 'structure_size_um': label, 'photon_for_threshold': photon_val})
        results[col] = photon_val

    summary_df = pd.DataFrame(rows)
    if save_csv:
        summary_df.to_csv(save_csv, index=False)

    return summary_df, results
#find_crossings_in_file('sdnr_results_60keV_all_sizes_scaled_12cm_2D_pixel_large_samples_217_total_phase_only_phase_v2.csv', threshold=5.0, save_csv='sdnr_60keV_fullCT_threshold_crossings_5_mean_phi_total_v2_217.csv')

def plot_size_vs_photons(file_path1, file_path2, file_path3, file_path4, save_fig=None):
    """
    Read a threshold crossings CSV and plot structure size vs photons required.
    Includes a secondary y-axis showing exposure time in seconds.
    
    file_path: path to CSV with columns: structure_size_um, photon_for_threshold, exposure_time_sec
    save_fig: optional path to save the figure
    """

    
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    df3 = pd.read_csv(file_path3)
    df4 = pd.read_csv(file_path4)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df1['structure_size_um'], df1['photon_for_threshold'], 
            marker=markers[0], linestyle=linestyles[0], linewidth=2, markersize=8, color=palette[0], label='12 cm sample')
    ax.plot(df2['structure_size_um'], df2['photon_for_threshold'], 
            marker=markers[1], linestyle=linestyles[1], linewidth=2, markersize=8, color=palette[1], label='5 cm sample')
    ax.plot(df3['structure_size_um'], df3['photon_for_threshold'], 
            marker=markers[2], linestyle=linestyles[2], linewidth=2, markersize=8, color=palette[2], label='5 mm sample')
    ax.plot(df4['structure_size_um'], df4['photon_for_threshold'], 
            marker=markers[3], linestyle=linestyles[3], linewidth=2, markersize=8, color=palette[3], label='0.5 mm sample')
    ax.set_yscale('log')
    ax.set_xlabel('Structure Size (μm)')

    ax.set_ylabel('Photons per Pixel for SDNR = 5')
    ax.set_title('Required Photons vs Structure Size CT style at 60 keV')
    #ax.set_ylim(df['photon_for_threshold'].min(), df['photon_for_threshold'].max())
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 110)
    # Create secondary y-axis on the right for exposure time
    #ax2 = ax.twinx()
    #ax2.set_ylabel('Exposure Time (sec)')
    #ax2.set_ylim(df['exposure_time_sec'].min(), df['exposure_time_sec'].max())
    plt.tight_layout()
    ax.legend()
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()
    
    return fig, ax

def plot_size_vs_photons_single1(file_path1, file_path2 ,file_path3,save_fig=None):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    df3 = pd.read_csv(file_path3)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(
        df1['structure_size_um'],
        df1['photon_for_threshold'],
        linestyle=linestyles[0],
        linewidth=1.5,
        color=palette[0],
        markersize=5,
        marker=markers[0],
        label = "differential phase"
    )
    ax.plot(
        df2['structure_size_um'],
        df2['photon_for_threshold'],
        linestyle=linestyles[1],
        linewidth=1.5,
        color=palette[1],
        markersize=5,
        marker=markers[1],
        label = "integrated phase"
    )

    ax.plot(
        df3['structure_size_um'],
        df3['photon_for_threshold'],
        linestyle=linestyles[2],
        linewidth=1.5,
        color=palette[2],
        markersize=5,
        marker=markers[2],
        label = "attenuation"
    )
    ax.set_yscale('log')
    ax.set_xlim(40, 200)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1)

    ax.set_xlabel(r'Structure Size ($\mu$m)')
    ax.set_ylabel('Photons per Pixel for SDNR = 5', size=FONT_SIZE)
    #ax.set_title('Required Photons vs Structure Size (fullCT style, 60 keV, 12cm sample)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    ax2 = ax.twinx()
    ax2.plot(df1['structure_size_um'], df1['photons_mAs'], color=palette[0], linestyle=linestyles[0], linewidth=2.5)
    ax2.plot(df2['structure_size_um'], df2['photons_mAs'], color=palette[1], linestyle=linestyles[1], linewidth=2.5)
    ax2.plot(df3['structure_size_um'], df3['photons_mAs'], color=palette[2], linestyle=linestyles[2], linewidth=2.5)
    ax2.set_ylabel("mAs")
    ax2.set_yscale('log')
    
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_fig}")
    else:
        plt.show()

    return fig, ax

def photons_to_mAs(pixel_size_um_x, pixel_size_um_y, flux_photons_per_sec_mm2_mA, file_path):
    
    df = pd.read_csv(file_path)
    pixel_area_um2 = pixel_size_um_x * pixel_size_um_y
    pixel_area_mm2 = pixel_area_um2 * 1e-6  # 1 um^2 = 1e-6 mm^2

    # Photons per pixel per second
    photons_per_pixel_per_sec_mA = flux_photons_per_sec_mm2_mA * pixel_area_mm2

    # compute mAs
    df['photons_mAs'] = (
        df['photon_for_threshold'] / photons_per_pixel_per_sec_mA
    )

    # overwrite the same file (or change path if you want a new file)
    df.to_csv(file_path, index=False)

#photons_to_mAs(pixel_size_um_x=1, pixel_size_um_y=217, flux_photons_per_sec_mm2_mA=2e6, file_path='sdnr_60keV_fullCT_threshold_crossings_5_mean_phi_total_v2_217.csv')

plot_size_vs_photons_single1('sdnr_60keV_fullCT_threshold_crossings_5_phi_217.csv',
                             'sdnr_60keV_fullCT_threshold_crossings_5_mean_phi_total_v2_217.csv', 
                             'sdnr_60keV_fullCT_threshold_crossings_5_mean_abs_only_v2_217.csv',
                             save_fig='size_vs_photons_sdnr5_60keV_2D_pixel_217.pdf')

"""
fig, ax = plot_size_vs_photons('sdnr_60keV_fullCT_threshold_crossings_5_12cm.csv', 
                                'sdnr_60keV_fullCT_threshold_crossings_5_5cm.csv',
                               'sdnr_60keV_fullCT_threshold_crossings_5_5mm.csv', 
                               'sdnr_60keV_fullCT_threshold_crossings_5_500um.csv',
                               save_fig='size_vs_photons_sdnr5_comparison_60keV.pdf')
"""
