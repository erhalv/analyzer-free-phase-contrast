import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parameter import *
from plotting import *
import seaborn as sns
import matplotlib
from matplotlib.ticker import FormatStrFormatter

d_sph = 100
data  = "intensity_tumor_background_60keV_all_sizes_talbot3_large_sizes_withabsorb.csv"

col_name = f"{int(round(d_sph))}um"

I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()

def estimate_phi_fourier(Iref, Isamp):
        omega = 2 * np.pi / (px_in_um*1e-6 / detector_pixel_size)
        x_walk = np.arange(len(Iref))
        
        centers = []
        a0_list = []
        a1_list = []
        phi_list = []

        for start in range(0, len(x_walk) - segment_size_in_pix + 1, segment_size_in_pix):
            print(f"Processing segment starting at pixel {start}")
            print(f"x range: {x_walk[start]} to {x_walk[start+segment_size_in_pix-1]}")
            x_seg = x_walk[start:start+segment_size_in_pix]
            Iref_seg = Iref[start:start+segment_size_in_pix]
            Isamp_seg = Isamp[start:start+segment_size_in_pix]
            fourier_Isamp = np.fft.fft(Isamp_seg)
            fourier_Iref = np.fft.fft(Iref_seg)
            N = len(Isamp_seg)
            k = int(N / (px_in_um*1e-6 / detector_pixel_size))

            mean_samp = fourier_Isamp[0].real / N
            amp_samp = 2 * np.abs(fourier_Isamp[k]) / N
            phase_samp = np.angle(fourier_Isamp[k])

            mean_ref = fourier_Iref[0].real / N
            amp_ref = 2 * np.abs(fourier_Iref[k]) / N
            phase_ref = np.angle(fourier_Iref[k])
            centers.append(np.mean(x_seg))
            a0_list.append(mean_samp/mean_ref)
            a1_list.append(amp_samp*mean_ref/(mean_samp*amp_ref))
            phi_list.append(phase_samp - phase_ref)

        return a0_list, a1_list, phi_list

def estimate_phi_mean_single(Iref, Isamp, file_name="cosine_fit_segments_new_2.csv"):
        omega = 2 * np.pi / (px_in_um*1e-6 / detector_pixel_size)
        x_walk = np.arange(len(Iref))
        
        centers = []
        a0_list = []
        a1_list = []
        phi_list = []

        for start in range(0, len(x_walk) - segment_size_in_pix + 1, segment_size_in_pix):
            print(f"Processing segment starting at pixel {start}")
            print(f"x range: {x_walk[start]} to {x_walk[start+segment_size_in_pix-1]}")
            x_seg = x_walk[start:start+segment_size_in_pix]
            Iref_seg = Iref[start:start+segment_size_in_pix]
            Isamp_seg = Isamp[start:start+segment_size_in_pix]

            # Linearized cosine model
            X = np.column_stack([
                np.ones_like(x_seg),
                np.cos(omega * x_seg),
                np.sin(omega * x_seg)
            ])

            beta_ref, _, _, _ = np.linalg.lstsq(X, Iref_seg, rcond=None)
            beta_samp, _, _, _ = np.linalg.lstsq(X, Isamp_seg, rcond=None)

            a0_ref = beta_ref[0]
            a0_samp = beta_samp[0]

            A_ref = beta_ref[1]
            A_samp = beta_samp[1]

            B_ref = beta_ref[2]
            B_samp = beta_samp[2]

            a1_ref = np.sqrt(A_ref**2 + B_ref**2)
            a1_samp = np.sqrt(A_samp**2 + B_samp**2)

            phi_ref = np.arctan2(-B_ref, A_ref) 
            phi_samp = np.arctan2(-B_samp, A_samp)

            centers.append(np.mean(x_seg))
            a0_list.append(a0_samp/a0_ref)
            a1_list.append(a1_samp*a0_ref/(a0_samp*a1_ref))
            phi_list.append(phi_samp - phi_ref)
        
        results = pd.DataFrame({
            "x_center": centers,
            "mean": a0_list,
            "visibility": a1_list,
            "phase_shift": phi_list
        })
        results.to_csv(file_name, index=False)
        print(f"Saved results to '{file_name}'.")

#estimate_phi_mean_single(I_ref, I_samp)
#plot_cosine_fit_1d_images('cosine_fit_segments_new_2.csv', 'cosine_fit_1d_images_new_2.pdf')

#plt.plot(I_ref)
plt.plot(I_no_tumor)
plt.plot(I_tumor)

plt.savefig("test_intensities")

def plot_phase_shift_image(img, file_name):
    n_segments = len(img)
    img = np.asarray(img)
    total_length_um = n_segments * segment_size_in_um
    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    xticks_um = np.arange(0, total_length_um + 1, 100)
    xtick_positions = (xticks_um - 0.5 * segment_size_in_um) / segment_size_in_um
    lim = (-0.03, 0.03)

    im = ax.imshow(
        img[np.newaxis, :],
        aspect='auto',
        cmap='gray',
        origin='lower',
        extent=[-0.5, n_segments - 0.5, 0, 1],
        #vmin=0.975, vmax=1
    )

    #ax.imshow(img, aspect='auto', cmap='gray', origin='lower')
    #ax.set_title(title, fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks_um])
    ax.set_xlabel("detector position (\mu m)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("differential phase")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


def plot_phase_shift_image_with_line(img, file_name):
    full_palette = sns.color_palette("deep", n_colors=6)
    palette = [full_palette[i] for i in [0, 1, 2, 5, 4, 3]]  # Skip indices 2 (light green) and 5 (brown)
    markers = ["x", ".", "^", "D" ]  # circle, square, triangle, diamond
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # solid, dashed, dash-dot, dotted, custom1, custom2

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
    img = np.asarray(img)
    n_segments = img.size
    total_length_um = n_segments * segment_size_in_um

    xticks_um = np.arange(0, total_length_um + 1, 100)
    xtick_positions = (xticks_um - 0.5 * segment_size_in_um) / segment_size_in_um

    #vmin, vmax = -0.04, 0.04
    #vmin, vmax = 0.99975, 1

    # two axes: image + line
    fig, ( ax_line,ax_img) = plt.subplots(
        1, 2, figsize=(6, 2.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.1},
        sharey=False
    )
    #fig.text(0.03, 0.9, "(c)", ha="left", va="top", fontweight="bold")

    # --- right: line plot of the same 1D data ---
    x_um = (np.arange(n_segments) + 0.5) * segment_size_in_um
    ax_line.plot(x_um, img, color=palette[0], linewidth=1.5)
    ax_line.set_xlabel("detector position [µm]")
    ax_line.set_ylabel("transmission")
    #ax_line.set_ylim(-0.045, 0.047)
    #ax_line.set_ylim(0.99975, 1)
    
    ax_line.set_xlim(0, total_length_um)
    ax_line.grid(True, alpha=0.3)

    # --- left: 2D image made from 1D signal ---
    im = ax_img.imshow(
        img[np.newaxis, :],
        aspect="auto",
        cmap="gray",
        origin="lower",
        extent=[-0.5, n_segments - 0.5, 0, 1],
        #vmin=vmin, vmax=vmax
    )

    ax_img.set_yticks([])
    ax_img.set_xticks(xtick_positions)
    ax_img.set_xticklabels([f"{x:.0f}" for x in xticks_um])
    ax_img.set_xlabel("detector position [µm]")

    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    #cbar.set_ticklabels(["-0.04", "-0.02", "0", "0.02", "0.04"])
    cbar.set_label("transmission")
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


a0_list, a1_list, phi_list = estimate_phi_fourier(I_ref, I_tumor)
a0_list_no, a1_list_no, phi_list_no = estimate_phi_fourier(I_ref, I_no_tumor)
def compute_total_phase(phi_tumor, phi_no_tumor):
    total_phi_tumor = np.cumsum(phi_tumor)
    total_phi_no_tumor = np.cumsum(phi_no_tumor)
    return total_phi_tumor, total_phi_no_tumor

#total_phi_tumor, total_phi_no_tumor  = compute_total_phase(phi_tumor=phi_list, phi_no_tumor=phi_list_no)
#print(total_phi_tumor)
#plot_phase_shift_image(a0_list, "test_fourier.pdf")
#plot_phase_shift_image_with_line(a0_list, "test_fourier5.pdf")
"""
df = pd.read_csv('intensity_tumor_background_38keV_80um_4micron.csv')

# no noise 1D Intensity at detector plane with tumor cell
I_tumor = df["I_tumor"].to_numpy()

# no noise 1D Intensity at detector plane without tumor cell
I_no_tumor = df["I_no_tumor"].to_numpy()

# no noise 1D Intensity at detector plane reference no sample
I_ref = df["I_ref"].to_numpy() 
"""


#plot_phase_shift_image(phi_tumor_array[10,:].T, 'little_look_tumor.pdf')
#plot_phase_shift_image(phi_no_tumor_array[10,:].T, 'little_look_no_tumor.pdf')


#photons_list, sdnr_list = photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, "sdnr_results_38keV_80um_4micron.csv")

#photon_values = pd.read_csv('sdnr_results_38keV_80um_4micron.csv')['photons']
#sdnr_list1 = pd.read_csv('sdnr_results_38keV_80um_4micron.csv')['sdnr']
"""
sdnr_list2 = pd.read_csv('SDNR\sdnr_results_20keV_40um.csv')['sdnr']
sdnr_list3 = pd.read_csv('SDNR\sdnr_results_20keV_80um.csv')['sdnr']
sdnr_list4 = pd.read_csv('SDNR\sdnr_results_38keV_20um.csv')['sdnr']
sdnr_list5 = pd.read_csv('SDNR\sdnr_results_38keV_40um.csv')['sdnr']
sdnr_list6 = pd.read_csv('SDNR\sdnr_results_38keV_80um.csv')['sdnr']
"""
#plot_sdnr_vs_photons(photon_values, sdnr_list1,file_name='sdnr_vs_photons_38keV_4micron.pdf')
