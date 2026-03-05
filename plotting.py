import matplotlib.pyplot as plt
import numpy as np
from grating import *
from sample import *
from propagator import *
from parameter import *
from detector import *
import os
import pandas as pd
import time

"""
This file has a bunch of random functions that I used at some point....
Most of them are kinda irrelevant and the only really important one is: save_visibility_epsilon
But I don't really use that one anymore either
"""

def plot_intensity_withG2(det, prop,  wavefld_bg, save_plot=True):  
    """
    Plots the intensity on the detector after it was convolved with the virtual G2
    """

    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)

    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    I_max_samp = np.max(Isamp_stepped)
    I_min_samp = np.min(Isamp_stepped)

    I_max_ref = np.max(Iref_stepped)
    I_min_ref = np.min(Iref_stepped)

    a_1s = (I_max_samp-I_min_samp)/2
    a_1r = (I_max_ref-I_min_ref)/2

    a_0s = np.mean(Isamp_stepped.real)
    a_0r = np.mean(Iref_stepped.real)

    visibility_s = a_1s.real/a_0s
    visibility_r = a_1r.real/a_0r

    visibility = visibility_s/visibility_r
    epsilon = -np.log(visibility) / (t_samp_in_mm * 1e-3)

    print(f"Visibility with sample: {visibility.real:.3f} at Energy: {E_in_keV:.1f} keV")
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")

    plt.plot(Iref_stepped, label='Iref with G2', linewidth=0.5, color='red')
    plt.plot(Isamp_stepped, label='Isamp with G2', linewidth=0.5, color='blue')
    plt.title(f"Intensity Profile at {E_in_keV:.1f} keV | Visibility with sample: {visibility.real:.3f} \n Thickness of sample: {t_samp_in_mm:.1f} mm | Mean Intensity: {np.mean(Isamp_stepped.real):.3f}")
    plt.xlabel('Pixels')
    plt.ylabel('Intensity')
    plt.legend()

    if save_plot:
        path_image = os.path.join("Test_intensity_with_G2.pdf")
        plt.savefig(path_image, dpi=600, bbox_inches='tight')
    del Iref_stepped, Isamp_stepped

def plot_intensity_withoutG2(det, prop,  wavefld_bg, save_plot=True, show=False): 
    """
    Plots the intensity on the detector as it is (without any convolution)
    """
    Iref, Isamp = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)
    x_walk = np.arange(len(Iref))
    plt.plot(Iref, label='Iref', linewidth=0.7, color='red')
    plt.plot(Isamp, label='Isamp', linewidth=0.5, color='blue')
    plt.title("Without G2")
    plt.xlabel('Pixels (1 micron size)')
    plt.tight_layout()
    plt.title(f'Energy: {E_in_keV:.1f} keV')
    # plt.xlim(500, 2000)
    # plt.ylim(-0.1, 1.5)
    # for x in range(500, 2001, 100):
    #     plt.axvline(x=x, color='green', linestyle='-', linewidth=1, alpha=1)
    plt.legend()
    plt.grid()
    plt.savefig('test1.pdf', dpi=600, bbox_inches='tight')

    if show:
        plt.show()

def save_visibility_epsilon(det, prop,  wavefld_bg, bin_grat,thick_samp_mm):
    """
    Calculates the visibility and the corresponding epsilon values 
    by convolving the intensity on the detector with the virtual G2 and then simply looking for
    the minimum and maximum values to calculate the visibility

    There is an updated version of this in calc_sdnr that looks at the intensity in sections instead
    """
    Iref_large, Isamp_large = prop.obtain_Iref_Isamp(wavefld_bg, bin_grat)
    
    G2 = det.create_g2()
    Iref_stepped, Isamp_stepped = det.phasestepping_conv(Isamp_large, Iref_large, G2)

    I_max_samp = np.max(Isamp_stepped)
    I_min_samp = np.min(Isamp_stepped)
    I_max_ref = np.max(Iref_stepped)
    I_min_ref = np.min(Iref_stepped)
    a_1s = (I_max_samp-I_min_samp)/2
    a_1r = (I_max_ref-I_min_ref)/2
    a_0s = np.mean(Isamp_stepped.real)
    a_0r = np.mean(Iref_stepped.real)
    visibility_s = a_1s.real/a_0s
    visibility_r = a_1r.real/a_0r
    visibility = visibility_s/visibility_r
    epsilon = -np.log(visibility) / (thick_samp_mm * 1e-3)
    print(f"Mean with sample: {np.mean(Isamp_stepped.real):.3f} at Energy: {E_in_keV:.1f} keV")
    return visibility, epsilon

def plot_single_slice_pair(slc2d_sph_padded, slc2d_bkg_padded, slice_idx=0, save_plot=True):
    """
    This can plot individual thickness maps to see what the look like
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axs[0].plot(slc2d_sph_padded[slice_idx], color='blue')
    axs[0].set_title(f"Slice {slice_idx} (Spheres)")
    axs[0].set_xlabel("Pixels")
    axs[0].set_ylabel("Value")

    axs[1].plot(slc2d_bkg_padded[slice_idx], color='orange')
    axs[1].set_title(f"Slice {slice_idx} (Background)")
    axs[1].set_xlabel("Pixels")

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"slice_pair_{slice_idx}.pdf", dpi=300, bbox_inches='tight')

def plot_cosine_fit_1d_images(csv_path,file_name):
    """
    Plot 3 grayscale 1D image strips (mean, amplitude, phase)
    from the cosine fit results CSV file.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Extract columns
    mean_a0 = df["mean"].to_numpy()
    amplitude_a1 = df["visibility"].to_numpy()
    phase_phi = df["phase_shift"].to_numpy()
    
    mean_img = mean_a0[np.newaxis, :]
    amp_img  = amplitude_a1[np.newaxis, :]
    phi_img  = phase_phi[np.newaxis, :]

    limits_mean = (0.96, 1)
    limits_visibility = (0.99,1)
    limits_phase = (-0.03, 0.03)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    images = [mean_img, amp_img, phi_img]
    titles = ["Absorption Image", "Visibility Image", "Differential Phase Shift Image"]
    limits = [limits_mean, limits_visibility, limits_phase]

    n_segments = len(df)
    total_length_um = n_segments * segment_size_in_um
    xticks_um = np.arange(0, total_length_um + 1, 100)
    #xtick_positions = xticks_um / segment_size_in_um  # position in segment units
    xtick_positions = (xticks_um - 0.5 * segment_size_in_um) / segment_size_in_um
    for ax, img, title, lim in zip(axes, images, titles, limits):
        #ax.imshow(img, aspect='auto', cmap='gray', origin='lower',
                  #extent=[-0.5, n_segments - 0.5, 0, 1])
        
        im = ax.imshow(
            img,
            aspect='auto',
            cmap='gray',
            origin='lower',
            extent=[-0.5, n_segments - 0.5, 0, 1],
            vmin=None if lim is None else lim[0],
            vmax=None if lim is None else lim[1],
        )
        #ax.imshow(img, aspect='auto', cmap='gray', origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{x:.0f}" for x in xticks_um])
        ax.set_xlabel("detector position (μm)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    main_title=f"Energy: {E_in_keV:.1f} keV material sphere: {name_sph} | material background: {name_bkg} \n sphere diameter: {d_sph_in_um} um sample thickness: {t_samp_in_mm} mm \n photons per pixel: {num_ph:.1e}"
    fig.suptitle(main_title, fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
