import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from parameter import *
import xraylib as xrl

data  = "intensity_tumor_background_60keV_all_sizes_talbot3.csv"
"""d_sph = 80
col_name = f"{int(round(d_sph))}um"

I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
"""
def compute_2D_Intensities(I_ref, I_no_tumor, I_tumor ,photons, d_sph):
    
    I_ref_fit = np.resize(I_ref, (int(len(I_ref)/segment_size_in_pix), len(I_ref))) 
    I_ref_noisy_2d = np.random.poisson(I_ref_fit * photons)
    
    I_no_tumor_fit = np.resize(I_no_tumor, (int(len(I_ref)/segment_size_in_pix), len(I_ref))) 
    I_no_tumor_noisy_2d = np.random.poisson(I_no_tumor_fit * photons)

    dim1_y_stacks = I_ref_noisy_2d.shape[0] #60
    dim2_x_det = I_ref_noisy_2d.shape[1]   #300

    tumor_rows = int(d_sph / segment_size_in_pix)  # diameter in pixels (rows)
    I_tumor_2d = np.resize(I_no_tumor, (int(len(I_ref)/segment_size_in_pix), len(I_ref))) 

    if tumor_rows > 0:
        r0 = (dim1_y_stacks - tumor_rows) // 2
        r1 = r0 + tumor_rows
        print("r0, r1:", r0, r1)
        # Overwrite center band with a tumor realization (independent noise)
        I_tumor_2d[r0:r1, :] = I_tumor
    I_tumor_noisy_2d = np.random.poisson(I_tumor_2d * photons)
    return I_ref_noisy_2d, I_no_tumor_noisy_2d, I_tumor_noisy_2d

# size of the arrays is (dim1_y_stacks, dix2_x_detector) = (60, 300)
#I_ref_noisy_2d, I_no_tumor_noisy_2d, I_tumor_noisy_2d = compute_2D_Intensities(I_ref, I_no_tumor, I_tumor ,photons=1e5, d_sph=d_sph)

def estimate_phi_fourier_rows(Iref, Isamp, segment_size_in_pix, px_in_um, detector_pixel_size):
    Isamp = np.asarray(Isamp)
    Iref  = np.asarray(Iref)

    R, L = Isamp.shape

    x_walk = np.arange(L)

    centers = []
    phi_list = []

    N = int(segment_size_in_pix)
    k = int(N / (px_in_um * 1e-6 / detector_pixel_size))

    # Loop over segments only
    for start in range(0, L - N + 1, N):
        # shape: (R, N) for both
        Isamp_seg = Isamp[:, start:start+N]
        Iref_seg  = Iref[:, start:start+N]

        # FFT along the segment axis (per row, vectorized)
        F_samp = np.fft.fft(Isamp_seg, axis=-1)  # (R, N)
        F_ref  = np.fft.fft(Iref_seg,  axis=-1)  # (R, N)

        phase_samp = np.angle(F_samp[:, k])      # (R,)
        phase_ref  = np.angle(F_ref[:,  k])      # (R,)

        phi = phase_samp - phase_ref             # (R,)

        phi_list.append(phi)                     # list of (R,)
        centers.append(np.mean(x_walk[start:start+N]))

    # stack into (R, n_segments)
    phi_out = np.stack(phi_list, axis=1)
    centers = np.asarray(centers)

    return phi_out

#phi_out = estimate_phi_fourier_rows(I_ref_noisy_2d, I_tumor_noisy_2d, segment_size_in_pix, px_in_um, detector_pixel_size)

def compute_phase_shift_fourier(I_ref, I_tumor, I_no_tumor, num_realizations, photons, d_sph):

    phi_tumor_results = []
    phi_no_tumor_results = []
    print ("photons per pixel",photons)
    for i in range(num_realizations): 
        I_ref_noisy_2d, I_no_tumor_noisy_2d, I_tumor_noisy_2d = compute_2D_Intensities(I_ref, I_no_tumor, I_tumor ,photons, d_sph)
        phi_no_tumor = estimate_phi_fourier_rows(I_ref_noisy_2d, I_no_tumor_noisy_2d, segment_size_in_pix, px_in_um, detector_pixel_size)
        phi_tumor = estimate_phi_fourier_rows(I_ref_noisy_2d, I_tumor_noisy_2d, segment_size_in_pix, px_in_um, detector_pixel_size)
        phi_tumor_results.append(phi_tumor)
        phi_no_tumor_results.append(phi_no_tumor)

    phi_tumor_array = np.array(phi_tumor_results)
    phi_no_tumor_array = np.array(phi_no_tumor_results)
    return phi_tumor_array, phi_no_tumor_array
"""
phi_tumor_array, phi_no_tumor_array = compute_phase_shift_fourier(I_ref, 
                                                                  I_tumor, 
                                                                  I_no_tumor, 
                                                                  num_realizations=50, 
                                                                  photons=1e5, 
                                                                  d_sph=d_sph
                                                                  )
"""
def compute_sdnr(G0, G1):
    """
    G0: array (N0, n) realizations under H0
    G1: array (N1, n) realizations under H1
    Returns: SDNR, SDNR_squared, weight_vector w, t_H0, t_H1
    """
    # means
    g0_bar = np.mean(G0, axis=0)
    g1_bar = np.mean(G1, axis=0)
    d = g1_bar - g0_bar  # Δg

    # covariances (sample covariance; rows are samples)
    K0 = np.cov(G0, rowvar=False, bias=False)   # shape (n,n)
    K1 = np.cov(G1, rowvar=False, bias=False)

    Ksum = K0 + K1

    # compute weight vector w = (K0+K1)^{-1} Δg
    w = np.linalg.solve(Ksum, d)

    # SDNR^2 = d^T (K0+K1)^{-1} d
    sdnr2 = float(d.dot(w))
    sdnr = np.sqrt(max(sdnr2, 0.0))

    # test-statistic distributions (linear observer)
    t_H0 = G0.dot(w)   # shape (N0,)
    t_H1 = G1.dot(w)   # shape (N1,)

    return sdnr, sdnr2, w, t_H0, t_H1

def photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, file_name, d_sphe):
    sdnr_list = []
    photons_list = np.logspace(9, 3, num=10, base=10.0, dtype=int)[::-1]

    for photons in photons_list:
        # compute phase-shifts for this photon count
        phi_tumor_array, phi_no_tumor_array = compute_phase_shift_fourier(I_ref, 
                                                                  I_tumor, 
                                                                  I_no_tumor, 
                                                                  num_realizations=500, 
                                                                  photons=photons, 
                                                                  d_sph=d_sphe
                                                                  )
        sdnr, sdnr2, w, t_H0, t_H1 = compute_sdnr(phi_no_tumor_array.reshape(500,-1), phi_tumor_array.reshape(500,-1))
        
        sdnr_list.append(sdnr)
    if not os.path.exists(file_name):
        # File does NOT exist — create it
        results = pd.DataFrame({
        "photons": photons_list,
        f"SDNR_{int(round(d_sphe))}um": sdnr_list
        })
        results.to_csv(file_name, index=False)
    else:
        # File exists — load and add new column
        results = pd.read_csv(file_name)
        
        # Add new column (ensure lengths match)
        results[f"SDNR_{int(round(d_sphe))}um"] = sdnr_list
        
        results.to_csv(file_name, index=False)
"""
d_sph = 50
col_name = f"{int(round(d_sph))}um"
I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, "sdnr_results_60keV_all_sizes_2D_pixel_5x5um.csv", d_sph)
"""

d_sphss = [50,60,70,80]
for d_sph in d_sphss:
    col_name = f"{int(round(d_sph))}um"
    I_tumor = pd.read_csv(data)[col_name].to_numpy()
    I_ref = pd.read_csv(data)["I_ref"].to_numpy()
    I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()
    photons_list, sdnr_list = photons_vs_sdnr(I_ref, I_tumor, I_no_tumor, "sdnr_results_60keV_all_sizes_2D_pixel_5x5um.csv", int(d_sph))
