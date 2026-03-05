import numpy as np
from scipy import constants as cte
import sys
from pathlib import Path
import xraydb as xdb
import xraylib as xrl

# --- Grating -----------------------------------------------------------------

#Grating period of G1 in um
px_in_um = 5

# --- Geometry ----------------------------------------------------------------

# Simulated pixel siz in m
sim_pix_size_in_m = 1e-8

# Image size in pix (can either be given directly or via amount of grating periods)
img_size_in_pix = int(round(300*1e-6/ sim_pix_size_in_m))
# img_size_in_pix = 1000
# img_size_in_pix = 64 * int(px_in_um * 1e-6 / sim_pix_size_in_m)

grating_periods =  img_size_in_pix/ (px_in_um * 1e-6 / sim_pix_size_in_m)

# Sample size in pix in X direction
samp_size_in_pix = img_size_in_pix
samp_size_in_m = samp_size_in_pix * sim_pix_size_in_m*1e6

# --- Source ------------------------------------------------------------------

# Energy in keV
E_in_keV = 60
E_in_J = E_in_keV * 1e3 * cte.e

# Wavelength in m 
l_in_m = (cte.physical_constants["Planck constant in eV s"][0] * cte.c) / \
         (E_in_keV * 1e3) 

# Wavevector magnitude in 1/m 
k_in_1_m = 2 * np.pi * (E_in_keV * 1e3) / \
           (cte.physical_constants["Planck constant in eV s"][0] * cte.c)

# Classical electron radius
r_e = cte.physical_constants["classical electron radius"][0]  # in m

# Calculation of Talbot distance (just to check)
talbot_in_m = 2 * (px_in_um * 1e-6)**2 / l_in_m 

# Here you can look at what the distance from G1 grating to detector would look like 
# (pick the fractional Talbot distance you're interested in)
grat2det_in_m = 3/4 * talbot_in_m  

# Distance from middle of sample to detector
prop_in_m = grat2det_in_m/2

#Amount of photons per pixel (flat field)
num_ph = 1e6

# --- Sample ------------------------------------------------------------------

# Total thickness of the sample in mm
t_samp_in_mm = 0.5

# diameter of central cancer lesion
d_sph_in_um = 100

#Thickness of slices (sections that the 2D sample gets subdivided into) in um 
t_slc_in_um = 1
t_slc_in_pix = int(t_slc_in_um * 1e-6 / sim_pix_size_in_m)

# --- Material Properties of Sample ---------------------------------------------
"""
mat_sph = "SiO2"
name_sph = "glass"
rho_sph_in_g_cm3 = 2.196


mat_bkg = "C2H6O"
name_bkg = "Ethanol"
rho_bkg_in_g_cm3 = 0.78945

mat_bkg = "H2O"
name_bkg = "Water"
rho_bkg_in_g_cm3 = 0.998

mat_sph = "H0.39234C0.15008N0.03487O0.31620Na0.00051Mg0.00096P0.03867S0.00109Ca0.06529" 
name_sph = "bone"
rho_sph_in_g_cm3 = 1.92 
"""

# name_bkg = "adipose tissue"
# mat_bkg="H0.62536C0.27525N0.00276O0.09606Na0.00024S0.00017Cl0.00016"
# rho_bkg_in_g_cm3 = 0.95 

# mat_bkg = "H2O"
# name_bkg = "Water"
# rho_bkg_in_g_cm3 = 0.998
#
# mat_sph = "SiO2"
# name_sph = "glass"
# rho_sph_in_g_cm3 = 2.196

mat_bkg = "N0.78O0.21Ar0.01"
name_bkg = "Air"
rho_bkg_in_g_cm3 = 0.001225

mat_sph = "C5H8O2"
name_sph = "PMMA"
rho_sph_in_g_cm3 = 1.19

# name_sph = "Tumor cell"
# mat_sph = "cancer"
#
# # experimental electron density from paper
# electron_density = 3.54*1e29 # in electrons/m3
# delta_sph = 2*np.pi * r_e * electron_density / (k_in_1_m**2)
# rho_sph_in_g_cm3 = 0
#
# # experimental absorption from paper
# mu_sph_in_1_m = 0.212 * 100

# --- Detector ----------------------------------------------------------------

# size of bins in case we do any binning
detector_pixel_size = 1*1e-6
binning_factor = int(detector_pixel_size/sim_pix_size_in_m)

# When we do reconstruction of the phase image and look at the intensity in sections we decide 
# here how big those sections are (I recommend to have it set to the grating period as be get a min 
# and a max in each section)
segment_size_in_um = px_in_um
segment_size_in_pix = int(round(segment_size_in_um * 1e-6 / sim_pix_size_in_m/binning_factor))
