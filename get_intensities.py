from grating import *
from sample import *
from propagator import *
import numpy as np
from parameter import *
from detector import *
from plotting import *
import time
import threading
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

"""
This file is similar to the main function but it runs through a bunch of different tumour sizes 
and saves them all in a .csv file (in the first run it saves the reference intensity and the intensity
when no tumor is present and then it goes through the  given in d_sphss)
"""

grat1d = Grating(px_in_um = px_in_um)
det = Detector(px_in_um= px_in_um)

d_sphss = [0,40,50,60,70,80,90,100]
for d_sph in d_sphss:

    samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                    d_sph_in_um = d_sph,
                    mat_sph = mat_sph, 
                    mat_bkg = mat_bkg,
                    rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                    rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3)
    
    prop = Propagator(grat = grat1d,
                        samp = samp2d,
                        det = det,
                        prop_in_m = prop_in_m)

    wavefld_bg = np.ones(img_size_in_pix)

    slice_profiles_path =  "slices_data.npz"
    slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices()
    np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
    Iref, Isamp = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)
    file_path = "intensity_tumor_background_60keV.csv"
    col_name = f"{int(round(d_sph))}um"

    if not os.path.exists(file_path):
        # File does NOT exist — create it
        results = pd.DataFrame({
            "I_ref": Iref,
            "I_no_tumor": Isamp
        })
        results.to_csv(file_path, index=False)
    else:
        # File exists — load and add new column
        results = pd.read_csv(file_path)
        
        # Add new column (ensure lengths match)
        results[col_name] = Isamp
        
        results.to_csv(file_path, index=False)

    # Remove saved slice profiles to free disk space now that results are saved
    if os.path.exists(slice_profiles_path):
        try:
            os.remove(slice_profiles_path)
        except OSError:
            pass