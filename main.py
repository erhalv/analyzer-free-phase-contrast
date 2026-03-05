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
This is the main file that can run individual simulations.
currently it creates a sample and saves it and then runs a simulation and saves the intensity 
in a .csv file
"""

def main():
    # --- Initialise sample, grating, detector, and propagator -------

    samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                    d_sph_in_um = d_sph_in_um,
                    mat_sph = mat_sph, 
                    mat_bkg = mat_bkg,
                    rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                    rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3,
                    shape='wedge')

    grat1d = Grating(px_in_um = px_in_um)
    det = Detector(px_in_um= px_in_um)

    prop = Propagator(grat = grat1d,
                            samp = samp2d,
                            det = det,
                            prop_in_m = prop_in_m)


    wavefld_bg = np.ones(img_size_in_pix)

    slice_profiles_path =  "slices_data.npz"

    # This stuff just puts a timer so you can see how long the simulation is already taking
    def print_elapsed_time(start_time, stop_event):
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            print(f"\rElapsed time: {elapsed:.1f} seconds", end="")
            time.sleep(1)
        print()  # Move to next line after stopping

    start = time.time()
    stop_event = threading.Event()
    timer_thread = threading.Thread(target=print_elapsed_time, args=(start, stop_event))
    timer_thread.start()


    try:

        if os.path.exists(slice_profiles_path):
            # Loads the sample saved in slices_data.npz from file if it exists 
            # (propagator will always take data from file)
            data = np.load(slice_profiles_path)
            slc2d_sph_padded = data['slc2d_sph_padded']
            slc2d_bkg_padded = data['slc2d_bkg_padded'] 
            print()
            print("Loaded slice profiles from file.")
        else:

            slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices()
            np.savez(slice_profiles_path, slc2d_sph_padded=slc2d_sph_padded, slc2d_bkg_padded=slc2d_bkg_padded)
            print()
            print("Created and saved slice profiles.")


        # !!You can put whatever you want to run here!! 
        # Currently it just calculates the reference and sample intensity and saves them in a .csv file
        # you can then change the diameter of the sphere and run again if you like 
        # (don't forget to delete the slices_data.npz file otherwise it will use the old sample)
        Iref, Isamp = prop.obtain_Iref_Isamp(wavefld_bg, prop.bin_grat)
        # file_path = "intensity_tumor_background.csv"
        file_path = "intensity_wedge.csv"

        print(Iref, Isamp)

        # Column name derived from sphere diameter in parameter file
        col_name = f"{int(round(d_sph_in_um))}um"

        if not os.path.exists(file_path):
            # File does NOT exist — create it
            results = pd.DataFrame({
                "I_ref": Iref,
                col_name: Isamp
            })
            results.to_csv(file_path, index=False)
        else:
            # File exists — load and add new column
            results = pd.read_csv(file_path)

            # Add new column (ensure lengths match)
            results[col_name] = Isamp

            results.to_csv(file_path, index=False)

        # 'Possible other things you could be looking at'
        # plot_intensity_withG2(det, prop,  wavefld_bg, save_plot=True)
        # plot_intensity_withoutG2(det, prop,  wavefld_bg, save_plot=True, show=True)
        # plot_cosine_fit_1d_images('cosine_fit_segments_30.csv', 'cosine_fit_1d_images30.pdf')    

    finally:

        end = time.time() 
        stop_event.set()
        timer_thread.join()
        print(f"Total simulation time: {end - start:.2f} seconds.")
        print()


    # --- Save all the simulation parameters in sim_param.csv --------------------------------------

    sim_param = {   
                    "Energy in keV": E_in_keV, 
                    "Simulated pixel size in m": sim_pix_size_in_m, 
                    "Simulated image size in pix": img_size_in_pix,
                    "Grating period in X in um": grat1d.px_in_um,                    
                    "Sample size in pix": samp_size_in_pix,
                    "Sample thickness in mm": samp2d.thickness_in_mm,
                    "Sphere diameter in um": samp2d.d_sph_in_um,
                    "Sphere material": samp2d.mat_sph,
                    "Background material": samp2d.mat_bkg,
                    "Number of slices": samp2d.num_slc,
                    "Thickness of a sample slice in pix": t_slc_in_pix,
                    "Talbot distance in m": round(prop.talbot_in_m, 1),
                    "Grating-to-detector distance in cm": round(prop.grat2det_in_m * 100, 1),
                    "Grating-to-sample distance in cm": round(prop.grat2samp_in_m * 100, 1),
                    "Sample-to-detector distance in cm": round(prop.samp2det_in_m * 100, 1),
                    "Propagation distance in cm": round(prop_in_m * 100, 1), 
                    "Simulation time in min": round((end - start) / 60, 1)
                }

    with open("sim_param.csv",
              "w", 
              newline = "") as file:
        w = csv.writer(file)
        w.writerows(sim_param.items())


if __name__ == '__main__':
    main()
