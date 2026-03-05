import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This file can simply do a Fourier aproximation of an intensity signal.
It can read data from a .csv file with columns of names: 10um, 20um..... I_ref, I_no_tumor 
and use that data to do the Fourier fitting 
"""

d_sph = 10
data  = "intensity_tumor_background_20keV_all_sizes.csv"

col_name = f"{int(round(d_sph))}um"

I_tumor = pd.read_csv(data)[col_name].to_numpy()
I_ref = pd.read_csv(data)["I_ref"].to_numpy()
I_no_tumor = pd.read_csv(data)["I_no_tumor"].to_numpy()

N = len(I_tumor)
fourier_output = np.fft.fft(I_tumor)


sim_pix_size_in_m = 1e-6
img_size_in_pix = int(round(300*1e-6/ sim_pix_size_in_m))
x = np.arange(img_size_in_pix)
period_in_um = 5
period_in_pixels = period_in_um * 1e-6 / sim_pix_size_in_m
k= int(N / period_in_pixels)

mean = fourier_output[0].real / N
amp1 = 2 * np.abs(fourier_output[k]) / N
phase1 = np.angle(fourier_output[k])

print(f"Calculated k: {k}")
freq = 1 /period_in_pixels

print(f"Mean: {mean}")
print(f"Amplitude of first harmonic: {amp1}")
print(f"Phase of first harmonic (radians): {phase1}")
print(f"Phase of first harmonic (degrees): {np.degrees(phase1)}")

fourier_fit  = mean + amp1 * np.cos(2 * np.pi * freq * x + phase1)

plt.figure(figsize=(10, 6))
plt.plot(I_tumor, label='I_tumor', color='blue')
plt.plot(fourier_fit, label='First Harmonic Fit', color='red', linestyle='--')
plt.title('Intensity Profile for Tumor Size 10um at 20keV')
plt.savefig('Fourier_fit_20keV_10um.png', dpi=300)