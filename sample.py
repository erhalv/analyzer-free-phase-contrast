import numpy as np
from numpy._core.fromnumeric import shape
from pandas._libs.parsers import sanitize_objects
import scipy.fft
from skimage.draw import disk, polygon
import xraylib as xrl
from typing import Tuple
from parameter import * 
import matplotlib.pyplot as plt

class Sample:                             
    
    def __init__(self, 
                 t_samp_in_mm: float, 
                 d_sph_in_um: float,  
                 mat_sph: str, 
                 mat_bkg: str,
                 rho_sph_in_g_cm3: float = None, 
                 rho_bkg_in_g_cm3: float = None,
                 shape = 'sphere') -> None:
        """
        Initializes an instance of the Sample class with the specified 
        parameters.

        Args:
            t_samp_in_mm (float): Sample thickness in mm.
            d_sph_in_um (float): Sphere diameter in um. 
            mat_sph (str): Sphere material. If it is an element, write its
                           symbol. If it is a compound, write its chemical
                           formula. 
            mat_bkg (str): Background material. If it is an element, write its
                           symbol. If it is a compound, write its chemical
                           formula. 
            rho_sph_in_g_cm3 (float, optional): Mass density of the sphere 
                                                material, in g/cm3. If it is a
                                                compound, write its mass density
                                                value. Defaults to None.
            rho_bkg_in_g_cm3 (float, optional): Mass density of the background 
                                                material, in g/cm3. If it is a
                                                compound, write its mass density
                                                value. Defaults to None.
        
        Using these arguments, the following variables are calculated:

            num_slc (int): Total number of sample slices.
            mu_sph_in_1_m (float): Linear attenuation coefficient of the sphere
                                   material, in 1/m.
            delta_sph (float): Phase coefficient of the sphere material.
            Z_bkg (int): Atomic number of the background mateiral.
            mu_bkg_in_1_m (float): Linear attenuation coefficient of the 
                                   background material, in 1/m.                                  
            delta_bkg (float): Phase coefficient of the background material.

        """
                     
        self.thickness_in_mm = t_samp_in_mm
        self.t_samp_in_pix = int((self.thickness_in_mm * 1e-3) / sim_pix_size_in_m)
        # self.t_samp_in_pix = 10

        self.d_sph_in_um = d_sph_in_um
        self.d_sph_in_pix = int((self.d_sph_in_um * 1e-6) / sim_pix_size_in_m)
        self.r_sph_in_pix = int(self.d_sph_in_pix / 2)  

        self.num_slc = int(self.t_samp_in_pix /t_slc_in_pix)
        print('nr of slices',self.num_slc)
          
        self.mat_sph = mat_sph
        self.mat_bkg = mat_bkg

        # Material properties sphere
        self.rho_sph_in_g_cm3 = rho_sph_in_g_cm3

        # Either hard experimental code values from papers or use material properties from database
        # self.delta_sph = delta_sph
        # self.mu_sph_in_1_m = mu_sph_in_1_m

        self.delta_sph = 1 - xrl.Refractive_Index_Re(self.mat_sph, 
                                                     E_in_keV,
                                                     self.rho_sph_in_g_cm3)
        # self.mu_sph_in_1_m = xrl.CS_Total_CP(self.mat_sph, E_in_keV) \
        #                          * self.rho_sph_in_g_cm3 * 100 



        #Material properties background
        self.rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3
        self.mu_bkg_in_1_m = xrl.CS_Total_CP(self.mat_bkg, E_in_keV) \
                                 * self.rho_bkg_in_g_cm3 * 100
        self.delta_bkg = 1 - xrl.Refractive_Index_Re(self.mat_bkg, 
                                                     E_in_keV,
                                                     self.rho_bkg_in_g_cm3)

        self.mu_sph_in_1_m = self.mu_bkg_in_1_m
        # Shape of the sample
        # Sphere of wedge
        self.shape = shape


    def create_wedge2d(self):

        # Create the 2D slice that will contain the wedge 
        slc2d_wdg = np.zeros((samp_size_in_pix,self.t_samp_in_pix), dtype=np.uint16) 

        # Coordinates of the vertices
        r = [samp_size_in_pix, 0,  samp_size_in_pix]
        c = [self.t_samp_in_pix//2, self.t_samp_in_pix//2, self.t_samp_in_pix//2 + self.r_sph_in_pix]

        rr, cc = polygon(r, c, shape=slc2d_wdg.shape)

        slc2d_wdg[rr, cc] = 1
        slc2d_wdg_real = np.abs(slc2d_wdg)

        # Create inverse array for background 
        slc2d_bkg = np.ones(slc2d_wdg_real.shape) - slc2d_wdg_real
        return slc2d_wdg_real, slc2d_bkg


    # --- Spheres -------------------------------------------------------------
    def create_slice2d(self) -> Tuple[np.ndarray, 
                                           np.ndarray]:
        """
        Creates the binary 2D array with single disk of set diameter in the middle
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the 2D binary
                                           arrays with spheres and the background
                                           of the slice, respectively.
        """
        # Create the 2D slice that will contain the projected spheres
        slc2d_sph = np.zeros((samp_size_in_pix,self.t_samp_in_pix), dtype=np.uint16) 
        print('shape', slc2d_sph.shape)

        # Gets the centre coordinates
        centre = [samp_size_in_pix//2, self.t_samp_in_pix//2]

        # Places the disk around the centre postition with given diameter
        rr, cc = disk(centre, self.r_sph_in_pix, shape=slc2d_sph.shape)
        slc2d_sph[rr, cc] = 1
        slc2d_sph_real = np.abs(slc2d_sph)

        # Create inverse array for background 
        slc2d_bkg = np.ones(slc2d_sph_real.shape) - slc2d_sph_real
        return slc2d_sph_real, slc2d_bkg

    def create_projected_1d_slices(self) -> Tuple[np.ndarray,np.ndarray]:
        """
        Takes the 2D sample and slices it into sections that are summed up so we can propegate through it
        in the thin slice aproximation. These individual vectors are thickness maps representing one section
        of the sample.
      
        Returns:
            Tuple[np.ndarray, np.ndarray]:  A tuple containing the 1D projected 
                                            sphere and background slices ready for propagation
        """
        slice_profiles_sph = []
        slice_profiles_bkg = []
        if self.shape == 'sphere':
            slc2d_sph_real, slc2d_bkg = self.create_slice2d()
        else: 
            slc2d_sph_real, slc2d_bkg = self.create_wedge2d()

        print(slc2d_sph_real.shape)
        print(slc2d_sph_real)

        for i in range(self.num_slc):
            start = i * t_slc_in_pix
            end = start + t_slc_in_pix

            slice_chunk_sph = slc2d_sph_real[:,start:end] 
            slice_chunk_bkg = slc2d_bkg[:, start:end]          
            
            profile_sph = np.sum(slice_chunk_sph, axis=1)           # sum over rows
            profile_bkg = np.sum(slice_chunk_bkg, axis=1)

            slice_profiles_sph.append(profile_sph)
            slice_profiles_bkg.append(profile_bkg)

        slice_profiles_sph = np.array(slice_profiles_sph)  # (num_slc, img_size_in_pix)
        slice_profiles_bkg = np.array(slice_profiles_bkg)
        
        # New slices with zero padding (I usually don't have any padding)
        slc2d_sph_padded = np.zeros((slice_profiles_sph.shape[0], img_size_in_pix))
        slc2d_bkg_padded = np.zeros((slice_profiles_bkg.shape[0], img_size_in_pix))
        x_offset = int((img_size_in_pix - samp_size_in_pix) / 2)
        slc2d_sph_padded[:, x_offset:x_offset + samp_size_in_pix] = slice_profiles_sph
        slc2d_bkg_padded[:, x_offset:x_offset + samp_size_in_pix] = slice_profiles_bkg
        return slc2d_sph_padded, slc2d_bkg_padded

    def samp_with_refract_property(self, 
                           samp_sph_in_m: np.ndarray,
                           samp_bkg_in_m: np.ndarray) -> np.ndarray:
        """
        Applies the refractive properties of the sample to the binary array. 
        Instead of having ones and zeros in the array we now have the complex numbers 
        correspning to the refractive indexes multiplied by the thickness of the structures.

        Args:
            samp_sph_in_m (np.ndarray): The positive projected slices (thickness maps)
            samp_bkg_in_m (np.ndarray): The negative/background projected slices (thickness maps)

        Returns:
            np.ndarray: Thickness maps with complex numbers in them ready for multiplication with wave vector
        """

        print("Sample created with sphere delta:", self.delta_sph, "and background delta:", self.delta_bkg)
        return np.exp(-1j * k_in_1_m * (self.delta_sph * \
                                samp_sph_in_m + self.delta_bkg * \
                                samp_bkg_in_m)) * \
               np.exp(-((self.mu_sph_in_1_m / 2) * samp_sph_in_m + \
                      (self.mu_bkg_in_1_m / 2) * samp_bkg_in_m)) 



if __name__ == '__main__': 
    samp2d = Sample(t_samp_in_mm = t_samp_in_mm,
                    d_sph_in_um = d_sph_in_um,
                    mat_sph = mat_sph, 
                    mat_bkg = mat_bkg, 
                    rho_sph_in_g_cm3 = rho_sph_in_g_cm3, 
                    rho_bkg_in_g_cm3 = rho_bkg_in_g_cm3,
                    shape='sphere')

    slc2d_sph_padded, slc2d_bkg_padded = samp2d.create_projected_1d_slices() 

    slc2d_sph_padded = slc2d_sph_padded[::, ::100]

    plt.imshow(slc2d_sph_padded, cmap='gray')
    plt.show()

    print(slc2d_sph_padded.shape)




