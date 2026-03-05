import numpy as np
from tqdm import tqdm
from grating import *
from sample import *
from detector import *
from parameter import *


class Propagator:

    def __init__(self, 
                 grat: Grating, 
                 samp: Sample,
                 det: Detector,
                 prop_in_m: float) -> None:
        """
        Initializes an instance of the wave field class with the specified 
        parameters.

        Args:
            grat (Grating): The grating object used in the setup.
            samp (Sample): The sample object being analyzed.
            det (Detector): The detector object capturing the wave field.

        Using these arguments, the following variables are calculated:

            talbot_in_m (float): Talbot distance, in m.
            grat2det_in_m (float): Grating-to-detector distance, in m.
            grat2samp_in_m (float): Grating-to-sample distance, in m.
            samp2det_in_m (float): Sample-to-detector distance, in m.
        """
        
        self.grat = grat
        self.samp = samp 
        self.det = det
        self.prop_in_m = prop_in_m

        #actual calculation of the talbot distance that is used for simulation
        self.talbot_in_m = 2 * (grat.px_in_um * 1e-6)**2 / l_in_m 
        
        # actual grating to detector distance used for simulation (pick the fraction that you want)
        self.grat2det_in_m = 3/4 * self.talbot_in_m  

        # Calculates the distance from the G1 grating to the from edge of the sample
        self.grat2samp_in_m = self.grat2det_in_m - self.prop_in_m - \
                              (samp.thickness_in_mm * 1e-3) / 2
        
        # Calculates the the distance of the back edge of the sample to the detector
        self.samp2det_in_m = self.prop_in_m - (samp.thickness_in_mm * 1e-3) / 2
        self.bin_grat = grat.create_grating()

    # --- Basic operations ----------------------------------------------------

    def inter_wavefld_grat(self, 
                           wavefld: np.ndarray) -> np.ndarray:
        """
        Applies the grating interaction to the wave field.

        This method modifies the input wave field based on the phase shift
        introduced by the grating and the attenuation due to the grating 
        material properties.

        Args:
            wavefld (np.ndarray): The input wave field to be modified.

        Returns:
            np.ndarray: The modified wave field after interaction with the 
                        grating.
        """

        return wavefld * self.bin_grat  

    def create_Fresnel_kernel(self, z_in_m: float) -> np.ndarray:
        """
        Creates a Fresnel propagation kernel in the Fourier domain to simulate 
        wave field propagation over a certain distance in the Fourier space.

        Args:
            z_in_m (float): The propagation distance, in m.

        Returns:
            np.ndarray: The Fresnel kernel for the specified propagation 
                        distance.
        """

        u = scipy.fft.fftfreq(img_size_in_pix, d=sim_pix_size_in_m)
        return np.exp(1j*(2 * np.pi / l_in_m) * z_in_m) * np.exp(-1j * np.pi * l_in_m * z_in_m * (u**2))

    def prop_wavefld(self, 
                    wavefld: np.ndarray, 
                    kernel: np.ndarray) -> np.ndarray:  
        """
        Propagates the wave field over a specified distance using the Fresnel 
        kernel.

        Args:
            wavefld (np.ndarray): The input wave field to be propagated.
            kernel (float): The Fresnel kernel in Fourier.

        Returns:
            np.ndarray: The propagated wave field after applying the Fresnel 
                        propagation.
        """

        return scipy.fft.ifftn(scipy.fft.fftn(wavefld) * kernel)  
    
    # --- Create Iref and Isamp -----------------------------------------------    

    def obtain_Iref_Isamp(self, 
                          wavefld: np.ndarray, 
                          bin_grat: np.ndarray) -> Tuple[np.ndarray, 
                                                         np.ndarray]:
        """
        Obtains the reference and sample intensity images. This is where stuff 
        actually happens!!! 

        Args:
            wavefld (np.ndarray): The input wave field to be processed.
            bin_grat (np.ndarray):The binary grating.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing a single 
                                           reference and sample intensity 
                                           image, respectively.
        """
        
        # Obtain the Fresnel kernels for each propagation distance
        kernel_grat2det = self.create_Fresnel_kernel(self.grat2det_in_m) 
        kernel_grat2samp = self.create_Fresnel_kernel(self.grat2samp_in_m) 
        kernel_samp2det = self.create_Fresnel_kernel(self.samp2det_in_m) 
        kernel_slc2slc = self.create_Fresnel_kernel(t_slc_in_pix * \
                                                    sim_pix_size_in_m)

        # Interaction of the initial wave field with the binary grating
        wavefld_ag = self.inter_wavefld_grat(wavefld)

        # --- Obtain Iref (propagation straight to the detector) --------------------------------------

        # Propagation of the wave field after the grating to the detector position
        wavefld_prop = self.prop_wavefld(wavefld_ag, kernel_grat2det) 
        del kernel_grat2det

        # Intensity of the wave fields in the detector plane
        Iref_large = np.abs(wavefld_prop)**2
        # Since we use a phase shifting grating the intensity is only shifted 
        # so we get peaks at 2 and we scale that to 1
        I_ref_large_scaled = Iref_large / 2
    
        # Binned intensity image in the detector plane
        Iref_binned = self.det.img_binning(I_ref_large_scaled)
        #del Iref_large

        # Scale 'Iref_binned' by multiplying for the number of photons per pixel
        # and apply Poisson noise
        #Iref = self.det.scale_img_with_Poisson_noise(Iref_binned, num_ph)

        #del Iref_binned
        
        # --- Obtain Isamp ----------------------------------------------------

        # Propagation of the wave field after the grating up to front end of the sample 
        wavefld_bs = self.prop_wavefld(wavefld_ag, kernel_grat2samp) 
        del kernel_grat2samp

        #Load the saved slice profiles (see main.py)
        slice_profiles_path = "slices_data.npz"
        data = np.load(slice_profiles_path)
        self.slc2d_sph_full = data['slc2d_sph_padded']
        slc2d_bkg_full = data['slc2d_bkg_padded']   

        # Assign the refractive properties to the thickness maps
        sample_compressed = self.samp.samp_with_refract_property(self.slc2d_sph_full* sim_pix_size_in_m,
                           slc2d_bkg_full * sim_pix_size_in_m)
        
        #now walk through the sample step by step
        for i in tqdm(range(self.samp.num_slc)):

            # Interaction between the wave field before the slice and the sample slice
            wavefld_as = wavefld_bs * sample_compressed[i, :]

            # Propagation of the wave field after the sample slice until the next sample slice
            wavefld_prop = self.prop_wavefld(wavefld_as, kernel_slc2slc)

            # The propagated wave field becomes the wave field before the next sample slice
            wavefld_bs = wavefld_prop
        
        del kernel_slc2slc

        # The last value of 'wavefld_bs' is the wave field after the whole sample
        wavefld_as = wavefld_bs
        del wavefld_bs

        # Propagation of the wave field after passing through the entire sample to the detector
        wavefld_prop = self.prop_wavefld(wavefld_as, kernel_samp2det)
        del wavefld_as, kernel_samp2det

        # Intensity of the wave fields in the detector plane
        Isamp_large = np.abs(wavefld_prop)**2

        I_samp_large_scaled = Isamp_large / 2

        # Binned intensity image in the detector plane
        Isamp_binned = self.det.img_binning(I_samp_large_scaled)
        #del Isamp_large

        # Scale 'Isamp_binned' by multiplying for the number of photons per 
        # pixel and apply Poisson noise
        #Isamp = self.det.scale_img_with_Poisson_noise(Isamp_binned, num_ph)
        #del Isamp_binned

        return Iref_binned, Isamp_binned
 
