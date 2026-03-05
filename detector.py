import numpy as np
import scipy.ndimage as nd
from numpy.linalg import lstsq
from parameter import *
import scipy.fft
import scipy.ndimage as nd

class Detector:

    def __init__(self,px_in_um) -> None:
        """
        Initializes an instance of the Detector class with the specified 
        parameters.

        Args:
            binning_factor (int): The factor by which to bin the image.
        """
        self.px_in_um = px_in_um
        self.px_in_pix = int((self.px_in_um * 1e-6) / sim_pix_size_in_m)
        self.binning_factor = binning_factor

    
    def scale_img_with_Poisson_noise(self,
                                     img: np.ndarray, 
                                     num_ph: int) -> np.ndarray:
        """
        Scale an image by multiplying it with the number of photons per pixel 
        and applying Poisson noise to account for photon statistics.

        Args:
            img (np.ndarray): The input image to be scaled.
            num_ph (int): The number of photons per pixel, representing photon 
                          counts.

        Returns:
            np.ndarray: The image with scaled pixel intensities and Poisson 
                        noise applied.
        
        """
        
        return np.random.poisson(img * num_ph)
    

    def conv_PSF_det(self, 
                     img: np.ndarray) -> np.ndarray:
        """
        Applies a Gaussian filter to simulate the blurring effect caused by 
        the detector's PSF. The Gaussian filter models the detector's PSF based 
        on its FWMH. 

        Args:
            img (np.ndarray): The input image to be filtered.

        Returns:
            np.ndarray: The resulting image after convolution with the Gaussian 
                        PSF.
        
        Note:
            For a normal distribution, the realtionship between the FWHM and
            the standar deviation (sigma) is given by:

            FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma
        """

        sigma = self.FWHM_PSF_in_pix / (2 * np.sqrt(2 * np.log(2)))  
        conv_img_PSF = nd.gaussian_filter(img, sigma)     
        
        return conv_img_PSF

    def img_binning(self, img: np.ndarray) -> np.ndarray:
        """
        Bins the input image by the specified binning factor.

        Args:
            img (np.ndarray): The input image to be binned.

        Returns:
            np.ndarray: The binned image.
        """

        n = len(img) // self.binning_factor
        img = img[:n * self.binning_factor]  # Trim to a multiple of binning_factor
        binned_img = img.reshape(n, self.binning_factor).mean(axis=1)
        return binned_img

    def create_g2(self):
        """
        Creates the G2 grating (just like the G1) but it's absorbing with the same grating period
        """
        x_walk=np.linspace(0, img_size_in_pix, img_size_in_pix)
        G2 = np.where(np.mod(x_walk, self.px_in_pix) < self.px_in_pix / 2, 0, 1)
        return G2

    
    def phasestepping_conv(self, Isamp, Iref, G2):
        """
        Convolves the G2 with the reference and sample intensities
        
        :param self: Description
        Isamp: Intensity on detector after propagating through sample
        Iref: Intensity on detector after propagating through vacuum
        G2: G2 grating created above
        """
        
        Iref_stepped = scipy.fft.ifftn(scipy.fft.fftn(Iref) * scipy.fft.fftn(G2))/ np.sum(G2)
        Isamp_stepped = scipy.fft.ifftn(scipy.fft.fftn(Isamp) * scipy.fft.fftn(G2))/ np.sum(G2)
        return Iref_stepped, Isamp_stepped
