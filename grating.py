import numpy as np
from parameter import *

class Grating:                             
    
    def __init__(self, 
                 px_in_um: float) -> None:
        """
        Initializes an instance of the Grating class with the specified 
        parameters.

        Args:
            px_in_um (float): Grating period in the X direction, in um.
                                       
        """
        self.px_in_um = px_in_um
        self.px_in_pix = int((self.px_in_um * 1e-6) / sim_pix_size_in_m)

    def create_grating(self) -> np.ndarray:
        """
        Creates a binary grating based on the specified period.

        Returns:
            np.ndarray: The binary grating array.
        """
        x_walk=np.linspace(0, img_size_in_pix, img_size_in_pix)
        # Here the type of grating is also defined: For me it's a pi/2 phase shifting grating with 
        # 0.5 durty cycle
        E0_pi2_phase = np.where(np.mod(x_walk, self.px_in_pix) < self.px_in_pix / 2, -1/2, 0)
        grating_1d = np.exp(1j * np.pi * E0_pi2_phase)
        return grating_1d
