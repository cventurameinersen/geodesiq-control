import os
from typing import Tuple, Optional

import numpy as np
import qutip as qt


class Fidelity:
    
    def __init__(self, pulse: np.ndarray, duration: float, fidelity_args: Optional[tuple] = None, fidelity_kwargs: Optional[dict] = None):
        """
        Initialize the PulseControl object with the control pulse and the rescaled time array as inputs.

        Parameters:
        -----------
        s: np.ndarray
            Rescaled time array (s = t/t_f) for the pulse.
        pulse: np.ndarray
            Control pulse values corresponding to the rescaled time array.
        duration: float
            Duration of the control pulse (t_f).
        pulse_args: tuple
            Positional arguments for the pulse synthesis (e.g., duration, filter parameters).
        pulse_kwargs: dict
            Keyword arguments for the pulse synthesis (e.g., duration, filter parameters).
        """

        
        self._pulse = pulse
        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(pulse))  # Rescaled time array corresponding to the pulse values 
        self._pulse_args = fidelity_args if fidelity_args is not None else ()
        self._pulse_kwargs = fidelity_kwargs if fidelity_kwargs is not None else {}



    def __call__(self):
        pass

    def time_evolution_operator(self):
        pass

    def compute_state_fidelity(self):
        pass

    def fidelity_sweep(self):
        pass


    