from typing import Optional, Tuple, Dict, Any, Union
import os

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes






class PulseControl:
    """
    A class to represent a control pulse for quantum optimal control. This class depends on the control pulse computed in the Hamiltonian class.
    The class allows for the synthesis of the pulse, filtering, and plotting of the pulse shape and its Fourier spectrum.
    """

    def __init__(self, s: np.ndarray, pulse: np.ndarray, *pulse_args: Any, **pulse_kwargs: Any):
        """
        Initialize the PulseControl object with the control pulse and the rescaled time array as inputs.

        Parameters:
        -----------
        s: np.ndarray
            Rescaled time array (s = t/t_f) for the pulse.
        pulse: np.ndarray
            Control pulse values corresponding to the rescaled time array.
        pulse_args: tuple
            Positional arguments for the pulse synthesis (e.g., duration, filter parameters).
        pulse_kwargs: dict
            Keyword arguments for the pulse synthesis (e.g., duration, filter parameters).
        """

        self._pulse_times = s
        self._pulse = pulse
        self._pulse_args = pulse_args
        self._pulse_kwargs = pulse_kwargs




    def __call__(self):
        """
        Call the appropriate pulse processing method based on stored arguments.
        Uses _pulse_args[0] as the method name and remaining args/kwargs as parameters.
        
        Returns
        -------
        output: Any
            Result from the called method.
            
        Raises
        ------
        ValueError
            If no method is specified or if the method name is not recognized.
        """
        if not self._pulse_args:
            raise ValueError("[geodesiq] No method specified in pulse_args.")
        
        method_name = self._pulse_args[0]
        remaining_args = self._pulse_args[1:]
        
        # Map method names to actual methods
        methods = {
            'discretized': self._discretized_pulse,
            'fourier': self._fourier_spectrum,
            'filter': self._filter_pulse,
            'plot': self._plot_pulse,
            'export': self._export_pulse,
        }
        
        if method_name not in methods:
            raise ValueError(f"[geodesiq] Unknown method: {method_name}. Available methods: {list(methods.keys())}")
        
        method = methods[method_name]
        return method(*remaining_args, **self._pulse_kwargs)



    
    def _discretized_pulse(self, linear_steps: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains the piecewise linear function from control pulse.
        
               
        Parameters
        ----------
        linear_steps: int
            Number of linear steps to use for the piecewise linear approximation of the control pulse.

        Returns
        -------
        new_s: np.ndarray
            Rescaled time array for the piecewise linear approximation.
        approx_sol: np.ndarray
            Control pulse values corresponding to the new rescaled time array for the piecewise linear approximation.
        
        """

        piecewise_linear = interp1d(self._pulse_times, self._pulse, kind='linear', fill_value="extrapolate")  

        new_s = np.linspace(self._pulse_times[0], self._pulse_times[-1], linear_steps)
        approx_sol = piecewise_linear(new_s)
        
        return new_s, approx_sol




    def _fourier_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Fourier spectrum of the control pulse.

        Returns
        -------
        frequencies: np.ndarray
            Positive frequencies corresponding to the Fourier spectrum.
        magnitude: np.ndarray
            Magnitude spectrum of the control pulse (absolute value of the FFT).

        """

        # Compute the sampling rate from the rescaled time array
        dt = np.mean(np.diff(self._pulse_times))
        
        # Compute FFT of the pulse and corresponding frequencies
        fft_values = np.fft.fft(self._pulse)
        freqs = np.fft.fftfreq(len(self._pulse), dt)
        
        # Get only positive frequencies and their magnitudes
        positive_freq_idx = freqs >= 0
        frequencies = freqs[positive_freq_idx]
        magnitude = np.abs(fft_values[positive_freq_idx])
        
        
        return frequencies, magnitude




    def _filter_pulse(self, cutoff_freq: float = 1e9, filter_order: int = 3) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to the control pulse.

        Parameters
        ----------
        cutoff_freq: float
            Cutoff frequency in Hz (default: 1 GHz = 1e9 Hz).
        filter_order: int
            Order of the Butterworth filter.

        Returns
        -------
        filtered_pulse: np.ndarray
            Returns the (butterworth-)filtered control pulse.

        """

        # Compute the sampling rate from the rescaled time array
        dt = np.mean(np.diff(self._pulse_times))
        sampling_rate = 1.0 / dt
        nyquist_freq = sampling_rate / 2.0
        
        # Normalize the cutoff frequency to the Nyquist frequency
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Ensure the normalized cutoff frequency is within valid range
        if normalized_cutoff >= 1.0:
            print(f"[geodesiq] Warning: Normalized cutoff frequency {normalized_cutoff:.2f} is too high. Setting to 0.99 to avoid instability.")
            normalized_cutoff = 0.99
        
        # Design Butterworth filter coefficients
        b, a = sp.signal.butter(filter_order, normalized_cutoff, btype='low')

        # Apply the filter to the pulse using filtfilt for zero-phase filtering
        filtered_pulse = sp.signal.filtfilt(b, a, self._pulse)

        return filtered_pulse




    def _plot_pulse(self, show: bool = True, **plot_kwargs) -> Tuple[Figure, Axes]:
        """
        Plot the (rescaled) control pulse.

        Parameters
        ----------
        show: bool
            Show plot before possibly adding plot_kwargs
        plot_kwargs: dict
            Dictionary of style changes to ax.plot()

        Returns
        -------
        fig, ax: Figure, Axes
            Figure and axes for the construction of a custom plot.

        """

        t, pulse = self._pulse_times, self._pulse

        fig, ax = plt.subplots()
        ax.plot(t, pulse, **plot_kwargs)
        ax.set_xlabel('Rescaled Time $t/t_f$')
        ax.set_ylabel('Control Pulse')

        if show:
            plt.show()

        return fig, ax
    




    def _export_pulse(self, filename: str = None, overwrite: bool = False) -> str:
        """
        Export pulse data to a txt file.

        Parameters
        ----------
        filename: str
            Name for the data file saved.
        overwrite: bool
            Ensures accidental overwrites.

        Returns
        -------
        filename: str

        """

        t, pulse = self._pulse_times, self._pulse

        if filename is None:
            raise ValueError("[geodesiq] Missing filename for saving.")
        
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"[geodesiq] File already exists (choose overwrite=True to remove safety check.): {filename}")
    

        data = np.column_stack((t, pulse))
        np.savetxt(filename, data, delimiter=",", header="t,pulse", comments="", fmt="%.8f")

        print(f"[geodesiq] File {filename} saved.")

        return filename