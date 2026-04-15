import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

from ._meta import PACKAGE_NAME
from .exceptions import MissingArgsError, ValidationError, IOErrorGeodesiQ
from .warnings import NumericalStabilityWarning


class PulseControl:
    """
    A class to represent a control pulse for quantum optimal control. This class depends on the control pulse computed in the Hamiltonian class.
    The class allows for the synthesis of the pulse, filtering, and plotting of the pulse shape and its Fourier spectrum.
    """

    def __init__(self, pulse: np.ndarray, duration: float, method: Optional[str] = None, 
                 pulse_args: Optional[tuple] = None, pulse_kwargs: Optional[dict] = None):
        """
        Initialize the PulseControl object with the control pulse and the rescaled time array as inputs.

        Parameters:
        -----------
        s: np.ndarray
            Rescaled time array (s = t/t_f) for the pulse.
        pulse: np.ndarray
            Control pulse values corresponding to the rescaled time array.
        duration: floatí
            Duration of the control pulse (t_f).
        pulse_args: tuple
            Positional arguments for the pulse synthesis (e.g., duration, filter parameters).
        pulse_kwargs: dict
            Keyword arguments for the pulse synthesis (e.g., duration, filter parameters).
        """

        
        self._pulse = pulse
        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(pulse))  # Rescaled time array corresponding to the pulse values 
        self._method = method
        self._pulse_args = pulse_args if pulse_args is not None else ()
        self._pulse_kwargs = pulse_kwargs if pulse_kwargs is not None else {}



    def __call__(self):
        """
        Call the appropriate pulse processing method based on stored arguments.
        
        Returns
        -------
        output: Any
            Result from the called method.
            
        Raises
        ------
        MissingArgsError
            If no method is specified
        ValidationError
            If the method name is not recognized
        """

        if self._method is None:
            return self # Return object if no specific method was given


        # if not self._pulse_args:
        #     raise MissingArgsError("No pulse arguments specified.")

        # Map method names to actual methods
        methods = {'discretized': self.discretized_pulse, 'fourier': self.fourier_spectrum,
                   'filtered': self.filtered_pulse, 'plot': self.plot_pulse, 'export': self.export_pulse, }

        if self._method not in methods:
            raise ValidationError(f"Unknown method: {self._method}. Available methods: {list(methods.keys())}")

        method = methods[self._method]
        return method(*self._pulse_args, **self._pulse_kwargs)









# ------------------------------------------------------------
#               Methods of PulseControl class
# ------------------------------------------------------------



    def discretized_pulse(self, linear_steps: int = 4) -> Tuple[np.ndarray, np.ndarray]:
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

    def fourier_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def filtered_pulse(self, cutoff_freq: float = 1e9, filter_order: int = 3) -> np.ndarray:
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
            normalized_cutoff = 0.99
            # raise NumericalStabilityWarning(
            #     f"Normalized cutoff frequency {normalized_cutoff:.2} is too high. Setting to 0.99 to avoid instability.")

        # Design Butterworth filter coefficients
        b, a = sp.signal.butter(filter_order, normalized_cutoff, btype='low')

        # Apply the filter to the pulse using filtfilt for zero-phase filtering
        filtered_pulse = sp.signal.filtfilt(b, a, self._pulse)

        return self._pulse_times, filtered_pulse

    def plot_pulse(self, show: bool = True, **plot_kwargs) -> Tuple[Figure, Axes]:
        """
        Plot the (real-time) control pulse.

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

        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(t, pulse, **plot_kwargs)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Control Pulse')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    def export_pulse(self, filename: str = None, overwrite: bool = False) -> str:
        """
        Export (real-time) pulse data to a txt file.

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
            raise MissingArgsError("Missing filename for saving.")

        if os.path.exists(filename) and not overwrite:
            raise IOErrorGeodesiQ(f"File already exists (choose overwrite=True to remove safety check.): {filename}")

        data = np.column_stack((t, pulse))
        np.savetxt(filename, data, delimiter=",", header="t,pulse", comments="", fmt="%.8f")

        print(f"[{PACKAGE_NAME}] File {filename} saved.")

        return filename
