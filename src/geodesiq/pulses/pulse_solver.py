from typing import Optional, Tuple, Dict, Any, Union
import os

import numpy as np
from scipy.integrate import romb, solve_ivp, trapezoid, simpson
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import pandas as pd







class PulseControl:
    
    def __init__(self, **kwargs):
        # Extract the functions (Hamiltonian and Partial)
        self.ham = kwargs.get("ham")
        self.partial_ham = kwargs.get("partial_ham")
        
        if not callable(self.H):
            raise ValueError("PulseControl requires a callable Hamiltonian function 'H'.")
        
        # PulseControl parameters
        self.control_name = kwargs.get("control_name")
        self.initial = kwargs.get("initial")
        self.final = kwargs.get("final")

        self.initial_state = kwargs.get("initial_state")
        self.final_state = kwargs.get("final_state")

        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")

        # Internally used variables
        self._eigenvalues = None
        self._matrix_elements = None
        self._energy_gaps = None

        self._pulse = None
        self._filtered_pulse = None
        self._pulse_times = None

        



    def _generate_hyperarrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the hamiltonian and partial hamiltonian for all control values.
        If partial_hamiltonian is not callable, then the partial hamiltonian is computed numerically.

        Parameters
        ----------
        

        Returns
        -------

        """
        def _numerical_partial_hamiltonian(ham: np.ndarray) -> np.ndarray:
            pass

        pass




    def _compute_matrix_elements_and_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transition matrix elements and energy gaps for all control values.

        Parameters
        ----------
        

        Returns
        -------

        """
        
        pass




    def _build_diad_list(self) -> np.ndarray:
        """
        Build di-ad transition list for diabatic-adiabatic control.

        Parameters
        ----------
        

        Returns
        -------

        """
        
        pass




    def generate_geometric_params(self) -> np.ndarray:
        """
        Compute the relevant quantum metric and the necessary boundary condition parameter for the solving of the ODE.

        Parameters
        ----------
        

        Returns
        -------

        """
        
        pass





    def compute_pulse(self, filter: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the control pulse given the corresponding quantum metric.

        Parameters
        ----------
        

        Returns
        -------

        """
        
        pass






    def plot_pulse(self, show: bool = True, **plot_kwargs) -> Tuple[Figure, Axes]:
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
    




    def export_csv(self, filename: str = None, overwrite: bool = False) -> str:
        """
        Export pulse data to a CSV file.

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
            raise ValueError("geodesiq: Missing filename for saving.")
        
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"geodesiq: File already exists (choose overwrite=True to remove safety check.): {filename}")
    

        df = pd.DataFrame({"t": t, "pulse": pulse})
        df.to_csv(filename, index=False)

        return filename
