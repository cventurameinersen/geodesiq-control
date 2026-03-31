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
        
        if not callable(self.ham):
            raise ValueError("PulseControl requires a callable Hamiltonian function 'ham'.")
        
        # PulseControl parameters
        self.control_name = kwargs.get("control_name")
        self.initial = kwargs.get("initial")
        self.final = kwargs.get("final")
        self.pulse_accuracy = kwargs.get("pulse_accuracy", 1000) # Setting 1000 as default

        self.initial_state = kwargs.get("initial_state")
        self.final_state = kwargs.get("final_state")

        self.alpha = kwargs.get("alpha")
        self.beta = kwargs.get("beta")
        self.dia_alpha = kwargs.get("dia_alpha")
        self.dia_beta = kwargs.get("dia_beta")

        # Internally used variables
        self._eigenvalues = None
        self._matrix_elements = None
        self._energy_gaps = None

        self._pulse = None
        self._filtered_pulse = None
        self._pulse_times = None



        
    def summary(self):
        print("\n ------------------ PulseControl Summary ------------------ \n")
        print(f" Control Name: {self.control_name}")
        print(f" Initial Control Value: {self.initial}")
        print(f" Final Control Value: {self.final}")
        print(f" Initial State Index: {self.initial_state}")
        print(f" Final State Index: {self.final_state}")
        print(f" Alpha (Adiabatic): {self.alpha}")
        print(f" Beta (Adiabatic): {self.beta}")
        print(f" Alpha (Diabatic): {self.dia_alpha}")
        print(f" Beta (Diabatic): {self.dia_beta}")
        print("\n ---------------------------------------------------------- \n")


    def _generate_ham_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the hamiltonian and partial hamiltonian for all control values.
        If partial_hamiltonian is not callable, then the partial hamiltonian is computed numerically.

        Parameters
        ----------
        No external parameters needed.

        Returns
        -------
        hamiltonians: np.ndarray
            Array of shape (pulse_accuracy, dim, dim) containing the Hamiltonian matrices for each control value.
        partial_hamiltonians: np.ndarray
            Array of shape (pulse_accuracy, dim, dim) containing the partial Hamiltonian matrices for each control value.

        """
        def _numerical_partial_hamiltonian(control_value: float) -> np.ndarray:
            # Numerical differentiation using a 5-point stencil method for better accuracy.
            scale = np.abs(control_value) if control_value != 0 else 1.0
            delta = (np.finfo(float).eps)**(1/5) * max(scale, 1.0) 
            
            h1 = self.ham(control_value - 2*delta)
            h2 = self.ham(control_value - delta)
            h3 = self.ham(control_value + delta)
            h4 = self.ham(control_value + 2*delta)

            return (-h4 + 8*h3 - 8*h2 + h1) / (12 * delta)


        if self.ham is None:
            raise ValueError("[geodesiq] Hamiltonian function is currently not defined.")
        

        control_values = np.linspace(self.initial, self.final, self.pulse_accuracy)
        hamiltonians = np.array([self.ham(control_value) for control_value in control_values])

        if callable(self.partial_ham):
            partial_hamiltonians = np.array([self.partial_ham(control_value) for control_value in control_values])
        else:
            partial_hamiltonians = np.array([_numerical_partial_hamiltonian(control_value) for control_value in control_values])

        return hamiltonians, partial_hamiltonians




    def _compute_matrix_elements_and_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transition matrix elements and energy gaps for all control values.

        Parameters
        ----------
        No external parameters needed.

        Returns
        -------
        matrix_elements: np.ndarray
            Array of shape (pulse_accuracy, dim, dim) containing the transition matrix elements for each control value.
        energy_gaps: np.ndarray
            Array of shape (pulse_accuracy, dim) containing the energy gaps for each control value.

        """
        hamiltonians, partial_hamiltonians = self._generate_ham_arrays()

        energies, eigenvectors = np.linalg.eigh(hamiltonians)
        self._eigenvalues = energies

        energy_gaps = np.diff(energies, axis=1)
        matrix_elements = np.einsum('...ij,...jk,...kl->...il', 
                                    eigenvectors.conj().transpose(0, 2, 1), 
                                    partial_hamiltonians, 
                                    eigenvectors)

        self._energy_gaps = energy_gaps
        self._matrix_elements = matrix_elements

        return matrix_elements, energy_gaps




    def _build_diad_list(self) -> np.ndarray:
        """
        Build di-ad transition list for diabatic-adiabatic control.

        Parameters
        ----------
        

        Returns
        -------

        """
        
        pass




    def _generate_geometric_params(self) -> np.ndarray:
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
            raise ValueError("[geodesiq] Missing filename for saving.")
        
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f"[geodesiq] File already exists (choose overwrite=True to remove safety check.): {filename}")
    

        df = pd.DataFrame({"t": t, "pulse": pulse})
        df.to_csv(filename, index=False)

        return filename
