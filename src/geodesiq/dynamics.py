import os
from typing import Tuple, Optional, Union

import numpy as np
import qutip as qt

from geodesiq import Hamiltonian


class Dynamics(Hamiltonian):
    
    def __init__(self, duration: float):
        """
        Initialize the Dynamics object as a subclass of the Hamiltonian class.

        Parameters:
        -----------
        duration: float
            Duration of the control pulse (t_f).
        """
        super().__init__()

        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(self._pulse))  # Real-time array 
        self._hamiltonian_parameters = self._parameters






    def _get_ham(self) -> qt.QObj:
        """
        Construct the time-dependent Hamiltonian using QuTiP QObj. 
        """

        pass



    def time_evolution_operator(self) -> np.ndarray:
        """
        Compute the time evolution operator using the pulse Hamiltonian.
        """
        pass



    def state_fidelity(self, initial_state: Optional[Union[np.ndarrray, int]] = None, 
                               final_state: Optional[Union[np.ndarrray, int]] = None, 
                               c_ops: Optional[np.ndarray] = None) -> float:
        """
        Compute the state transfer fidelity with Lindblad master equation. Depending on whether initial/final states are explicitly given, then the time evolution is constructed.
        If integers of None are given then eigenstate evolution is assumed.

        Parameters:
        -----------
        initial_state: np.ndarray or int
            Initial state for pulsed time evolution.
        final_state: np.ndarray or int
            Final state for pulsed time evolution.
        c_ops: np.ndarray
            Collapse oprators for the Lindblad master equation.

        """
        pass



    def fidelity_sweep(self, sweep_dict: dict,
                             initial_state: Optional[Union[np.ndarrray, int]] = None, 
                             final_state: Optional[Union[np.ndarrray, int]] = None, 
                             c_ops: Optional[np.ndarray] = None) -> np.ndarray:     
        """
        
        """
        pass


    