import os
from typing import Tuple, Optional, Union

import numpy as np
import qutip as qt

from geodesiq import Hamiltonian
from geodesiq.exceptions import ValidationError


class Dynamics(Hamiltonian):
    
    def __init__(self, duration: float):
        """
        Initialize the Dynamics object as a subclass of the Hamiltonian class. This class deals with observables
        due to the time evolution of the pulsed Hamiltonian. 

        Parameters:
        -----------
        duration: float
            Duration of the control pulse (t_f).
        """
        super().__init__()

        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(self._pulse))  # Real-time array 




    def _get_ham(self, t: float, qt_args: dict) -> qt.Qobj:
        """
        Construct the time-dependent Hamiltonian using QuTiP QObj. 
        """
        x_array = qt_args["pulse"]
        t_array = qt_args["times"]

        x_t = np.interp(t, t_array, x_array)

        return qt.Qobj(self.H_func(x_t, **self._parameters))



    def time_evolution_operator(self) -> np.ndarray:
        """
        Compute the time evolution operator using the pulse Hamiltonian.

        Parameters:
        -----------
        times: np.ndarray
            Real-time array (0,tfs) for all evolutions times.
        """
        return qt.propagator(self._get_ham, self._pulse_times, args={"pulse": self._pulse, "times": self._pulse_times})
        



    def state_fidelity(self, initial_state: Optional[Union[np.ndarrray, int]] = None, 
                               final_state: Optional[Union[np.ndarrray, int]] = None, 
                                     c_ops: Optional[np.ndarray] = []) -> float:
        """
        Compute the state transfer fidelity with Lindblad master equation. Depending on whether initial/final states are explicitly 
        given, then the time evolution is constructed. If integers of None are given then eigenstate evolution is assumed.

        Parameters:
        -----------
        initial_state: np.ndarray or int
            Initial state for pulsed time evolution.
        final_state: np.ndarray or int
            Final state for pulsed time evolution.
        c_ops: np.ndarray
            Collapse oprators for the Lindblad master equation.

        """
        if isinstance(initial_state, (int, None)) and isinstance(final_state, (int, None)):
            psi_init = self._initial_state_eigenvector
            psi_target = self._final_state_eigenvector
        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            if initial_state.shape != self._H_func(0, **self._parameters).shape[0] or final_state.shape != self._H_func(0, **self._parameters).shape[0]:
                raise ValidationError("Initial and final states must have the same dimension as the Hamiltonian.")
            psi_init = initial_state
            psi_target = final_state
        else:
            raise ValidationError("Initial and final states must be either integers or numpy arrays with correct dimensions.")
        
        H_T = qt.QobjEvo(self._get_ham, args={"pulse": self._pulse, "times": self._pulse_times})
        psi_f = qt.mesolve(H_T, psi_init, self._pulse_times, c_ops=c_ops).states[-1]
        state_fidelity = qt.fidelity(psi_target, psi_f) ** 2
        
        return state_fidelity



    def average_gate_fidelity(self):
        pass

    def fidelity_sweep(self, sweep_dict: dict,
                             initial_state: Optional[Union[np.ndarrray, int]] = None, 
                             final_state: Optional[Union[np.ndarrray, int]] = None, 
                             c_ops: Optional[np.ndarray] = None) -> np.ndarray:     
        """
        
        """
        pass


    