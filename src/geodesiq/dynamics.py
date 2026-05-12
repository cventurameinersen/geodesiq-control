import os
from typing import Tuple, Optional, Union, Callable, Any

import numpy as np
import qutip as qt

from geodesiq import Hamiltonian
from .exceptions import ValidationError


class Dynamics:
    
    def __init__(self, duration: float, hamiltonian: Hamiltonian):
        """
        Initialize the Dynamics object, which depends on an instance of the Hamiltonian class. This class deals with observables
        due to the time evolution of the pulsed Hamiltonian. 

        Parameters:
        -----------
        duration: float
            Duration of the control pulse (t_f).
        hamiltonian: Hamiltonian
            An instance of the Hamiltonian class containing the control pulse and system parameters.
        """

        # Attributes of the Hamiltonian instance
        self._hamiltonian = hamiltonian
        self._H_func = hamiltonian._H_func
        self._parameters = hamiltonian._parameters
        self._control_sol = hamiltonian._control_sol
        self._initial_state_eigenvector = hamiltonian._initial_state_eigenvector
        self._final_state_eigenvector = hamiltonian._final_state_eigenvector


        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(self._control_sol))  # Real-time array 




    def _get_ham(self, t: float, args: dict) -> qt.Qobj:
        """
        Construct the time-dependent Hamiltonian using QuTiP QObj. 
        """
        x_array = args["pulse"]
        t_array = args["times"]

        x_t = np.interp(t, t_array, x_array)

        return qt.Qobj(self._H_func(x_t, **self._parameters))



    def time_evolution_operator(self) -> np.ndarray:
        """
        Compute the time evolution operator using the pulse Hamiltonian.
        """
        return qt.propagator(self._get_ham, self._pulse_times, args={"pulse": self._control_sol, "times": self._pulse_times})
        



    def state_fidelity(self, initial_state: Optional[np.ndarray] = None, 
                               final_state: Optional[np.ndarray] = None, 
                                     c_ops: Optional[list] = []) -> float:
        """
        Compute the state transfer fidelity with Lindblad master equation. Depending on whether initial/final states are explicitly 
        given, then the time evolution is constructed. If integers of None are given then eigenstate evolution is assumed.

        Parameters:
        -----------
        initial_state: Optional[np.ndarray or int]
            Initial state for pulsed time evolution.
        final_state: Optional[np.ndarray or int]
            Final state for pulsed time evolution.
        c_ops: Optional[list]
            Collapse operators (passed as a list of Qobj or np.ndarray) for the Lindblad master equation.

        """
        if initial_state is None and final_state is None:
            psi_init = qt.Qobj(self._initial_state_eigenvector)
            psi_target = qt.Qobj(self._final_state_eigenvector)
        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            if initial_state.shape != self._H_func(0, **self._parameters).shape[0] or final_state.shape != self._H_func(0, **self._parameters).shape[0]:
                raise ValidationError("Initial and final states must have the same dimension as the Hamiltonian.")
            psi_init = qt.Qobj(initial_state)
            psi_target = qt.Qobj(final_state)
        else:
            raise ValidationError("Initial and final states must be either integers or numpy arrays with correct dimensions.")
        
        c_ops = [qt.Qobj(op) for op in c_ops]
        print(psi_init, psi_target, c_ops)
        
        H_T = qt.QobjEvo(self._get_ham, args={"pulse": self._control_sol, "times": self._pulse_times})
        psi_f = qt.mesolve(H_T, psi_init, self._pulse_times, c_ops=c_ops).states[-1]
        state_fidelity = qt.fidelity(psi_target, psi_f) ** 2
        
        return state_fidelity



    def average_gate_fidelity(self, gate: Optional[qt.Qobj] = None, target_gate: Optional[Union[qt.Qobj, np.ndarray]] = None):
        """
        Compute average gate fidelity given the pulsed time evolution operator in the explicit real-time duration given.

        Parameters:
        -----------
        gate: Optional[qt.Qobj]
            The resulting pulsed gate operation.
        target_gate: Optional[Union[qt.Qobj, np.ndarray]]
            The target gate operation.
        """

        target_gate = qt.Qobj(target_gate) if isinstance(target_gate, np.ndarray) else target_gate
        
        if gate is None:
            U = self.time_evolution_operator()
            gate_fid = [qt.average_gate_fidelity(oper=U[j], target=target_gate) for j in range(len(U))]
        else:
            gate_fid = [qt.average_gate_fidelity(oper=gate[j], target=target_gate) for j in range(len(gate))]
        
        return gate_fid
        



    