from typing import Optional, List

import numpy as np
import qutip as qt

from .exceptions import ValidationError
from .hamiltonian import Hamiltonian


class Dynamics:

    def __init__(self, duration: float, hamiltonian: Hamiltonian):
        """
        Initialize the Dynamics object, which depends on an instance of the Hamiltonian class. This class deals with
        observables due to the time evolution of the pulsed Hamiltonian.

        Parameters:
        -----------
        duration: float
            Duration of the control pulse (t_f).
        hamiltonian: Hamiltonian
            An instance of the Hamiltonian class containing the control pulse and system parameters.
        """

        # Attributes of the Hamiltonian instance
        self._H_func = hamiltonian._H_func
        self._parameters = hamiltonian._parameters
        self._control_pulse = hamiltonian._control_pulse
        self._control_sol = hamiltonian._control_sol
        self._initial_state = hamiltonian._initial_state
        self._final_state = hamiltonian._final_state
        self._control_name = hamiltonian._control_name

        self._duration = duration
        self._pulse_times = duration * np.linspace(0, 1, len(self._control_sol))  # Real-time array

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the time-dependent Hamiltonian using QuTiP Qobj.
        """

        control_val_t = np.interp(t, self._pulse_times, self._control_sol)
        ham_kwargs = {self._control_name: control_val_t, **self._parameters}

        return qt.Qobj(self._H_func(**ham_kwargs))

    def time_evolution_operator(self) -> List[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse Hamiltonian.
        """
        return qt.propagator(self._get_ham, self._pulse_times)

    def state_fidelity(self, initial_state: Optional[np.ndarray | int | qt.Qobj] = None,
                       final_state: Optional[np.ndarray | int | qt.Qobj] = None,
                       c_ops: Optional[List[qt.Qobj] | List[np.ndarray]] = None) -> float:
        """
        Compute the state transfer fidelity with Lindblad master equation. Depending on whether initial/final states are
        explicitly given, then the time evolution is constructed. If integers of None are given then eigenstate
        evolution is assumed.

        Parameters:
        -----------
        initial_state: Optional[np.ndarray, int or qt.Qobj]
            Initial state for pulsed time evolution.
        final_state: Optional[np.ndarray, int or qt.Qobj]
            Final state for pulsed time evolution.
        c_ops: Optional[list]
            Collapse operators (passed as a list of Qobj or np.ndarray) for the Lindblad master equation.

        """
        if c_ops is None:
            pass
        elif isinstance(c_ops, list):
            c_ops = [qt.Qobj(op) if isinstance(op, np.ndarray) else op for op in c_ops]
        else:
            raise ValidationError("Collapse operators must be provided as a list of Qobj or numpy arrays.")

        if initial_state is None and final_state is None:
            init_kwargs = {self._control_name: self._control_pulse[0], **self._parameters}
            _, init_eigenstates = qt.Qobj(self._H_func(**init_kwargs)).eigenstates()
            psi_init = init_eigenstates[self._initial_state]

            final_kwargs = {self._control_name: self._control_pulse[-1], **self._parameters}
            _, final_eigenstates = qt.Qobj(self._H_func(**final_kwargs)).eigenstates()
            psi_target = final_eigenstates[self._final_state]

        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            dummy_kwargs = {self._control_name: 0, **self._parameters}
            ham_shape = self._H_func(**dummy_kwargs).shape

            if initial_state.shape[0] != ham_shape[0] or final_state.shape[0] != ham_shape[0]:
                raise ValidationError(
                    f"Initial and final states must have the same dimension as the Hamiltonian. Shape of Hamiltonian:"
                    f" {ham_shape}. Shape of initial state: {initial_state.shape}")

            psi_init = qt.Qobj(initial_state)
            psi_target = qt.Qobj(final_state)
        elif isinstance(initial_state, qt.Qobj) and isinstance(final_state, qt.Qobj):
            psi_init = initial_state
            psi_target = final_state
        else:
            raise ValidationError(
                "Initial and final states must be either integers, numpy arrays with correct dimensions or Qobj "
                "instances.")

        options = {'store_final_state': True, 'store_states': False}

        psi_f = qt.mesolve(self._get_ham, psi_init, self._pulse_times, c_ops=c_ops, options=options).final_state
        state_fidelity = qt.fidelity(psi_target, psi_f) ** 2

        return state_fidelity

    def average_gate_fidelity(self, gate: Optional[qt.Qobj | List[qt.Qobj]] = None,
                              target_gate: Optional[qt.Qobj | np.ndarray] = None) -> List[float]:
        """
        Compute average gate fidelity given the pulsed time evolution operator in the explicit real-time duration given.

        Parameters:
        -----------
        gate: Optional[qt.Qobj]
            The resulting pulsed gate operation.
        target_gate: Optional[qt.Qobj | np.ndarray]
            The target gate operation.


        Returns:
        --------
        gate_fid: List[float]
            A list of average gate fidelities for each time step in the pulse duration.
        """

        target_gate = qt.Qobj(target_gate) if isinstance(target_gate, np.ndarray) else target_gate

        if gate is None:
            U = self.time_evolution_operator()
            gate_fid = [qt.average_gate_fidelity(oper=U[j], target=target_gate) for j in range(len(U))]
        else:
            if isinstance(gate, qt.Qobj):
                gate = [gate]  # Convert to list for consistent processing
            elif not isinstance(gate, list) or not all(isinstance(g, qt.Qobj) for g in gate):
                raise ValidationError("Gate must be a Qobj or a list of Qobj instances.")
            gate_fid = [qt.average_gate_fidelity(oper=gate[j], target=target_gate) for j in range(len(gate))]

        return gate_fid
