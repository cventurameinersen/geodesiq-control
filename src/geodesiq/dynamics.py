from typing import Any, List, Optional

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
        self._H_func: Any = hamiltonian._H_func
        self._parameters: dict[str, Any] = dict(hamiltonian._parameters)
        self._control_pulse: np.ndarray | None = (
            np.asarray(hamiltonian._control_pulse) if hamiltonian._control_pulse is not None else None
        )
        self._control_sol: np.ndarray | None = (
            np.asarray(hamiltonian._control_sol) if hamiltonian._control_sol is not None else None
        )
        self._initial_state: int | None = hamiltonian._initial_state
        self._final_state: int | None = hamiltonian._final_state
        self._control_name: str | None = hamiltonian._control_name

        if self._control_pulse is None or self._control_sol is None or self._control_name is None:
            raise ValidationError(
                "Dynamics requires a solved Hamiltonian with control pulse, control solution and control name set."
            )

        self._control_pulse = np.asarray(self._control_pulse, dtype=float)
        self._control_sol = np.asarray(self._control_sol, dtype=float)

        self._duration = duration
        self._pulse_times: np.ndarray = duration * np.linspace(0, 1, len(self._control_sol))  # Real-time array

    def _control_kwargs(self, control_value: float) -> dict[str, Any]:
        control_name = self._control_name
        assert control_name is not None
        return {control_name: control_value, **self._parameters}

    def _eigenstate(self, control_value: float, state_index: int) -> qt.Qobj:
        hamiltonian = qt.Qobj(self._H_func(**self._control_kwargs(control_value)))
        _, eigenstates = hamiltonian.eigenstates()
        return eigenstates[state_index]

    def _get_ham(self, t: float) -> qt.Qobj:
        """
        Construct the time-dependent Hamiltonian using QuTiP Qobj.
        """

        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()
        control_sol: list[float] = np.asarray(self._control_sol, dtype=float).tolist()
        control_val_t = float(np.interp(t, pulse_times, control_sol))

        return qt.Qobj(self._H_func(**self._control_kwargs(control_val_t)))

    def time_evolution_operator(self) -> List[qt.Qobj]:
        """
        Compute the time evolution operator using the pulse Hamiltonian.
        """
        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()
        propagator = qt.propagator(self._get_ham, pulse_times)
        if isinstance(propagator, list):
            return propagator
        return [propagator]

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
            c_ops = None
        elif isinstance(c_ops, list):
            c_ops = [qt.Qobj(op) if isinstance(op, np.ndarray) else op for op in c_ops]
        else:
            raise ValidationError("Collapse operators must be provided as a list of Qobj or numpy arrays.")

        control_pulse = self._control_pulse
        assert control_pulse is not None
        pulse_times: list[float] = np.asarray(self._pulse_times, dtype=float).tolist()

        if initial_state is None and final_state is None:
            assert self._initial_state is not None and self._final_state is not None
            psi_init = self._eigenstate(float(control_pulse[0]), self._initial_state)
            psi_target = self._eigenstate(float(control_pulse[-1]), self._final_state)

        elif isinstance(initial_state, int) and isinstance(final_state, int):
            psi_init = self._eigenstate(float(control_pulse[0]), initial_state)
            psi_target = self._eigenstate(float(control_pulse[-1]), final_state)

        elif isinstance(initial_state, np.ndarray) and isinstance(final_state, np.ndarray):
            ham_shape = self._H_func(**self._control_kwargs(0)).shape

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

        result = qt.mesolve(self._get_ham, psi_init, pulse_times, c_ops=c_ops, options=options)
        psi_f = result.final_state
        if psi_f is None:
            raise ValidationError("Time evolution did not return a final state.")

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
