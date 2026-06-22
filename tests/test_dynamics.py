from typing import Any, cast

import numpy as np
import pytest
import qutip as qt

from geodesiq.dynamics import Dynamics
from geodesiq.exceptions import ValidationError

# ------------------------------------------------------------
# Dummy Hamiltonian() object as pytest.fixture
# ------------------------------------------------------------

class DummyHamiltonian:
    """A minimal Hamiltonian object to isolate Dynamics testing."""

    def __init__(self, control_sol=None):
        # A simple 2-level system Hamiltonian function: H(x) = x * sigma_x
        self._H_func = lambda x, **kwargs: x * qt.sigmax().full()
        self._parameters = {}
        self._control_pulse = np.array([1.0, 1.0, 1.0])  # Used to select endpoint eigenstates
        self._control_sol = np.array([1.0, 1.0, 1.0]) if control_sol is None else np.array(control_sol)
        self._initial_state = 0
        self._final_state = 1
        self._control_name = 'x'


@pytest.fixture
def mock_hamiltonian():
    return DummyHamiltonian()


@pytest.fixture
def default_dynamics(mock_hamiltonian):
    duration = 2.0
    return Dynamics(duration=duration, hamiltonian=cast(Any, mock_hamiltonian))


@pytest.fixture
def varying_dynamics():
    duration = 2.0
    varying_hamiltonian = DummyHamiltonian(control_sol=[0.0, 2.0, 4.0])
    return Dynamics(duration=duration, hamiltonian=cast(Any, varying_hamiltonian))


# ------------------------------------------------------------
# Testing initialization and internal method _get_ham()
# ------------------------------------------------------------

def test_initialization(mock_hamiltonian):
    """Verify attributes are correctly extracted from the Hamiltonian object."""
    duration = 5.0
    dyn = Dynamics(duration=duration, hamiltonian=cast(Any, mock_hamiltonian))

    assert dyn._duration == 5.0
    assert len(dyn._pulse_times) == len(mock_hamiltonian._control_sol)
    assert dyn._pulse_times[-1] == 5.0  # Check proper scaling of time array


def test_get_ham(default_dynamics):
    """Test the internal QuTiP Hamiltonian constructor method."""
    H_qobj = default_dynamics._get_ham(t=1.0)

    assert isinstance(H_qobj, qt.Qobj)


def test_get_ham_interpolation(varying_dynamics):
    """Hamiltonian values should follow linear interpolation of the solved control pulse."""
    H_qobj = varying_dynamics._get_ham(t=0.5)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]])

    np.testing.assert_allclose(H_qobj.full(), expected)


def test_initialization_requires_solved_hamiltonian(mock_hamiltonian):
    """Dynamics should reject Hamiltonian objects missing solved-control fields."""
    mock_hamiltonian._control_sol = None

    with pytest.raises(ValidationError, match="requires a solved Hamiltonian"):
        Dynamics(duration=1.0, hamiltonian=cast(Any, mock_hamiltonian))


# ------------------------------------------------------------
# Testing gate and state transfer fidelity
# ------------------------------------------------------------

def test_time_evolution_operator(default_dynamics):
    """Ensure the propagator computes successfully and returns expected elements."""
    U_list = default_dynamics.time_evolution_operator()

    assert isinstance(U_list, list)
    assert len(U_list) == len(default_dynamics._pulse_times)
    assert isinstance(U_list[0], qt.Qobj)
    assert U_list[0].isket is False  # Propagators must be operators (matrices)


def test_time_evolution_operator_starts_as_identity(default_dynamics):
    """The propagator at t=0 should be the identity operator."""
    U_list = default_dynamics.time_evolution_operator()

    np.testing.assert_allclose(U_list[0].full(), np.eye(2))


def test_time_evolution_operator_wraps_single_qobj(default_dynamics, monkeypatch):
    """When qutip.propagator returns a single Qobj, the API should still return a list."""
    monkeypatch.setattr(qt, "propagator", lambda *args, **kwargs: qt.identity(2))

    U_list = default_dynamics.time_evolution_operator()

    assert isinstance(U_list, list)
    assert len(U_list) == 1
    assert isinstance(U_list[0], qt.Qobj)


def test_state_fidelity_eigenstates(default_dynamics):
    """Test fidelity execution when initial_state/final_state are implicitly None."""
    # This triggers the if-branch using the initial/final state indices
    fidelity = default_dynamics.state_fidelity(initial_state=None, final_state=None)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity < 1.0


def test_state_fidelity_explicit_arrays(default_dynamics):
    """Test fidelity calculation with explicitly passed state vector numpy arrays."""
    # Create 2-level state vectors: ground |0> and excited |1>
    psi_i = np.array([[1.0], [0.0]])
    psi_f = np.array([[0.0], [1.0]])

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f)
    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity < 1.0


def test_state_fidelity_qobj_states(default_dynamics):
    """Qobj inputs should be accepted directly without array conversion."""
    psi_i = qt.basis(2, 0)
    psi_f = qt.basis(2, 1)

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_invalid_dimensions(default_dynamics):
    """Verify that array dimensional mismatches throw a ValidationError."""
    # Our dummy Hamiltonian has dimensions 2x2. Let's pass a 3-level state.
    bad_state = np.array([[1.0], [0.0], [0.0]])
    valid_state = np.array([[1.0], [0.0]])

    with pytest.raises(ValidationError, match="must have the same dimension"):
        default_dynamics.state_fidelity(initial_state=bad_state, final_state=valid_state)


def test_state_fidelity_type_mismatch_error(default_dynamics):
    """Passing mixed or invalid types (like strings) must raise a ValidationError."""
    with pytest.raises(ValidationError, match="either integers, numpy arrays with correct dimensions"):
        default_dynamics.state_fidelity(initial_state=cast(Any, "invalid_type"), final_state=1)


def test_state_fidelity_invalid_c_ops_container(default_dynamics):
    """Collapse operators must be provided as a list container."""
    with pytest.raises(ValidationError, match="Collapse operators must be provided as a list"):
        default_dynamics.state_fidelity(c_ops=cast(Any, "not_a_list"))


def test_state_fidelity_c_ops_numpy_array_conversion(default_dynamics):
    """Numpy-array collapse operators should be converted and accepted by mesolve."""
    psi_i = qt.basis(2, 0)
    psi_f = qt.basis(2, 1)
    c_ops = [0.1 * qt.sigmaz().full()]

    fidelity = default_dynamics.state_fidelity(initial_state=psi_i, final_state=psi_f, c_ops=c_ops)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_integer_state_indices(default_dynamics):
    """Integer state indices should select eigenstates at pulse boundaries."""
    fidelity = default_dynamics.state_fidelity(initial_state=0, final_state=1)

    assert isinstance(fidelity, float)
    assert 0.0 <= fidelity <= 1.0


def test_state_fidelity_raises_when_mesolve_returns_no_final_state(default_dynamics, monkeypatch):
    """A mesolve result without final_state should raise a ValidationError."""

    class DummyResult:
        final_state = None

    monkeypatch.setattr(qt, "mesolve", lambda *args, **kwargs: DummyResult())

    with pytest.raises(ValidationError, match="did not return a final state"):
        default_dynamics.state_fidelity(initial_state=qt.basis(2, 0), final_state=qt.basis(2, 1))


def test_average_gate_fidelity(default_dynamics):
    """Ensure average gate fidelity is calculated correctly against a target operator."""
    # Target gate: Identity gate for a 2-level system
    target = qt.identity(2)

    gate_fid_list = default_dynamics.average_gate_fidelity(target_gate=target)

    assert isinstance(gate_fid_list, list)
    assert len(gate_fid_list) == len(default_dynamics._pulse_times)
    assert all(0.0 <= f <= 1.0 for f in gate_fid_list)


def test_average_gate_fidelity_accepts_numpy_target(default_dynamics):
    """A numpy target gate should be promoted to Qobj internally."""
    gate_fid_list = default_dynamics.average_gate_fidelity(target_gate=np.eye(2))

    assert isinstance(gate_fid_list, list)
    assert len(gate_fid_list) == len(default_dynamics._pulse_times)
    assert all(0.0 <= f <= 1.0 for f in gate_fid_list)


def test_average_gate_fidelity_single_qobj_gate(default_dynamics):
    """A single Qobj gate should be accepted and treated as a one-item list."""
    gate_fid_list = default_dynamics.average_gate_fidelity(gate=qt.identity(2), target_gate=qt.identity(2))

    assert gate_fid_list == [pytest.approx(1.0)]


def test_average_gate_fidelity_list_of_qobj_gate(default_dynamics):
    """A list of Qobj gates should be accepted directly."""
    gate_fid_list = default_dynamics.average_gate_fidelity(gate=[qt.identity(2)], target_gate=qt.identity(2))

    assert gate_fid_list == [pytest.approx(1.0)]


def test_average_gate_fidelity_invalid_gate_type(default_dynamics):
    """Invalid gate types should raise a ValidationError."""
    with pytest.raises(ValidationError, match="Gate must be a Qobj or a list of Qobj instances"):
        default_dynamics.average_gate_fidelity(gate=cast(Any, [qt.identity(2), np.eye(2)]), target_gate=qt.identity(2))
