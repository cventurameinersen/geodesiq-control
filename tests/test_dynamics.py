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
    def __init__(self):
        # A simple 2-level system Hamiltonian function: H(x) = x * sigma_x
        self._H_func = lambda x, **kwargs: x * qt.sigmax().full()
        self._parameters = {}
        self._control_pulse = np.array([1.0, 1.0, 1.0]) # Constant pulse for simplicity
        self._control_sol = np.array([1.0, 1.0, 1.0])
        self._initial_state = 0  
        self._final_state = 1    

@pytest.fixture
def mock_hamiltonian():
    return DummyHamiltonian()

@pytest.fixture
def default_dynamics(mock_hamiltonian):
    duration = 2.0
    return Dynamics(duration=duration, hamiltonian=mock_hamiltonian)







# ------------------------------------------------------------
# Testing initialization and internal method _get_ham()
# ------------------------------------------------------------

def test_initialization(mock_hamiltonian):
    """Verify attributes are correctly extracted from the Hamiltonian object."""
    duration = 5.0
    dyn = Dynamics(duration=duration, hamiltonian=mock_hamiltonian)
    
    assert dyn._duration == 5.0
    assert len(dyn._pulse_times) == len(mock_hamiltonian._control_sol)
    assert dyn._pulse_times[-1] == 5.0  # Check proper scaling of time array


def test_get_ham(default_dynamics):
    """Test the internal QuTiP Hamiltonian constructor method."""
    args = {
        "pulse": np.array([0.0, 1.0]),
        "times": np.array([0.0, 1.0])
    }
    H_qobj = default_dynamics._get_ham(t=1.0, args=args)
    
    assert isinstance(H_qobj, qt.Qobj)




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


def test_state_fidelity_invalid_dimensions(default_dynamics):
    """Verify that array dimensional mismatches throw a ValidationError."""
    # Our dummy Hamiltonian has dimensions 2x2. Let's pass a 3-level state.
    bad_state = np.array([[1.0], [0.0], [0.0]])
    valid_state = np.array([[1.0], [0.0]])
    
    with pytest.raises(ValidationError, match="must have the same dimension"):
        default_dynamics.state_fidelity(initial_state=bad_state, final_state=valid_state)



def test_state_fidelity_type_mismatch_error(default_dynamics):
    """Passing mixed or invalid types (like strings) must raise a ValidationError."""
    with pytest.raises(ValidationError, match="must be either integers or numpy arrays"):
        default_dynamics.state_fidelity(initial_state="invalid_type", final_state=1)



def test_average_gate_fidelity(default_dynamics):
    """Ensure average gate fidelity is calculated correctly against a target operator."""
    # Target gate: Identity gate for a 2-level system
    target = qt.identity(2)
    
    gate_fid_list = default_dynamics.average_gate_fidelity(target_gate=target)
    
    assert isinstance(gate_fid_list, list)
    assert len(gate_fid_list) == len(default_dynamics._pulse_times)
    assert all(0.0 <= f < 1.0 for f in gate_fid_list)