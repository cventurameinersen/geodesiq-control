import os
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from geodesiq.pulses import PulseControl 
from geodesiq.exceptions import MissingArgsError, ValidationError




# ------------------------------------------------------------
# Ramp pulse as pytest.fixtures
# ------------------------------------------------------------

@pytest.fixture
def sample_pulse_data():
    """Generates a simple linear ramp for deterministic testing."""
    duration = 10.0  
    t = np.linspace(0, duration, 100)
    pulse = 20*t-10  # Linear ramp from -10 to 10 over 10 seconds
    return pulse, duration

@pytest.fixture
def default_pulse(sample_pulse_data):
    """Provides a basic PulseControl instance without an explicit method."""
    pulse, duration = sample_pulse_data
    return PulseControl(pulse=pulse, duration=duration)




# ------------------------------------------------------------
# Testing the __call__ method
# ------------------------------------------------------------

def test_call_returns_self_if_method_none(default_pulse):
    """If method=None, __call__ should return the instance itself."""
    result = default_pulse()
    assert result is default_pulse

def test_call_routes_to_discretized(sample_pulse_data):
    """Verify __call__ successfully forwards arguments to discretized_pulse."""
    pulse, duration = sample_pulse_data
    new_times, approx_sol = PulseControl(pulse, duration, method="discretized", pulse_kwargs={"linear_steps": 10})
    assert len(new_times) == 10
    assert len(approx_sol) == 10

def test_call_raises_validation_error_on_unknown_method(sample_pulse_data):
    """__call__ should catch invalid methods before execution."""
    pulse, duration = sample_pulse_data
    pc = PulseControl(pulse, duration, method="invalid_method_name")
    
    with pytest.raises(ValidationError, match="Unknown method"):
        pc()




# ------------------------------------------------------------
# Testing discretized_pulse and fourier_spectrum methods
# ------------------------------------------------------------

def test_discretized_pulse_bounds(default_pulse):
    """Ensure discretization keeps the original boundaries of the pulse time."""
    new_times, approx_sol = default_pulse.discretized_pulse(linear_steps=5)
    
    assert len(new_times) == 5
    assert len(approx_sol) == 5
    assert new_times[0] == default_pulse._pulse_times[0]
    assert new_times[-1] == default_pulse._pulse_times[-1]



def test_fourier_spectrum_with_linear_ramp(sample_pulse_data):
    """Verify FFT properties for a linear ramp from -10 to 10 over 10 seconds."""
    pulse, duration = sample_pulse_data
    pc = PulseControl(pulse, duration)
    
    frequencies, magnitude = pc.fourier_spectrum()
    assert np.all(frequencies >= 0)






# ------------------------------------------------------------
# Testing plot_pulse method
# ------------------------------------------------------------

def test_plot_pulse_without_showing(default_pulse):
    """Verify plotting logic executes and returns Matplotlib objects without blocking."""
    fig, ax = default_pulse.plot_pulse(show=False)
    
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == 'Time $t$'






# ------------------------------------------------------------
# Testing export_pulse method
# ------------------------------------------------------------

def test_export_pulse_as_npy(default_pulse, tmp_path):
    """Verify .npy file export works seamlessly using a temporary test directory."""
    
    test_file_base = os.path.join(tmp_path, "test_pulse")
    returned_filename = default_pulse.export_pulse(filename=test_file_base, file_extension="npy")
    expected_full_path = test_file_base + ".npy"
    assert os.path.exists(expected_full_path)
    
    # Load back data to verify integrity
    loaded_data = np.load(expected_full_path, allow_pickle=True).item()
    np.testing.assert_array_equal(loaded_data["pulse"], default_pulse._pulse)



def test_export_pulse_missing_filename_raises_error(default_pulse):
    """Exporting without a destination file should trip a MissingArgsError."""
    with pytest.raises(MissingArgsError, match="Missing filename"):
        default_pulse.export_pulse(filename=None)