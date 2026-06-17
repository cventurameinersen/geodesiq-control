import numpy as np
import pytest
from scipy.integrate import solve_ivp

from geodesiq import Hamiltonian, InvalidControlParameterError, ImmutableConfigurationError, \
    MissingControlParameterError, ValidationError


# ---------------------------------------------------------------------------
# Helpers – simple 2×2 Landau-Zener model:  H = [[lam, delta], [delta, -lam]]
# ---------------------------------------------------------------------------

def lz_hamiltonian(lam, delta=1.0):
    return np.array([[lam, delta], [delta, -lam]])


def lz_partial(lam, delta=1.0):
    """∂H/∂λ is constant for the LZ model."""
    return np.array([[1.0, 0.0], [0.0, -1.0]])


def _get_pyplot():
    """Lazy matplotlib import so plotting tests can be skipped when optional dependency is missing."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bare_ham():
    """A Hamiltonian instance with no partial derivative provided."""
    return Hamiltonian(lz_hamiltonian)


@pytest.fixture
def ham_with_partial():
    """A Hamiltonian instance with an analytical partial derivative."""
    return Hamiltonian(lz_hamiltonian, partial_H_func=lz_partial)


@pytest.fixture
def configured_ham(ham_with_partial):
    """A fully configured Hamiltonian ready for solve_problem()."""
    ham_with_partial.set_parameters(delta=0.5)
    ham_with_partial.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0,
                                 beta=2.0, num_steps=2 ** 8 + 1, )
    return ham_with_partial


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_stores_H_func(self, bare_ham):
        assert bare_ham.H_func is lz_hamiltonian

    def test_init_numerical_partial_flag_when_no_partial(self, bare_ham):
        assert bare_ham._flag_numerical_partial_H is True

    def test_init_analytical_partial_flag_when_partial_given(self, ham_with_partial):
        assert ham_with_partial._flag_numerical_partial_H is False

    def test_init_flags_all_false(self, bare_ham):
        assert bare_ham._flags["eigenproblem_solved"] is False
        assert bare_ham._flags["metric_computed"] is False
        assert bare_ham._flags["ode_solved"] is False


# ---------------------------------------------------------------------------
# Property setters – validation
# ---------------------------------------------------------------------------

class TestSetterValidation:
    def test_control_name_rejects_non_string(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="string"):
            bare_ham.control_name = 42

    def test_pulse_initial_rejects_non_number(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="number"):
            bare_ham.pulse_initial = "abc"

    def test_pulse_final_rejects_non_number(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="number"):
            bare_ham.pulse_final = [1, 2]

    def test_initial_state_rejects_non_int(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="integer"):
            bare_ham.initial_state = 1.5

    def test_alpha_rejects_non_number(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="number"):
            bare_ham.alpha = "two"

    def test_beta_rejects_non_number(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="number"):
            bare_ham.beta = "two"

    def test_num_steps_rejects_non_positive(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="positive integer"):
            bare_ham.num_steps = -5

    def test_num_steps_rejects_float(self, bare_ham):
        with pytest.raises(InvalidControlParameterError, match="positive integer"):
            bare_ham.num_steps = 3.5


# ---------------------------------------------------------------------------
# H_func / partial_H_func immutability
# ---------------------------------------------------------------------------

class TestFuncImmutability:
    def test_H_func_cannot_be_overwritten(self, bare_ham):
        with pytest.raises(ImmutableConfigurationError, match="already set"):
            bare_ham.H_func = lambda lam: np.eye(2)

    def test_partial_H_func_cannot_be_overwritten(self, ham_with_partial):
        with pytest.raises(ImmutableConfigurationError, match="already set"):
            ham_with_partial.partial_H_func = lambda lam: np.eye(2)

    def test_partial_H_func_can_be_set_once_if_missing(self, bare_ham):
        bare_ham.partial_H_func = lz_partial
        assert bare_ham.partial_H_func is lz_partial
        assert bare_ham._flag_numerical_partial_H is False


# ---------------------------------------------------------------------------
# set_parameters
# ---------------------------------------------------------------------------

class TestSetParameters:
    def test_set_parameters_stores_values(self, bare_ham):
        bare_ham.set_parameters(delta=0.5, gamma=1.0)
        assert bare_ham._parameters == {"delta": 0.5, "gamma": 1.0}

    def test_set_parameters_updates_incrementally(self, bare_ham):
        bare_ham.set_parameters(delta=0.5)
        bare_ham.set_parameters(gamma=1.0)
        assert bare_ham._parameters == {"delta": 0.5, "gamma": 1.0}

    def test_set_parameters_same_values_keeps_eigenproblem_flag(self, configured_ham):
        configured_ham._flags["eigenproblem_solved"] = True
        configured_ham.set_parameters(delta=0.5)
        assert configured_ham._flags["eigenproblem_solved"] is True

    def test_set_parameters_resets_eigenproblem_flag(self, configured_ham):
        configured_ham._flags["eigenproblem_solved"] = True
        configured_ham.set_parameters(delta=2.0)
        assert configured_ham._flags["eigenproblem_solved"] is False


# ---------------------------------------------------------------------------
# set_control
# ---------------------------------------------------------------------------

class TestSetControl:
    def test_set_control_assigns_all_values(self, bare_ham):
        bare_ham.set_control(control_name="lam", pulse_initial=-3.0, pulse_final=3.0, initial_state=0, alpha=2.0,
                             beta=2.0, num_steps=129, )
        assert bare_ham.control_name == "lam"
        assert bare_ham.pulse_initial == -3.0
        assert bare_ham.pulse_final == 3.0
        assert bare_ham.initial_state == 0
        assert bare_ham.alpha == 2.0
        assert bare_ham.beta == 2.0
        assert bare_ham.num_steps == 129

    def test_set_control_partial_update_preserves_others(self, bare_ham):
        bare_ham.set_control(control_name="lam", pulse_initial=-3.0, pulse_final=3.0, initial_state=0, alpha=2.0,
                             beta=2.0, num_steps=129)
        bare_ham.set_control(alpha=4.0)  # only update alpha
        assert bare_ham.control_name == "lam"
        assert bare_ham.alpha == 4.0


# ---------------------------------------------------------------------------
# _check_control_parameters
# ---------------------------------------------------------------------------

class TestCheckControlParameters:
    def test_missing_all_raises(self, bare_ham):
        with pytest.raises(MissingControlParameterError, match="Missing control"):
            bare_ham._check_control_parameters()

    def test_missing_single_param_mentioned(self, bare_ham):
        bare_ham.set_control(control_name="lam", pulse_initial=-1.0, pulse_final=1.0, initial_state=0, beta=2.0)
        with pytest.raises(MissingControlParameterError, match="alpha"):
            bare_ham._check_control_parameters()

    def test_no_error_when_all_set(self, configured_ham):
        configured_ham._check_control_parameters()  # should not raise


# ---------------------------------------------------------------------------
# Flag reset cascade
# ---------------------------------------------------------------------------

class TestFlagResets:
    def test_changing_control_name_resets_eigenproblem(self, configured_ham):
        configured_ham._flags["eigenproblem_solved"] = True
        configured_ham.control_name = "x"
        assert configured_ham._flags["eigenproblem_solved"] is False

    def test_changing_alpha_resets_metric(self, configured_ham):
        configured_ham._flags["metric_computed"] = True
        configured_ham.alpha = 3.0
        assert configured_ham._flags["metric_computed"] is False

    def test_changing_num_steps_resets_eigenproblem(self, configured_ham):
        configured_ham._flags["eigenproblem_solved"] = True
        configured_ham.num_steps = 513
        assert configured_ham._flags["eigenproblem_solved"] is False
        assert configured_ham._flags["metric_computed"] is False
        assert configured_ham._flags["ode_solved"] is False


# ---------------------------------------------------------------------------
# __str__ / __repr__
# ---------------------------------------------------------------------------

class TestStringRepresentations:
    def test_str_contains_control_name(self, configured_ham):
        s = str(configured_ham)
        assert "lam" in s

    def test_repr_contains_summary_banner(self, configured_ham):
        r = repr(configured_ham)
        assert "Hamiltonian Control Summary" in r


# ---------------------------------------------------------------------------
# solve_problem (integration-level)
# ---------------------------------------------------------------------------

class TestSolveProblem:
    def test_solve_problem_sets_all_flags(self, configured_ham):
        configured_ham.solve_problem()
        assert configured_ham._flags["eigenproblem_solved"] is True
        assert configured_ham._flags["metric_computed"] is True
        assert configured_ham._flags["ode_solved"] is True

    def test_solve_problem_populates_energies(self, configured_ham):
        configured_ham.solve_problem()
        assert configured_ham._energies is not None
        assert configured_ham._energies.shape[0] == configured_ham.num_steps

    def test_solve_problem_control_sol_matches_boundaries(self, configured_ham):
        configured_ham.solve_problem()
        sol = configured_ham._control_sol
        assert sol is not None
        np.testing.assert_allclose(sol[0], configured_ham.pulse_initial, atol=1e-4)
        np.testing.assert_allclose(sol[-1], configured_ham.pulse_final, atol=1e-1)

    def test_solve_problem_recomputes_ode_when_pulse_accuracy_changes(self, configured_ham):
        configured_ham.solve_problem(pulse_accuracy=25)
        first_s = configured_ham._s.copy()
        first_sol = configured_ham._control_sol.copy()
        assert configured_ham._previous_pulse_accuracy == 25

        configured_ham.solve_problem(pulse_accuracy=60)

        assert configured_ham._flags["ode_solved"] is True
        assert configured_ham._previous_pulse_accuracy == 60
        assert first_s.shape == (25,)
        assert first_sol.shape == (25,)
        assert configured_ham._s.shape == (60,)
        assert configured_ham._control_sol.shape == (60,)

    def test_solve_problem_with_numerical_partial(self):
        """When no analytical partial is given, the numerical derivative is used."""
        ham = Hamiltonian(lz_hamiltonian)  # no partial_H_func
        ham.set_parameters(delta=0.5)
        ham.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0, beta=2.0,
                        num_steps=2 ** 8 + 1, )
        ham.solve_problem()
        assert ham._flags["ode_solved"] is True
        assert ham._control_sol is not None

    def test_control_sol_property_triggers_solve(self, configured_ham):
        """Accessing `control_sol` property should auto-solve if needed."""
        assert configured_ham._flags["ode_solved"] is False
        sol = configured_ham.control_sol
        assert sol is not None
        assert configured_ham._flags["ode_solved"] is True

    def test_solve_problem_accepts_custom_solver_function_and_kwargs(self, configured_ham):
        calls = {}

        def custom_solver(fun, t_span, y0, t_eval=None, **kwargs):
            calls["kwargs"] = kwargs
            return solve_ivp(fun, t_span, y0, t_eval=t_eval, **kwargs)

        configured_ham.solve_problem(solver=custom_solver,
                                     solver_kwargs={"method": "RK23", "rtol": 1e-5, "atol": 1e-7}, )

        assert calls["kwargs"]["method"] == "RK23"
        assert np.isclose(calls["kwargs"]["rtol"], 1e-5)
        assert np.isclose(calls["kwargs"]["atol"], 1e-7)

    def test_solve_problem_accepts_custom_metric_integrator_and_kwargs(self, configured_ham):
        calls = {}

        def custom_metric_integrator(values, dx=1.0, scale=1.0):
            calls["dx"] = dx
            calls["scale"] = scale
            return np.trapezoid(values, dx=dx) * scale

        configured_ham.solve_problem(metric_integrator=custom_metric_integrator,
                                     metric_integrator_kwargs={"scale": 1.0}, )

        dx = float(np.abs(configured_ham._control_pulse[1] - configured_ham._control_pulse[0]))
        expected = np.trapezoid(np.sqrt(configured_ham._metric_tensor), dx=dx)

        assert np.isclose(calls["dx"], dx)
        assert np.isclose(calls["scale"], 1.0)
        assert np.isclose(configured_ham._a_tilde, expected)

    def test_solve_problem_recomputes_ode_when_solver_changes(self, configured_ham):
        configured_ham.solve_problem()
        baseline = configured_ham._control_sol.copy()

        def synthetic_solver(_, __, y0, t_eval=None, **kwargs):
            return t_eval, np.array([np.full_like(t_eval, y0[0] + kwargs.get("offset", 0.0), dtype=float)])

        configured_ham.solve_problem(solver=synthetic_solver, solver_kwargs={"offset": 0.123})

        assert np.allclose(configured_ham._control_sol, configured_ham.pulse_initial + 0.123)
        assert not np.allclose(configured_ham._control_sol, baseline)

    def test_solve_problem_rejects_non_callable_solver(self, configured_ham):
        with pytest.raises(ValidationError, match="solver must be a callable"):
            configured_ham.solve_problem(solver="RK45")

    def test_solve_problem_rejects_non_callable_metric_integrator(self, configured_ham):
        with pytest.raises(ValidationError, match="metric_integrator must be a callable"):
            configured_ham.solve_problem(metric_integrator="romb")


# ---------------------------------------------------------------------------
# eigenenergies / control_pulse properties
# ---------------------------------------------------------------------------

class TestEigenGetters:
    def test_eigenenergies_property_triggers_eigenproblem_solve(self, configured_ham):
        assert configured_ham._flags["eigenproblem_solved"] is False

        energies = configured_ham.eigenenergies

        assert energies is not None
        assert configured_ham._flags["eigenproblem_solved"] is True
        assert energies.shape[0] == configured_ham.num_steps

    def test_control_pulse_property_triggers_eigenproblem_solve(self, configured_ham):
        assert configured_ham._flags["eigenproblem_solved"] is False

        pulse = configured_ham.control_pulse

        assert pulse is not None
        assert configured_ham._flags["eigenproblem_solved"] is True
        np.testing.assert_allclose(pulse, np.linspace(configured_ham.pulse_initial, configured_ham.pulse_final,
                                                      configured_ham.num_steps), )

    def test_eigenenergies_property_raises_when_eigensystem_controls_missing(self, bare_ham):
        with pytest.raises(MissingControlParameterError, match="eigensystem"):
            _ = bare_ham.eigenenergies

    def test_control_pulse_property_raises_when_eigensystem_controls_missing(self, bare_ham):
        with pytest.raises(MissingControlParameterError, match="eigensystem"):
            _ = bare_ham.control_pulse


# ---------------------------------------------------------------------------
# Property setters – None keeps previous value
# ---------------------------------------------------------------------------

class TestSetterNoneNoOp:
    def test_control_name_none_keeps_value(self, configured_ham):
        configured_ham.control_name = None
        assert configured_ham.control_name == "lam"

    def test_pulse_initial_none_keeps_value(self, configured_ham):
        configured_ham.pulse_initial = None
        assert configured_ham.pulse_initial == -5.0

    def test_pulse_final_none_keeps_value(self, configured_ham):
        configured_ham.pulse_final = None
        assert configured_ham.pulse_final == 5.0

    def test_initial_state_none_keeps_value(self, configured_ham):
        configured_ham.initial_state = None
        assert configured_ham.initial_state == 0

    def test_alpha_none_keeps_value(self, configured_ham):
        configured_ham.alpha = None
        assert configured_ham.alpha == 2.0

    def test_beta_none_keeps_value(self, configured_ham):
        configured_ham.beta = None
        assert configured_ham.beta == 2.0


# ---------------------------------------------------------------------------
# set_parameters – overwrite existing key
# ---------------------------------------------------------------------------

class TestSetParametersOverwrite:
    def test_overwrite_existing_key(self, bare_ham):
        bare_ham.set_parameters(delta=0.5)
        bare_ham.set_parameters(delta=1.0)
        assert bare_ham._parameters["delta"] == 1.0

    def test_overwrite_preserves_other_keys(self, bare_ham):
        bare_ham.set_parameters(delta=0.5, gamma=2.0)
        bare_ham.set_parameters(delta=1.0)
        assert bare_ham._parameters == {"delta": 1.0, "gamma": 2.0}


# ---------------------------------------------------------------------------
# _solve_eigenproblem – caching
# ---------------------------------------------------------------------------

class TestSolveEigenproblemCaching:
    def test_eigenproblem_skips_when_flag_set(self, configured_ham):
        """Calling _solve_eigenproblem twice should reuse the cached result."""
        configured_ham._control_pulse = np.linspace(-5.0, 5.0, configured_ham.num_steps)
        configured_ham._solve_eigenproblem()
        energies_first = configured_ham._energies.copy()

        # Second call should skip (flag already True) and keep the same data
        configured_ham._solve_eigenproblem()
        np.testing.assert_array_equal(configured_ham._energies, energies_first)


# ---------------------------------------------------------------------------
# _compute_metric_tensor
# ---------------------------------------------------------------------------

class TestComputeMetricTensor:
    def test_metric_tensor_populated(self, configured_ham):
        configured_ham.solve_problem()
        assert configured_ham._metric_tensor is not None
        assert configured_ham._metric_tensor.shape[0] == configured_ham.num_steps

    def test_metric_tensor_non_negative(self, configured_ham):
        configured_ham.solve_problem()
        assert np.all(configured_ham._metric_tensor >= 0)

    def test_a_tilde_positive(self, configured_ham):
        configured_ham.solve_problem()
        assert configured_ham._a_tilde > 0


# ---------------------------------------------------------------------------
# Numerical vs analytical partial derivative
# ---------------------------------------------------------------------------

class TestNumericalPartialAccuracy:
    def test_numerical_partial_matches_analytical(self):
        """The numerical derivative should closely match the analytical one."""
        ham_num = Hamiltonian(lz_hamiltonian)
        ham_num.set_parameters(delta=0.5)
        ham_num.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0,
                            beta=2.0, num_steps=2 ** 8 + 1)

        ham_ana = Hamiltonian(lz_hamiltonian, partial_H_func=lz_partial)
        ham_ana.set_parameters(delta=0.5)
        ham_ana.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0,
                            beta=2.0, num_steps=2 ** 8 + 1)

        ham_num.solve_problem()
        ham_ana.solve_problem()

        np.testing.assert_allclose(ham_num._control_sol, ham_ana._control_sol, atol=1e-2,
                                   err_msg="Numerical and analytical solutions should be close")


# ---------------------------------------------------------------------------
# control_sol monotonicity
# ---------------------------------------------------------------------------

class TestControlSolMonotonicity:
    def test_control_sol_monotonically_increasing(self, configured_ham):
        """For pulse_initial < pulse_final the solution should be non-decreasing."""
        configured_ham.solve_problem()
        diffs = np.diff(configured_ham._control_sol)
        assert np.all(diffs >= -1e-8)


# ---------------------------------------------------------------------------
# plot_eigenvalues
# ---------------------------------------------------------------------------

class TestPlotEigenvalues:
    def test_plot_eigenvalues_creates_figure_and_lines(self, configured_ham):
        plt = _get_pyplot()
        fig, ax = configured_ham.plot_eigenvalues()

        assert fig is not None
        assert ax is not None
        assert len(ax.lines) == 2

        plt.close(fig)

    def test_plot_eigenvalues_uses_user_figure_axis_and_kwargs(self, configured_ham):
        plt = _get_pyplot()
        fig, ax = plt.subplots()

        out_fig, out_ax = configured_ham.plot_eigenvalues(fig=fig, ax=ax, linestyle="--", color="black")

        assert out_fig is fig
        assert out_ax is ax
        assert all(line.get_linestyle() == "--" for line in ax.lines)
        assert all(line.get_color() == "black" for line in ax.lines)

        plt.close(fig)

    def test_plot_eigenvalues_allows_custom_axis_labels_and_title(self, configured_ham):
        plt = _get_pyplot()
        fig, ax = configured_ham.plot_eigenvalues(xlabel="lambda", ylabel="Eigenenergy", title="Spectrum")

        assert ax.get_xlabel() == "lambda"
        assert ax.get_ylabel() == "Eigenenergy"
        assert ax.get_title() == "Spectrum"

        plt.close(fig)

    def test_plot_eigenvalues_keeps_default_labels_when_not_overridden(self, configured_ham):
        plt = _get_pyplot()
        fig, ax = configured_ham.plot_eigenvalues()

        assert ax.get_xlabel() == configured_ham.control_name
        assert ax.get_ylabel() == "Energy"
        assert ax.get_title() == "Hamiltonian Eigenvalues"

        plt.close(fig)

    def test_plot_eigenvalues_reuses_cached_calculation(self, bare_ham, monkeypatch):
        _get_pyplot()
        bare_ham.set_parameters(delta=0.5)
        bare_ham.set_control(control_name="lam", pulse_initial=-5.0, pulse_final=5.0, initial_state=0, alpha=2.0,
                             beta=2.0, num_steps=33)

        call_counter = {"count": 0}
        original_eigh = np.linalg.eigh

        def counting_eigh(*args, **kwargs):
            call_counter["count"] += 1
            return original_eigh(*args, **kwargs)

        monkeypatch.setattr(np.linalg, "eigh", counting_eigh)

        fig1, _ = bare_ham.plot_eigenvalues()
        fig2, _ = bare_ham.plot_eigenvalues()

        assert call_counter["count"] == 1

        plt = _get_pyplot()
        plt.close(fig1)
        plt.close(fig2)


# ---------------------------------------------------------------------------
# solve_problem raises on missing parameters
# ---------------------------------------------------------------------------

class TestSolveProblemErrors:
    def test_solve_problem_raises_without_control(self, bare_ham):
        with pytest.raises(MissingControlParameterError, match="Missing control"):
            bare_ham.solve_problem()


# ---------------------------------------------------------------------------
# synthesize_pulse
# ---------------------------------------------------------------------------

# ToDo: Recover when pulse is properly implemented
# class TestSynthesizePulse:
#     def test_synthesize_pulse_returns_pulse_control(self, configured_ham):
#         pulse = configured_ham.synthesize_pulse(duration=1.0)
#         assert pulse is not None
#
#     def test_pulse_property_triggers_solve(self, configured_ham):
#         """Accessing `pulse` property should auto-solve if needed."""
#         assert configured_ham._flags["ode_solved"] is False
#         pulse = configured_ham.pulse
#         assert pulse is not None
#         assert configured_ham._flags["ode_solved"] is True


# ---------------------------------------------------------------------------
# _generate_summary content
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    def test_summary_contains_set_indicators(self, configured_ham):
        summary = configured_ham._generate_summary()
        assert "✅ set" in summary  # H_func is set

    def test_summary_shows_parameters(self, configured_ham):
        summary = configured_ham._generate_summary()
        assert "delta" in summary

    def test_summary_bare_ham_shows_not_set(self, bare_ham):
        summary = bare_ham._generate_summary()
        assert "❌ not set" in summary


# ---------------------------------------------------------------------------
# Test analytical results for Landau-Zener model
# ---------------------------------------------------------------------------

class TestLandauZenerAnalytical:
    def test_linear(self):
        """For the LZ model, the optimal control for n_+ = 0 is a linear ramp from pulse_initial to pulse_final."""
        ham = Hamiltonian(lz_hamiltonian)

        delta = 0.5
        pulse_0 = -5.0
        pulse_f = -pulse_0

        ham.set_parameters(delta=delta)
        ham.set_control(control_name="lam", pulse_initial=pulse_0, pulse_final=pulse_f, initial_state=0, alpha=0,
                        beta=0, num_steps=2 ** 8 + 1)
        ham.solve_problem()
        s = ham._s

        expected_control = pulse_0 * (1 - 2 * s)

        np.testing.assert_allclose(ham.control_sol, expected_control, atol=1e-2,
                                   err_msg="Optimal control for LZ n_+ = 0 is wrong")

    def test_n1(self):
        """For the LZ model, the optimal control for n_+ = 1 has a closed form solution"""
        ham = Hamiltonian(lz_hamiltonian)

        delta = 0.7
        pulse_0 = -10.0
        pulse_f = -pulse_0

        ham.set_parameters(delta=delta)
        ham.set_control(control_name="lam", pulse_initial=pulse_0, pulse_final=pulse_f, initial_state=0, alpha=1,
                        beta=1, num_steps=2 ** 8 + 1)
        ham.solve_problem()
        s = ham._s

        expected_control = delta * np.sinh((1 - 2 * s) * np.arcsinh(pulse_0 / delta))

        np.testing.assert_allclose(ham.control_sol, expected_control, atol=1e-2,
                                   err_msg="Optimal control for LZ n_+ = 1 is wrong")

    def test_geoQUAD(self):
        """For the LZ model, the optimal control for n_+ = 2 has a closed form solution"""
        ham = Hamiltonian(lz_hamiltonian)

        delta = 1.4
        pulse_0 = -3.4
        pulse_f = -pulse_0

        ham.set_parameters(delta=delta)
        ham.set_control(control_name="lam", pulse_initial=pulse_0, pulse_final=pulse_f, initial_state=0, alpha=2,
                        beta=2, num_steps=2 ** 8 + 1)
        ham.solve_problem()
        s = ham._s

        expected_control = -delta * np.tan((2 * s - 1) * np.arctan(pulse_0 / delta))

        np.testing.assert_allclose(ham.control_sol, expected_control, atol=1e-2,
                                   err_msg="Optimal control for LZ n_+ = 2 is wrong")

    def test_FAQUAD(self):
        """For the LZ model, the optimal control for n_+ = 3 has a closed form solution"""
        ham = Hamiltonian(lz_hamiltonian)

        delta = 2.4
        pulse_0 = -8.02
        pulse_f = -pulse_0

        ham.set_parameters(delta=delta)
        ham.set_control(control_name="lam", pulse_initial=pulse_0, pulse_final=pulse_f, initial_state=0, alpha=3,
                        beta=3, num_steps=2 ** 8 + 1)
        ham.solve_problem()
        s = ham._s

        expected_control = (1 - 2 * s) * delta * pulse_0 / np.sqrt(delta ** 2 - 4 * (s - 1) * s * pulse_0 ** 2)

        np.testing.assert_allclose(ham.control_sol, expected_control, atol=1e-2,
                                   err_msg="Optimal control for LZ n_+ = 3 is wrong")
