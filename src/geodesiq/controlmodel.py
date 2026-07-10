from typing import Any, Callable, Optional, cast

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import romb, solve_ivp
from scipy.interpolate import interp1d

from ._utils import Flags, build_diab
from .exceptions import (ImmutableConfigurationError, InvalidControlParameterError, MissingControlParameterError,
                         SolverError, ValidationError, MetricComputationError)
from .pulses import PulseControl


class ControlModel:
    """
    A class to represent a parameter-dependent ControlModel and solve the optimization problem for finding the optimal
    control pulse. The ControlModel is defined as a function of a control parameter (e.g., lambda) and can also depend
    on other parameters. The class provides methods to set the parameters, compute the metric tensor, solve the ODE
    for the control pulse, and synthesize the control pulse based on the solution of the optimization problem.
    """

    _SINGULARITY_RTOL = 1e-12

    def __init__(self, H_func: Callable[..., np.ndarray], partial_H_func: Optional[Callable[..., np.ndarray]] = None,
                 _flags_verbose: bool = False):
        """
        Initialize the ControlModel class with the ControlModel function and its partial derivative (if provided).

        Parameters
        ----------
        H_func : Callable[..., np.ndarray]
            A function that takes the control parameter and other parameters as input and returns the ControlModel
            matrix as a numpy array. The function should be defined such that it can accept the control parameter as a
            keyword argument, e.g., H_func(lambda=..., param1=..., param2=..., ...).
        partial_H_func : Optional[Callable[..., np.ndarray]]
            An optional function that takes the control parameter and other parameters as input and returns the partial
            derivative of the ControlModel with respect to the control parameter as a numpy array.
            If this function is not provided, the class will compute the numerical partial derivative.
        """
        self._H_func = H_func
        self._partial_H_func = partial_H_func

        # Initialize flags needed to track the state of the computations
        self._flags = Flags(_verbose=_flags_verbose)
        self._flags.add('eigenproblem_solved')
        self._flags.add('dia_list_computed')
        self._flags.add('metric_computed', parents=['eigenproblem_solved', 'dia_list_computed'])
        self._flags.add('ode_solved', parent='metric_computed')

        if self._partial_H_func is not None:
            self._flag_numerical_partial_H = False
        else:
            self._flag_numerical_partial_H = True

        # Initialize parameters and control settings
        self._parameters: dict[str, Any] = {}
        self._control_name: str | None = None
        self._pulse_initial: float | None = None
        self._pulse_final: float | None = None
        self._initial_state: int | None = None
        self._final_state: int | None = None
        self._alpha: float | None = None
        self._beta: float | None = None
        self._dia_alpha: float | None = None
        self._dia_beta: float | None = None
        self._num_steps: int | None = None

        # Initialize energy gaps and matrix elements to None (to be computed in self.solve_problem())
        self._energies: np.ndarray | None = None
        self._matrix_elements: np.ndarray | None = None

        # Initialize metric tensor and normalization factor to None (to be computed in self.solve_problem())
        self._dia_list: np.ndarray | None = None
        self._metric_tensor: np.ndarray | None = None
        self._a_tilde: float | None = None

        # Initialize pulse parameters
        self._s: np.ndarray | None = None
        self._control_pulse: np.ndarray | None = None
        self._control_sol: np.ndarray | None = None
        self._pulse: Any = None

        # Numerical integration configuration (user-overridable in solve_problem).
        self._solver = solve_ivp
        self._solver_kwargs: dict[str, Any] = {}
        self._metric_integrator = romb
        self._metric_integrator_kwargs: dict[str, Any] = {}
        self._previous_pulse_accuracy: int | None = None  # To track changes in pulse accuracy for ODE solving

    def __call__(self, *args, **kwargs) -> np.ndarray:
        # Return the ControlModel function if the object is called directly, allowing for easy evaluation of the
        #   ControlModel at specific control values
        # Runtime kwargs override stored defaults for explicit one-off evaluations.
        return self.H_func(*args, **{**self._parameters, **kwargs})

    def _evaluation_kwargs(self, control_value: float) -> dict[str, Any]:
        if self.control_name is None:
            raise MissingControlParameterError("control_name must be set before evaluating the Hamiltonian.")
        control_name = self.control_name
        assert control_name is not None
        return {**self._parameters, control_name: float(control_value)}

    def evaluate_hamiltonian(self, control_value: float) -> np.ndarray:
        """Evaluate and validate the Hamiltonian at one control value."""
        if not isinstance(control_value, (int, float, np.integer, np.floating)) or isinstance(control_value, bool):
            raise InvalidControlParameterError("Control value must be a finite real number.")
        if not np.isfinite(control_value):
            raise InvalidControlParameterError("Control value must be finite.")
        matrix = self.H_func(**self._evaluation_kwargs(float(control_value)))

        return matrix

    # Setters and getters are defined for the critical attributes of the ControlModel class, so we have control over how
    # the user modifies them. It is important that the correct flags are updated when these attributes are changed
    @property
    def H_func(self) -> Callable[..., np.ndarray]:
        return self._H_func

    @H_func.setter
    def H_func(self, func: Callable[..., np.ndarray]):
        if self._H_func is None:
            self._H_func = func
        else:
            raise ImmutableConfigurationError("H_func is already set and cannot be changed. If you want to change it,"
                                              " please create a new instance of the ControlModel class.")

    @property
    def partial_H_func(self) -> Callable[..., np.ndarray] | None:
        return self._partial_H_func

    @partial_H_func.setter
    def partial_H_func(self, func: Callable[..., np.ndarray]):
        if self._partial_H_func is None:
            self._partial_H_func = func
            self._flag_numerical_partial_H = False  # Update the numerical partial flag
            self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag
        else:
            raise ImmutableConfigurationError(
                "partial_H_func is already set and cannot be changed. If you want to change it,"
                " please create a new instance of the ControlModel class.")

    @property
    def control_name(self) -> str | None:
        return self._control_name

    @control_name.setter
    def control_name(self, name: str | None):
        if name is None:  # Keep the previous value
            return
        if not isinstance(name, str):
            raise InvalidControlParameterError("Control name must be a string.")
        if name == self._control_name:  # Keep the previous value
            return
        self._control_name = name
        self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the control name changes

    @property
    def pulse_initial(self) -> float:
        return cast(float, self._pulse_initial)

    @pulse_initial.setter
    def pulse_initial(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Pulse initial value must be a number.")

        if self._pulse_final is not None:
            if np.isclose(value, self._pulse_final):
                raise ValueError("The initial and final control values must be different.")

        if value == self._pulse_initial:  # Keep the previous value
            return

        self._pulse_initial = value
        self._flags[
            'eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the pulse initial value changes

    @property
    def pulse_final(self) -> float:
        return cast(float, self._pulse_final)

    @pulse_final.setter
    def pulse_final(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Pulse final value must be a number.")

        if self._pulse_initial is not None:
            if np.isclose(value, self._pulse_initial):
                raise ValueError("The initial and final control values must be different.")

        if value == self._pulse_final:  # Keep the previous value
            return
        self._pulse_final = value
        self._flags[
            'eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the pulse final value changes

    @property
    def initial_state(self) -> int:
        return cast(int, self._initial_state)

    @initial_state.setter
    def initial_state(self, value: int | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise InvalidControlParameterError("Initial state index must be an integer.")

        if value < 0:
            raise InvalidControlParameterError("Initial state index must be positive.")

        if value == self._initial_state:  # Keep the previous value
            return

        self._initial_state = value

        if self._final_state is None:  # If not final state, assume for the moment that is the same as the initial one
            self.final_state = value

        self._flags['metric_computed'] = False  # Reset the  metric computed flag if the initial state index changes
        self._flags['dia_list_computed'] = False  # Reset the diabatic computed flag if the pulse initial value changes

        # If the initial state is the same as the final state, we can mark the diabatic passage list as computed
        if self._initial_state == self._final_state:
            self._flags['dia_list_computed'] = True

    @property
    def final_state(self) -> int:
        return cast(int, self._final_state)

    @final_state.setter
    def final_state(self, value: int | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise InvalidControlParameterError("Final state index must be an integer.")

        if value < 0:
            raise InvalidControlParameterError("Final state index must be positive.")

        if value == self._final_state:  # Keep the previous value
            return

        self._final_state = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if the final state index changes
        self._flags['dia_list_computed'] = False  # Reset the diabatic computed flag if the pulse initial value changes

        # If the initial state is the same as the final state, we can mark the diabatic passage list as computed
        if self._initial_state == self._final_state:
            self._flags['dia_list_computed'] = True

    @property
    def alpha(self) -> float:
        return cast(float, self._alpha)

    @alpha.setter
    def alpha(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Alpha must be a number.")

        if value == self._alpha:  # Keep the previous value
            return

        self._alpha = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if alpha changes

    @property
    def beta(self) -> float:
        return cast(float, self._beta)

    @beta.setter
    def beta(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Beta must be a number.")

        if value == self._beta:  # Keep the previous value
            return

        self._beta = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if beta changes

    @property
    def dia_alpha(self) -> float | None:
        return self._dia_alpha

    @dia_alpha.setter
    def dia_alpha(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Diabatic alpha must be a number.")

        if value == self._dia_alpha:  # Keep the previous value
            return

        self._dia_alpha = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if alpha changes

    @property
    def dia_beta(self) -> float | None:
        return self._dia_beta

    @dia_beta.setter
    def dia_beta(self, value: float | None):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Diabatic beta must be a number.")

        if value == self._dia_beta:  # Keep the previous value
            return

        self._dia_beta = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if beta changes

    @property
    def num_steps(self) -> int:
        return cast(int, self._num_steps)

    @num_steps.setter
    def num_steps(self, value: int | None):
        if value is None:  # Keep the previous value
            return

        if value == self._num_steps:  # Keep the previous value
            return

        if not isinstance(value, int) or value < 3:
            raise InvalidControlParameterError(
                "Number of steps must be an integer >= 3 to support pulse interpolation.")

        if value == self._num_steps:  # Keep the previous value
            return

        self._num_steps = value
        self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the number of steps changes

    @property
    def control_sol(self) -> np.ndarray:
        if not self._flags['ode_solved']:
            self.solve_problem()  # Attempt to solve the ODE if not already solved
        assert self._control_sol is not None
        return self._control_sol

    @property
    def pulse(self) -> np.ndarray:
        all_flags_check = self._flags.all()
        if not all_flags_check:
            self.solve_problem()  # Attempt to solve the pulse if not already solved
        assert self._pulse is not None
        return cast(np.ndarray, self._pulse)

    @property
    def eigenenergies(self) -> np.ndarray:
        """Return ControlModel eigenenergies, solving the eigenproblem if needed."""
        if not self._flags['eigenproblem_solved']:
            self._check_eigensystem_parameters()
            self._solve_eigenproblem()
        assert self._energies is not None
        return self._energies

    @property
    def control_pulse(self) -> np.ndarray:
        """Return the control-parameter grid, solving the eigenproblem if needed."""
        if not self._flags['eigenproblem_solved']:
            self._check_eigensystem_parameters()
            self._solve_eigenproblem()
        assert self._control_pulse is not None
        return self._control_pulse

    def set_parameters(self, **parameters):
        """
        Set the parameters for the ControlModel. This method allows you to specify any parameters that are needed to
        compute the ControlModel and its partial derivative. The parameters should be provided as keyword arguments,
        e.g., set_parameters(param1=value1, param2=value2, ...), and they will be stored. If a single parameter is
        updated, others will not be affected.
        """

        if self.control_name in parameters:
            raise InvalidControlParameterError(f"{self.control_name!r} is the control variable and cannot also "
                                               "be supplied as a fixed parameter.")

        new_params = {**self._parameters, **parameters}

        if new_params.keys() != self._parameters.keys():
            self._parameters = new_params  # Update the parameters with the new values
            self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag
        else:
            for key, value in new_params.items():
                if not np.allclose(value, self._parameters[key]):
                    self._parameters = new_params  # Update the parameters with the new values
                    self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag
                    break

    def set_control(self, control_name: Optional[str] = None, pulse_initial: Optional[float] = None,
                    pulse_final: Optional[float] = None, initial_state: Optional[int] = None,
                    final_state: Optional[int] = None, alpha: Optional[float] = None, beta: Optional[float] = None,
                    dia_alpha: Optional[float] = None, dia_beta: Optional[float] = None,
                    num_steps: Optional[int] = None):
        """
        Set the control parameters for the optimization problem. This method allows you to specify the control
        parameters such as the name of the control parameter, the initial and final values of the control pulse, ....
        Control parameters can be set individually, and if any parameter is not provided, it will retain its previous
        value, so you can update only the parameters you want without affecting the others.

        Parameters
        ----------
        control_name : Optional[str]
            The name of the control parameter (e.g., "lambda"). This is used to identify the control parameter in the
             ControlModel function.
        pulse_initial : Optional[float]
            The initial value of the control pulse. This is the value of the control parameter at the beginning of the
             pulse.
        pulse_final : Optional[float]
            The final value of the control pulse. This is the value of the control parameter at the end of the pulse.
        initial_state : Optional[int]
            The index of the initial state in the energy spectrum. This is used to compute the metric tensor and the
             ODE for the control pulse.
        final_state : Optional[int]
            The index of the final state in the energy spectrum. This is used to compute the metric tensor and the ODE
            for the control pulse. If not provided, it will be assumed to be the same as the initial state.
        alpha : Optional[float]
            The exponent alpha used in the metric tensor computation. This parameter controls the weighting of the
             energy gaps in the metric tensor.
        beta : Optional[float]
            The exponent beta used in the metric tensor computation. This parameter controls the weighting of the
            matrix elements in the metric tensor.
        dia_alpha : Optional[float]
            The exponent alpha used in the diabatic passage contribution to the metric tensor. This parameter controls
             the weighting of the energy gaps in the diabatic passage contribution to the metric tensor.
        dia_beta : Optional[float]
            The exponent beta used in the diabatic passage contribution to the metric tensor. This parameter controls
             the weighting of the matrix elements in the diabatic passage contribution to the metric tensor.
        num_steps : Optional[int]
            The number of steps to use in the discretization of the control pulse. This determines the resolution of
            the control pulse and the accuracy of the numerical solution. If omitted and no previous value exists,
            a default of ``2**10 + 1`` is used.
        """

        candidate_name = self.control_name if control_name is None else control_name
        if candidate_name is not None and candidate_name in self._parameters:
            raise InvalidControlParameterError(
                f"Control name {candidate_name!r} collides with a stored Hamiltonian parameter.")
        numeric_types = (int, float, np.integer, np.floating)
        candidate_initial = self.pulse_initial if pulse_initial is None else pulse_initial
        candidate_final = self.pulse_final if pulse_final is None else pulse_final
        if (isinstance(candidate_initial, numeric_types) and not isinstance(candidate_initial, bool) and isinstance(
                candidate_final, numeric_types) and not isinstance(candidate_final, bool) and float(
            candidate_initial) == float(candidate_final)):
            raise InvalidControlParameterError("Pulse initial and final values must be different.")

        # Update the final endpoint first when needed to avoid transient equality during range changes.
        self.control_name = control_name
        if pulse_initial is not None and pulse_final is not None and pulse_initial == self._pulse_final:
            self.pulse_final = pulse_final
            self.pulse_initial = pulse_initial
        else:
            self.pulse_initial = pulse_initial
            self.pulse_final = pulse_final
        self.initial_state = initial_state
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dia_alpha = dia_alpha
        self.dia_beta = dia_beta

        # Keep explicit user value; otherwise lazily initialize to default on first configuration.
        if num_steps is None and self._num_steps is None:
            self.num_steps = 2 ** 10 + 1
        else:
            self.num_steps = num_steps

    def solve_problem(self, pulse_accuracy: int = 1000, solver: Optional[Callable] = None,
                      solver_kwargs: Optional[dict] = None, metric_integrator: Optional[Callable] = None,
                      metric_integrator_kwargs: Optional[dict] = None, ):
        """
        Solve the optimization problem to find the optimal control pulse. This method computes the metric tensor based
        on the energies and matrix elements of the ControlModel, and then solves the ODE for the control pulse using the
        computed metric tensor. Note that this method does not synthesize the pulse itself, for that you need to call
        the synthesize_pulse() method after solving the problem.

        Parameters
        ----------
        pulse_accuracy : int
            The number of points to use in the numerical solution of the ODE for the control pulse. Higher values will
             yield a more accurate solution but will also increase the computational cost.
        solver : Optional[Callable]
            Callable used to integrate the ODE. Defaults to ``scipy.integrate.solve_ivp``.
            Expected signature is solver(fun, t_span, y0, t_eval=..., **kwargs), and the return
            value must expose ``t`` and ``y`` (or be a ``(t, y)`` tuple).
        solver_kwargs : Optional[dict]
            Additional keyword arguments forwarded to ``solver``.
        metric_integrator : Optional[Callable]
            Callable used to compute ``a_tilde`` from ``sqrt(metric_tensor)``. Defaults to
            ``scipy.integrate.romb``.
        metric_integrator_kwargs : Optional[dict]
            Additional keyword arguments forwarded to ``metric_integrator``.
        """
        self._check_control_parameters()

        self._configure_integration(solver=solver, solver_kwargs=solver_kwargs, metric_integrator=metric_integrator,
                                    metric_integrator_kwargs=metric_integrator_kwargs, )

        self._solve_eigenproblem()
        self._compute_metric_tensor()

        self._solve_ode(pulse_accuracy)

    def _configure_integration(self, solver: Optional[Callable], solver_kwargs: Optional[dict],
                               metric_integrator: Optional[Callable], metric_integrator_kwargs: Optional[dict], ):
        selected_solver = solve_ivp if solver is None else solver
        selected_metric_integrator = romb if metric_integrator is None else metric_integrator
        if solver_kwargs is None:
            selected_solver_kwargs = {}
        elif isinstance(solver_kwargs, dict):
            selected_solver_kwargs = dict(solver_kwargs)
        else:
            raise ValidationError("solver_kwargs must be a dictionary.")

        if metric_integrator_kwargs is None:
            selected_metric_integrator_kwargs = {}
        elif isinstance(metric_integrator_kwargs, dict):
            selected_metric_integrator_kwargs = dict(metric_integrator_kwargs)
        else:
            raise ValidationError("metric_integrator_kwargs must be a dictionary.")

        if not callable(selected_solver):
            raise ValidationError("solver must be a callable integration function.")

        if not callable(selected_metric_integrator):
            raise ValidationError("metric_integrator must be a callable integration function.")

        if (
                selected_metric_integrator is not self._metric_integrator or selected_metric_integrator_kwargs != self._metric_integrator_kwargs):  # noqa: E501
            self._metric_integrator = selected_metric_integrator
            self._metric_integrator_kwargs = selected_metric_integrator_kwargs
            self._flags['metric_computed'] = False

        if selected_solver is not self._solver or selected_solver_kwargs != self._solver_kwargs:
            self._solver = selected_solver
            self._solver_kwargs = selected_solver_kwargs
            self._flags['ode_solved'] = False

    def _solve_dia_list(self):
        """
        Compute the diabatic passage list based on the initial and final states. This method is called internally by
        solve_problem() to compute the diabatic passage list, which is used in the computation of the metric tensor.
        The results are stored in self._dia_list for later use in computing the metric tensor.
        """
        if self._flags['dia_list_computed']:
            return  # If the diabatic passage list is already computed, skip the computation

        dim = self.eigenenergies.shape[1]  # Get the dimension of the ControlModel from the energies array
        self._dia_list = build_diab(initial_state=self.initial_state, final_state=self.final_state, dim=dim)
        self._flags['dia_list_computed'] = True  # Mark the diabatic passage list as computed to avoid recomputation

    def _solve_eigenproblem(self):
        """
        Solve the eigenproblem for the ControlModel at each control value to obtain the energies and matrix elements.
        This method is called internally by solve_problem() if the eigenproblem has not been solved yet. The results are
        stored in self.eigenenergies and self._matrix_elements for later use in computing the metric tensor.
        """

        if self._flags['eigenproblem_solved']:
            return  # If the eigenproblem is already solved, skip the computation

        self._check_eigensystem_parameters()
        control_name = self.control_name
        pulse_initial = self.pulse_initial
        pulse_final = self.pulse_final
        num_steps = self.num_steps
        assert control_name is not None

        control_pulse = np.linspace(pulse_initial, pulse_final, num=num_steps)
        self._control_pulse = control_pulse

        full_hamiltonian = np.array([self.H_func(**{control_name: lam, **self._parameters}) for lam in control_pulse])

        self._energies, eigenvectors = np.linalg.eigh(full_hamiltonian)

        if self._flag_numerical_partial_H:
            # Compute the numerical partial derivative of H with respect to the control parameter
            full_partial_H = self._compute_numerical_partial_H()

        else:
            # Use the provided analytical partial derivative function
            full_partial_H = np.array(
                [self.partial_H_func(**{control_name: lam, **self._parameters}) for lam in control_pulse])

        self._matrix_elements = np.abs(
            np.einsum('...ij,...jk,...kl->...il', eigenvectors.conj().transpose(0, 2, 1), full_partial_H, eigenvectors))

        self._flags['eigenproblem_solved'] = True  # Mark the eigenproblem as solved to avoid recomputation

    def _metric_ratio(self, numerator: np.ndarray, denominator: np.ndarray, alpha: float, beta: float,
                      transition: tuple[int, int], ) -> np.ndarray:
        if alpha > 0:
            gap_scale = max(1.0, float(np.max(np.abs(self._energies)))) if self._energies is not None else 1.0
            tolerance = self._SINGULARITY_RTOL * gap_scale
            if np.any(denominator <= tolerance):
                raise MetricComputationError("Degenerate or near-degenerate energy gap encountered for transition "
                                             f"{transition}; provide a regularized model or avoid the degeneracy.")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = numerator ** beta / denominator ** alpha
        return np.asarray(ratio, dtype=float)

    def _compute_metric_tensor(self):
        """
        Compute the metric tensor G_tensor based on the energies and matrix elements of the ControlModel. If the
        eigenproblem has not been solved yet, solve it first to obtain the energies and matrix elements.
        """
        if self._flags['metric_computed']:
            return

        if self._initial_state == self._final_state:
            self._compute_G_adiabatic()
        else:
            self._solve_dia_list()  # Ensure the diabatic passage list is computed before computing the metric tensor
            self._compute_G_diabatic()

        if self._metric_tensor is None or self._control_pulse is None:
            raise MetricComputationError("Metric tensor was not produced.")
        metric = np.asarray(self._metric_tensor, dtype=float)
        if not np.all(np.isfinite(metric)):
            raise MetricComputationError("Metric tensor contains NaN or infinite values.")

        scale = max(1.0, float(np.max(np.abs(metric))))
        tolerance = 1e-12 * scale
        if np.any(metric < -tolerance):
            raise MetricComputationError("Metric tensor contains negative values.")
        metric = np.maximum(metric, 0.0)
        if np.any(metric <= tolerance):
            locations = self._control_pulse[metric <= tolerance]
            sample = ", ".join(f"{value:.6g}" for value in locations[:3])
            raise MetricComputationError("Metric tensor is zero or numerically singular" + (
                f" near control value(s) {sample}." if sample else "."))

        dx = float(np.abs(self._control_pulse[1] - self._control_pulse[0]))
        metric_values = np.sqrt(self._metric_tensor)

        # scipy.integrate.romb requires sample count n = 2**k + 1.
        if self._metric_integrator is romb:
            n_samples = metric_values.size
            power_minus_one = n_samples - 1
            if power_minus_one <= 0 or (power_minus_one & (power_minus_one - 1)) != 0:
                raise InvalidControlParameterError(
                    f"num_steps={n_samples} is incompatible with romb. Use num_steps=2**k+1, "
                    "or pass a different metric_integrator to solve_problem(...).")

        try:
            self._a_tilde = float(self._metric_integrator(metric_values, dx=dx, **self._metric_integrator_kwargs))
        except TypeError as exc:
            raise ValidationError(
                "Invalid metric_integrator call: ensure it accepts arguments like (values, dx=..., **kwargs).") from exc

        self._flags['metric_computed'] = True  # Mark the metric as computed to avoid recomputation

    def _compute_G_diabatic(self):
        num, dim = self._energies.shape
        metric = np.zeros(num, dtype=float)
        alpha, beta = self.alpha, self.beta

        dia_alpha, dia_beta = self.dia_alpha, self.dia_beta
        if alpha is None or beta is None or dia_alpha is None or dia_beta is None:
            raise MissingControlParameterError("Adiabatic and diabatic metric exponents are required.")

        for m in range(dim):
            for n in range(dim):
                if n == m:
                    continue
                adiabatic = bool(self._dia_list[m, n])
                denominator = np.abs(self._energies[:, n] - self._energies[:, m])
                numerator = self._matrix_elements[:, m, n]
                metric += self._metric_ratio(numerator, denominator, alpha=alpha if adiabatic else dia_alpha,
                                             beta=beta if adiabatic else dia_beta, transition=(m, n), )
        self._metric_tensor = metric

    def _compute_G_adiabatic(self):
        """
        Compute the adiabatic contribution to the metric tensor G_tensor based on the energies and matrix elements
        of the
        """

        num, dim = np.shape(self.eigenenergies)
        metric = np.zeros(num, dtype=float)
        alpha, beta = self.alpha, self.beta
        if alpha is None or beta is None:
            raise MissingControlParameterError("Adiabatic metric exponents are required.")
        for state in range(dim):
            if state == self.initial_state:
                continue
            denominator = np.abs(self._energies[:, state] - self._energies[:, self.initial_state])
            numerator = self._matrix_elements[:, self.initial_state, state]
            metric += self._metric_ratio(numerator, denominator, alpha=alpha, beta=beta,
                                         transition=(self.initial_state, state), )
        self._metric_tensor = metric

    def _compute_numerical_partial_H(self, order: int = 8, ) -> np.ndarray:
        """
        Evaluate dH/dx at every point of a real-valued grid.

        H_func is assumed not to be vectorized: it accepts one scalar x
        and returns a real- or complex-valued Hamiltonian.

        Parameters
        ----------
        order
            Order of the finite-difference formula.

        Returns
        -------
        dH_dx
            Derivative with shape (n_points, *H_shape).
        """
        if self._control_pulse is None:
            raise ValueError("x_grid is unavailable before the eigenproblem grid is initialized.")

        x_grid = np.asarray(self._control_pulse, dtype=float)

        if x_grid.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")

        if x_grid.size < 2:
            raise ValueError("x_grid must contain at least two points.")

        if not np.all(np.isfinite(x_grid)):
            raise ValueError("x_grid must contain only finite values.")

        unique_x = np.unique(x_grid)

        if unique_x.size < 2:
            raise ValueError("x_grid must contain at least two distinct points.")

        initial_step = float(np.min(np.diff(unique_x)))

        tolerances = {"atol": 1e-10, "rtol": 1e-8, }

        # Scalar evaluation to determine the Hamiltonian shape.
        H_reference = np.asarray(self._H_func(**self._evaluation_kwargs(float(x_grid[0]))), dtype=np.complex128, )

        H_shape = H_reference.shape
        n_elements = H_reference.size

        def evaluate_and_pack(x_column: np.ndarray, ) -> np.ndarray:
            """
            Evaluate the non-vectorized Hamiltonian at one scalar x.

            x_column has shape (1,), since x is the only independent
            variable.
            """
            x = float(x_column[0])

            H = np.asarray(self._H_func(**self._evaluation_kwargs(x)), dtype=np.complex128, )

            if H.shape != H_shape:
                raise ValueError("H_func returned inconsistent shapes: "
                                 f"expected {H_shape}, got {H.shape}.")

            H_flat = H.ravel()

            # scipy.differentiate.jacobian expects real outputs.
            return np.concatenate((H_flat.real, H_flat.imag,))

        def packed_hamiltonian(x_array: np.ndarray, ) -> np.ndarray:
            """
            Adapt the scalar H_func to SciPy's vectorized interface.

            x_array has shape (1, ...), where the trailing axes contain
            evaluation points introduced by SciPy.
            """
            return np.apply_along_axis(evaluate_and_pack, axis=0, arr=x_array, )

        # Use one-sided differences at the two domain boundaries.
        x_min = np.min(x_grid)
        x_max = np.max(x_grid)

        step_direction = np.zeros_like(x_grid, dtype=int)
        step_direction[x_grid - x_min < initial_step] = 1
        step_direction[x_max - x_grid < initial_step] = -1

        result = jacobian(packed_hamiltonian, x_grid[np.newaxis, :], order=order, initial_step=initial_step,
                          step_direction=step_direction[np.newaxis, :], tolerances=tolerances, )

        # result.df shape:
        # (2 * n_elements, 1, n_points)
        derivative = np.asarray(result.df[:, 0, :])

        dH_flat = (derivative[:n_elements] + 1j * derivative[n_elements:])

        # Convert from (*H_shape, n_points) to
        # (n_points, *H_shape).
        dH_dx = dH_flat.reshape(H_shape + (x_grid.size,))

        return np.moveaxis(dH_dx, -1, 0)

    def _solve_ode(self, pulse_accuracy: int):
        """
        Solve the ODE for the control pulse using the computed metric tensor and the normalization factor a_tilde.
        """

        if self._previous_pulse_accuracy != pulse_accuracy:
            self._flags['ode_solved'] = False  # Reset the ODE solved flag if the pulse accuracy changes

        if self._flags['ode_solved']:
            return  # If the ODE is already solved, skip the computation

        if self._control_pulse is None or self._metric_tensor is None or self._a_tilde is None:
            raise SolverError("ODE cannot be solved before the control pulse and metric tensor are available.")

        sig = np.sign(self._control_pulse[1] - self._control_pulse[0])

        factor_interpolation = interp1d(self._control_pulse, self._a_tilde / np.sqrt(self._metric_tensor),
                                        kind='quadratic', fill_value="extrapolate")

        def model(_, y):
            return sig * factor_interpolation(y)

        s = np.linspace(0, 1, pulse_accuracy)
        kwargs = dict(self._solver_kwargs)
        if self._solver is solve_ivp:
            kwargs = {'dense_output': True, 'method': 'RK45', 'atol': 1e-8, 'rtol': 1e-6, **kwargs}

        try:
            sol = self._solver(model, [0, 1], [self._control_pulse[0]], t_eval=s, **kwargs)
        except TypeError as exc:
            raise ValidationError(
                "Invalid solver call: ensure solver accepts (fun, t_span, y0, t_eval=..., **kwargs).") from exc

        if hasattr(sol, "success") and not sol.success:
            raise SolverError(getattr(sol, "message", "ODE solver failed."))

        if isinstance(sol, tuple):
            if len(sol) != 2:
                raise SolverError("Solver tuple output must be a (t, y) pair.")
            t_arr = sol[0]
            y_arr = sol[1]
        elif hasattr(sol, "t") and hasattr(sol, "y"):
            t_arr, y_arr = sol.t, sol.y
        else:
            raise SolverError("Solver output must expose 't' and 'y', or return a (t, y) tuple.")

        y_arr = np.asarray(y_arr)
        self._s = np.asarray(t_arr, dtype=float)
        self._control_sol = np.asarray(y_arr[0] if y_arr.ndim == 2 else y_arr, dtype=float)
        self._previous_pulse_accuracy = pulse_accuracy
        self._flags['ode_solved'] = True

    def _check_eigensystem_parameters(self):
        """Validate control settings needed to compute and compute the eigenvalues."""
        missing_params = []

        if self.control_name is None:
            missing_params.append("control_name")
        if self.pulse_initial is None:
            missing_params.append("pulse_initial")
        if self.pulse_final is None:
            missing_params.append("pulse_final")
        if self.num_steps is None:
            missing_params.append("num_steps")

        if missing_params:
            missing_msg = ", ".join(missing_params)
            raise MissingControlParameterError(f"Missing control parameters for eigensystem: {missing_msg}. "
                                               "Please set them using set_control(...).")

    def plot_eigenvalues(self, fig=None, ax=None, legend: bool = True, legend_kwargs: Optional[dict] = None,
                         xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None,
                         **plot_kwargs):
        """
        Plot ControlModel eigenvalues as a function of the control parameter.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : Optional[dict]
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : Optional[str]
            Label for the x-axis. Defaults to ``control_name`` when not provided.
        ylabel : Optional[str]
            Label for the y-axis. Defaults to ``"Energy"`` when not provided.
        title : Optional[str]
            Plot title. Defaults to ``"ControlModel Eigenvalues"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot`` for each energy branch.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        self._check_eigensystem_parameters()
        control_name = self.control_name
        assert control_name is not None

        if ax is None:
            import matplotlib.pyplot as plt

            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        elif fig is None:
            fig = ax.figure

        for level in range(self.eigenenergies.shape[1]):
            ax.plot(self._control_pulse, self.eigenenergies[:, level], label=f"E{level}", **plot_kwargs)

        ax.set_xlabel(control_name if xlabel is None else xlabel)
        ax.set_ylabel("Energy" if ylabel is None else ylabel)
        ax.set_title("ControlModel Eigenvalues" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax

    def plot_metric_tensor(self, fig=None, ax=None, legend: bool = True, legend_kwargs: Optional[dict] = None,
                           xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None,
                           **plot_kwargs):
        """
        Plot the metric tensor (G tensor) as a function of the control parameter.

        If the metric tensor is not available yet, it is computed from the current
        ControlModel/control configuration.

        Parameters
        ----------
        fig, ax
            Optional matplotlib figure/axis. If not provided, they are created.
        legend : bool
            Whether to draw a legend.
        legend_kwargs : Optional[dict]
            Extra kwargs forwarded to ``ax.legend``.
        xlabel : Optional[str]
            Label for the x-axis. Defaults to ``control_name`` when not provided.
        ylabel : Optional[str]
            Label for the y-axis. Defaults to ``"G tensor"`` when not provided.
        title : Optional[str]
            Plot title. Defaults to ``"G tensor"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot``.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        self._check_control_parameters()
        self._solve_eigenproblem()
        self._compute_metric_tensor()
        control_name = self.control_name
        assert control_name is not None
        assert self._control_pulse is not None
        assert self._metric_tensor is not None

        if ax is None:
            import matplotlib.pyplot as plt

            if fig is None:
                fig, ax = plt.subplots()
            else:
                ax = fig.add_subplot(111)
        elif fig is None:
            fig = ax.figure

        ax.plot(self._control_pulse, self._metric_tensor, label="G", **plot_kwargs)

        ax.set_xlabel(control_name if xlabel is None else xlabel)
        ax.set_ylabel("G tensor" if ylabel is None else ylabel)
        ax.set_title("G tensor" if title is None else title)

        if legend:
            ax.legend(**(legend_kwargs or {}))

        return fig, ax

    def synthesize_pulse(self, duration: float, method: Optional[str] = None, pulse_args: Optional[tuple] = None,
                         pulse_kwargs: Optional[dict] = None):
        """
        Synthesize the control pulse based on the solution of the optimization problem. If the problem has not been
        solved yet, this method will automatically solve it first.

        Parameters
        ----------
        duration : float
            The total duration of the control pulse.
        method : Optional[str]
            Name of the method to use for the pulse synthesis
        pulse_args : Optional[tuple]
            Positional arguments for the pulse synthesis (e.g., duration, filter parameters).
        pulse_kwargs : Optional[dict]
            Keyword arguments for the pulse synthesis (e.g., duration, filter parameters).

        Returns
        -------
        PulseControl
             An instance of the PulseControl class representing the synthesized control pulse.
        """
        self.solve_problem()
        if self._control_sol is None:
            raise SolverError("Cannot synthesize a pulse before the control solution is available.")
        pulse = PulseControl(self._control_sol, duration, method, pulse_args, pulse_kwargs)
        self._pulse = pulse
        return pulse()

    def _check_control_parameters(self):
        """
        Check if all necessary control parameters are set before solving the problem. If any parameter is missing,
        raise a ConfigurationError with a clear message indicating which parameters are missing and how to set them.
        """
        missing_params = []

        if self.control_name is None:
            missing_params.append("control_name")
        if self.pulse_initial is None:
            missing_params.append("pulse_initial")
        if self.pulse_final is None:
            missing_params.append("pulse_final")
        if self.initial_state is None:
            missing_params.append("initial_state")
        if self.final_state is None:
            missing_params.append("final_state")
        if self.alpha is None:
            missing_params.append("alpha")
        if self.beta is None:
            missing_params.append("beta")
        if self.num_steps is None:
            missing_params.append("num_steps")

        if self._initial_state != self._final_state:
            if self.dia_alpha is None:
                missing_params.append("dia_alpha")
            if self.dia_beta is None:
                missing_params.append("dia_beta")

        if missing_params:
            missing_msg = ", ".join(missing_params)
            raise MissingControlParameterError(
                f"Missing control parameters: {missing_msg}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)}).")

    def _generate_summary(self) -> str:
        """
        Generate a summary string of the current control parameters and settings. This method creates a formatted
        string that provides a clear overview of the control parameters, their values, and any relevant settings for
        the optimization problem. The summary can be used for logging, debugging, or displaying the current state of the
        ControlModel object.
        """
        hamiltonian_params = (", ".join(
            f"{key}: {value}" for key, value in self._parameters.items()) if self._parameters else "❌ not set")
        alpha_beta = (f"({self.alpha if self.alpha is not None else '❌ not set'}, "
                      f"{self.beta if self.beta is not None else '❌ not set'})")
        diabatic_alpha_beta = ("("
                               f"{self.dia_alpha if self.dia_alpha is not None else '❌ not set'}, "
                               f"{self.dia_beta if self.dia_beta is not None else '❌ not set'}"
                               ")")

        summary_lines = ["------------------ ControlModel Control Summary ------------------",
                         f"ControlModel: {'✅ set' if self.H_func is not None else '❌ not set'}",
                         f"Partial ControlModel: {'✅ set' if self.partial_H_func is not None else '❌ not set'}",
                         f"ControlModel parameters: {hamiltonian_params}",
                         f"Control name → {self.control_name if self.control_name is not None else '❌ not set'}",
                         f"Pulse initial → {self.pulse_initial if self.pulse_initial is not None else '❌ not set'}",
                         f"Pulse final → {self.pulse_final if self.pulse_final is not None else '❌ not set'}",
                         f"Initial state index →"
                         f" {self.initial_state if self.initial_state is not None else '❌ not set'}",
                         f"Final state index → {self.final_state if self.final_state is not None else '❌ not set'}",
                         f"(Alpha, Beta) → {alpha_beta}", f"(Diabatic Alpha, Diabatic Beta) → {diabatic_alpha_beta}",
                         f"Eigenproblem solved → {'✅ yes' if self._flags['eigenproblem_solved'] else '❌ no'}",
                         f"Metric computed → {'✅ yes' if self._flags['metric_computed'] else '❌ no'}",
                         f"ODE solved → {'✅ yes' if self._flags['ode_solved'] else '❌ no'}",
                         "---------------------------------------------------------------", ]
        return "\n".join(summary_lines)

    def print_summary(self):
        """
        Print a summary of the current control parameters and settings. This method provides a clear overview of the
        control parameters, their values, and any relevant settings for the optimization problem.
        """
        print(self._generate_summary())

    def __str__(self):
        return (f"ControlModel(control_name={self.control_name}, pulse_initial={self.pulse_initial}, "
                f"pulse_final={self.pulse_final}, initial_state={self.initial_state}, alpha={self.alpha}, "
                f"beta={self.beta}, num_steps={self.num_steps})")

    def __repr__(self):
        return self._generate_summary()
