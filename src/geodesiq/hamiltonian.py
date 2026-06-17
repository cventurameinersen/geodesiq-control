from typing import Optional, Callable, Any

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import solve_ivp, romb
from scipy.interpolate import interp1d

from ._utils import Flags, build_diab
from .exceptions import (ImmutableConfigurationError, MissingControlParameterError, InvalidControlParameterError,
                         ValidationError, SolverError, )
from .pulses import PulseControl


class Hamiltonian:
    """
    A class to represent a parameter-dependent Hamiltonian and solve the optimization problem for finding the optimal
    control pulse. The Hamiltonian is defined as a function of a control parameter (e.g., lambda) and can also depend
    on other _parameters. The class provides methods to set the _parameters, compute the metric tensor, solve the ODE
    for the control pulse, and synthesize the control pulse based on the solution of the optimization problem.
    """

    def __init__(self, H_func: Callable[[Any], np.ndarray],
                 partial_H_func: Optional[Callable[[Any], np.ndarray]] = None, _flags_verbose: bool = False):
        """
        Initialize the Hamiltonian class with the Hamiltonian function and its partial derivative (if provided).

        Parameters
        ----------
        H_func : Callable[[Any], np.ndarray]
            A function that takes the control parameter and other _parameters as input and returns the Hamiltonian
            matrix as a numpy array. The function should be defined such that it can accept the control parameter as a
            keyword argument, e.g., H_func(lambda=..., param1=..., param2=..., ...).
        partial_H_func : Optional[Callable[[Any], np.ndarray]]
            An optional function that takes the control parameter and other _parameters as input and returns the partial
            derivative of the Hamiltonian with respect to the control parameter as a numpy array.
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

        # Initialize _parameters and control settings
        self._parameters = {}
        self._control_name = None
        self._pulse_initial = None
        self._pulse_final = None
        self._initial_state = None
        self._final_state = None
        self._alpha = None
        self._beta = None
        self._dia_alpha = None
        self._dia_beta = None
        self._num_steps = None

        # Initialize energy gaps and matrix elements to None (to be computed in self.solve_problem())
        self._energies = None
        self._matrix_elements = None

        # Initialize metric tensor and normalization factor to None (to be computed in self.solve_problem())
        self._dia_list = None
        self._metric_tensor = None
        self._a_tilde = None

        # Initialize pulse _parameters
        self._s = None
        self._control_pulse = None
        self._control_sol = None
        self._pulse = None

        # Numerical integration configuration (user-overridable in solve_problem).
        self._solver = solve_ivp
        self._solver_kwargs = {}
        self._metric_integrator = romb
        self._metric_integrator_kwargs = {}
        self._previous_pulse_accuracy = None  # To track changes in pulse accuracy for ODE solving

    def __call__(self, *args, **kwargs):
        # Return the Hamiltonian function if the object is called directly, allowing for easy evaluation of the
        #   Hamiltonian at specific control values
        return self.H_func(*args, **{**kwargs, **self._parameters})

    # Setters and getters are defined for the critical attributes of the Hamiltonian class, so we have control over how
    # the user modifies them. It is important that the correct flags are updated when these attributes are changed
    @property
    def H_func(self):
        return self._H_func

    @H_func.setter
    def H_func(self, func):
        if self._H_func is None:
            self._H_func = func
        else:
            raise ImmutableConfigurationError("H_func is already set and cannot be changed. If you want to change it,"
                                              " please create a new instance of the Hamiltonian class.")

    @property
    def partial_H_func(self):
        return self._partial_H_func

    @partial_H_func.setter
    def partial_H_func(self, func):
        if self._partial_H_func is None:
            self._partial_H_func = func
            self._flag_numerical_partial_H = False  # Update the numerical partial flag
            self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag
        else:
            raise ImmutableConfigurationError(
                "partial_H_func is already set and cannot be changed. If you want to change it,"
                " please create a new instance of the Hamiltonian class.")

    @property
    def control_name(self):
        return self._control_name

    @control_name.setter
    def control_name(self, name):
        if name is None:  # Keep the previous value
            return
        if not isinstance(name, str):
            raise InvalidControlParameterError("Control name must be a string.")
        if name == self._control_name:  # Keep the previous value
            return
        self._control_name = name
        self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the control name changes

    @property
    def pulse_initial(self):
        return self._pulse_initial

    @pulse_initial.setter
    def pulse_initial(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Pulse initial value must be a number.")

        if value == self._pulse_initial:  # Keep the previous value
            return

        self._pulse_initial = value
        self._flags[
            'eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the pulse initial value changes

    @property
    def pulse_final(self):
        return self._pulse_final

    @pulse_final.setter
    def pulse_final(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Pulse final value must be a number.")

        if value == self._pulse_final:  # Keep the previous value
            return
        self._pulse_final = value
        self._flags[
            'eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the pulse final value changes

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, int):
            raise InvalidControlParameterError("Initial state index must be an integer.")

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
    def final_state(self):
        return self._final_state

    @final_state.setter
    def final_state(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, int):
            raise InvalidControlParameterError("Final state index must be an integer.")

        if value == self._final_state:  # Keep the previous value
            return

        self._final_state = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if the final state index changes
        self._flags['dia_list_computed'] = False  # Reset the diabatic computed flag if the pulse initial value changes

        # If the initial state is the same as the final state, we can mark the diabatic passage list as computed
        if self._initial_state == self._final_state:
            self._flags['dia_list_computed'] = True

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Alpha must be a number.")

        if value == self._alpha:  # Keep the previous value
            return

        self._alpha = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if alpha changes

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Beta must be a number.")

        if value == self._beta:  # Keep the previous value
            return

        self._beta = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if beta changes

    @property
    def dia_alpha(self):
        return self._dia_alpha

    @dia_alpha.setter
    def dia_alpha(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Diabatic alpha must be a number.")

        if value == self._dia_alpha:  # Keep the previous value
            return

        self._dia_alpha = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if alpha changes

    @property
    def dia_beta(self):
        return self._dia_beta

    @dia_beta.setter
    def dia_beta(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Diabatic beta must be a number.")

        if value == self._dia_beta:  # Keep the previous value
            return

        self._dia_beta = value
        self._flags['metric_computed'] = False  # Reset the metric computed flag if beta changes

    @property
    def num_steps(self):
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        if value == self._num_steps:  # Keep the previous value
            return

        if not isinstance(value, int) or value <= 0:
            raise InvalidControlParameterError("Number of steps must be a positive integer.")

        if value == self._num_steps:  # Keep the previous value
            return

        self._num_steps = value
        self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag if the number of steps changes

    @property
    def control_sol(self):
        if self._flags['ode_solved']:
            return self._control_sol
        else:
            self.solve_problem()  # Attempt to solve the ODE if not already solved
            return self._control_sol

    @property
    def pulse(self):
        all_flags_check = self._flags.all()
        if all_flags_check:
            return self._pulse
        else:
            self.solve_problem()  # Attempt to solve the pulse if not already solved
            return self._pulse

    @property
    def eigenenergies(self):
        """Return Hamiltonian eigenenergies, solving the eigenproblem if needed."""
        if self._flags['eigenproblem_solved']:
            return self._energies

        self._check_eigensystem_parameters()
        self._solve_eigenproblem()
        return self._energies

    @property
    def control_pulse(self):
        """Return the control-parameter grid, solving the eigenproblem if needed."""
        if self._flags['eigenproblem_solved']:
            return self._control_pulse

        self._check_eigensystem_parameters()
        self._solve_eigenproblem()
        return self._control_pulse

    def set_parameters(self, **params):
        """
        Set the _parameters for the Hamiltonian. This method allows you to specify any _parameters that are needed to
        compute the Hamiltonian and its partial derivative. The _parameters should be provided as keyword arguments,
        e.g., set_parameters(param1=value1, param2=value2, ...), and they will be stored. If a single parameter is
        updated, others will not be affected.
        """

        new_params = {**self._parameters, **params}

        if new_params == self._parameters:  # If the new parameters are the same as the current ones, do nothing
            return
        else:
            self._parameters = new_params  # Update the _parameters with the new values
            self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag

    def set_control(self, control_name: Optional[str] = None, pulse_initial: Optional[float] = None,
                    pulse_final: Optional[float] = None, initial_state: Optional[int] = None,
                    final_state: Optional[int] = None, alpha: Optional[float] = None, beta: Optional[float] = None,
                    dia_alpha: Optional[float] = None, dia_beta: Optional[float] = None, num_steps: int = 2 ** 10 + 1):
        """
        Set the control _parameters for the optimization problem. This method allows you to specify the control
        _parameters such as the name of the control parameter, the initial and final values of the control pulse, ....
        Control _parameters can be set individually, and if any parameter is not provided, it will retain its previous
        value, so you can update only the _parameters you want without affecting the others.

        Parameters
        ----------
        control_name : Optional[str]
            The name of the control parameter (e.g., "lambda"). This is used to identify the control parameter in the
             Hamiltonian function.
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
        num_steps : int
            The number of steps to use in the discretization of the control pulse. This determines the resolution of
             the control pulse and the accuracy of the numerical solution. Higher values will yield a more accurate
              solution but will also increase the computational cost.
        """

        self.control_name = control_name
        self.pulse_initial = pulse_initial
        self.pulse_final = pulse_final
        self.initial_state = initial_state
        self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.dia_alpha = dia_alpha
        self.dia_beta = dia_beta
        self.num_steps = num_steps

    def solve_problem(self, pulse_accuracy: int = 1000, solver: Optional[Callable] = None,
                      solver_kwargs: Optional[dict] = None, metric_integrator: Optional[Callable] = None,
                      metric_integrator_kwargs: Optional[dict] = None, ):
        """
        Solve the optimization problem to find the optimal control pulse. This method computes the metric tensor based
        on the energies and matrix elements of the Hamiltonian, and then solves the ODE for the control pulse using the
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
                selected_metric_integrator is not self._metric_integrator or selected_metric_integrator_kwargs != self._metric_integrator_kwargs):
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

        dim = self.eigenenergies.shape[1]  # Get the dimension of the Hamiltonian from the energies array
        self._dia_list = build_diab(initial_state=self.initial_state, final_state=self.final_state, dim=dim)
        self._flags['dia_list_computed'] = True  # Mark the diabatic passage list as computed to avoid recomputation

    def _solve_eigenproblem(self):
        """
        Solve the eigenproblem for the Hamiltonian at each control value to obtain the energies and matrix elements.
        This method is called internally by solve_problem() if the eigenproblem has not been solved yet. The results are
        stored in self.eigenenergies and self._matrix_elements for later use in computing the metric tensor.
        """

        if self._flags['eigenproblem_solved']:
            return  # If the eigenproblem is already solved, skip the computation

        self._control_pulse = np.linspace(self.pulse_initial, self.pulse_final, num=self.num_steps)

        full_hamiltonian = np.array(
            [self.H_func(**{self.control_name: lam, **self._parameters}) for lam in self._control_pulse])

        self._energies, eigenvectors = np.linalg.eigh(full_hamiltonian)

        if self._flag_numerical_partial_H:
            # Compute the numerical partial derivative of H with respect to the control parameter
            full_partial_H = self._compute_numerical_partial_H()

        else:
            # Use the provided analytical partial derivative function
            full_partial_H = np.array(
                [self.partial_H_func(**{self.control_name: lam, **self._parameters}) for lam in self._control_pulse])

        self._matrix_elements = np.abs(
            np.einsum('...ij,...jk,...kl->...il', eigenvectors.conj().transpose(0, 2, 1), full_partial_H, eigenvectors))

        self._flags['eigenproblem_solved'] = True  # Mark the eigenproblem as solved to avoid recomputation

    def _compute_metric_tensor(self):
        """
        Compute the metric tensor G_tensor based on the energies and matrix elements of the Hamiltonian. If the
        eigenproblem has not been solved yet, solve it first to obtain the energies and matrix elements.
        """
        if self._flags['metric_computed']:
            return

        if self._initial_state == self._final_state:
            self._compute_G_adiabatic()
        else:
            self._solve_dia_list()  # Ensure the diabatic passage list is computed before computing the metric tensor
            self._compute_G_diabatic()

        dx = float(np.abs(self._control_pulse[1] - self._control_pulse[0]))
        metric_values = np.sqrt(self._metric_tensor)

        try:
            self._a_tilde = float(self._metric_integrator(metric_values, dx=dx, **self._metric_integrator_kwargs))
        except TypeError as exc:
            raise ValidationError(
                "Invalid metric_integrator call: ensure it accepts arguments like (values, dx=..., **kwargs).") from exc

        self._flags['metric_computed'] = True  # Mark the metric as computed to avoid recomputation

    def _compute_G_diabatic(self):
        num, dim = np.shape(self.eigenenergies)

        counter_n = 0
        counter_m = 0
        diad_tensor = np.zeros([num, dim - 1, dim])

        for m in range(dim):
            for n in range(dim):

                if n != m:
                    ad_idx = self._dia_list[m][n]

                    numerator = self._matrix_elements[:, m, n]
                    denominator = np.abs(self.eigenenergies[:, n] - self.eigenenergies[:, m])

                    diad_tensor[:, counter_n, counter_m] = np.heaviside(ad_idx, 1) * (
                            ad_idx * (numerator ** self.beta / (denominator ** self.alpha)) + (1 - ad_idx) * (
                            numerator ** self.dia_beta / (denominator ** self.dia_alpha)))

                    counter_n += 1

            counter_m += 1
            counter_n = 0

        self._metric_tensor = np.sum(diad_tensor, axis=(1, 2))

    def _compute_G_adiabatic(self):
        """
        Compute the adiabatic contribution to the metric tensor G_tensor based on the energies and matrix elements
        of the
        """

        num, dim = np.shape(self.eigenenergies)
        counter = 0  # Temp variable to save the number of G_tensor computed
        G_tensor = np.zeros([num, dim - 1])
        for i in range(dim):
            if i != self.initial_state:
                numerator = self._matrix_elements[:, self.initial_state, i]
                denominator = np.abs(self.eigenenergies[:, i] - self.eigenenergies[:, self.initial_state])
                G_tensor[:, counter] = numerator ** self.beta / (denominator ** self.alpha)

                counter += 1

        self._metric_tensor = np.sum(G_tensor, axis=1)

    def _compute_numerical_partial_H(self) -> np.ndarray:
        """
        Compute the numerical partial derivative of the Hamiltonian with respect to the control parameter using finite
        differences. This method is used when the analytical partial derivative function is not provided.
        """

        def H_of_z_vec(xi):
            xs = xi[0]  # shape (...)

            xs = np.asarray(xs)

            # Evaluate once to infer matrix shape and dtype
            sample = np.asarray(self.H_func(**{self.control_name: xs.flat[0], **self._parameters}))
            flat_dim = sample.size

            out = np.empty((flat_dim, *xs.shape), dtype=sample.dtype)

            for idx in np.ndindex(xs.shape):
                out[(slice(None), *idx)] = np.asarray(
                    self.H_func(**{self.control_name: xs[idx], **self._parameters})).ravel()

            return out

        res = jacobian(H_of_z_vec, np.array([self._control_pulse])).df

        dim = int(np.sqrt(res.shape[0]))
        res = res.reshape((dim, dim, self.num_steps))
        return res.transpose(2, 0, 1)  # Reshape and transpose to get the correct shape (num_steps, dim, dim)

    def _solve_ode(self, pulse_accuracy: int):
        """
        Solve the ODE for the control pulse using the computed metric tensor and the normalization factor a_tilde.
        """

        if self._previous_pulse_accuracy != pulse_accuracy:
            self._flags['ode_solved'] = False  # Reset the ODE solved flag if the pulse accuracy changes

        if self._flags['ode_solved']:
            return  # If the ODE is already solved, skip the computation

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

        if isinstance(sol, tuple) and len(sol) == 2:
            t_arr, y_arr = sol
        elif hasattr(sol, "t") and hasattr(sol, "y"):
            t_arr, y_arr = sol.t, sol.y
        else:
            raise SolverError("Solver output must expose 't' and 'y', or return a (t, y) tuple.")

        y_arr = np.asarray(y_arr)
        self._s = np.asarray(t_arr)
        self._control_sol = y_arr[0] if y_arr.ndim == 2 else y_arr
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
            raise MissingControlParameterError(f"Missing control _parameters for eigensystem: {missing_msg}. "
                                               "Please set them using set_control(...).")

    def plot_eigenvalues(self, fig=None, ax=None, legend: bool = True, legend_kwargs: Optional[dict] = None,
                         xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None,
                         **plot_kwargs):
        """
        Plot Hamiltonian eigenvalues as a function of the control parameter.

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
            Plot title. Defaults to ``"Hamiltonian Eigenvalues"`` when not provided.
        **plot_kwargs
            Extra kwargs forwarded to ``ax.plot`` for each energy branch.

        Returns
        -------
        tuple
            ``(fig, ax)`` with the generated plot.
        """
        self._check_eigensystem_parameters()

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

        ax.set_xlabel(self.control_name if xlabel is None else xlabel)
        ax.set_ylabel("Energy" if ylabel is None else ylabel)
        ax.set_title("Hamiltonian Eigenvalues" if title is None else title)

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
        self._pulse = PulseControl(self._control_sol, duration, method, pulse_args, pulse_kwargs)
        return self._pulse()

    def _check_control_parameters(self):
        """
        Check if all necessary control _parameters are set before solving the problem. If any parameter is missing,
        raise a ConfigurationError with a clear message indicating which _parameters are missing and how to set them.
        """
        missing_params = []

        if self.control_name is None:
            missing_params.append("name")
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

        if self._initial_state != self._final_state:
            if self.dia_alpha is None:
                missing_params.append("dia_alpha")
            if self.dia_beta is None:
                missing_params.append("dia_beta")

        if missing_params:
            missing_msg = ", ".join(missing_params)
            raise MissingControlParameterError(
                f"Missing control _parameters: {missing_msg}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)}).")

    def _generate_summary(self) -> str:
        """
        Generate a summary string of the current control _parameters and settings. This method creates a formatted
        string that provides a clear overview of the control _parameters, their values, and any relevant settings for
        the optimization problem. The summary can be used for logging, debugging, or displaying the current state of the
        Hamiltonian object.
        """
        summary_lines = ["------------------ Hamiltonian Control Summary ------------------",
                         f"Hamiltonian: {'✅ set' if self.H_func is not None else '❌ not set'}",
                         f"Partial Hamiltonian: {'✅ set' if self.partial_H_func is not None else '❌ not set'}",
                         f"Hamiltonian parameters: "
                         f"{', '.join(key + ': ' + str(value) for key, value in self._parameters.items()) if self._parameters else '❌ not set'}",
                         f"Control name → {self.control_name if self.control_name is not None else '❌ not set'}",
                         f"Pulse initial → {self.pulse_initial if self.pulse_initial is not None else '❌ not set'}",
                         f"Pulse final → {self.pulse_final if self.pulse_final is not None else '❌ not set'}",
                         f"Initial state index →"
                         f" {self.initial_state if self.initial_state is not None else '❌ not set'}",
                         f"Final state index → {self.final_state if self.final_state is not None else '❌ not set'}",
                         f"(Alpha, Beta) → ({self.alpha if self.alpha is not None else '❌ not set'}, "
                         f"{self.beta if self.beta is not None else '❌ not set'})",
                         f"(Diabatic Alpha, Diabatic Beta) → ({self.dia_alpha if self.dia_alpha is not None else '❌ not set'}, "
                         f"{self.dia_beta if self.dia_beta is not None else '❌ not set'})",
                         f"Eigenproblem solved → {'✅ yes' if self._flags['eigenproblem_solved'] else '❌ no'}",
                         f"Metric computed → {'✅ yes' if self._flags['metric_computed'] else '❌ no'}",
                         f"ODE solved → {'✅ yes' if self._flags['ode_solved'] else '❌ no'}",
                         "---------------------------------------------------------------"]
        return "\n".join(summary_lines)

    def print_summary(self):
        """
        Print a summary of the current control _parameters and settings. This method provides a clear overview of the
        control _parameters, their values, and any relevant settings for the optimization problem.
        """
        print(self._generate_summary())

    def __str__(self):
        return (f"Hamiltonian(control_name={self.control_name}, pulse_initial={self.pulse_initial}, "
                f"pulse_final={self.pulse_final}, initial_state={self.initial_state}, alpha={self.alpha}, "
                f"beta={self.beta}, num_steps={self.num_steps})")

    def __repr__(self):
        return self._generate_summary()
