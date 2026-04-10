from typing import Optional, Callable, Any

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import solve_ivp, romb
from scipy.interpolate import interp1d

from ._utils import Flags
from .exceptions import ImmutableConfigurationError, MissingControlParameterError, InvalidControlParameterError
from .pulses import PulseControl


class Hamiltonian:
    """
    A class to represent a parameter-dependent Hamiltonian and solve the optimization problem for finding the optimal
    control pulse. The Hamiltonian is defined as a function of a control parameter (e.g., lambda) and can also depend
    on other _parameters. The class provides methods to set the _parameters, compute the metric tensor, solve the ODE
    for the control pulse, and synthesize the control pulse based on the solution of the optimization problem.
    """

    def __init__(self, H_func: Callable[[Any], np.ndarray],
                 partial_H_func: Optional[Callable[[Any], np.ndarray]] = None):
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
        self._flags = Flags()
        self._flags.add('eigenproblem_solved')
        self._flags.add('metric_computed', parent='eigenproblem_solved')
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
        self._alpha = None
        self._beta = None
        self._num_steps = None

        # Initialize energy gaps and matrix elements to None (to be computed in self.solve_problem())
        self._energies = None
        self._matrix_elements = None

        # Initialize metric tensor and normalization factor to None (to be computed in self.solve_problem())
        self._metric_tensor = None
        self._a_tilde = None

        # Initialize pulse _parameters
        self._s = None
        self._control_pulse = None
        self._control_sol = None
        self._pulse = None

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
        self._initial_state = value
        self._flags['metric_computed'] = False  # Reset the  metric computed flag if the initial state index changes

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:  # Keep the previous value
            return

        if not isinstance(value, (int, float)):
            raise InvalidControlParameterError("Alpha must be a number.")
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
        self._beta = value
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

    def set_parameters(self, **params):
        """
        Set the _parameters for the Hamiltonian. This method allows you to specify any _parameters that are needed to
        compute the Hamiltonian and its partial derivative. The _parameters should be provided as keyword arguments,
        e.g., set_parameters(param1=value1, param2=value2, ...), and they will be stored. If a single parameter is
        updated, others will not be affected.
        """
        self._parameters = {**self._parameters, **params}  # Update the _parameters with the new values
        self._flags['eigenproblem_solved'] = False  # Reset the eigenproblem solved flag

    def set_control(self, control_name: Optional[str] = None, pulse_initial: Optional[float] = None,
                    pulse_final: Optional[float] = None, initial_state: Optional[int] = None,
                    alpha: Optional[float] = None, beta: Optional[float] = None, num_steps: int = 2 ** 10 + 1):
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
        alpha : Optional[float]
            The exponent alpha used in the metric tensor computation. This parameter controls the weighting of the
             energy gaps in the metric tensor.
        beta : Optional[float]
            The exponent beta used in the metric tensor computation. This parameter controls the weighting of the
            matrix elements in the metric tensor.
        num_steps : int
            The number of steps to use in the discretization of the control pulse. This determines the resolution of
             the control pulse and the accuracy of the numerical solution. Higher values will yield a more accurate
              solution but will also increase the computational cost.
        """

        self.control_name = control_name
        self.pulse_initial = pulse_initial
        self.pulse_final = pulse_final
        self.initial_state = initial_state
        # self.final_state = final_state
        self.alpha = alpha
        self.beta = beta
        self.num_steps = num_steps

    def solve_problem(self, pulse_accuracy: int = 1000, solver_kwargs: Optional[dict] = None):
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
        solver_kwargs : Optional[dict]
            Additional keyword arguments for the ODE solver (if any). This allows you to customize the behavior of the
             ODE solver, such as the integration method, tolerances, etc.
        """
        self._check_control_parameters()

        self._control_pulse = np.linspace(self.pulse_initial, self.pulse_final, num=self.num_steps)

        self._solve_eigenproblem()
        self._compute_metric_tensor()

        if solver_kwargs is None:
            solver_kwargs = {}

        self._solve_ode(pulse_accuracy, solver_kwargs)

    def _solve_eigenproblem(self):
        """
        Solve the eigenproblem for the Hamiltonian at each control value to obtain the energies and matrix elements.
        This method is called internally by solve_problem() if the eigenproblem has not been solved yet. The results are
        stored in self._energies and self._matrix_elements for later use in computing the metric tensor.
        """
        if self._flags['eigenproblem_solved']:
            return  # If the eigenproblem is already solved, skip the computation

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
        num, dim = np.shape(self._energies)

        # ToDo: Implement the diabatic passage
        counter = 0  # Temp variable to save the number of G_tensor computed
        G_tensor = np.zeros([num, dim - 1])
        for i in range(dim):
            if i != self.initial_state:
                num = self._matrix_elements[:, self.initial_state, i]
                den = np.abs(self._energies[:, i] - self._energies[:, self.initial_state])
                G_tensor[:, counter] = num ** self.beta / (den ** self.alpha)

                counter += 1

        self._metric_tensor = np.sum(G_tensor, axis=1)
        self._a_tilde = float(
            romb(np.sqrt(self._metric_tensor), dx=float(np.abs(self._control_pulse[1] - self._control_pulse[0]))))

        self._flags['metric_computed'] = True  # Mark the metric as computed to avoid recomputation

        """
        # diad_list = self._build_diad_list(dim=dim)
        # diad_tensor = np.zeros([num, dim - 1, dim])
        # counter_n = 0
        # counter_m = 0
        #
        # for m in range(dim):
        #     for n in range(dim):
        #
        #         if n != m:
        #             ad_idx = diad_list[m][n]
        #
        #             num = np.abs(np.einsum('ia,iab,ib->i', states[..., m].conj(), full_partial_H, states[..., n],
        #                                    optimize='greedy'))
        #
        #             den = np.abs(energies[:, n] - energies[:, m])
        #
        #             diad_tensor[:, counter_n, counter_m] = np.heaviside(ad_idx, 1) * (
        #                     ad_idx * (num ** self.beta / (den ** self.alpha)) + (1 - ad_idx) * (
        #                     num ** (self.dia_beta) / (den ** (self.dia_alpha))))
        #
        #             counter_n += 1
        #
        #     counter_m += 1
        #     counter_n = 0
        
        # diad_tensor = np.sum(diad_tensor, axis=(1, 2))
        """

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

    def _solve_ode(self, pulse_accuracy: int, solve_kwargs):
        """
        Solve the ODE for the control pulse using the computed metric tensor and the normalization factor a_tilde.
        """
        if self._flags['ode_solved']:
            return  # If the ODE is already solved, skip the computation

        sig = np.sign(self._control_pulse[1] - self._control_pulse[0])

        factor_interpolation = interp1d(self._control_pulse, self._a_tilde / np.sqrt(self._metric_tensor),
                                        kind='quadratic', fill_value="extrapolate")

        def model(_, y):
            return sig * factor_interpolation(y)

        s = np.linspace(0, 1, pulse_accuracy)
        sol = solve_ivp(model, [0, 1], [self._control_pulse[0]], t_eval=s, dense_output=True, method='RK45', atol=1e-8,
                        rtol=1e-6, **solve_kwargs)
        self._s = sol.t
        self._control_sol = sol.y[0]
        self._flags['ode_solved'] = True

    def synthesize_pulse(self, duration: float, kwargs_filter: Optional[dict] = None,
                         convoluted_array: Optional[np.ndarray] = None):
        """
        Synthesize the control pulse based on the solution of the optimization problem. If the problem has not been
        solved yet, this method will automatically solve it first.

        Parameters
        ----------
        duration : float
            The total duration of the control pulse.
        kwargs_filter : Optional[dict]
            Additional keyword arguments for the pulse filter (if any).
        convoluted_array : Optional[np.ndarray]
            An optional array to convolve with the control pulse (if any).

        Returns
        -------
        PulseControl
             An instance of the PulseControl class representing the synthesized control pulse.
        """
        self.solve_problem()
        self._pulse = PulseControl(self._s,
                                   self._control_sol)  # ToDo: Use the actual Pulse class instead of a placeholder
        return self._pulse

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
        if self.initial_state is None:  # ToDo: Include checks when diabatic is implemented
            missing_params.append("initial_state")
        if self.alpha is None:
            missing_params.append("alpha")
        if self.beta is None:
            missing_params.append("beta")

        if missing_params:
            missing_msg = ", ".join(missing_params)
            raise MissingControlParameterError(
                f"Missing control _parameters: {missing_msg}. Please set them using set_control"
                f"({', '.join(f'{name}=<...>' for name in missing_params)}).")

    def _generate_summary(self) -> str:
        """
        Generate a summary string of the current control _parameters and settings. This method creates a formatted string
        that provides a clear overview of the control _parameters, their values, and any relevant settings for the
        optimization problem. The summary can be used for logging, debugging, or displaying the current state of the
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
                         f"(Alpha, Beta) → ({self.alpha if self.alpha is not None else '❌ not set'}, "
                         f"{self.beta if self.beta is not None else '❌ not set'})",
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
