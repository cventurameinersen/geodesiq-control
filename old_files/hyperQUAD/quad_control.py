from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
from scipy.integrate import romb, solve_ivp, trapezoid, simpson
from scipy.interpolate import interp1d


import matplotlib.pyplot as plt

import qutip as qt

from tqdm import tqdm

from joblib import Parallel, delayed

import time



class HyperQuadControl:
    
    
    def __init__(self, hamiltonian: Optional[np.ndarray] = None, partial_hamiltonian: Optional[np.ndarray] = None,
        initial_state: Union[int, np.ndarray] = 0, final_state: Union[int, np.ndarray] = 0,
        control_init: Optional[np.ndarray] = None, control_final: Optional[np.ndarray] = None, pulse_accuracy: int = 2 ** 16 + 1, discrete_length: int = 4,
        alpha: float = 2.0, beta: float = 2.0, dia_alpha: float = 0.0, dia_beta: float = 0.0,
        pulse_time: float = 10, control_method: str = 'hyperQUAD',
        ham_args: Tuple[Any, ...] = (), ham_kwargs: Optional[Dict[str, Any]] = None, 
        diad_list: Optional[np.ndarray] = None, custom_pulse: Optional[np.ndarray] = None, ODE_method: str = 'RK45', integration_method: str = 'romb',
        qsn_samples: int = 10, qsn_variance: float = 0.1):
        
        self.hamiltonian = hamiltonian 
        self.partial_hamiltonian = partial_hamiltonian
        
        self.initial_state = initial_state
        self.final_state = final_state
        
        self.control_init = control_init
        self.control_final = control_final
        self.pulse_time = pulse_time
        self.pulse_accuracy = pulse_accuracy
        self.discrete_length = discrete_length
        
        self.alpha = alpha
        self.beta = beta
        
        self.dia_alpha = dia_alpha
        self.dia_beta = dia_beta
        
        self.control_method = control_method
        
        self.ham_args = ham_args
        self.ham_kwargs = ham_kwargs if ham_kwargs is not None else {}

        self.diad_list = diad_list

        self.custom_pulse = None
        self.ODE_method = 'RK45'  # Default ODE method, for stiff problems 'Radau'
        self.integration_method = 'romb' # romb, trapezoid, simpson

        self.qsn_samples = qsn_samples
        self.qsn_variance = qsn_variance

        # Internally used variables
        self._control_vals = None
        self._full_hamiltonian = None
        self._full_partial_hamiltonian = None

        self._eigenvalues = None
        self._eigenstates = None



    def summary(self):
        print(" ")
        print("------- HyperQuadControl Summary -------")
        print(" ")
        print(f"  Hamiltonian: {'✅ set' if self.hamiltonian is not None else '❌ not set'}")
        print(f"  Partial Hamiltonian: {'✅ set' if self.partial_hamiltonian is not None else '❌ not set'}")
        print(f"  Control init → {self.control_init}")
        print(f"  Control final → {self.control_final}")
        print(f"  Initial state index → {self.initial_state}")
        print(f"  Final state index → {self.final_state}")
        print(f"  Pulse time → {self.pulse_time}")
        print(f"  Pulse accuracy → {self.pulse_accuracy}")
        print(f"  Discrete length → {self.discrete_length}")
        print(f"  (Alpha, Beta) → ({self.alpha}, {self.beta})")
        print(f"  (Dia_Alpha, Dia_Beta) → ({self.dia_alpha}, {self.dia_beta})")
        print(f"  Control method → {self.control_method}")
        print(f"  Hamiltonian args → {self.ham_args}")
        print(f"  Diad list: {'✅ set' if self.diad_list is not None else '❌ not set'}")
        print(f"  Custom pulse: {'✅ set' if self.custom_pulse is not None else '❌ not set'}")
        print(f"  ODE method → {self.ODE_method}")
        print(f"  Integration method → {self.integration_method}")
        print(f"  Quasistatic noise (samples, variance): → ({self.qsn_samples}, {self.qsn_variance})")
        print(" ")



    def update_params(self, **kwargs):
        """
        Update internal parameters of the HyperQuadControl instance.
    
        Parameters
        ----------
        kwargs : dict
            Dictionary of parameter names and new values. Supported keys:
            - 'alpha'
            - 'beta'
            - 'dia_alpha'
            - 'dia_beta'
            - 'pulse_accuracy'
            - 'discrete_length'
            - 'control_init'
            - 'control_final'
            - 'pulse_time'
            - 'hamiltonian'
            - 'partial_hamiltonian'
            - 'ham_args'
            - 'ham_kwargs'
            - 'initial_state'
            - 'final_state'
            - 'control_method'
            - 'custom_pulse'
            - 'ODE_method'
            - 'integration_method'
            - 'qsn_samples'
            - 'qsn_variance'
        """
        valid_params = {
            'alpha', 'beta', 'dia_alpha', 'dia_beta', 'pulse_accuracy', 'discrete_length',
            'control_init', 'control_final', 'pulse_time',
            'hamiltonian', 'partial_hamiltonian',
            'ham_args', 'ham_kwargs', 'initial_state', 'final_state', 'control_method',
            'custom_pulse', 'ODE_method', 'integration_method',
            'qsn_samples', 'qsn_variance'
        }
    
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: '{key}' is not recognized.")
                
                
            
                
                



    def _generate_hyperarrays(self):
        """
        Compute (partial) hamiltonian and control function for times t in [0, pulse_time]
        
        Parameters
        ----------

        Returns
        -------
        

        """
        
        if self.hamiltonian is None:
            raise ValueError("HyperQuadControl Error ⚠️: No Hamiltonian function provided.")
        if self.partial_hamiltonian is None:
            raise ValueError("HyperQuadControl Error ⚠️: No partial Hamiltonian provided.")

        self._control_vals = np.linspace(self.control_init, self.control_final, self.pulse_accuracy)

        self._full_hamiltonian = np.array([
            self.hamiltonian(control_val, *self.ham_args, **self.ham_kwargs)
            for control_val in self._control_vals
        ])
        # self._full_partial_hamiltonian = np.array([
        #     self.partial_hamiltonian for _ in range(len(self._control_vals))
        # ])
        self._full_partial_hamiltonian = np.array([
            self.partial_hamiltonian(control_val, *self.ham_args, **self.ham_kwargs)
            for control_val in self._control_vals
        ])
        
        return self._control_vals, self._full_hamiltonian, self._full_partial_hamiltonian


    def _generate_eigenstuff(self):
        """
        Compute eigenvalues and eigenstates of the full Hamiltonian.
        
        Parameters
        ----------

        Returns
        -------
        energies : np.ndarray
            Eigenvalues of the full Hamiltonian.
        states : np.ndarray
            Eigenstates of the full Hamiltonian.
        """
        
        control_vals, full_ham, full_partial_ham = self._generate_hyperarrays()
        
        if self._eigenvalues is not None and self._eigenstates is not None:
            # If already computed, return cached values
            return self._eigenvalues, self._eigenstates
        else:
            # Compute eigenvalues and eigenstates
            self._eigenvalues, self._eigenstates = np.linalg.eigh(full_ham)
            return self._eigenvalues, self._eigenstates

    
    def _build_diad_list(self, dim: int, adia_depth: int = 6):
        """
        Create a dim x dim list such that: 
            - adiabatic = 1, 
            - diabatic = 0,
            - pass (disallowed) = -1 (e.g., degenerate or disallowed transitions)
    
        Parameters
        ----------
        dim : int
            Dimension of the Hilbert space (number of eigenstates).
        adia_depth : int, optional
            Number of extra levels to include around initial/final states to prevent overshooting.
    
        Returns
        -------
        diad_list : np.ndarray
            dim x dim matrix with entries as specified.
        """
        diad_list = -1 * np.eye(dim, dtype=int)  # Diagonal entries are -1 by default
    
        init = self.initial_state
        final = self.final_state
    
        # Define the active subspace for allowed transitions
        min_state = max(0, min(init, final) - adia_depth)
        max_state = min(dim - 1, max(init, final) + adia_depth)
    
        # Loop through all pairs in the active region
        for i in range(min_state, max_state + 1):
            for j in range(min_state, max_state + 1):
                if i == j:
                    continue  # already set to -1 on diagonal
                # Diabatic if in the core path (between init and final)
                if min(init, final) <= i <= max(init, final) and min(init, final) <= j <= max(init, final):
                    diad_list[i, j] = 0
                else:
                    diad_list[i, j] = 1  # Adiabatic transition outside the core path
    
        if self.diad_list is not None:
            # If already computed, return cached value
            return self.diad_list
        else:
            self.diad_list = diad_list
            return self.diad_list
        


        
    def plot_eigenvalues(self, shift=1, linewidth=2, bdry: bool = True):
        """
        Plot eigenvalues of hamiltonian as a function of control parameter 
        
        Parameters
        ----------

        Returns
        -------
        

        """
        control_vals, full_ham, full_partial_ham = self._generate_hyperarrays()
        energies, states = self._generate_eigenstuff()
        
        
        mod_control_vals = np.linspace(control_vals[0]+np.sign(control_vals[0])*shift, control_vals[-1]+np.sign(control_vals[-1])*shift, self.pulse_accuracy)
        
        fig, ax = plt.subplots()
        ax.plot(mod_control_vals, energies, linewidth=linewidth)
        plt.axvline(x=self.control_init, color='red', linestyle='--', linewidth=linewidth)
        plt.axvline(x=self.control_final, color='red', linestyle='--', linewidth=linewidth)
        
        idx_init = np.argmin(np.abs(mod_control_vals - self.control_init))
        idx_final = np.argmin(np.abs(mod_control_vals - self.control_final))
        
        if bdry:
            ax.plot(self.control_init, energies[idx_init, self.initial_state], 'ro', markersize=10)  
            ax.plot(self.control_final, energies[idx_final, self.final_state], 'ro', markersize=10)  
        
        plt.xlabel("Control parameter", fontsize=15)
        plt.ylabel("Energies", fontsize=15)
        plt.tick_params(labelsize=15)
        plt.grid(True)
        
        
        
    
    def generate_adiabatic_params(self):
        
        """
        Compute hypermetric tensor G and rescaled adiabaticity a_tilde.
        
               
        Parameters
        ----------
        
        Returns
        -------
        
        
        
        """
        
        control_vals, full_ham, full_partial_ham = self._generate_hyperarrays()
        energies, states = self._generate_eigenstuff()
        
        num, dim = np.shape(energies)
        
            
        def G_tensor(energies, states, num, dim):
            
            counter = 0
            G_tensor = np.zeros([num, dim-1])
            m = self.initial_state
            
            for n in range(dim): 
                if n != m:
                    
                    num = np.abs(
                        np.einsum('ia,iab,ib->i', 
                                  states[..., m].conj(), 
                                  full_partial_ham, 
                                  states[..., n],
                                  optimize='greedy')
                    ) 
            
                    den = np.abs(energies[:, n] - energies[:, m]) 
                    
                    G_tensor[:, counter] = num**self.beta / (den**self.alpha)
                    
                    counter += 1
                            
                    
        
            G_tensor = np.sum(G_tensor, axis=1)
            
            return G_tensor
        
        
        
        
        
        diad_list = self._build_diad_list(dim=dim)
        
        
        def diad_tensor(energies, states, num, dim):
            counter_n = 0
            counter_m = 0
            diad_tensor = np.zeros([num, dim-1, dim])
            
            for m in range(dim):
                for n in range(dim):
                    
                    if n != m:
                        
                        ad_idx = diad_list[m][n]
                                                
                        num = np.abs(
                            np.einsum('ia,iab,ib->i', 
                                      states[..., m].conj(), 
                                      full_partial_ham, 
                                      states[..., n],
                                      optimize='greedy')
                        ) 
                
                        den = np.abs(energies[:, n] - energies[:, m]) 
                        
                        diad_tensor[:, counter_n, counter_m] = np.heaviside(ad_idx, 1) * ( ad_idx * ( num**self.beta / (den**self.alpha) ) 
                                                                                         + (1-ad_idx) * ( num**(self.dia_beta) / (den**(self.dia_alpha)) ) )
                        
                        counter_n += 1
                
                counter_m += 1
                counter_n = 0
                
            diad_tensor = np.sum(diad_tensor, axis=(1,2))
            
            return diad_tensor
        
        
        
        
        if self.control_method == 'hyperQUAD' or self.control_method == 'discrete_hyperQUAD' or self.control_method == 'filtered_hyperQUAD':
            metric_tensor = G_tensor(energies=energies, states=states, num=num, dim=dim)
        elif self.control_method == 'diad':
            metric_tensor = diad_tensor(energies=energies, states=states, num=num, dim=dim)
        else:
            raise ValueError(f"HyperQuadControl Error ⚠️: No method '{self.control_method}' known.")
        


        if self.integration_method == 'romb':
            f_int = romb
        elif self.integration_method == 'trapezoid':
            f_int = trapezoid
        elif self.integration_method == 'simpson':
            f_int = simpson
        else:
            raise ValueError(f"HyperQuadControl Error ⚠️: No method '{self.integration_method}' known.")
            
        a_tilde = f_int(np.sqrt(metric_tensor), dx=np.abs(control_vals[1] - control_vals[0]))
        
        
        return control_vals, metric_tensor, float(a_tilde)
    
        
    
    def check_metric_stability(self, diad: bool = False, depth: int = 3):
        """
        Check stability of the hypermetric tensor G.
        
        Parameters
        ----------
        threshold : float, optional
            Determines the max threshold diabatic metric tensor.
        
        Returns
        -------
        Text
            Figure if the (maximum value of the) hypermetric tensor is stable
        """
        steps = 4
        alphas = np.linspace(-depth, depth, steps)
        betas = alphas
        alpha_beta_list = [(alpha, beta) for alpha in alphas for beta in betas]

        max_metrics = []

        for alpha, beta in tqdm(alpha_beta_list, desc="Checking stability"):
            if diad:
                self.update_params(dia_alpha=alpha, dia_beta=beta)
            else:
                self.update_params(alpha=alpha, beta=beta)

            _, metric_tensor, _ = self.generate_adiabatic_params()
            max_metric = np.max(metric_tensor)
            max_metrics.append(max_metric)

        max_metrics = np.array(max_metrics).reshape(steps, steps)
        plt.figure(figsize=(6, 5))
        X, Y = np.meshgrid(alphas, betas)
        cp = plt.contourf(X, Y, np.log10(max_metrics), levels=20, cmap='viridis')
        plt.colorbar(cp, label=r'$\log_{10}(\max G)$')
        plt.xlabel('Alpha', fontsize=14)
        plt.ylabel('Beta', fontsize=14)
        plt.tight_layout()
        plt.show()

        
        
    
    
    
    
    def compute_optimal_pulse(self):
        """
        Solve adiabtic protocol given G, a_tilde.
        
               
        Parameters
        ----------

        Returns
        -------
        
        """
        control_vals, G_tensor, a_tilde = self.generate_adiabatic_params()
        
        
        
        sig = np.sign(control_vals[1] - control_vals[0])

        factor_interpolation = interp1d(control_vals, a_tilde / np.sqrt(G_tensor), kind='quadratic', fill_value="extrapolate")
    
        def model(t, y):  
            return sig * factor_interpolation(y)
    
        s = np.linspace(0, 1, self.pulse_accuracy)
        sol = solve_ivp(model, [0, 1], [control_vals[0]], t_eval=s, method=self.ODE_method, atol=1e-15, rtol=1e-13, dense_output=True) #atol=1e-8 (1e-15), rtol=1e-4 (1e-12)
        s = sol.t
        control_sol = sol.y[0]
    
    
        return s, control_sol
        
   
        
        
    def discretized_pulse(self):
        """
        Obtains the piecewise linear function from hypergeometric pulse.
        
               
        Parameters
        ----------

        Returns
        -------
        
        
        
        """
        s, control_sol = self.compute_optimal_pulse()
        
        piecewise_linear = interp1d(s, control_sol, kind='linear', fill_value="extrapolate")  

        new_s = np.linspace(s[0], s[-1], self.discrete_length)
        approx_sol = piecewise_linear(new_s)
        
        return new_s, approx_sol
        
        
    def filtered_pulse(self):
        """
        Obtains the filtered piecewise linear function from hypergeometric pulse.
        
               
        Parameters
        ----------

        Returns
        -------
        
        
        
        """
        
        s, discrete_sol = self.discretized_pulse()
        
        pass
        
 
        
 
    
 
    def plot_pulse(self):
        """
        Plotting script to show the hypergeometric optimal pulse shape.
        
               
        Parameters
        ----------

        Returns
        -------
        
        
        
        """
        if self.control_method == 'hyperQUAD' or self.control_method == 'diad' or self.control_method == 'discrete_hyperQUAD' or self.control_method == 'filtered_hyperQUAD':
            s, control_sol = self.compute_optimal_pulse()
        elif self.control_method == 'custom':
            if self.custom_pulse is None:
                raise ValueError("Custom pulse not set. Please set 'custom_pulse' before using 'custom' control method.")
            else:
                s = np.linspace(0, 1, self.pulse_accuracy)
                control_sol = self.custom_pulse
    
        plt.figure(figsize=(6, 4))
        plt.plot(s, control_sol, linewidth=2, label = fr"$\alpha, \beta, \hat{{\alpha}}, \hat{{\beta}}$ = {self.alpha}, {self.beta}, {self.dia_alpha}, {self.dia_beta}")
        plt.xlabel("Rescaled pulse time $s$", fontsize=13)
        plt.ylabel("Control pulse", fontsize=13)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plt.show()



    def plot_metric_tensor(self):
        """
        Plotting script to show the hypergeometric metric tensor as function of control.
        
        Parameters
        ----------

        Returns
        -------
        
        
        
        """
        control_vals, metric_tensor, _ = self.generate_adiabatic_params()
    
        plt.figure(figsize=(6, 4))
        plt.plot(control_vals, metric_tensor, linewidth=2, label = fr"$\alpha, \beta, \hat{{\alpha}}, \hat{{\beta}}$ = {self.alpha}, {self.beta}, {self.dia_alpha}, {self.dia_beta}")
        plt.xlabel("Control parameter", fontsize=13)
        plt.ylabel("Metric tensor", fontsize=13)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plt.show()
        
        
        
       
        
        
    def compute_fidelity_old(self, c_ops = None, infidelity: bool = False):
        """
        Script to compute the state-transfer fidelity using the hypergeometric pulses.
        
               
        Parameters
        ----------

        Returns
        -------
        
        """
        
        control_vals, _ , _ = self._generate_hyperarrays()
        
        
        if self.control_method == 'hyperQUAD' or self.control_method == 'diad':
            _, control_sol = self.compute_optimal_pulse()
        elif self.control_method == 'discrete_hyperQUAD':
            _, control_sol = self.discretized_pulse()
        elif self.control_method == 'filtered_hyperQUAD':
            _, control_sol = self.filtered_pulse()
        elif self.control_method == 'custom':
            if self.custom_pulse is None:
                raise ValueError("Custom pulse not set. Please set 'custom_pulse' before using 'custom' control method.")
            else:
                control_sol = self.custom_pulse
        else:
            raise ValueError(f"Unknown control method: {self.control_method}")
        
        times = np.linspace(0, self.pulse_time, len(control_sol))
    
        H_0 = qt.Qobj(self.hamiltonian(0, *self.ham_args))
        H_driving = qt.Qobj(self.partial_hamiltonian)
        H_T = [H_0, [H_driving, qt.coefficient(control_sol, tlist=times)]]
        H_T = qt.QobjEvo(H_T)
    
        # psi_0 = qt.Qobj(self.hamiltonian(control_vals[0], *self.ham_args)).groundstate()[1]
        init_eigenvals, init_eigenstates = qt.Qobj(self.hamiltonian(control_vals[0], *self.ham_args)).eigenstates()
        psi_init = init_eigenstates[self.initial_state]

        final_eigenvals, final_eigenstates = qt.Qobj(self.hamiltonian(control_vals[-1], *self.ham_args)).eigenstates()
        psi_target = final_eigenstates[self.final_state]
        #psi_target = qt.Qobj(self.hamiltonian(control_vals[-1], *self.ham_args)).groundstate()[1]
    
        if c_ops is not None:
            if not isinstance(c_ops, list):
                c_ops = [qt.Qobj(c_ops)]
            else:
                c_ops = [qt.Qobj(op) for op in c_ops]

    
        psi_f = qt.mesolve(H_T, psi_init, times, c_ops=c_ops).states[-1]
        
        if infidelity:
            return 1 - qt.fidelity(psi_target, psi_f) ** 2
    
        else:
            return qt.fidelity(psi_target, psi_f) ** 2
        
        
        
    def compute_fidelity(self, c_ops=None, infidelity: bool = False, quasistatic: bool = False, qsn_parallel: bool = False):
        """
        Compute the state-transfer fidelity for a fully time-dependent Hamiltonian
        based on a single control function x(t) that may affect different terms differently.

        Parameters
        ----------
        c_ops : list or Qobj, optional
            Collapse operators for open system evolution. Default is None.
        infidelity : bool, optional
            If True, return 1 - fidelity^2. Default is False.

        Returns
        -------
        float
            Fidelity or infidelity between the evolved state and target state.
        """

        # Generate control values (hypergeometric pulses)
        control_vals, _, _ = self._generate_hyperarrays()

        # Get control pulse based on selected method
        if self.control_method in ['hyperQUAD', 'diad']:
            _, control_sol = self.compute_optimal_pulse()
        elif self.control_method == 'discrete_hyperQUAD':
            _, control_sol = self.discretized_pulse()
        elif self.control_method == 'filtered_hyperQUAD':
            _, control_sol = self.filtered_pulse()
        elif self.control_method == 'custom':
            if self.custom_pulse is None:
                raise ValueError("Custom pulse not set. Please set 'custom_pulse' before using 'custom' control method.")
            control_sol = self.custom_pulse
        else:
            raise ValueError(f"Unknown control method: {self.control_method}")

        # Time array
        times = np.linspace(0, self.pulse_time, len(control_sol))

        # Initial and target states
        _ , init_eigenstates = qt.Qobj(self.hamiltonian(control_sol[0], *self.ham_args)).eigenstates()
        psi_init = init_eigenstates[self.initial_state]

        _ , final_eigenstates = qt.Qobj(self.hamiltonian(control_sol[-1], *self.ham_args)).eigenstates()
        psi_target = final_eigenstates[self.final_state]

        # Prepare collapse operators if any
        if c_ops is not None:
            if not isinstance(c_ops, list):
                c_ops = [qt.Qobj(c_ops)]
            else:
                c_ops = [qt.Qobj(op) for op in c_ops]


        def _compute_fidelity(control_sol, times, c_ops, infidelity, quasistatic):
            
            def H_t_func(t, args):
                ham_args = args["ham_args"]
                x_array = args["x_array"]
                t_array = args["t_array"]
                x_t = np.interp(t, t_array, x_array)
                return qt.Qobj(self.hamiltonian(x_t, *ham_args))

            qsn_control_sol = control_sol*(1+np.random.normal(0, self.qsn_variance)) if quasistatic else  control_sol

            H_T = qt.QobjEvo(H_t_func, args={
                "ham_args": self.ham_args,
                "x_array": qsn_control_sol,
                "t_array": times
            })

            psi_f = qt.mesolve(H_T, psi_init, times, c_ops=c_ops).states[-1]

            fid = qt.fidelity(psi_target, psi_f) ** 2
            return 1 - fid if infidelity else fid


        num_samples = self.qsn_samples

        if quasistatic and (not qsn_parallel):
            fidelities = []
            for _ in range(num_samples):
                fidelity = _compute_fidelity(control_sol, times, c_ops, infidelity, quasistatic=quasistatic)
                fidelities.append(fidelity)
            return np.mean(fidelities), np.median(fidelities), np.std(fidelities)

        elif quasistatic and qsn_parallel:
            fidelities = Parallel(n_jobs=int(num_samples/4))(delayed(_compute_fidelity)(control_sol, times, c_ops, infidelity, quasistatic=True) for _ in range(num_samples))
            fidelities = np.array(fidelities)
            return np.mean(fidelities), np.median(fidelities), np.std(fidelities)

        else:
            return _compute_fidelity(control_sol, times, c_ops, infidelity, quasistatic=quasistatic)



        # Define fully time-dependent Hamiltonian
        # def H_t_func(t, args):
        #     ham_args = args["ham_args"]
        #     x_array = args["x_array"]
        #     t_array = args["t_array"]
        #     x_t = np.interp(t, t_array, x_array)
        #     return qt.Qobj(self.hamiltonian(x_t, *ham_args))

        # qsn_control_sol = control_sol*(1+np.random.normal(0, self.qsn_variance)) if quasistatic else control_sol

        # H_T = qt.QobjEvo(H_t_func, args={
        #     "ham_args": self.ham_args,
        #     "x_array": qsn_control_sol,
        #     "t_array": times
        # })

        # # Solve dynamics
        # psi_f = qt.mesolve(H_T, psi_init, times, c_ops=c_ops).states[-1]

        # # Return fidelity or infidelity
        # fid = qt.fidelity(psi_target, psi_f) ** 2
        # return 1 - fid if infidelity else fid

    
    
    
    
    
    def _apply_sweep_value(self, key, value):
        if isinstance(key, str):
            self.update_params(**{key: value})
        elif isinstance(key, tuple) and key[0] == 'ham_args':
            args = list(self.ham_args)
            args[key[1]] = value
            self.update_params(ham_args=tuple(args))
        elif isinstance(key, tuple) and key[0] == 'ham_kwargs':
            kwargs = self.ham_kwargs.copy()
            kwargs[key[1]] = value
            self.update_params(ham_kwargs=kwargs)
        else:
            raise ValueError(f"Unsupported sweep key: {key}")
            
            
            
            

    def fidelity_sweep(self, sweep_dict: Dict, c_ops=None, infidelity: bool = False, quasistatic: bool = False, qsn_parallel: bool = False):
        """
        Flexible 1D or 2D fidelity sweep over parameters.
    
        Parameters
        ----------
        sweep_dict : dict
            Keys are parameter names or tuple-indexed values (e.g., ('ham_args', 0)).
            Values are either scalars (no sweep) or arrays (sweep values).
            - 1 array → 1D sweep.
            - 2 arrays → 2D grid sweep.
    
        c_ops : optional
            Collapse operators passed to compute_fidelity().
            
    
        Returns
        -------
        1D or 2D array of fidelities, depending on sweep dimensionality.
        """
    
        sweep_params = {k: v for k, v in sweep_dict.items() if hasattr(v, '__len__') and not isinstance(v, str)}
        fixed_params = {k: v for k, v in sweep_dict.items() if k not in sweep_params}
    
        sweep_keys = list(sweep_params.keys())
    
        original_params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'dia_alpha': self.dia_alpha,
            'dia_beta': self.dia_beta,
            'pulse_accuracy': self.pulse_accuracy,
            'control_init': self.control_init,
            'control_final': self.control_final,
            'hamiltonian': self.hamiltonian,
            'partial_hamiltonian': self.partial_hamiltonian,
            'ham_args': self.ham_args,
            'ham_kwargs': self.ham_kwargs.copy(),
            'initial_state': self.initial_state,
            'final_state': self.final_state,
            'pulse_time': self.pulse_time,
        }
    
        # Apply fixed parameters
        self.update_params(**{k: v for k, v in fixed_params.items() if isinstance(k, str)})
    
        # 1D Sweep
        if len(sweep_keys) == 1:
            key = sweep_keys[0]
            values = sweep_params[key]
            fidelities = []
            fidelities_median = []
            fidelities_std = []

            if quasistatic:
                for val in tqdm(values, desc=f"Sweeping {key}"):
                    self._apply_sweep_value(key, val)
                    mean_fid, median_fid, std_fid = self.compute_fidelity(c_ops=c_ops, infidelity=infidelity, quasistatic=quasistatic, qsn_parallel=qsn_parallel)
                    fidelities.append(mean_fid)
                    fidelities_median.append(median_fid)
                    fidelities_std.append(std_fid)
                self.update_params(**original_params)
                return np.array(fidelities), np.array(fidelities_median), np.array(fidelities_std)

            else:
                for val in tqdm(values, desc=f"Sweeping {key}"):
                    self._apply_sweep_value(key, val)
                    fidelity = self.compute_fidelity(c_ops=c_ops, infidelity=infidelity, quasistatic=quasistatic, qsn_parallel=qsn_parallel)
                    fidelities.append(fidelity) 
                self.update_params(**original_params)
                return np.array(fidelities)
    
        # 2D Sweep
        elif len(sweep_keys) == 2:
            key1, key2 = sweep_keys
            values1, values2 = sweep_params[key1], sweep_params[key2]
            fidelities = np.zeros((len(values1), len(values2)))
    
            outer_loop = tqdm(enumerate(values1), total=len(values1), desc=f"Sweeping {key1}")
            for i, val1 in outer_loop:
                self._apply_sweep_value(key1, val1)
                inner_loop = tqdm(enumerate(values2), total=len(values2), desc=f"Sweeping {key2}", leave=False)
                for j, val2 in inner_loop:
                    self._apply_sweep_value(key2, val2)
                    fidelities[i, j] = self.compute_fidelity(c_ops=c_ops, infidelity=infidelity, quasistatic=quasistatic)
    
            self.update_params(**original_params)
            return fidelities
    
        else:
            raise ValueError("Only 1D or 2D sweeps are supported (1 or 2 array-like values in sweep_dict).")
        


    def fidelity_sweep_parallel(self, sweep_dict: Dict, c_ops=None, 
                            infidelity: bool = False, n_jobs: int = 3):
        sweep_params = {k: v for k, v in sweep_dict.items() if hasattr(v, '__len__') and not isinstance(v, str)}
        fixed_params = {k: v for k, v in sweep_dict.items() if k not in sweep_params}
        sweep_keys = list(sweep_params.keys())

        original_params = {
            'alpha': self.alpha,
            'beta': self.beta,
            'pulse_accuracy': self.pulse_accuracy,
            'control_init': self.control_init,
            'control_final': self.control_final,
            'hamiltonian': self.hamiltonian,
            'partial_hamiltonian': self.partial_hamiltonian,
            'ham_args': self.ham_args,
            'ham_kwargs': self.ham_kwargs.copy(),
            'initial_state': self.initial_state,
            'pulse_time': self.pulse_time,
        }

        # Apply fixed params
        self.update_params(**{k: v for k, v in fixed_params.items() if isinstance(k, str)})

        # 1D Sweep
        if len(sweep_keys) == 1:
            key = sweep_keys[0]
            values = sweep_params[key]

            def compute_single_1d(val):
                self._apply_sweep_value(key, val)
                return self.compute_fidelity(c_ops=c_ops, infidelity=infidelity)

            fidelities = Parallel(n_jobs=n_jobs)(
                delayed(compute_single_1d)(val) for val in tqdm(values, desc=f"Parallel sweeping {key}")
            )

            self.update_params(**original_params)
            return np.array(fidelities)

        # 2D Sweep
        elif len(sweep_keys) == 2:
            key1, key2 = sweep_keys
            values1, values2 = sweep_params[key1], sweep_params[key2]

            def compute_single_2d(i, j, val1, val2):
                self.update_params(**original_params)  # reset before each combo
                self._apply_sweep_value(key1, val1)
                self._apply_sweep_value(key2, val2)
                return i, j, self.compute_fidelity(c_ops=c_ops, infidelity=infidelity)

            tasks = [(i, j, val1, val2) for i, val1 in enumerate(values1) for j, val2 in enumerate(values2)]
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_single_2d)(i, j, val1, val2)
                for i, j, val1, val2 in tqdm(tasks, desc="Parallel 2D sweep")
            )

            fidelities = np.zeros((len(values1), len(values2)))
            for i, j, fid in results:
                fidelities[i, j] = fid

            self.update_params(**original_params)
            return fidelities

        else:
            raise ValueError("Only 1D or 2D sweeps are supported (1 or 2 array-like values in sweep_dict).")
    
    
        
        
        
        
        
        
        
    
    
    
    
        
        
        
        