# ============================================
# 1. Import the package
# ============================================

import numpy as np

import geodesiq as gq


# ============================================
# 2. Define a custom Hamiltonian
#    H(lambda, **params) -> matrix
# ============================================

def H_landau_zener(lam, x):
    # lam is the control parameter to be pulsed
    return np.array([[lam, x], [x, -lam]], dtype=complex)


def partial_H(lam, x):
    # partial derivative of H with respect to lam
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# ============================================
# 3. Create a control problem
# ============================================

problem = gq.Hamiltonian(H_landau_zener, partial_H_func=partial_H)
problem.set_parameters(x=1.0)
problem.set_control(name="lam", pulse_initial=-10.0, pulse_final=+10.0, initial_state=0, final_state=1, alpha=4,
                    beta=2)  # initial eigenstate to transfer adiabatically
problem.solve_problem()  # <--- The calculation (heavy work) is done here.

# ============================================
# 4. Synthesize the pulse
# ============================================

# ↓ If problem.solve_problem() hasn't been called, this will call it automatically with default settings. (Show a message)
filtered_pulse: Pulse = problem.synthesize_pulse(duration=20.0, kwargs_filter={'order': 5, 'cutoff': 0.8,
                                                                               'filter_name': 'butterworth'})
pulse_fast = problem.synthesize_pulse(duration=10.0, ...)
convoluted_pulse = problem.synthesize_pulse(duration=2, convoluted_array=np.array(...))

filtered_pulse.plot()  # plot the pulse shape
filtered_pulse.plot_spectrum()  # plot the Fourier spectrum of the pulse

# ============================================
# 5. Simulate the dynamics and compute fidelity
# ============================================

# This will use QuTiP to simulate the dynamics under the given pulse and compute the fidelity of the state transfer.
result = problem.simulate(pulse=filtered_pulse, **qutip_kwargs)
print("Final fidelity:", result.final_fidelity)
result.plot_populations()
result.plot_fidelity_vs_time()

# ============================================
# 6. Add noise models
# ============================================

noise_result = problem.simulate(pulse=filtered_pulse,
                                noise=gq.noise.Quasistatic(sigma_x=0.02, sigma_z=0.01, samples=500))

print("Average noisy fidelity:", noise_result.mean_fidelity)
noise_result.plot_fidelity_histogram()

# ============================================
# 7. Scan over alpha, beta, and duration
#     to find the best protocol
# ============================================
alphas = [2, 4, 6]
betas = [2, 4]
fidelity = gq.parameter_scan(problem=problem, alphas=alphas, betas=betas, durations=np.linspace(5.0, 50.0, 40))

gq.plot_heatmap(x=alphas, y=betas, value=np.max(fidelity, axis=2))

# ============================================
# 8. Export pulse for experiment
# ============================================

filtered_pulse.export_csv("pulse.csv")
