# `geodesiq`: Geometric optimal control
[Documentation](#documentation) | [Installation](#installation) | [Example code](#example-code) | [Citing geodesiq](#citing-geodesiq)

`geodesiq` is a Python package for optimal pulse control of Hamiltonian parameters for generic quantum systems.




# Documentation
Documentation is available [here](www.github.com).

# Installation
To install `geodesiq`, you can use the standard Python package installer:
```bash
pip install geodesiq
```

# Example code
Here is an example code based on the two-level Landau-Zener problem $H[z(t)]=z(t)\,\sigma_z+x\, \sigma_x$ with control parameter $z(t)$. To compute the optimal pulse, you establish the base Hamiltonian and (optionally) the partial derivative of the Hamiltonian with respect to the control parameter.

```python
import numpy as np
from geodesiq import Hamiltonian

# ----- Define Hamiltonian and its gradient -----
def H_fun(x, z):
    return np.array([[z, x],
                     [x, -z]])

def H_partial(x, z):
    return np.array([[1, 0],
                     [0, -1]])

hamiltonian = Hamiltonian(H_fun, H_partial)

# ----- Set system and control parameters -----
alpha = 2
beta = 2
x = 1
z0 = -10
zf = -z0

hamiltonian.set_parameters(x=x)
hamiltonian.set_control(control_name='z', pulse_initial=z0, pulse_final=zf,
                        initial_state=0, alpha=alpha, beta=beta)

# ----- Solve for optimal pulse -----
hamiltonian.solve_problem()
```

# Citing `geodesiq`
If you use `geodesiq` in your research, please cite the reference paper available [here](www.github.com).
