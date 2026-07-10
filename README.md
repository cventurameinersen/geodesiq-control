<p align="center">
  <img src="images/geodesiq_logo.png" alt="Logo"/>
</p>

# `geodesiq`: Geometric optimal control

[Installation](#installation) | [Example code](#example-code) | [Citing geodesiq](#citing-geodesiq)

`geodesiq` is a Python package for optimal pulse control of Hamiltonian parameters for generic quantum systems.


[//]: # (# Documentation)

[//]: # (Documentation is available [here]&#40;https://github.com/geodesiq-control/geodesiq-control&#41;.)

# Installation

To install `geodesiq`, you can use the standard Python package installer:

```bash
pip install geodesiq
```

# Example code

Here is an example code based on the two-level Landau-Zener problem $H[z(t)]=z(t)\sigma_z+x \sigma_x$ with control
parameter $z(t)$. To compute the optimal pulse, you define the Hamiltonian and, optionally, its partial
derivative with respect to the control parameter.

```python
import numpy as np
from geodesiq import ControlModel


# ----- Define the Hamiltonian and its derivative -----
def H_fun(x, z):
    return np.array([[z, x], [x, -z]])


def H_partial(x, z):
    return np.array([[1, 0], [0, -1]])


model = ControlModel(H_fun, H_partial)

# ----- Set system and control parameters -----
alpha = 2
beta = 2
x = 1
z0 = -10
zf = -z0

model.set_parameters(x=x)
model.set_control(control_name='z', pulse_initial=z0, pulse_final=zf, initial_state=0, alpha=alpha, beta=beta)

# ----- Solve for optimal pulse -----
model.solve_problem()
```

## Public API

Top-level imports are intentionally kept small and explicit:

- `ControlModel`
- `PulseControl`
- `Dynamics`
- `GeodesiQError` and the typed exception hierarchy
- `GeodesiQWarning` and related warnings
- `__version__`

# Citing `geodesiq`

If you use `geodesiq` in your research, please cite the reference paper available [here](https://github.com/geodesiq-control/geodesiq-control).

## Development

Use the following checks before submitting changes:

```bash
uv run ruff check .
uv run mypy
uv run pytest
uv run python -m build
```