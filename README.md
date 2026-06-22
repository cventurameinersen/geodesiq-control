# `geodesiq`: geometric optimal control

`geodesiq` is a Python package for computing optimal quantum-control pulses from a parameter-dependent Hamiltonian.

[Installation](#installation) | [Quickstart](#quickstart) | [Public API](#public-api) | [Development](#development)

## Installation

Install from PyPI:

```bash
pip install geodesiq
```

For development:

```bash
uv sync --group dev
```

## Quickstart

The public API exposes `Hamiltonian`, `PulseControl`, `Dynamics`, the package version, and the custom exception/warning types.

```python
import numpy as np

from geodesiq import Hamiltonian


def lz_hamiltonian(lam: float, delta: float = 1.0) -> np.ndarray:
    return np.array([[lam, delta], [delta, -lam]])


def lz_partial(lam: float, delta: float = 1.0) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]])


ham = Hamiltonian(lz_hamiltonian, partial_H_func=lz_partial)
ham.set_parameters(delta=0.5)
ham.set_control(
    control_name="lam",
    pulse_initial=-5.0,
    pulse_final=5.0,
    initial_state=0,
    alpha=2.0,
    beta=2.0,
    num_steps=257,
)

ham.solve_problem()
print(ham.control_sol)
```

## Public API

Top-level imports are intentionally kept small and explicit:

- `Hamiltonian`
- `PulseControl`
- `Dynamics`
- `GeodesiQError` and the typed exception hierarchy
- `GeodesiQWarning` and related warnings
- `__version__`

## Development

Use the following checks before submitting changes:

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python -m build
```
