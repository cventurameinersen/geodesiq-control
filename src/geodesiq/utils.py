import numpy as np


class Flags:
    """
    A class to manage flags with hierarchical dependencies.

    Each flag can have one or more parents. If any parent flag is down (False),
    all descendants reachable from that parent are set down as well.

    Example
    -------
    >>> flags = Flags()
    >>> flags.add("solver")
    >>> flags.add("gradient", parent="solver")
    >>> flags.add("hessian", parents=["gradient", "solver"])
    >>> flags["solver"]    # True
    >>> flags["solver"] = False
    >>> flags["gradient"]  # False  (parent is down)
    >>> flags["hessian"]   # False  (grandparent is down)
    """

    def __init__(self):
        self._values: dict[str, bool] = {}
        self._parents: dict[str, list[str]] = {}
        self._children: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, name: str, value: bool = False, parent: str | None = None,
            parents: list[str] | tuple[str, ...] | set[str] | None = None, ) -> None:
        """Register a new flag.

        Parameters
        ----------
        name:
            Unique identifier for the flag.
        value:
            Initial value (default ``False``).
        parent:
            Name of an already-registered flag that this one depends on.
            Backward-compatible alias for a single entry in ``parents``.
        parents:
            Names of already-registered flags that this one depends on.

        Raises
        ------
        KeyError
            If *name* is already registered or any parent does not exist.
        ValueError
            If both *parent* and *parents* are given.
        """
        if name in self._values:
            raise KeyError(f"Flag '{name}' is already registered.")

        if parent is not None and parents is not None:
            raise ValueError("Use either 'parent' or 'parents', not both.")

        if parents is None:
            normalized_parents = [parent] if parent is not None else []
        else:
            normalized_parents = list(dict.fromkeys(parents))

        for parent_name in normalized_parents:
            if parent_name not in self._values:
                raise KeyError(f"Parent flag '{parent_name}' does not exist.")

        self._values[name] = value
        self._parents[name] = normalized_parents
        self._children[name] = []

        for parent_name in normalized_parents:
            if name not in self._children[parent_name]:
                self._children[parent_name].append(name)

    def get(self, name: str) -> bool:
        """Return the value of a flag.

        The effective value is ``False`` when the flag itself or any of
        its ancestors is ``False``.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        self._check_exists(name)
        return self._values[name]

    def set(self, name: str, value: bool) -> None:
        """Set the stored value of a flag.

        Parameters
        ----------
        name:
            Flag to update.
        value:
            New boolean value.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        self._check_exists(name)
        self._values[name] = value

        for child in self._children[name]:
            if not value:
                self.set(child, False)

    def all(self) -> bool:
        """Return ``True`` if all flags are effectively up."""
        return all(self.get(name) for name in self._values)

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> bool:
        return self.get(name)

    def __setitem__(self, name: str, value: bool) -> None:
        self.set(name, value)

    def __repr__(self) -> str:
        lines = []
        for name, raw in self._values.items():
            parents = self._parents[name]
            lines.append(f"  {name}: stored={raw}," + (f" parents={parents!r}" if parents else ""))
        return "Flags(\n" + "\n".join(lines) + "\n)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exists(self, name: str) -> None:
        if name not in self._values:
            raise KeyError(f"Flag '{name}' is not registered.")


def build_diab(initial_state: int, final_state: int, dim: int) -> np.ndarray:
    diad_list = -1 * np.eye(dim, dtype=int)  # Diagonal entries are -1 by default

    min_state = min(initial_state, final_state)
    max_state = max(initial_state, final_state)

    for i in range(dim):
        for j in range(i + 1, dim):
            pass  # ToDo continue

    # ToDo: use the symmetry of the problem to compute the lower triangular part of the matrix
