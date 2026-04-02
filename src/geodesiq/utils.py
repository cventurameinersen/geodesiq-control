class Flags:
    """
    A class to manage flags with hierarchical dependencies.

    Each flag can have a parent. If a parent flag is down (False),
    all its descendants are set to down as well, regardless of their own stored value.

    Example
    -------
    >>> flags = Flags()
    >>> flags.add("solver")
    >>> flags.add("gradient", parent="solver")
    >>> flags.add("hessian",  parent="gradient")
    >>> flags["solver"]    # True
    >>> flags["solver"] = False
    >>> flags["gradient"]  # False  (parent is down)
    >>> flags["hessian"]   # False  (grandparent is down)
    """

    def __init__(self):
        self._values: dict[str, bool] = {}
        self._parent: dict[str, str | None] = {}
        self._children: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, name: str, value: bool = False, parent: str | None = None) -> None:
        """Register a new flag.

        Parameters
        ----------
        name:
            Unique identifier for the flag.
        value:
            Initial value (default ``False``).
        parent:
            Name of an already-registered flag that this one depends on.
            If the parent is down, this flag is also considered down.

        Raises
        ------
        KeyError
            If *name* is already registered or *parent* does not exist.
        """
        if name in self._values:
            raise KeyError(f"Flag '{name}' is already registered.")
        if parent is not None and parent not in self._values:
            raise KeyError(f"Parent flag '{parent}' does not exist.")

        self._values[name] = value
        self._parent[name] = parent
        self._children[name] = []

        if parent is not None:
            self._children[parent].append(name)

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
            parent = self._parent[name]
            lines.append(f"  {name}: stored={raw}," + (f" parent='{parent}'" if parent else ""))
        return "Flags(\n" + "\n".join(lines) + "\n)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_exists(self, name: str) -> None:
        if name not in self._values:
            raise KeyError(f"Flag '{name}' is not registered.")
