# Contributing

Thanks for contributing to `geodesiq`.

## Development setup

1. Create a branch from `dev`.
2. Commit your changes there.
3. Open a pull request into `dev`.
4. Once `dev` is stable, maintainers open a pull request from `dev` into `main`.

Direct pushes to `main` are not allowed.
Only `dev` is merged into `main` (no feature branch PRs directly into `main`).

Automation flow:
- CI runs on pushes to `dev` and PRs targeting `dev`.
- Release runs only when a PR from `dev` into `main` is merged.
- Publish runs only after a successful `Release` workflow from `main`.

To recreate the development environment, run:
```bash
uv sync --group dev
```

If you do not use `uv`, create a virtual environment and install equivalent dev dependencies from `pyproject.toml`.

## Run quality checks

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m build
uv run python -m twine check dist/*
```

## Pull request guidelines

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update `README.md` and `CHANGELOG.md` when user-facing behavior changes.
- Ensure all checks pass before requesting review.

## Release notes

- Add entries to the `[Unreleased]` section in `CHANGELOG.md`.
- At release time, move those entries under a versioned heading.

## Bumping the version

Use the `dev/bump_version.py` script to update the version consistently across all project files (`src/geodesiq/_meta.py` and `CHANGELOG.md`).

**Bump by increment type:**
```bash
python dev/bump_version.py --patch   # e.g. 0.1.0 → 0.1.1
python dev/bump_version.py --minor   # e.g. 0.1.1 → 0.2.0
python dev/bump_version.py --major   # e.g. 0.2.0 → 1.0.0
```

**Set an explicit version:**
```bash
python dev/bump_version.py 1.2.3
```

The script will:
1. Update `__version__` in `src/geodesiq/_meta.py`.
2. Insert a new dated section for the new version in `CHANGELOG.md`.

After running the script, commit the changes, push to `dev`, and open a pull request into `main` to trigger the release workflow.

