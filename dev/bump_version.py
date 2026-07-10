#!/usr/bin/env python
"""
Script to automate version bumping across all project files.

Usage:
    python scripts/bump_version.py <new_version>
    python scripts/bump_version.py --patch
    python scripts/bump_version.py --minor
    python scripts/bump_version.py --major

Example:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py --minor  # bumps X.Y to X.Y+1
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch)."""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}. Expected: X.Y.Z")
    try:
        major, minor, patch = (int(p) for p in parts)
        return major, minor, patch
    except ValueError as err:
        raise ValueError(f"Version parts must be integers: {version_str}") from err


def version_to_string(major: int, minor: int, patch: int) -> str:
    """Convert (major, minor, patch) to string."""
    return f"{major}.{minor}.{patch}"


def bump_version(version_str: str, bump_type: str) -> str:
    """Bump version based on type: 'major', 'minor', or 'patch'."""
    major, minor, patch = parse_version(version_str)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

    return version_to_string(major, minor, patch)


def get_current_version(meta_file: Path) -> str:
    """Extract current version from _meta.py."""
    content = meta_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {meta_file}")
    return match.group(1)


def update_meta_file(meta_file: Path, new_version: str) -> None:
    """Update version in _meta.py."""
    content = meta_file.read_text()
    updated = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new_version}"', content)
    meta_file.write_text(updated)
    print(f"✓ Updated {meta_file.relative_to(meta_file.parent.parent.parent)}")


def update_changelog(changelog_file: Path, new_version: str) -> None:
    """Update CHANGELOG.md with new version section.

    Renames the existing [Unreleased] heading to the new version and inserts
    a fresh empty [Unreleased] section above it.
    """
    content = changelog_file.read_text()

    date = datetime.now().strftime("%Y-%m-%d")
    new_unreleased = "## [Unreleased]\n\n"
    versioned_heading = f"## [{new_version}] - {date}"

    # Replace "## [Unreleased]" with a new empty [Unreleased] + versioned heading
    updated = re.sub(
        r"## \[Unreleased\]",
        f"{new_unreleased}{versioned_heading}",
        content,
        count=1,
    )

    if updated == content:
        raise ValueError("Could not find '## [Unreleased]' section in CHANGELOG.md")

    changelog_file.write_text(updated)
    print(f"✓ Updated {changelog_file.relative_to(changelog_file.parent.parent.parent)}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    meta_file = project_root / "src" / "geodesiq" / "_meta.py"
    changelog_file = project_root / "CHANGELOG.md"

    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    # Get current version
    current_version = get_current_version(meta_file)
    print(f"Current version: {current_version}\n")

    # Determine new version
    if arg.startswith("--"):
        bump_type = arg[2:]
        if bump_type not in ["major", "minor", "patch"]:
            print(f"Invalid bump type: {bump_type}")
            sys.exit(1)
        new_version = bump_version(current_version, bump_type)
    else:
        # Validate provided version format
        new_version = arg
        try:
            parse_version(new_version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"New version: {new_version}\n")

    # Update files
    try:
        update_meta_file(meta_file, new_version)
        update_changelog(changelog_file, new_version)
        print(f"\n✓ Version successfully bumped to {new_version}")
        print("  Remember to: commit, push the changes to dev, and create a pull request to main")
    except Exception as e:
        print(f"Error during update: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
