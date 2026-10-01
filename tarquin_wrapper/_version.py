# Single source of truth for the package version. pyproject.toml reads it at build time,
# and binaries.py derives the GitHub release / container image tag ("v<version>") from
# it, so every wheel pairs with the TARQUIN binaries built alongside it.
__version__ = "0.1.0"
