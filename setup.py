"""Backwards-compatible shim: the project metadata lives in ``pyproject.toml``.

Kept so that ``python setup.py develop`` / legacy tooling keeps working.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
