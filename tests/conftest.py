"""Shared test configuration.

Tests resolve project files from this file's location rather than relying on
pytest being launched from the repository root. This keeps the suite isolated
from the caller's current working directory.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
