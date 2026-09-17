"""Pytest configuration: add project root to sys.path so test modules can import model_tester and config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
