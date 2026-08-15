import os
import sys

# Ensure the repository root (where main.py lives) is importable regardless
# of how pytest is invoked (e.g. `pytest` from repo root vs. from tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)