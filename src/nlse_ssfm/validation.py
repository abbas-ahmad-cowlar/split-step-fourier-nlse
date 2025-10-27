"""
SSFM Validation Suite
=====================

Reusable validation checks for the split-step Fourier solver.
Each public function runs a specific physics or numerical test and
returns a dictionary with ``passed``, measured values, and optional
``rows`` for ``run_all_validation_checks()``.

Usage::

    python -m nlse_ssfm.validation   # run all checks from CLI
"""

import numpy as np
from .nlse_utils import (
    create_grid, gaussian_pulse, sech_pulse, compute_energy, compute_spectrum,
)
from .ssfm import ssfm_propagate


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _row(test, expected, measured, passed):
    return {
        "test": test,
        "expected": expected,
        "measured": measured,
        "passed": bool(passed),
    }
