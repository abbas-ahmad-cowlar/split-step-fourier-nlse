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


def _save_fig(fig, save_path):
    """Save figure if *save_path* is not None, creating parent dirs."""
    if save_path is None:
        return
    from pathlib import Path
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight")
