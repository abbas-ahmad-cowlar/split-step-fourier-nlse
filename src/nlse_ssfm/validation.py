"""
SSFM Validation Suite
=====================

Reusable validation checks for the split-step Fourier solver.
Each public function runs a specific physics or numerical test and
returns a dictionary with ``passed``, measured values, and optional
``rows`` for ``run_all_validation_checks()``.

