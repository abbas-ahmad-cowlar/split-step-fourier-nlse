"""
NLSE Utility Module
====================

Shared functions for normalized NLSE split-step Fourier simulations:
- Grid creation (time and frequency domains)
- Pulse definitions (Gaussian, sech/soliton)
- Energy computation
- Spectrum computation
- RMS width and instantaneous-frequency diagnostics
- Plotting helpers for propagation maps
"""

import numpy as np


