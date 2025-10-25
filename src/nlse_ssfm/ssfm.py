"""
Split-Step Fourier Method (SSFM) Solver
========================================

Symmetric split-step Fourier solver for the Nonlinear Schrodinger Equation (NLSE):

    i du/dxi + (s/2) d^2u/dtau^2 + N^2|u|^2 u = 0

Convention: Agrawal (Nonlinear Fiber Optics, 6th ed.)
    s = -sign(beta2) = +1 for anomalous dispersion
    N^2 = gamma * P0 * L_D (soliton number squared)
"""

import numpy as np


