"""Split-Step Fourier Solver package."""

__version__ = "0.1.0"

from .ssfm import dispersion_step, nonlinear_step, ssfm_propagate
from .nlse_utils import (
    create_grid,
    gaussian_pulse,
    sech_pulse,
    compute_energy,
    compute_spectrum,
    compute_spectrum_density,
    normalized_spectrum,
    rms_width,
    instantaneous_frequency,
