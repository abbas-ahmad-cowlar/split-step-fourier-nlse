"""
Higher-Order Soliton Tests (N=2, N=3)
Phase 3, Step 3.3 of the SSFM project.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from nlse_ssfm.nlse_utils import create_grid, sech_pulse, compute_energy
from nlse_ssfm.ssfm import ssfm_propagate


