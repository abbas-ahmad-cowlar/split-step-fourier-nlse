"""Energy conservation check for N=1 and N=2 solitons."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nlse_ssfm.nlse_utils import create_grid, sech_pulse, compute_energy
from nlse_ssfm.ssfm import ssfm_propagate


