"""Convergence study - verify 2nd-order accuracy of Strang splitting."""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nlse_ssfm.nlse_utils import create_grid, sech_pulse
from nlse_ssfm.ssfm import ssfm_propagate

