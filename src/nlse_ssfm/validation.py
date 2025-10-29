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


# ------------------------------------------------------------------
# 1. Convergence study
# ------------------------------------------------------------------

def run_convergence_study(save_path=None):
    """Check Strang step-size convergence for the N=1 soliton."""
    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    xi_max = 5 * np.pi / 2
    Nz_values = [50, 100, 200, 500, 1000, 2000]
    errors, complex_errors, dxi_values = [], [], []

    for Nz in Nz_values:
        dxi = xi_max / Nz
        dxi_values.append(dxi)
        xi_arr, u_hist = ssfm_propagate(u0, tau, omega,
                                         xi_max=xi_max, N_z=Nz, s=1, N_sq=1.0)
        err = np.max(np.abs(np.abs(u_hist[-1]) - np.abs(u0)))
        errors.append(err)
        u_exact = u0 * np.exp(1j * xi_max / 2)
        ov = np.vdot(u_exact, u_hist[-1])
        gp = ov / abs(ov)
        cerr = np.max(np.abs(u_hist[-1] - gp * u_exact))
        complex_errors.append(cerr)

    dxi_arr = np.array(dxi_values)
    err_arr = np.array(errors)
    cerr_arr = np.array(complex_errors)

    # Shape slope
    floor = max(1e-12, 5 * np.min(err_arr))
    valid = err_arr > floor
    if np.sum(valid) < 3:
        valid = err_arr > 1e-12
    slope, intercept = np.polyfit(np.log10(dxi_arr[valid]),
                                   np.log10(err_arr[valid]), 1)

    # Complex slope
    cfloor = max(1e-12, 5 * np.min(cerr_arr))
    cvalid = cerr_arr > cfloor
    if np.sum(cvalid) < 3:
        cvalid = cerr_arr > 1e-12
    cslope, cintercept = np.polyfit(np.log10(dxi_arr[cvalid]),
                                     np.log10(cerr_arr[cvalid]), 1)

    passed = abs(slope - 2) < 0.3 and abs(cslope - 2) < 0.4

    if save_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(dxi_arr, err_arr, "ko-", ms=8, lw=2, label="Shape error")
        ax.loglog(dxi_arr, cerr_arr, "s-", color="purple", ms=6, lw=1.5,
                  label="Phase-aligned complex error")
        dfit = np.logspace(np.log10(dxi_arr.min()),
                           np.log10(dxi_arr.max()), 100)
        ax.loglog(dfit, 10**intercept * dfit**slope, "r--", lw=1.5,
                  label=f"Shape fit: slope = {slope:.2f}")
        ax.loglog(dfit, 10**cintercept * dfit**cslope, "--", color="purple",
                  lw=1.2, label=f"Complex fit: slope = {cslope:.2f}")
        dref = np.logspace(-2.5, -0.5, 50)
        ax.loglog(dref, 1e-1*(dref/dref[0])**2, "b:", lw=1, alpha=0.5,
                  label=r"$O(d\xi^2)$")
        ax.loglog(dref, 1e-1*(dref/dref[0])**1, "g:", lw=1, alpha=0.5,
                  label=r"$O(d\xi)$")
        ax.set_xlabel(r"Step size $d\xi$", fontsize=13)
        ax.set_ylabel("Max error", fontsize=13)
        ax.set_title("SSFM Convergence Study (N=1 Soliton)", fontsize=14)
        ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(bottom=1e-14)
        _save_fig(fig, save_path)
        import matplotlib.pyplot as plt
        plt.close(fig)

    table = list(zip(Nz_values, dxi_values, errors, complex_errors))
    return {
        "passed": passed,
        "slope": slope, "intercept": intercept,
        "complex_slope": cslope, "complex_intercept": cintercept,
        "table": table,
        "fit_points": int(np.sum(valid)),
        "complex_fit_points": int(np.sum(cvalid)),
        "rows": [
            _row("Convergence shape slope", "2.0 +/- 0.3",
                 f"{slope:.2f}", abs(slope - 2) < 0.3),
            _row("Convergence complex slope", "2.0 +/- 0.4",
                 f"{cslope:.2f}", abs(cslope - 2) < 0.4),
        ],
    }


# ------------------------------------------------------------------
# 2. Energy conservation
# ------------------------------------------------------------------

def run_energy_conservation_checks(save_path=None):
    """Check max |E(xi)/E(0)-1| for N=1 and N=2 soliton propagation."""
    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    E0 = compute_energy(u0, dtau)

    xi1, uh1 = ssfm_propagate(u0, tau, omega, xi_max=5*np.pi/2,
                               N_z=500, s=1, N_sq=1.0)
    en1 = np.array([compute_energy(uh1[i], dtau) for i in range(len(xi1))])
    dev1 = np.max(np.abs(en1 / E0 - 1.0))

    xi2, uh2 = ssfm_propagate(u0, tau, omega, xi_max=np.pi,
                               N_z=1000, s=1, N_sq=4.0)
    en2 = np.array([compute_energy(uh2[i], dtau) for i in range(len(xi2))])
    dev2 = np.max(np.abs(en2 / E0 - 1.0))

    if save_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(xi1, en1 / E0 - 1, 'b-', lw=1.5)
        ax1.set_xlabel(r'$\xi$', fontsize=12)
        ax1.set_ylabel(r'$E(\xi)/E(0) - 1$', fontsize=12)
        ax1.set_title('N=1 Soliton', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax2.plot(xi2, en2 / E0 - 1, 'r-', lw=1.5)
        ax2.set_xlabel(r'$\xi$', fontsize=12)
        ax2.set_ylabel(r'$E(\xi)/E(0) - 1$', fontsize=12)
        ax2.set_title('N=2 Soliton', fontsize=13)
        ax2.grid(True, alpha=0.3)
        fig.suptitle('Energy Conservation', fontsize=14)
        _save_fig(fig, save_path)
        plt.close(fig)

    return {
        "passed": dev1 < 1e-10 and dev2 < 1e-10,
        "max_E_dev_N1": dev1, "max_E_dev_N2": dev2,
        "rows": [
            _row("Energy conservation N=1", "< 1e-10", f"{dev1:.2e}",
                 dev1 < 1e-10),
            _row("Energy conservation N=2", "< 1e-10", f"{dev2:.2e}",
                 dev2 < 1e-10),
        ],
    }


# ------------------------------------------------------------------
# 3. Gaussian broadening
# ------------------------------------------------------------------

