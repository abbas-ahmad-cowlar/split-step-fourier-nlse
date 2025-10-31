"""Convergence study - verify 2nd-order accuracy of Strang splitting."""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nlse_ssfm.nlse_utils import create_grid, sech_pulse
from nlse_ssfm.ssfm import ssfm_propagate


def test_convergence_study():

    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    xi_max = 5 * np.pi / 2

    Nz_values = [50, 100, 200, 500, 1000, 2000]
    errors = []
    complex_errors = []
    dxi_values = []

    print("Running convergence study...")
    print(f"{'N_z':>6} | {'dxi':>12} | {'Shape err':>12} | {'Complex err':>12} | {'Time':>8}")
    print("-" * 66)

    for Nz in Nz_values:
        dxi = xi_max / Nz
        dxi_values.append(dxi)
        t0 = time.time()
        xi_arr, u_hist = ssfm_propagate(u0, tau, omega, xi_max=xi_max,
                                         N_z=Nz, s=1, N_sq=1.0)
        elapsed = time.time() - t0
        err = np.max(np.abs(np.abs(u_hist[-1]) - np.abs(u0)))
        errors.append(err)
        u_exact = u0 * np.exp(1j * xi_max / 2)
        overlap = np.vdot(u_exact, u_hist[-1])
        gp = overlap / abs(overlap)
        cerr = np.max(np.abs(u_hist[-1] - gp * u_exact))
        complex_errors.append(cerr)
        print(f"{Nz:>6} | {dxi:>12.6f} | {err:>12.4e} | {cerr:>12.4e} | {elapsed:>8.3f}")

    dxi_arr = np.array(dxi_values)
    err_arr = np.array(errors)
    cerr_arr = np.array(complex_errors)

    # Fit shape-error slope
    floor = max(1e-12, 5 * np.min(err_arr))
    valid = err_arr > floor
    if np.sum(valid) < 3:
        valid = err_arr > 1e-12
    log_dxi = np.log10(dxi_arr[valid])
    log_err = np.log10(err_arr[valid])
    slope, intercept = np.polyfit(log_dxi, log_err, 1)

    print(f"\n=== CONVERGENCE ORDER ===")
    print(f"Shape-error slope: {slope:.2f} (expected 2.00)")

    # Fit complex-error slope
    cfloor = max(1e-12, 5 * np.min(cerr_arr))
    cvalid = cerr_arr > cfloor
    if np.sum(cvalid) < 3:
        cvalid = cerr_arr > 1e-12
    cslope, cintercept = np.polyfit(np.log10(dxi_arr[cvalid]),
                                     np.log10(cerr_arr[cvalid]), 1)
    print(f"Complex-error slope: {cslope:.2f}")

    if abs(slope - 2.0) < 0.3:
        print(f"[PASS] CONVERGENCE VERIFIED -- slope = {slope:.2f}")
    else:
        print(f"[FAIL] UNEXPECTED SLOPE = {slope:.2f}")

    assert abs(slope - 2.0) < 0.3, f"Shape-error slope {slope:.2f} deviates from expected 2.0"
    assert abs(cslope - 2.0) < 0.3, f"Complex-error slope {cslope:.2f} deviates from expected 2.0"

    # Safe default
    for i, (nz, e) in enumerate(zip(Nz_values, errors)):
        if e < 1e-5:
            print(f"\nRecommended default: N_z = {nz} (dxi = {dxi_values[i]:.6f})")
            break

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(dxi_arr, err_arr, "ko-", ms=8, lw=2, label="Shape error")
    ax.loglog(dxi_arr, cerr_arr, "s-", color="purple", ms=6, lw=1.5,
              label="Phase-aligned complex error")

    dxi_fit = np.logspace(np.log10(dxi_arr.min()), np.log10(dxi_arr.max()), 100)
    ax.loglog(dxi_fit, 10**intercept * dxi_fit**slope, "r--", lw=1.5,
              label=f"Shape fit: slope = {slope:.2f}")
    ax.loglog(dxi_fit, 10**cintercept * dxi_fit**cslope, "--", color="purple",
              lw=1.2, label=f"Complex fit: slope = {cslope:.2f}")

    dxi_ref = np.logspace(-2.5, -0.5, 50)
    ax.loglog(dxi_ref, 1e-1*(dxi_ref/dxi_ref[0])**2, "b:", lw=1, alpha=0.5,
              label=r"$O(d\xi^2)$ reference")
    ax.loglog(dxi_ref, 1e-1*(dxi_ref/dxi_ref[0])**1, "g:", lw=1, alpha=0.5,
              label=r"$O(d\xi)$ reference")

    ax.set_xlabel(r"Step size $d\xi$", fontsize=13)
    ax.set_ylabel("Max error", fontsize=13)
    ax.set_title("SSFM Convergence Study (N=1 Soliton)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-14)
    fig.tight_layout()
    fig.savefig("figures/nb04_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plot saved: figures/nb04_convergence.png")
