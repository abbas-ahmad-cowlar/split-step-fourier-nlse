"""Energy conservation check for N=1 and N=2 solitons."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nlse_ssfm.nlse_utils import create_grid, sech_pulse, compute_energy
from nlse_ssfm.ssfm import ssfm_propagate


def test_energy():

    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    E0 = compute_energy(u0, dtau)

    # N=1
    xi_arr, u_hist = ssfm_propagate(u0, tau, omega, xi_max=5*np.pi/2,
                                     N_z=500, s=1, N_sq=1.0)
    energies_N1 = np.array([compute_energy(u_hist[i], dtau)
                            for i in range(len(xi_arr))])
    ratio_N1 = energies_N1 / E0
    max_dev_N1 = np.max(np.abs(ratio_N1 - 1.0))
    print(f"N=1: Max |E(xi)/E(0) - 1| = {max_dev_N1:.2e}")
    assert max_dev_N1 < 1e-10, f"N=1 energy FAILED: {max_dev_N1}"
    print("   [PASS] Energy conserved to machine precision")

    # N=2
    xi_arr2, u_hist2 = ssfm_propagate(u0, tau, omega, xi_max=np.pi,
                                        N_z=1000, s=1, N_sq=4.0)
    energies_N2 = np.array([compute_energy(u_hist2[i], dtau)
                            for i in range(len(xi_arr2))])
    ratio_N2 = energies_N2 / E0
    max_dev_N2 = np.max(np.abs(ratio_N2 - 1.0))
    print(f"\nN=2: Max |E(xi)/E(0) - 1| = {max_dev_N2:.2e}")
    assert max_dev_N2 < 1e-10, f"N=2 energy FAILED: {max_dev_N2}"
    print("   [PASS] Energy conserved to machine precision")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ylim = max(1e-14, 1.2 * max(max_dev_N1, max_dev_N2))

    axes[0].plot(xi_arr, ratio_N1 - 1.0, "b-", lw=1.5)
    axes[0].axhline(y=0, color="k", ls="-", lw=0.5)
    axes[0].set_xlabel(r"$\xi = z/L_D$", fontsize=12)
    axes[0].set_ylabel(r"$E(\xi)/E(0) - 1$", fontsize=12)
    axes[0].set_title(r"Energy Conservation: $N=1$ Soliton", fontsize=13)
    axes[0].set_ylim(-ylim, ylim)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(0,0))
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.05, 0.95, f"Max deviation: {max_dev_N1:.2e}",
                 transform=axes[0].transAxes, fontsize=11, va="top",
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    axes[1].plot(xi_arr2, ratio_N2 - 1.0, "r-", lw=1.5)
    axes[1].axhline(y=0, color="k", ls="-", lw=0.5)
    axes[1].set_xlabel(r"$\xi = z/L_D$", fontsize=12)
    axes[1].set_ylabel(r"$E(\xi)/E(0) - 1$", fontsize=12)
    axes[1].set_title(r"Energy Conservation: $N=2$ Soliton", fontsize=13)
    axes[1].set_ylim(-ylim, ylim)
    axes[1].ticklabel_format(axis="y", style="scientific", scilimits=(0,0))
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.05, 0.95, f"Max deviation: {max_dev_N2:.2e}",
                 transform=axes[1].transAxes, fontsize=11, va="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("SSFM Energy Conservation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/nb04_energy_conservation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nPlot saved: figures/nb04_energy_conservation.png")
