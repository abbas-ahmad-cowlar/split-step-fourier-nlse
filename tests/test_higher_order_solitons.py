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


def test_higher_order():

    # ============================================================
    # N=2 SOLITON -- Periodic Breathing
    # ============================================================
    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    E0 = compute_energy(u0, dtau)
    print(f"N=2: Initial energy = {E0:.4f} (expected: 2.0)")
    assert abs(E0 - 2.0) < 0.01
    assert abs(np.abs(u0[len(tau)//2]) - 1.0) < 1e-10

    xi_max = np.pi  # 2 soliton periods
    N_z = 1000
    xi_arr, u_hist = ssfm_propagate(u0, tau, omega, xi_max=xi_max,
                                     N_z=N_z, s=1, N_sq=4.0)

    # CHECK 1: Energy conservation
    energies = np.array([compute_energy(u_hist[i], dtau)
                         for i in range(len(xi_arr))])
    max_E_dev = np.max(np.abs(energies / E0 - 1.0))
    assert max_E_dev < 1e-10, f"Energy not conserved: {max_E_dev}"
    print(f"   Energy conservation: max deviation = {max_E_dev:.2e} [PASS]")

    # CHECK 2: Recurrence at xi = pi/2
    idx_zsol = np.argmin(np.abs(xi_arr - np.pi/2))
    u_at_zsol = u_hist[idx_zsol]
    recurrence_error_1 = np.max(np.abs(np.abs(u_at_zsol) - np.abs(u0)))
    print(f"   Recurrence at xi = pi/2: max err = {recurrence_error_1:.4e}")

    # CHECK 3: Recurrence at xi = pi
    u_at_2zsol = u_hist[-1]
    recurrence_error_2 = np.max(np.abs(np.abs(u_at_2zsol) - np.abs(u0)))
    print(f"   Recurrence at xi = pi: max err = {recurrence_error_2:.4e}")

    n2_ok = recurrence_error_1 < 0.05 and recurrence_error_2 < 0.05
    assert n2_ok, (
        f"N=2 recurrence failed: errors {recurrence_error_1:.3e}, "
        f"{recurrence_error_2:.3e}"
    )
    print("   N=2 recurrence: PASSED at xi=pi/2 and xi=pi")

    # CHECK 4: Breathing
    peak_intensities = np.max(np.abs(u_hist)**2, axis=1)
    max_peak = np.max(peak_intensities)
    print(f"   Max peak intensity during breathing: {max_peak:.2f}")
    assert max_peak > 3.0, "N=2 should compress above initial peak"
    print("N=2 soliton test passed\n")

    # ============================================================
    # N=3 SOLITON -- Complex Breathing
    # ============================================================
    tau3, omega3, dtau3 = create_grid(N_t=4096, tau_window=20.0)
    u0_3 = sech_pulse(tau3)
    E0_3 = compute_energy(u0_3, dtau3)
    print(f"N=3: Initial energy = {E0_3:.4f} (expected: 2.0)")
    assert abs(E0_3 - 2.0) < 0.01

    xi_max_3 = np.pi
    N_z_3 = 2000
    xi_arr3, u_hist3 = ssfm_propagate(u0_3, tau3, omega3, xi_max=xi_max_3,
                                        N_z=N_z_3, s=1, N_sq=9.0)

    # CHECK 1: Energy
    energies3 = np.array([compute_energy(u_hist3[i], dtau3)
                          for i in range(len(xi_arr3))])
    max_E_dev3 = np.max(np.abs(energies3 / E0_3 - 1.0))
    assert max_E_dev3 < 1e-10, f"Energy not conserved: {max_E_dev3}"
    print(f"   Energy conservation: max deviation = {max_E_dev3:.2e} [PASS]")

    # CHECK 2: Recurrence at xi = pi/2
    idx_zsol3 = np.argmin(np.abs(xi_arr3 - np.pi/2))
    u_at_zsol3 = u_hist3[idx_zsol3]
    recurrence_error_3 = np.max(np.abs(np.abs(u_at_zsol3) - np.abs(u0_3)))
    print(f"   Recurrence at xi = pi/2: max error = {recurrence_error_3:.4e}")

    n3_ok = recurrence_error_3 < 0.1
    assert n3_ok, f"N=3 recurrence failed: error {recurrence_error_3:.3e}"
    print("   N=3 recurrence at z_sol: PASSED")

    # CHECK 3: Multi-peak breathing detection
    central_mask = np.abs(tau3) < 8
    tau_central = tau3[central_mask]
    probe_xis = np.linspace(0.1, np.pi/2 - 0.1, 12)
    best_peak_count = 0
    best_probe_xi = None
    best_peak_locations = None

    for xi_probe in probe_xis:
        idx_probe = np.argmin(np.abs(xi_arr3 - xi_probe))
        intensity_probe = np.abs(u_hist3[idx_probe])**2
        intensity_central = intensity_probe[central_mask]
        peaks_probe, _ = find_peaks(
            intensity_central,
            prominence=0.05 * np.max(intensity_central),
            distance=50,
        )
        if len(peaks_probe) > best_peak_count:
            best_peak_count = len(peaks_probe)
            best_probe_xi = xi_arr3[idx_probe]
            best_peak_locations = tau_central[peaks_probe]

    assert best_peak_count >= 2, "N=3 multi-peak breathing diagnostic failed"
    print(f"   Multi-peak breathing confirmed at xi = {best_probe_xi:.4f}")
    print(f"   Peak count: {best_peak_count}, locations: {best_peak_locations}")
    print("N=3 soliton test passed\n")

    # ============================================================
    # PROPAGATION MAPS
    # ============================================================

    # Re-run N=1 for composite (short range = pi)
    tau1, omega1, dtau1 = create_grid(N_t=2048, tau_window=20.0)
    u0_1 = sech_pulse(tau1)
    xi_arr1, u_hist1 = ssfm_propagate(u0_1, tau1, omega1,
                                        xi_max=np.pi, N_z=500, s=1, N_sq=1.0)

    # Composite figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    tau_range = 5
    titles = [r"$N=1$: Fundamental Soliton",
              r"$N=2$: Second-Order Soliton",
              r"$N=3$: Third-Order Soliton"]
    datasets = [
        (tau1, xi_arr1, u_hist1, 1.0),
        (tau, xi_arr, u_hist, 4.0),
        (tau3, xi_arr3, u_hist3, 9.0),
    ]

    for idx, (t, xi, uh, nsq) in enumerate(datasets):
        ax = axes[idx]
        mask = (t >= -tau_range) & (t <= tau_range)
        t_plot = t[mask]
        intensity = np.abs(uh[:, mask])**2
        im = ax.pcolormesh(t_plot, xi, intensity, cmap="hot",
                           shading="auto", vmin=0, vmax=nsq * 1.5)
        ax.set_xlabel(r"$\tau$", fontsize=12)
        ax.set_title(titles[idx], fontsize=13)
        for k in range(1, 3):
            ax.axhline(y=k * np.pi/2, color="cyan", linestyle="--",
                       alpha=0.5, linewidth=0.8)
        fig.colorbar(im, ax=ax, label=r"$|u|^2$", shrink=0.8)

    axes[0].set_ylabel(r"$\xi = z/L_D$", fontsize=13)
    fig.suptitle("Soliton Propagation: N=1, 2, 3", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("figures/nb03_soliton_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print("Plot saved: figures/nb03_soliton_comparison.png")

    # Individual N=2 map
    fig, ax = plt.subplots(figsize=(10, 6))
    mask2 = (tau >= -5) & (tau <= 5)
    im2 = ax.pcolormesh(tau[mask2], xi_arr, np.abs(u_hist[:, mask2])**2,
                         cmap="inferno", shading="auto", vmin=0, vmax=8)
    ax.set_xlabel(r"$\tau$", fontsize=13)
    ax.set_ylabel(r"$\xi = z/L_D$", fontsize=13)
    ax.set_title(r"Second-Order Soliton ($N=2$): $|u(\xi,\tau)|^2$", fontsize=14)
    for k in range(1, 3):
        ax.axhline(y=k*np.pi/2, color="cyan", linestyle="--", alpha=0.5)
    fig.colorbar(im2, ax=ax, label=r"$|u|^2$")
    fig.tight_layout()
    fig.savefig("figures/nb03_soliton_N2_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Individual N=3 map
    fig, ax = plt.subplots(figsize=(10, 6))
    mask3 = (tau3 >= -5) & (tau3 <= 5)
    im3 = ax.pcolormesh(tau3[mask3], xi_arr3, np.abs(u_hist3[:, mask3])**2,
                         cmap="inferno", shading="auto", vmin=0, vmax=20)
    ax.set_xlabel(r"$\tau$", fontsize=13)
    ax.set_ylabel(r"$\xi = z/L_D$", fontsize=13)
    ax.set_title(r"Third-Order Soliton ($N=3$): $|u(\xi,\tau)|^2$", fontsize=14)
    for k in range(1, 3):
        ax.axhline(y=k*np.pi/2, color="cyan", linestyle="--", alpha=0.5)
    fig.colorbar(im3, ax=ax, label=r"$|u|^2$")
    fig.tight_layout()
    fig.savefig("figures/nb03_soliton_N3_map.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plots saved: nb03_soliton_N2_map.png, nb03_soliton_N3_map.png")
