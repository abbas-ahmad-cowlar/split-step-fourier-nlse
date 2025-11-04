"""
THE ACID TEST -- Fundamental Soliton (N=1) Validation
=====================================================
Phase 3, Step 3.2 of the SSFM project.
"""
import numpy as np
from nlse_ssfm.nlse_utils import create_grid, sech_pulse, compute_energy
from nlse_ssfm.ssfm import ssfm_propagate


def test_acid():

    # Setup grid
    tau, omega, dtau = create_grid(N_t=2048, tau_window=20.0)
    u0 = sech_pulse(tau)
    E0 = compute_energy(u0, dtau)
    print(f"Initial energy: {E0:.6f} (expected: 2.000000)")
    assert abs(E0 - 2.0) < 0.01

    # Propagation parameters
    xi_max = 5 * np.pi / 2  # 5 soliton periods
    N_z = 500
    print(f"Propagation: xi_max = {xi_max:.4f} = 5*pi/2")
    print(f"Steps: N_z = {N_z}, dxi = {xi_max/N_z:.6f}")

    # RUN THE SOLVER
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega,
        xi_max=xi_max, N_z=N_z, s=1, N_sq=1.0
    )

    print("\n=== SOLITON VALIDATION RESULTS ===\n")
    results = {}

    # CHECK 1: Shape preservation at endpoint
    u_final = u_hist[-1]
    max_amp_err = np.max(np.abs(np.abs(u_final) - np.abs(u0)))
    results["shape"] = max_amp_err < 1e-4
    tag = "PASS" if results["shape"] else "FAIL"
    print(f"CHECK 1 -- Shape preservation: [{tag}]")
    print(f"   Max | |u(xi_final)| - sech(tau) | = {max_amp_err:.2e}")

    # CHECK 2: Peak intensity at every step
    center_idx = len(tau) // 2
    peak_intensities = np.abs(u_hist[:, center_idx])**2
    max_peak_dev = np.max(np.abs(peak_intensities - 1.0))
    results["peak"] = max_peak_dev < 1e-4
    tag = "PASS" if results["peak"] else "FAIL"
    print(f"CHECK 2 -- Peak intensity: [{tag}]")
    print(f"   Max | |u(0,xi)|^2 - 1 | = {max_peak_dev:.2e}")

    # CHECK 3: Energy conservation at every step
    energies = np.array([compute_energy(u_hist[i], dtau) for i in range(len(xi_arr))])
    max_E_dev = np.max(np.abs(energies / E0 - 1.0))
    results["energy"] = max_E_dev < 1e-10
    tag = "PASS" if results["energy"] else "FAIL"
    print(f"CHECK 3 -- Energy conservation: [{tag}]")
    print(f"   Max |E(xi)/E(0) - 1| = {max_E_dev:.2e}")

    # CHECK 4: Shape at ALL steps
    max_shape_all = 0.0
    worst_step = 0
    for i in range(len(xi_arr)):
        step_err = np.max(np.abs(np.abs(u_hist[i]) - np.abs(u0)))
        if step_err > max_shape_all:
            max_shape_all = step_err
            worst_step = i
    results["all_steps"] = max_shape_all < 1e-4
    tag = "PASS" if results["all_steps"] else "FAIL"
    print(f"CHECK 4 -- Shape at all steps: [{tag}]")
    print(f"   Max error across ALL {len(xi_arr)} steps = {max_shape_all:.2e}")
    print(f"   Worst step: {worst_step} (xi = {xi_arr[worst_step]:.4f})")

    # CHECK 5: Complex phase accuracy
    u_ana_final = u0 * np.exp(1j * xi_arr[-1] / 2)
    phase_num = np.angle(u_final[center_idx])
    phase_ana = xi_arr[-1] / 2
    phase_error = abs(np.angle(np.exp(1j * (phase_num - phase_ana))))
    complex_err_exact = np.max(np.abs(u_final - u_ana_final))

    overlap = np.vdot(u_ana_final, u_final)
    global_phase = overlap / abs(overlap)
    complex_err_aligned = np.max(np.abs(u_final - global_phase * u_ana_final))

    results["phase"] = complex_err_exact < 5e-3 and complex_err_aligned < 1e-4
    tag = "PASS" if results["phase"] else "FAIL"
    print(f"CHECK 5 -- Phase accuracy: [{tag}]")
    print(f"   Phase error at peak = {phase_error:.4f} rad")
    print(f"   Exact-phase complex error = {complex_err_exact:.2e} (expected <5e-3)")
    print(f"   Phase-aligned complex error = {complex_err_aligned:.2e} (expected <1e-4)")

    # VERDICT
    print()
    sep = "=" * 50
    print(sep)
    all_pass = all(results.values())
    if all_pass:
        print("ACID TEST PASSED -- The solver is CORRECT")
        print(f"   Soliton survived 5 full periods (xi = {xi_max:.4f})")
        print(f"   Max shape error: {max_shape_all:.2e}")
        print(f"   Max energy drift: {max_E_dev:.2e}")
    else:
        print("ACID TEST FAILED -- DO NOT PROCEED")
        for k, v in results.items():
            if not v:
                print(f"   FAILED check: {k}")
    print(sep)

    assert all_pass, f"Acid test failed checks: {[k for k, v in results.items() if not v]}"
