"""
NLSE Utility Module
====================

Shared functions for normalized NLSE split-step Fourier simulations:
- Grid creation (time and frequency domains)
- Pulse definitions (Gaussian, sech/soliton)
- Energy computation
- Spectrum computation
- RMS width and instantaneous-frequency diagnostics
- Plotting helpers for propagation maps
"""

import numpy as np


def create_grid(N_t=1024, tau_window=20.0):
    """Create time and frequency grids for SSFM simulations.

    Sets up the discrete time array and corresponding angular frequency
    array needed by the split-step Fourier solver. The frequency grid
    is in native FFT ordering (zero-frequency at index 0).

    Physics:
        The time grid spans [-tau_window, +tau_window] in normalized units (tau = t/T0).
        The frequency grid omega satisfies the Nyquist criterion: omega_max = pi/dtau.
        Both grids are used by the SSFM: time domain for the nonlinear step,
        frequency domain for the dispersion step.

    Args:
        N_t (int): Number of time grid points. Use powers of 2
            for predictable FFT performance and simple pedagogy, although NumPy
            FFTs also support non-power-of-2 sizes. Default: 1024 (2^10).
            Use 4096 for high-resolution.
        tau_window (float): Half-width of the time window in normalized units.
            The full window spans [-tau_window, +tau_window]. Default: 20.0.
            Must be large enough that pulses decay to ~0 at the edges.

    Returns:
        tau (np.ndarray): Time array of shape (N_t,), spanning
            [-tau_window, +tau_window), dtype float64.
        omega (np.ndarray): Angular frequency array of shape (N_t,),
            in native FFT ordering, dtype float64.
        dtau (float): Time step size = 2*tau_window/N_t.
    """
    if not isinstance(N_t, (int, np.integer)):
        raise TypeError("N_t must be an integer")
    if N_t <= 0:
        raise ValueError("N_t must be positive")
    if N_t & (N_t - 1) != 0:
        raise ValueError(
            "N_t must be a power of 2 for predictable FFT performance"
        )
    if not np.isfinite(tau_window) or tau_window <= 0:
        raise ValueError("tau_window must be a positive finite number")

    tau = np.linspace(-tau_window, tau_window, N_t, endpoint=False)
    dtau = tau[1] - tau[0]
    omega = 2 * np.pi * np.fft.fftfreq(N_t, d=dtau)
    return tau, omega, dtau


def gaussian_pulse(tau, chirp=0):
    """Create a normalized Gaussian pulse envelope.

    Physics:
        The Gaussian pulse u(tau) = exp(-(1+iC)tau^2/2) is the standard test pulse
        in fiber optics. Under dispersion-only propagation, its width evolves as
        T(xi) = T0*sqrt(1 + xi^2), providing an exact analytical benchmark.

        The chirp parameter C introduces a linear frequency sweep across the
        pulse: omega_inst(tau) = C*tau. C > 0 means the leading edge is red-shifted.

    Args:
        tau (np.ndarray): Normalized time array from create_grid().
        chirp (float): Chirp parameter C. Default: 0 (transform-limited).
            C > 0: up-chirped. C < 0: down-chirped.

    Returns:
        u (np.ndarray): Complex pulse envelope, shape same as tau,
            dtype complex128. Peak amplitude = 1.0 at tau = 0.
    """
    return np.exp(-(1 + 1j * chirp) * tau**2 / 2)


def sech_pulse(tau, amplitude=1.0):
    """Create a soliton (hyperbolic secant) pulse envelope.

    Physics:
        The sech pulse u(tau) = amplitude*sech(tau) = amplitude/cosh(tau)
        is the standard soliton shape for the NLSE.

        Key property: integral |sech(tau)|^2 dtau = 2 (used as validation check).

        In this project, soliton order is represented by the solver
        coefficient N_sq. For a second-order soliton, use
        sech_pulse(tau) and pass N_sq=4 to ssfm_propagate().

    Args:
        tau (np.ndarray): Normalized time array from create_grid().
        amplitude (float): Optional amplitude scaling for generic tests.
            Default: 1.0. This is not the soliton order.

    Returns:
        u (np.ndarray): Complex pulse envelope, shape same as tau,
            dtype complex128. Peak amplitude = amplitude at tau = 0.
    """
    abs_tau = np.abs(tau)
    sech = 2 * np.exp(-abs_tau) / (1 + np.exp(-2 * abs_tau))
    return (amplitude * sech).astype(np.complex128)
