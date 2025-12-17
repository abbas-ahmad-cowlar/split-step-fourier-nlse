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


def compute_energy(u, dtau):
    """Compute total pulse energy (conserved quantity of the NLSE).

    Physics:
        The NLSE conserves the integral E = integral |u(xi,tau)|^2 dtau during
        propagation. This is analogous to probability conservation
        integral |psi|^2 dx = 1 in quantum mechanics. If E(xi)/E(0) drifts
        from 1.0, the solver has a bug.

        Formula: E = sum |u_k|^2 * dtau (rectangular quadrature)

    Args:
        u (np.ndarray): Complex pulse envelope array of shape (N_t,).
        dtau (float): Time step size from create_grid().

    Returns:
        energy (float): Total pulse energy (dimensionless in normalized units).
    """
    return np.sum(np.abs(u)**2) * dtau


def compute_spectrum(u):
    """Compute unnormalized plotting spectrum |FFT(u)|^2 with zero-frequency centered.

    Physics:
        The power spectrum |U(omega)|^2 shows the frequency content of the pulse.
        Under SPM-only propagation, the spectrum broadens while the temporal
        shape |u(tau)|^2 stays constant. The spectrum is fftshift-ed so that
        omega = 0 is at the center -- suitable for direct plotting.

    Args:
        u (np.ndarray): Complex pulse envelope array of shape (N_t,).

    Returns:
        spectrum (np.ndarray): Unnormalized plotting spectrum |U(omega)|^2 of shape
            (N_t,), with zero-frequency at the center (fftshift applied).
    """
    U = np.fft.fftshift(np.fft.fft(u))
    return np.abs(U)**2


def compute_spectrum_density(u, dtau):
    """Compute dtau-normalized spectral intensity for Parseval checks.

    Uses the Fourier convention U(omega) = (1/sqrt(2pi)) integral u(tau) exp(-i*omega*tau) dtau.
    With domega = 2pi/(N*dtau), sum(|U|^2)*domega should match
    sum(|u|^2)*dtau to numerical precision for well-resolved pulses.
    """
    U = np.fft.fftshift(np.fft.fft(u)) * dtau / np.sqrt(2 * np.pi)
    return np.abs(U)**2


def normalized_spectrum(u):
    """Return fftshifted |FFT(u)|^2 normalized to a peak of 1 for plotting."""
    spectrum = compute_spectrum(u)
    peak = np.max(spectrum)
    if peak == 0:
        return spectrum
    return spectrum / peak


def rms_width(u, tau, dtau):
    """Compute RMS pulse width using intensity-weighted moments.

    Args:
        u (np.ndarray): Complex pulse envelope.
        tau (np.ndarray): Time grid.
        dtau (float): Time step.

    Returns:
        sigma (float): RMS pulse width.
    """
    intensity = np.abs(u)**2
    energy = np.sum(intensity) * dtau
    if energy <= 0 or not np.isfinite(energy):
        raise ValueError("pulse energy must be positive and finite")
    mean_tau = np.sum(tau * intensity) * dtau / energy
    variance = np.sum((tau - mean_tau)**2 * intensity) * dtau / energy
    return np.sqrt(variance)


def instantaneous_frequency(u, dtau, intensity_floor=None):
    """Compute instantaneous frequency omega_i = -d(arg(u))/dtau.

    If intensity_floor is provided, values where |u|^2 is below
    intensity_floor * max(|u|^2) are set to NaN. This prevents meaningless
    phase in the pulse tails from contaminating chirp plots.
    """
    phase = np.unwrap(np.angle(u))
    chirp = -np.gradient(phase, dtau)
    if intensity_floor is not None:
        intensity = np.abs(u)**2
        mask = intensity >= intensity_floor * np.max(intensity)
        chirp = chirp.astype(float)
        chirp[~mask] = np.nan
    return chirp


def plot_propagation_map(ax, tau, xi, u_hist, tau_lim=None, cmap="inferno",
                         title=None, vmax=None):
    """Plot |u(xi,tau)|^2 on an existing Matplotlib axis and return the mesh."""
    mask = np.ones_like(tau, dtype=bool)
    if tau_lim is not None:
        mask = (tau >= tau_lim[0]) & (tau <= tau_lim[1])
    intensity = np.abs(u_hist[:, mask])**2
    tau_plot = tau[mask]
    if tau_plot.size < 2 or xi.size < 2:
        raise ValueError(
            "plot_propagation_map requires at least two tau and xi points"
        )
    tau_edges = np.concatenate([
        [tau_plot[0] - 0.5 * (tau_plot[1] - tau_plot[0])],
        0.5 * (tau_plot[1:] + tau_plot[:-1]),
        [tau_plot[-1] + 0.5 * (tau_plot[-1] - tau_plot[-2])],
    ])
    xi_edges = np.concatenate([
        [xi[0] - 0.5 * (xi[1] - xi[0])],
        0.5 * (xi[1:] + xi[:-1]),
        [xi[-1] + 0.5 * (xi[-1] - xi[-2])],
    ])
    im = ax.pcolormesh(
        tau_edges,
        xi_edges,
        intensity,
        shading="auto",
        cmap=cmap,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\xi$")
    if title:
        ax.set_title(title)
    return im


def save_figure(fig, path, dpi=300):
    """Apply tight layout, create the parent directory, and save a figure."""
    from pathlib import Path
    import warnings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_spectrum_evolution(ax, omega, fields, labels, omega_lim=None,
                            normalize=True, linewidth=1.5):
    """Plot fftshifted spectra for several fields on an existing axis."""
    omega_shifted = np.fft.fftshift(omega)
    mask = np.ones_like(omega_shifted, dtype=bool)
    if omega_lim is not None:
        mask = (omega_shifted >= omega_lim[0]) & (omega_shifted <= omega_lim[1])

    for field, label in zip(fields, labels):
        spectrum = normalized_spectrum(field) if normalize else compute_spectrum(field)
        ax.plot(omega_shifted[mask], spectrum[mask], linewidth=linewidth, label=label)

    ax.set_xlabel(r"$\omega$")
    ylabel = (r"Normalized $|\tilde{u}(\omega)|^2$" if normalize
              else r"$|\tilde{u}(\omega)|^2$")
    ax.set_ylabel(ylabel)
    if omega_lim is not None:
        ax.set_xlim(*omega_lim)
    ax.legend()
    ax.grid(True, alpha=0.3)
