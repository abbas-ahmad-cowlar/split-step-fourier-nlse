"""
Split-Step Fourier Method (SSFM) Solver
========================================

Symmetric split-step Fourier solver for the Nonlinear Schrodinger Equation (NLSE):

    i du/dxi + (s/2) d^2u/dtau^2 + N^2|u|^2 u = 0

Convention: Agrawal (Nonlinear Fiber Optics, 6th ed.)
    s = -sign(beta2) = +1 for anomalous dispersion
    N^2 = gamma * P0 * L_D (soliton number squared)
"""

import numpy as np


def dispersion_step(u, omega, s, dxi_half):
    """Apply one half-step of dispersion in the frequency domain.

    Physics:
        Dispersion causes different frequency components to accumulate
        different phases as the pulse propagates. In the frequency domain,
        this is a simple multiplication by exp(-i*s*omega^2*dxi_half/2).

        From i du/dxi + (s/2) d^2u/dtau^2 = 0, the frequency-domain evolution is
        dU/dxi = -i*s*omega^2*U/2.

        This function is used TWICE per symmetric split-step iteration:
        once before and once after the nonlinear step.

    Args:
        u (np.ndarray): Complex pulse envelope in time domain, shape (N_t,).
        omega (np.ndarray): Angular frequency array from create_grid(),
            in native FFT ordering, shape (N_t,).
        s (float): Sign parameter. s = +1 for anomalous dispersion
            (beta2 < 0), s = -1 for normal dispersion (beta2 > 0).
        dxi_half (float): Half of the step size in normalized distance.
            For symmetric splitting, pass dxi/2 where dxi = xi_max/N_z.

    Returns:
        u_out (np.ndarray): Pulse after dispersion half-step, shape (N_t,),
            dtype complex128.
    """
    U = np.fft.fft(u)
    phase = np.exp(-1j * s * omega**2 * dxi_half / 2)
    return np.fft.ifft(U * phase)


def nonlinear_step(u, N_sq, dxi):
    """Apply one full step of Kerr nonlinearity in the time domain.

    Physics:
        The Kerr effect creates an intensity-dependent phase shift:
        phi_NL(tau) = N^2 * |u(tau)|^2 * dxi. This is Self-Phase Modulation (SPM).

        Key property: |u*exp(i*phi)|^2 = |u|^2 -- the temporal intensity is
        UNCHANGED by SPM. Only the phase (and therefore the spectrum) changes.

        This function is called ONCE per symmetric split-step iteration,
        sandwiched between the two dispersion half-steps.

    Args:
        u (np.ndarray): Complex pulse envelope in time domain, shape (N_t,).
            This should be the pulse AFTER the first dispersion half-step.
        N_sq (float): Soliton number squared, N^2 = gamma*P0*L_D.
            N_sq = 0: no nonlinearity (linear propagation).
            N_sq = 1: fundamental soliton condition.
        dxi (float): Full step size in normalized distance.

    Returns:
        u_out (np.ndarray): Pulse after nonlinear step, shape (N_t,),
            dtype complex128. Has same |u|^2 as input.
    """
    return u * np.exp(1j * N_sq * np.abs(u)**2 * dxi)


def ssfm_propagate(u0, tau, omega, xi_max, N_z, s=1, N_sq=1.0,
                   save_every=1, return_history=True, callback=None):
    """Propagate a pulse through the NLSE using the symmetric split-step Fourier method.

    Physics:
        Solves the normalized NLSE:
            i du/dxi + (s/2) d^2u/dtau^2 + N^2|u|^2 u = 0

        using Strang (symmetric) operator splitting:
            u(xi + dxi) ~ exp(D*dxi/2) * exp(N*dxi) * exp(D*dxi/2) * u(xi)

        This gives second-order accuracy O(dxi^2) in the step size.

    Algorithm:
        For each of the N_z steps:
        1. Half-step dispersion: FFT -> phase multiply -> IFFT
        2. Full-step nonlinearity: multiply by exp(iN^2|u|^2*dxi) in time domain
        3. Half-step dispersion: FFT -> phase multiply -> IFFT

    Args:
        u0 (np.ndarray): Initial pulse envelope, complex array of shape (N_t,).
        tau (np.ndarray): Time grid from create_grid(), shape (N_t,).
        omega (np.ndarray): Angular frequency grid from create_grid(),
            in native FFT ordering, shape (N_t,).
        xi_max (float): Total propagation distance in normalized units.
        N_z (int): Number of propagation steps. Step size dxi = xi_max/N_z.
        s (float): Sign parameter = -sign(beta2).
            s = +1 for anomalous dispersion, s = -1 for normal, s = 0 for SPM-only.
        N_sq (float): Soliton number squared = gamma*P0*L_D.
            N_sq = 0: dispersion-only. N_sq = 1: fundamental soliton.
        save_every (int): Save every Nth step when return_history=True.
        return_history (bool): If True, return saved history array.
            If False, return only the final field.
        callback (callable or None): Optional callback(step, xi, u_copy)
            called after each step. Receives a copy of u.

    Returns:
        If return_history=True:
            xi_saved (np.ndarray): Saved propagation distances.
            u_history (np.ndarray): Saved complex propagation history.
        If return_history=False:
            xi_max (float): Final propagation distance.
            u_final (np.ndarray): Final complex field only.
    """
    u0 = np.asarray(u0, dtype=np.complex128)
    tau = np.asarray(tau, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if not isinstance(N_z, (int, np.integer)) or N_z <= 0:
        raise ValueError("N_z must be a positive integer")
    if not np.isfinite(xi_max) or xi_max < 0:
        raise ValueError("xi_max must be a non-negative finite number")
    if u0.ndim != 1 or tau.ndim != 1 or omega.ndim != 1:
        raise ValueError("u0, tau, and omega must be one-dimensional arrays")
    if not (len(u0) == len(tau) == len(omega)):
        raise ValueError("u0, tau, and omega must have matching lengths")
    if len(u0) < 2:
        raise ValueError("arrays must contain at least two points")
    if not (np.all(np.isfinite(u0)) and np.all(np.isfinite(tau))
            and np.all(np.isfinite(omega))):
        raise ValueError("u0, tau, and omega must contain only finite values")
    dtau_values = np.diff(tau)
    if not np.all(dtau_values > 0):
        raise ValueError("tau grid must be strictly increasing")
    dtau = dtau_values[0]
    if not np.allclose(dtau_values, dtau, rtol=1e-12, atol=1e-14):
        raise ValueError("tau grid must be uniformly spaced")
    expected_omega = 2 * np.pi * np.fft.fftfreq(len(tau), d=dtau)
    if not np.allclose(omega, expected_omega, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "omega must match 2*pi*np.fft.fftfreq(len(tau), d=dtau)"
        )
    if not np.isfinite(s) or not np.isfinite(N_sq):
        raise ValueError("s and N_sq must be finite numbers")
    if N_sq < 0:
        raise ValueError("N_sq must be non-negative")
    if not isinstance(save_every, (int, np.integer)) or save_every <= 0:
        raise ValueError("save_every must be a positive integer")

    dxi = xi_max / N_z

    xi_array = np.linspace(0, xi_max, N_z + 1)
    save_steps = list(range(0, N_z + 1, save_every))
    if save_steps[-1] != N_z:
        save_steps.append(N_z)
    save_step_set = set(save_steps)
    if return_history:
        xi_saved = xi_array[save_steps]
        u_history = np.zeros((len(save_steps), len(u0)), dtype=np.complex128)
        u_history[0] = u0.copy()
        save_idx = 1

    # Precompute the Strang half-step dispersion phase once per run.
    # This is exp(-i*s*omega^2*dxi/4), equivalent to dxi_half=dxi/2.
    half_dispersion_phase = np.exp(-1j * s * omega**2 * dxi / 4)

    u = u0.copy()
    for i in range(N_z):
        # Symmetric split-step: D/2 -> N -> D/2
        u = np.fft.ifft(np.fft.fft(u) * half_dispersion_phase)
        u = nonlinear_step(u, N_sq, dxi)
        u = np.fft.ifft(np.fft.fft(u) * half_dispersion_phase)
        step = i + 1
        if callback is not None:
            callback(step, xi_array[step], u.copy())
        if return_history and step in save_step_set:
            u_history[save_idx] = u.copy()
            save_idx += 1

    if return_history:
        return xi_saved, u_history
    return xi_max, u
