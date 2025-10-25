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
