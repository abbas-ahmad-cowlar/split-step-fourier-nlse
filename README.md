# Split-Step Fourier Solver for Nonlinear Fiber Optics

A Python implementation of the symmetric split-step Fourier method (SSFM) for solving
the nonlinear Schrödinger equation (NLSE) — the master equation governing optical pulse
propagation in fibers. Demonstrates dispersion, self-phase modulation, soliton dynamics,
and soliton collisions.

## The Physics

The NLSE describes how an optical pulse evolves in a nonlinear dispersive fiber:

$$i\frac{\partial u}{\partial \xi} + \frac{s}{2}\frac{\partial^2 u}{\partial \tau^2} + N^2|u|^2 u = 0$$

Two competing effects — **group velocity dispersion** (temporal broadening) and
**self-phase modulation** (spectral broadening) — can form a self-consistent nonlinear
balance in anomalous dispersion. The fundamental result is the optical **soliton**:
a pulse that propagates without changing shape.

