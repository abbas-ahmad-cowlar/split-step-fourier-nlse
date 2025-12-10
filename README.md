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

## Key Results

### Soliton Propagation: N=1, 2, 3
![Soliton comparison](figures/nb03_soliton_comparison.png)
*Left: Fundamental soliton (N=1) — unchanged. Center: N=2 — periodic breathing.
Right: N=3 — complex multi-peak dynamics.*

### Soliton Collision
![Soliton collision](figures/nb05_soliton_collision.png)
*Two solitons collide and recover their shapes within numerical tolerance — the hallmark of integrability in the ideal NLSE.*

### Solver Validation
![Convergence](figures/nb04_convergence.png)
*Second-order convergence confirmed via log-log analysis (slope ≈ 2).*

### More Results
![SPM spectrum](figures/nb02_spectral_evolution.png)
*Self-phase modulation broadens the spectral intensity while temporal intensity remains fixed.*

![Energy conservation](figures/nb04_energy_conservation.png)
*The SSFM preserves the NLSE energy invariant to floating-point precision.*

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [Dispersion](notebooks/01_dispersion.ipynb) | Pulse broadening, chirp, anomalous vs normal GVD |
| 02 | [Self-Phase Modulation](notebooks/02_spm.ipynb) | Kerr effect, spectral broadening, phase structure |
| 03 | [Solitons](notebooks/03_solitons.ipynb) | N=1, 2, 3 soliton dynamics — the showcase |
| 04 | [Validation](notebooks/04_validation.ipynb) | Convergence, energy conservation, ground truth cert |
| 05 | [Advanced](notebooks/05_advanced.ipynb) | Soliton collisions, integrability |

