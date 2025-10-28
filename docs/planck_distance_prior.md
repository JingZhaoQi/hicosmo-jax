# Planck 2018 Distance Prior Likelihood

This note documents the implementation in `hicosmo/likelihoods/planck_distance.py`.

## Observables and Covariance

We adopt the compressed Planck 2018 TT,TE,EE+lowE results for the shift parameters

- $R = \sqrt{\Omega_m H_0^2} D_A(z_\star) / c$
- $\ell_A = \pi D_A(z_\star) / r_s(z_\star)$
- $\Omega_b h^2$

with observed values $(R, \ell_A, \Omega_b h^2) = (1.750235, 301.4707, 0.02235976)$ and the inverse covariance published in the Planck legacy release.

## Theoretical Predictions

The likelihood evaluates the model vector using the HIcosmo `LCDM` background API:

- The baryon and matter physical densities are read directly from the cosmology parameters.
- The decoupling redshift $z_\star$ is computed via the Hu & Sugiyama (1996) fitting formula.
- Distances rely on the built-in `LCDM.H_z` background evolution; the code integrates $D_A(z)$ with a JAX-native trapezoidal rule to retain compatibility with NumPyro.
- The comoving sound horizon at decoupling uses the Eisenstein & Hu (1998) analytic expression, evaluated at $z_\star$.

All calculations are performed with JAX primitives only, so the likelihood can be used inside HIcosmo’s NumPyro-based samplers.

## Validation

A quick MCMC run (`python example_planck_distance_mcmc.py`) sampling $H_0$, $\Omega_m$, and $\Omega_b$ yields posterior medians

- $H_0 = 67.4 \pm 1.0$ km s$^{-1}$ Mpc$^{-1}$
- $\Omega_m = 0.315 \pm 0.017$
- $\Omega_b = 0.049 \pm 0.001$

which are consistent with the official Planck 2018 base-$\Lambda$CDM constraints.

The script deposits a corner plot under `results/planck_distance_corner.pdf` for visual inspection.

## Tests

`tests/likelihoods/test_planck_distance.py` checks that

- The likelihood evaluated at Planck’s best-fit parameters is finite and close to the published values.
- Deviations in the cosmological parameters decrease the log-likelihood as expected.
- The derived $(R, \ell_A)$ predicted by the model match the Planck data within the expected tolerance.
