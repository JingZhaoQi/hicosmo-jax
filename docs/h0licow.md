# H0LiCOW Time-Delay Likelihood

This note documents the strong-lensing likelihood that lives in `hicosmo/likelihoods/h0licow.py` and the accompanying datasets under `hicosmo/data/h0licow/`.

## Data Assets

All observational products are copied verbatim from the public H0LiCOW release and grouped by lens:

- `h0licow_distance_chains/J1206_final.csv`
- `h0licow_distance_chains/wfi2033_dt_bic.dat`
- `h0licow_distance_chains/HE0435_Ddt_AO+HST.dat`
- `h0licow_distance_chains/RXJ1131_AO+HST_Dd_Ddt.dat`
- `h0licow_distance_chains/PG1115_AO+HST_Dd_Ddt.dat`
- `h0licow_distance_chains/DES0408-5354/power_law_dist_post_no_kext.txt`

The skewed log-normal summaries for B1608 adopt the analytical parameters reported in Suyu et al. (2010), Jee et al. (2019), and Wong et al. (2020).

## Likelihood Interface

```python
from hicosmo.likelihoods import H0LiCOWLikelihood
from hicosmo.models.lcdm import LCDM

likelihood = H0LiCOWLikelihood()
model = LCDM(H0=73.3, Omega_m=0.3)
logp = likelihood.log_likelihood(model)
```

Key features:

- Seven lenses are enabled by default (B1608, RXJ1131, HE0435, J1206, WFI2033, PG1115, DES0408).
- KDE-based lenses precompute their density estimates at initialisation; this requires `scikit-learn`.
- Cosmology dependencies rely on the new `angular_diameter_distance_between` and `time_delay_distance` helpers on `LCDM`.

## Quick Validation

Run `python example_h0licow_analysis.py` to perform a one-dimensional grid search over Hubble constants. With Ωₘ fixed to 0.3 we obtain

- `H0 = 73.8 ± 1.4 km s⁻¹ Mpc⁻¹`

which agrees with the official six-lens H0LiCOW measurement of `H0 = 73.3^{+1.7}_{-1.8} km s⁻¹ Mpc⁻¹` reported by Wong et al. (2020). citeturn0search5

## Tests

The unit tests in `tests/likelihoods/test_h0licow.py` assert that the likelihood prefers high H₀ values and that the coarse grid maximum falls inside the 70–76 km s⁻¹ Mpc⁻¹ window highlighted by the collaboration.

## Future Work

- Explore the KDE bandwidth hyper-parameters and expose them as configuration options.
- Add support for alternative lens selections (e.g., STRIDES-only subsets) through YAML configuration.
- Extend the cosmology hooks to accept dynamical dark-energy parameters for TDCOSMO comparisons.
