# Union3 Supernova Data

## Data Source

Rubin, D., et al. 2023, arXiv:2311.12098
"Union Through UNITY: Cosmology with 2,000 SNe Using a Unified Bayesian Framework"

Official data release: https://github.com/rubind/union3_release

## Data Format

The Union3 release provides a compressed Gaussian approximation of the full
UNITY1.5 hierarchical Bayesian posterior. The original FITS file
(`mu_mat_union3_cosmo=2_mu.fits`) encodes a (N+1)x(N+1) matrix where N=22
redshift bins:

- `[0, 1:]` = redshift bin centers (22 values, z = 0.05 to 2.26)
- `[1:, 0]` = median distance moduli from MCMC (22 values)
- `[1:, 1:]` = inverse covariance matrix of distance moduli (22x22)

The FITS file has been pre-converted to `union3_data.npz` (numpy format) to
avoid a runtime dependency on astropy.

## Files

| File | Description |
|------|-------------|
| `union3_data.npz` | NumPy compressed archive (redshifts, mu, cov_inv) |
| `mu_mat_union3_cosmo=2_mu.fits` | Original FITS file from data release |
| `DATA_INFO.md` | This file |

## Key Properties

- **Sample**: 2087 SNe Ia from 24 datasets
- **Redshift range**: 0.05 < z < 2.26
- **Bins**: 22 redshift bins
- **Standardization**: UNITY1.5 Bayesian hierarchical model
- **M_B**: Already marginalized in UNITY framework (no M_B parameter needed)
- **cosmo=2**: Spline-interpolated non-parametric distance model (22 free nodes). Official flat LCDM constraint (Ωm=0.344) uses cosmo=1 (direct parametric MCMC within UNITY). Fitting flat LCDM to cosmo=2 compressed Gaussian yields Ωm≈0.356 (0.4σ offset due to compression approximation).

## Usage

```python
from hicosmo.likelihoods import SN_likelihood
from hicosmo.models import LCDM

union3 = SN_likelihood(LCDM, "union3")
log_L = union3(H0=70.0, Omega_m=0.3)
```

## Citation

If using this data, cite:
- Rubin, D., et al. 2023, arXiv:2311.12098

## Conversion Date

Data downloaded and converted: 2026-03-28
