# HIcosmo

<p align="center">
  <strong>High-performance Inference for Cosmology</strong>
</p>

<p align="center">
  <a href="https://jingzhaoqi.github.io/hicosmo-jax/en/">Documentation (English)</a> •
  <a href="https://jingzhaoqi.github.io/hicosmo-jax/zh/">文档 (中文)</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#performance">Performance</a>
</p>

---

## Overview

**HIcosmo** (**H**igh-performance **I**nference for **Cosmo**logy) is a JAX-based cosmological parameter estimation framework, featuring automatic differentiation, JIT compilation, and GPU acceleration. It is described in the accompanying paper, [arXiv:2606.28175](https://arxiv.org/abs/2606.28175).

### Key Features

- **Minimal API**: a few lines of code for cosmological inference
- **JAX Native**: Automatic differentiation, JIT compilation, GPU acceleration
- **Multi-probe Analysis**: Supernovae, BAO, CMB distance priors, local $H_0$, strong-lensing time delays
- **Data Included**: the observational data used by the likelihoods are in `hicosmo/data/`
- **High-performance Sampling**: NumPyro NUTS + emcee dual backends
- **Fisher Forecasting**: 21cm intensity mapping survey predictions
- **Publication-quality Visualization**: GetDist integration

### Project Scale

| Metric | Value |
|--------|-------|
| Python Files | 71 |
| Lines of Code | ~28,500 |
| Cosmological Models | 4 (LCDM, wCDM, CPL, ILCDM) |
| Likelihood Functions | 10+ |
| 21cm Survey Configurations | SKA1-MID Band 1 and Band 2, MeerKAT, BINGO, Tianlai, CHIME |

---

## Installation

### Requirements

- Python 3.9+
- JAX 0.4.20+
- NumPyro 0.15+

### Install

```bash
# Clone repository
git clone https://github.com/JingZhaoQi/hicosmo-jax.git
cd hicosmo-jax

# Development mode install
pip install -e ".[dev]"

# Or core dependencies only
pip install -e .
```

Install in development mode (`-e`), so that the data files in `hicosmo/data/` are found.

### Dependencies

See `requirements.txt`:

```
jax>=0.4.20
jaxlib>=0.4.20
numpyro>=0.15.0
optax>=0.2.0
emcee>=3.1.0
numpy>=1.24
scipy>=1.10
astropy>=5.0
matplotlib>=3.7
getdist>=1.4
h5py>=3.8
pandas>=2.0
tqdm>=4.60
rich>=13.0
PyYAML>=6.0
corner>=2.2
```

---

## Quick Start

### A Few Lines for Cosmological Inference

```python
import hicosmo as hc
hc.init(8)  # Optional: enable 8-core parallel

from hicosmo import hicosmo
inf = hicosmo("LCDM", ["sn", "bao"], ["H0", "Omega_m"])
samples = inf.run()
```

### Complete Example

```python
# 1. Initialize (must be first)
import hicosmo as hc
hc.init(8)  # 8 JAX devices parallel

# 2. Import modules
from hicosmo.samplers import MCMC
from hicosmo.models import LCDM
from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
from hicosmo.visualization import Plotter

# 3. Create likelihoods
sne = SN_likelihood(LCDM, "pantheon+")
bao = BAO_likelihood(LCDM, "desi2024")

# 4. Parameter configuration
params = {
    'H0': {'init': 70, 'min': 60, 'max': 80},
    'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
}

# 5. Run MCMC (supports + operator for combining likelihoods)
mcmc = MCMC(params, sne + bao, chain_name='lcdm_joint')
samples = mcmc.run(num_samples=10000, num_chains=4)

# 6. Visualization
Plotter('lcdm_joint').corner(['H0', 'Omega_m'], filename='corner.pdf')
```

---

## Features

### 1. Cosmological Models

| Model | Description | Dark Energy EoS |
|-------|-------------|-----------------|
| **LCDM** | Standard ΛCDM | $w = -1$ |
| **wCDM** | Constant dark energy | $w = w_0$ |
| **CPL** | Chevallier-Polarski-Linder | $w(a) = w_0 + w_a(1-a)$ |
| **ILCDM** | Interacting dark energy | $Q = \beta H \rho_c$ |

```python
from hicosmo.models import LCDM, wCDM, CPL

# Create model instances
lcdm = LCDM(H0=67.36, Omega_m=0.3153)
wcdm = wCDM(H0=70, Omega_m=0.3, w0=-1.1)
cpl = CPL(H0=70, Omega_m=0.3, w0=-1.0, wa=0.2)

# Compute physical quantities
z = 1.0
d_L = lcdm.luminosity_distance(z)       # Luminosity distance [Mpc]
d_A = lcdm.angular_diameter_distance(z) # Angular diameter distance [Mpc]
E_z = lcdm.E_z(z)                        # Dimensionless Hubble parameter
```

### 2. Likelihood Functions

#### Type Ia Supernovae

```python
from hicosmo.likelihoods import SN_likelihood

# Pantheon+ (1590 SNe Ia with z > 0.01)
sne = SN_likelihood(LCDM, "pantheon+")

# Pantheon+SH0ES (with Cepheid calibration, 1657 entries)
sne_shoes = SN_likelihood(LCDM, "pantheon+shoes")

# DES-Dovekie and Union3
sne_des = SN_likelihood(LCDM, "desy5")
sne_u3 = SN_likelihood(LCDM, "union3")
```

#### Baryon Acoustic Oscillations

```python
from hicosmo.likelihoods import BAO_likelihood

# DESI DR1 (2024) and DR2
bao = BAO_likelihood(LCDM, "desi2024")
bao_dr2 = BAO_likelihood(LCDM, "desi_dr2")

# Earlier data sets: "sdss_dr12", "sdss_dr16", "boss_dr12", "sixdf"

# omega_b modes: 'fixed', 'free', 'bbn_prior', 'h0rd'
bao_bbn = BAO_likelihood(LCDM, "desi2024", omega_b_mode='bbn_prior')
```

#### Cosmic Microwave Background

```python
from hicosmo.likelihoods import Planck

# Planck 2018 distance priors on (R, l_a, omega_b h^2)
cmb = Planck(LCDM)
```

#### Local H0 and Strong Gravitational Lensing

```python
from hicosmo.likelihoods import SH0ESLikelihood, H0LiCOW, TDCOSMO

# SH0ES local H0 measurement
shoes = SH0ESLikelihood()

# H0LiCOW (6 lens systems)
h0licow = H0LiCOW(LCDM)

# TDCOSMO hierarchical Bayesian analysis
tdcosmo = TDCOSMO(LCDM)
```

### 3. MCMC Sampling

```python
from hicosmo.samplers import MCMC

likelihood = sne + bao  # params, sne and bao as in the Complete Example

# NumPyro NUTS (default)
mcmc = MCMC(params, likelihood, sampler='numpyro')
samples = mcmc.run(num_samples=10000, num_chains=4, num_warmup=1000)

# emcee ensemble sampler
mcmc = MCMC(params, likelihood, sampler='emcee')
```

### 4. Fisher Matrix Forecasting

```python
from hicosmo.fisher import IntensityMappingFisher, load_survey
from hicosmo.models import wCDM

# Load survey configuration
survey = load_survey('ska1_wide_band1')

# Create Fisher analyzer (fiducial cosmology: Planck 2018 defaults)
fisher = IntensityMappingFisher(survey, wCDM())

# Forecast w0, marginalized over H0, Omega_m and sigma8
result = fisher.parameter_forecast(
    ['w0'], marginalize_over=['H0', 'Omega_m', 'sigma8']
)

print(f"sigma(w0) = {result['errors'][0]:.3f}")
```

#### Supported Surveys

| Survey | Configuration name | Type | Redshift Range |
|--------|--------------------|------|----------------|
| SKA1-MID Band 1 | `ska1_wide_band1` | Single-dish | 0.35-3.06 |
| SKA1-MID Band 2 | `ska1_mid_band2` | Single-dish | 0-0.49 |
| MeerKAT | `meerkat` | Single-dish | 0.40-1.45 |
| BINGO | `bingo` | Single-dish | 0.13-0.48 |
| Tianlai | `tianlai` | Cylinder array | 0.50-1.58 |
| CHIME | `chime` | Cylinder array | 0.78-2.55 |

`list_available_surveys()` lists all configuration files. A new survey needs only a new YAML file.

### 5. Visualization

```python
from hicosmo.visualization import Plotter

# Load a saved chain by its chain_name (or a path to the .h5 file)
plotter = Plotter('lcdm_joint')

# Corner plot (confidence ellipses)
plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

# Chain traces
plotter.traces(['H0', 'Omega_m'], filename='traces.pdf')

# Multi-chain comparison (chains saved earlier with chain_name='lcdm_cmb' and 'lcdm_shoes')
plotter = Plotter(
    ['lcdm_cmb', 'lcdm_shoes'],
    chain_labels=['Planck', 'SH0ES']
)
plotter.corner(['H0'], filename='h0_tension.pdf')
```

---

## Project Structure

```
hicosmo/
├── hicosmo.py              # hicosmo() one-call interface
│
├── models/                  # Cosmological models
│   ├── base.py             # CosmologyBase abstract class
│   ├── lcdm.py             # Standard ΛCDM
│   ├── wcdm.py             # Constant dark energy
│   ├── cpl.py              # CPL parameterization
│   └── ilcdm.py            # Interacting dark energy
│
├── likelihoods/            # Likelihood functions
│   ├── sn/                 # Supernovae (Pantheon+, DES-Dovekie, Union3)
│   ├── bao/                # BAO (DESI DR1/DR2, SDSS/BOSS/eBOSS, 6dFGS)
│   ├── cmb/                # CMB distance priors
│   ├── h0/                 # Direct H0 (SH0ES)
│   └── lensing/            # Strong lensing time delays (H0LiCOW, TDCOSMO)
│
├── samplers/               # MCMC sampling system
│   ├── inference.py        # MCMC high-level API
│   ├── numpyro_backend.py  # NumPyro NUTS backend
│   └── emcee_backend.py    # emcee backend
│
├── fisher/                 # Fisher matrix analysis
│   ├── intensity_mapping.py # 21cm IM forecasting
│   └── surveys/            # Survey configuration files (YAML)
│
├── data/                   # Observational data used by the likelihoods
│
├── visualization/          # Visualization system
│   └── plotter.py          # Unified Plotter interface
│
└── utils/                  # Utilities
    ├── constants.py        # Physical constants
    └── jax_tools.py        # Sound horizon, integration and other JAX helpers
```

---

## Performance

Benchmarks, their measurement conditions and the validation against Cobaya are given in section 4 of the accompanying paper ([arXiv:2606.28175](https://arxiv.org/abs/2606.28175)). In the problems tested there, HIcosmo reaches an effective sample size per unit time 6.5 to 27 times higher than Cobaya on the same CPU, mainly because of the gradient-based NUTS sampler and a compressed form of the supernova likelihood.

---

## Tutorials

The `tutorials/` directory contains Jupyter notebooks for learning HIcosmo:

1. `01_quickstart.ipynb` - Getting started
2. `02_cosmological_models.ipynb` - Model basics
3. `03_supernovae_likelihood.ipynb` - SNe Ia analysis
4. `04_bao_likelihood.ipynb` - BAO constraints
5. `05_cmb_lensing_likelihood.ipynb` - CMB distance priors
6. `06_mcmc_sampling.ipynb` - MCMC techniques
7. `07_fisher_forecasting.ipynb` - Fisher analysis
8. `08_visualization.ipynb` - Plotting results
9. `09_advanced_analysis.ipynb` - Advanced topics
10. `10_multiprobe_constraints.ipynb` - Joint analysis

---

## Citation

If you use HIcosmo in your research, please cite:

```bibtex
@article{Qi:2026hicosmo,
  author = {Qi, Jing-Zhao and Zhang, Jing-Fei and Zhang, Xin},
  title = {HIcosmo: a differentiable JAX-based framework for cosmology inference},
  eprint = {2606.28175},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  year = {2026}
}
```

If you use the likelihoods, please also cite the papers of the corresponding data sets.

---

## License

MIT License

---

## Contact

- **Author**: Jingzhao Qi
- **Issues**: [GitHub Issues](https://github.com/JingZhaoQi/hicosmo-jax/issues)

---

<p align="center">
  <sub>High-performance Inference for Cosmology — Built with JAX</sub>
</p>
