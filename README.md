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

**HIcosmo** (**H**igh-performance **I**nference for **Cosmo**logy) is a modern JAX-based cosmological parameter estimation framework, featuring automatic differentiation, JIT compilation, and GPU/TPU acceleration.

### Key Features

- **Minimal API**: 3 lines of code for cosmological inference
- **JAX Native**: Automatic differentiation, JIT compilation, GPU/TPU acceleration
- **Multi-probe Analysis**: Supernovae, BAO, CMB, Strong Lensing Time Delays
- **High-performance Sampling**: NumPyro NUTS + emcee dual backends
- **Fisher Forecasting**: 21cm intensity mapping survey predictions
- **Publication-quality Visualization**: GetDist integration

### Project Scale

| Metric | Value |
|--------|-------|
| Python Files | 90+ |
| Lines of Code | ~35,000 |
| Cosmological Models | 4 (LCDM, wCDM, CPL, ILCDM) |
| Likelihood Functions | 10+ |
| Supported Surveys | SKA1, Tianlai, BINGO, MeerKAT, CHIME |

---

## Installation

### Requirements

- Python 3.9+
- JAX 0.4+
- NumPyro 0.12+

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

### Dependencies

```
jax>=0.4.0
jaxlib>=0.4.0
numpyro>=0.12.0
optax>=0.2.0
getdist>=1.3.0
astropy>=5.0
emcee>=3.1.0
```

---

## Quick Start

### 4 Lines for Cosmological Inference

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
wcdm = wCDM(H0=70, Omega_m=0.3, w=-1.1)
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

# Pantheon+ (1701 SNe Ia)
sne = SN_likelihood(LCDM, "pantheon+")

# Pantheon+SH0ES (with Cepheid calibration)
sne_shoes = SN_likelihood(LCDM, "pantheon+shoes")
```

#### Baryon Acoustic Oscillations

```python
from hicosmo.likelihoods import BAO_likelihood

# DESI 2024 latest data
bao = BAO_likelihood(LCDM, "desi2024")

# omega_b modes: 'free', 'bbn_prior', 'fixed'
bao_bbn = BAO_likelihood(LCDM, "desi2024", omega_b_mode='bbn_prior')
```

#### Cosmic Microwave Background

```python
from hicosmo.likelihoods import Planck2018DistancePriorsLikelihood

# Planck 2018 distance priors (l_A, R, z_*)
cmb = Planck2018DistancePriorsLikelihood()
```

#### Strong Gravitational Lensing

```python
from hicosmo.likelihoods import H0LiCOWLikelihood, TDCOSMOLikelihood

# H0LiCOW (6 lens systems)
h0licow = H0LiCOWLikelihood()

# TDCOSMO hierarchical Bayesian analysis
tdcosmo = TDCOSMOLikelihood()
```

### 3. MCMC Sampling

```python
from hicosmo.samplers import MCMC

# NumPyro NUTS (default)
mcmc = MCMC(params, likelihood, sampler='numpyro')
samples = mcmc.run(num_samples=10000, num_chains=4, num_warmup=1000)

# emcee ensemble sampler
mcmc = MCMC(params, likelihood, sampler='emcee')
```

### 4. Fisher Matrix Forecasting

```python
from hicosmo.fisher import IntensityMappingFisher, load_survey
from hicosmo.models import CPL

# Load survey configuration
survey = load_survey('ska1_mid_band2')

# Create Fisher analyzer
fisher = IntensityMappingFisher(
    survey=survey,
    fiducial_cosmology=CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
)

# Run forecast
result = fisher.forecast(
    free_params=['w0', 'wa'],
    external_priors={'Omega_m': 0.007}
)

print(f"σ(w0) = {result.constraints['w0']:.3f}")
print(f"FoM = {result.figure_of_merit:.1f}")
```

#### Supported Surveys

| Survey | Type | Redshift Range |
|--------|------|----------------|
| SKA1-MID Band 1 | Single-dish | 0.35-3.05 |
| SKA1-MID Band 2 | Single-dish | 0.1-0.58 |
| Tianlai | Interferometer | 0.5-2.5 |
| BINGO | Single-dish | 0.13-0.45 |
| MeerKAT | Interferometer | 0.1-0.58 |
| CHIME | Interferometer | 0.8-2.5 |

### 5. Visualization

```python
from hicosmo.visualization import Plotter

# Load from file
plotter = Plotter('mcmc_chain.pkl')

# Corner plot (confidence ellipses)
plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

# Chain traces
plotter.chains(['H0', 'Omega_m'], filename='chains.pdf')

# Multi-chain comparison
plotter = Plotter(
    ['planck_chain.pkl', 'shoes_chain.pkl'],
    chain_labels=['Planck', 'SH0ES']
)
plotter.corner(['H0'], filename='h0_tension.pdf')
```

---

## Project Structure

```
hicosmo/
├── core/                    # Core infrastructure
│   └── base.py             # CosmologyBase abstract class
│
├── models/                  # Cosmological models
│   ├── lcdm.py             # Standard ΛCDM
│   ├── wcdm.py             # Constant dark energy
│   ├── cpl.py              # CPL parameterization
│   └── ilcdm.py            # Interacting dark energy
│
├── likelihoods/            # Likelihood functions
│   ├── sn/                 # Supernovae (Pantheon+)
│   ├── bao/                # BAO (DESI, SDSS, 6dFGS)
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
│   └── intensity_mapping.py # 21cm IM forecasting
│
├── visualization/          # Visualization system
│   └── plotter.py          # Unified Plotter interface
│
└── utils/                  # Utilities
    └── constants.py        # Physical constants
```

---

## Performance Benchmarks

| Operation | qcosmc (scipy) | HIcosmo (JAX) | Speedup |
|-----------|----------------|---------------|---------|
| Distance (1000 pts) | 0.15s | 0.02s | **7.5x** |
| MCMC (10k samples) | 180s | 45s | **4.0x** |
| Fisher matrix | 2.1s | 0.5s | **4.2x** |
| BAO likelihood | 21.8ms | 1.24ms | **17.6x** |

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
@software{hicosmo,
  author = {Qi, Jingzhao},
  title = {HIcosmo: High-performance Inference for Cosmology},
  year = {2024},
  url = {https://github.com/JingZhaoQi/hicosmo-jax}
}
```

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
