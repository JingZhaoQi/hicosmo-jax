# HIcosmo 🌌 📡

[![CI/CD](https://github.com/JingZhaoQi/hicosmo-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/JingZhaoQi/hicosmo-jax/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)
[![NumPyro](https://img.shields.io/badge/NumPyro-0.13.0+-orange.svg)](https://num.pyro.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-performance universal cosmology framework with enhanced neutral hydrogen capabilities**

**HIcosmo** (HI = neutral hydrogen, I = Roman numeral 1) is a universal cosmological parameter estimation framework with enhanced functionality and optimizations for neutral hydrogen cosmology and 21cm surveys. Built from the ground up with modern high-performance computing principles, it leverages JAX for automatic differentiation, JIT compilation, and GPU acceleration to deliver 5-10x performance improvements while maintaining scientific accuracy.

**Author**: Jingzhao Qi

## 🚀 Key Features

### ⚡ High-Performance Computing
- **JAX-powered**: Automatic differentiation, JIT compilation, GPU support
- **5-10x faster**: Compared to traditional scipy-based implementations
- **Vectorized operations**: Efficient batch calculations with `vmap`
- **Memory optimized**: Minimal memory footprint with smart array operations

### 🧮 Advanced Cosmology
- **25+ cosmological models**: From ΛCDM to exotic dark energy and modified gravity
- **Complete distance calculations**: Comoving, luminosity, angular diameter (with curvature)
- **Growth functions**: Linear growth factor, growth rate, fσ8 parameter
- **Time evolution**: Lookback time, age of universe, sound horizon
- **Specialized calculations**: Time-delay distances, redshift drift, critical density

### 📊 Production-Ready Analysis
- **Modern MCMC**: NumPyro NUTS sampler with adaptive tuning
- **Multi-probe data**: SNe Ia, BAO, CMB, H₀, strong lensing, GW, FRB, 21cm
- **Fisher matrix**: Automatic differentiation for forecasting
- **Real-time diagnostics**: R̂ statistics, ESS, convergence monitoring
- **Cobaya compatible**: Seamless integration with existing workflows

## 🏗️ Architecture Highlights

### Pure Functional Design
```python
@jit
def E_z(z, params):
    """JIT-compiled Hubble parameter - no side effects"""
    return jnp.sqrt(params['Omega_m'] * (1 + z)**3 + params['Omega_Lambda'])
```

### Inheritance Over Conditionals
```python
class LCDM(CosmologyBase):
    @staticmethod
    @jit
    def E_z(z, params):
        return jnp.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)

class wCDM(CosmologyBase):  
    @staticmethod
    @jit
    def E_z(z, params):
        return jnp.sqrt(Omega_m * (1+z)**3 + Omega_DE * rho_DE_z(z, params))
```

### Automatic Differentiation
```python
# Fisher matrix computed exactly via autodiff
fisher_matrix = jax.hessian(log_likelihood)(params)
```

## 📈 Performance Comparison

| Operation | qcosmc (scipy) | HIcosmo (JAX) | Speedup |
|-----------|----------------|---------------|---------|
| Distance calculation (1000 points) | 0.15s | 0.02s | **7.5x** |
| MCMC sampling (10k samples) | 180s | 45s | **4.0x** |
| Fisher matrix | 2.1s | 0.5s | **4.2x** |
| Growth function | 0.08s | 0.01s | **8.0x** |

*Benchmarks on Intel i7-10700K, NVIDIA RTX 3080*

## 🛠️ Installation

### Basic Installation
```bash
pip install hicosmo-jax
```

### Development Installation  
```bash
git clone https://github.com/JingZhaoQi/hicosmo-jax.git
cd hicosmo-jax
pip install -e ".[dev]"
```

### GPU Support
```bash
pip install "hicosmo-jax[gpu]"
```

## 🚀 Quick Start

```python
import jax.numpy as jnp
from hicosmo.models import LCDM
from hicosmo.likelihoods.sne import PantheonPlus
from hicosmo.samplers import MCMCSampler

# 1. Set up cosmological model
model = LCDM.planck2018()

# 2. Load observational data  
likelihood = PantheonPlus()
likelihood.initialize()

# 3. Run MCMC analysis
sampler = MCMCSampler(model, likelihood)
samples = sampler.run(num_samples=2000, num_chains=4)

# 4. Analyze results
summary = sampler.get_summary()
corner_plot = sampler.plot_corner()
```

## 📚 Comprehensive Model Library

[Contributor Guide](AGENTS.md)

### Dark Energy Models
- **ΛCDM**: Standard cosmological constant
- **wCDM**: Constant equation of state  
- **CPL**: Chevallier-Polarski-Linder parameterization
- **JBP**: Jassal-Bagla-Padmanabhan model
- **Holographic DE**: Various cutoff implementations

### Modified Gravity
- **f(R) Gravity**: Power-law modifications
- **f(T) Gravity**: Torsion-based theories  
- **DGP**: Braneworld models
- **Scalar-Tensor**: Jordan-Fierz-Brans-Dicke

### Exotic Models  
- **Chaplygin Gas**: Generalized and modified versions
- **Interacting DE**: Various coupling mechanisms
- **Running Vacuum**: Time-varying cosmological term
- **Early Dark Energy**: Pre-recombination modifications

## 🔬 Multi-Probe Observatory

### Current Surveys
- **Pantheon+**: 1701 Type Ia supernovae
- **DESI DR1**: Latest BAO measurements
- **Planck 2018**: CMB compressed likelihood
- **SH0ES**: Local H₀ distance ladder
- **H0LiCOW**: Strong lensing time delays

### Future Surveys
- **Roman Space Telescope**: Next-gen SNe survey
- **Euclid**: Weak lensing and BAO
- **SKA**: 21cm intensity mapping
- **LSST**: Photometric supernovae
- **Lisa**: Space-based gravitational waves

## 🎯 Scientific Applications

### Parameter Constraints
```python
# Joint analysis of multiple probes
combined = CombinedLikelihood([
    PantheonPlus(),
    DESIDR1(),  
    PlanckCMB(),
    SH0ES()
])

results = run_mcmc(combined, model='wCDM')
print(f"H₀ = {results.H0:.2f} ± {results.H0_err:.2f} km/s/Mpc")
print(f"w = {results.w:.3f} ± {results.w_err:.3f}")
```

### Survey Forecasting
```python
# Predict constraints for future surveys
fisher = FisherMatrix(model=LCDM(), surveys=['Roman', 'Euclid'])
forecasts = fisher.marginalize(['H0', 'Omega_m', 'w'])
fisher.plot_ellipses()
```

### Model Comparison
```python
# Bayesian evidence calculation
models = [LCDM(), wCDM(), CPL()]
evidence = model_comparison(data, models)
print(f"Best model: {evidence.best_model}")
print(f"Bayes factor: {evidence.bayes_factor:.2f}")
```

## 🏆 Why HIcosmo?

### Scientific Rigor
- **Numerical precision**: High-accuracy integrations
- **Parameter validation**: Physical consistency checks  
- **Error propagation**: Automatic uncertainty quantification
- **Reproducible**: Fixed random seeds, version control

### Modern Software Engineering
- **Type safety**: Complete type annotations
- **Testing**: >90% code coverage
- **Documentation**: Comprehensive API docs
- **CI/CD**: Automated testing and deployment

### Community Focused
- **Open source**: MIT license
- **Collaborative**: GitHub-based development
- **Educational**: Extensive tutorials and examples
- **Extensible**: Plugin architecture for new models

## 📖 Documentation

- [**Architecture Guide**](docs/ARCHITECTURE.md) - Design principles and structure
- [**Development Guide**](docs/DEVELOPMENT.md) - Contributing guidelines
- [**API Reference**](https://hicosmo-jax.readthedocs.io) - Complete function reference
- [**Tutorials**](examples/) - Step-by-step examples
- [**Benchmarks**](benchmarks/) - Performance comparisons

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/JingZhaoQi/hicosmo-jax.git
cd hicosmo-jax
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v --cov=hicosmo
```

## 📝 Citation

If you use HIcosmo in your research, please cite:

```bibtex
@software{hicosmo2024,
  author = {Jingzhao Qi and HIcosmo Team},
  title = {HIcosmo: High-performance JAX-based cosmological parameter estimation},
  year = {2024},
  url = {https://github.com/JingZhaoQi/hicosmo-jax},
  version = {0.1.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

HIcosmo builds upon the excellent foundation of:
- **qcosmc** - Original cosmological analysis package
- **JAX** - Numerical computing and automatic differentiation  
- **NumPyro** - Probabilistic programming and MCMC
- **Cobaya** - Cosmological parameter sampling
- **GetDist** - MCMC analysis and visualization

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JingZhaoQi/hicosmo-jax&type=Date)](https://star-history.com/#JingZhaoQi/hicosmo-jax&Date)

---

**HIcosmo** - Empowering the next generation of precision cosmology 🚀
