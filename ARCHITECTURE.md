# HIcosmo Codebase Architecture Overview

**Project Status**: Active development (v0.1.0)
**Language**: Python 3.9+
**Core Framework**: JAX with NumPyro for MCMC
**Total Codebase**: ~22,900 lines across 66 Python files
**Key Performance Goal**: 5-10x faster than traditional scipy-based frameworks

---

## Executive Summary

HIcosmo (HI = neutral hydrogen, I = Roman numeral 1) is a modern, high-performance cosmological parameter estimation framework built from the ground up with JAX for automatic differentiation, JIT compilation, and GPU acceleration. It emphasizes clean architecture through single responsibility, inheritance over conditionals, and pure functional design.

**Core Philosophy** (from CLAUDE.md):
- **奥卡姆剃刀原则** (Occam's Razor): Only implement necessary functionality
- **基类简洁** (Minimal Base Classes): No conditional branches in base classes
- **继承重写** (Inheritance Specialization): Override methods in subclasses for special cases
- **单一责任** (Single Responsibility): Each class does one thing, each method one calculation
- **纯函数优先** (Pure Functions First): All cosmological calculations are pure functions with JAX

---

## Project Structure

```
hicosmo/
├── core/                    # Fundamental cosmology infrastructure
│   ├── base.py             # Abstract base classes (CosmologyBase, DistanceCalculator)
│   ├── fast_integration.py # Ultra-high performance integration engine (568 lines)
│   └── unified_parameters.py # Single parameter management system
├── models/                  # Cosmological models
│   ├── lcdm.py            # Lambda-CDM model (1,113 lines - largest file)
│   ├── wcdm.py            # Constant dark energy EoS
│   └── cpl.py             # Chevallier-Polarski-Linder parametrization
├── likelihoods/            # Observational data likelihoods
│   ├── pantheonplus.py    # SNe Ia supernova likelihood
│   ├── bao_base.py        # Baryon Acoustic Oscillations base
│   ├── bao_datasets.py    # Multiple BAO datasets (SDSS, DESI, etc.)
│   ├── h0licow.py         # Strong lensing H0 measurements
│   ├── planck_distance.py # CMB distance priors
│   ├── sh0es.py           # SH0ES distance ladder
│   ├── tdcosmo.py         # TDCOSMO strong lens time delays
│   └── base.py            # Abstract base for all likelihoods
├── samplers/               # MCMC and parameter estimation
│   ├── core.py            # Lightweight NumPyro wrapper (808 lines)
│   ├── inference.py       # High-level MCMC interface (943 lines)
│   ├── config.py          # Parameter configuration (584 lines)
│   ├── utils.py           # Parameter mapping and utilities (655 lines)
│   ├── persistence.py     # Checkpoint/resume system (799 lines)
│   ├── multicore.py       # Multi-chain parallelization
│   ├── constants.py       # MCMC constants
│   └── init.py            # Elegant initialization system
├── fisher/                 # Fisher matrix analysis & forecasting
│   ├── fisher_matrix.py   # Automatic differentiation-based Fisher (598 lines)
│   ├── figures_of_merit.py # FoM calculations (657 lines)
│   ├── numerical_derivatives.py # Numerical gradient computation
│   ├── forecasting.py     # Survey forecasting (643 lines)
│   └── intensity_mapping.py # 21cm cosmology
├── visualization/          # Scientific plotting (minimalist redesign)
│   ├── core.py            # Base visualization classes
│   ├── plotting.py        # Main plotting functions (GetDist wrapper)
│   ├── multi_chain.py     # Multi-chain visualization
│   ├── styles.py          # Professional styling
│   └── __init__.py        # Clean function interface
├── utils/                  # Utilities
│   ├── constants.py       # Physical constants and defaults
│   └── integration.py     # Integration utilities
├── data/                   # Observational datasets
│   ├── bao_data/          # BAO measurements
│   ├── sne/               # Supernovae data
│   ├── h0licow/           # Strong lensing data
│   ├── tdcosmo/           # TDCOSMO data
│   └── cmb/               # CMB power spectra
├── parameters/             # Parameter management (legacy, partially deprecated)
├── cmb/                    # CMB power spectrum calculations
├── powerspectrum/          # Matter power spectrum
├── perturbations/          # Linear perturbation theory
├── interfaces/             # External library compatibility (CAMB, CLASS)
├── forecasts/              # Survey forecasting
└── configs/                # Configuration files
```

---

## Core Architecture Layers

### Layer 1: Cosmology Foundation (`hicosmo/core/`)

**Purpose**: Minimal, focused interface for all cosmological calculations

#### `CosmologyBase` (Abstract Base Class)
- **Core Interface** (mandatory for all models):
  - `E_z(z)`: Dimensionless Hubble parameter H(z)/H₀
  - `comoving_distance(z)`: Physical distances
  - `angular_diameter_distance(z)`: Angular distance
  - `luminosity_distance(z)`: Luminosity distance
  - `distance_modulus(z)`: Distance modulus in magnitudes

- **Optional Interface** (with defaults):
  - `w_z(z)`: Dark energy equation of state
  - `get_parameters()`: Parameter dictionary access
  - `update_parameters()`: Parameter updates

- **Design Philosophy**: 
  - Minimal abstract methods (no bloat)
  - No implementation details in base class
  - Subclasses implement specific physics

#### `FastIntegration` (The Performance Engine)
- **Purpose**: Ultra-fast numerical integration for comoving distance calculations
- **Key Stats**: 
  - **3,400x faster** than original scipy-based implementation
  - **Line Count**: 568 lines with complete modularity
  - **Method Selection**: Automatic adaptive method selection
    - **Ultra-Fast**: Single point JIT with 8-point Gaussian quadrature
    - **Vectorized**: Batch computation with NumPy-based approach
    - **Interpolation**: Pre-computed lookup tables for large batches
    - **Optimized Batch**: JAX vmap for maximum vectorization

- **Precision Modes**:
  - `'fast'`: 8-point Gauss integration (speed-optimized)
  - `'balanced'`: 12-point Gauss integration (default)
  - `'precise'`: 16-point Gauss integration (accuracy-optimized)

- **Architecture**:
  - Fully parametric design (no hardcoding of cosmological parameters)
  - JIT-compiled integration kernel
  - LRU caching for repeated calculations
  - Pre-computation of lookup tables for batch operations

#### `CosmologicalParameters` (Unified Parameter Management)
- **Purpose**: Single source of truth for all parameter handling
- **Features**:
  - Built-in parameter specifications with ranges
  - Validation on input
  - Default values (Planck 2018, WMAP9)
  - Derived parameter computation
  - Type checking and error handling

### Layer 2: Cosmological Models (`hicosmo/models/`)

**Architecture Pattern**: Inheritance-based specialization

#### Model Hierarchy
```
CosmologyBase (abstract)
    ├── LCDM (1,113 lines)
    │   ├── Flat + matter + radiation + dark energy
    │   ├── Sound horizon calculation (Eisenstein & Hu 1998)
    │   ├── Growth functions (Carroll, Press & Turner 1992)
    │   └── Validation: closure relation (Ω_total = 1 ± 1e-6)
    ├── wCDM (constant w_0 dark energy)
    │   └── Overrides E_z() and w_z()
    └── CPL (Chevallier-Polarski-Linder parametrization)
        └── Time-dependent w(z) = w0 + wa(1-a)
```

#### LCDM Implementation (The Gold Standard)
- **1,113 lines** - largest production model
- **Complete Feature Set**:
  - Flat and non-flat geometries (Omega_k support)
  - Radiation component with neutrino masses
  - Sound horizon/acoustic scale
  - Recombination history
  - Growth functions and linear growth rates
  - All distance measures (comoving, angular, luminosity, modulus)
  - Time evolution (lookback time, age, Hubble time)
  - Specialized: time-delay distances, redshift drift

- **Validation**:
  - Physical parameter ranges (H0: 20-200 km/s/Mpc, Omega_m: 0.01-1.0)
  - Density closure checks
  - Negative density detection

- **Performance Optimization**:
  - FastIntegration engine for distance calculations
  - JAX JIT compilation of E_z()
  - Parameter validation at initialization (not per-call)

### Layer 3: Likelihood System (`hicosmo/likelihoods/`)

**Design Philosophy**: Simple interface: `likelihood.log_likelihood(model)`

#### Available Likelihoods

1. **PantheonPlus** (SNe Ia)
   - 1,701 Type Ia supernovae
   - Marginalized or free M_B parameter
   - Support for Pantheon+SH0ES combined data
   - Covariance matrix with systematic uncertainties

2. **BAO Likelihoods** (Baryon Acoustic Oscillations)
   - `BAOLikelihood`: Base class for generic BAO
   - Datasets:
     - SDSS DR12 (BOSS)
     - SDSS DR16 (eBOSS)
     - DESI 2024
     - 6dFGS
     - Custom datasets
   - Measurements: D_M/r_d and H*r_d

3. **Strong Lensing** (H0LiCOW)
   - Time-delay cosmography
   - Multiple lens systems (HE0435-1223, PG1115+080, etc.)
   - Time-delay distance D_Δt = c * Δt_obs / (1 + z_lens)

4. **CMB Distance Priors** (Planck 2018)
   - Compressed likelihood from Planck
   - Angular scale l_A, curvature R, recombination redshift z_*

5. **H0 Measurements**
   - SH0ES distance ladder constraints
   - Strong constraint on Hubble tension parameter space

6. **TDCOSMO** (TDCOSMO Collaboration)
   - Strong lens time-delay measurements
   - Kappa prior for mass sheet degeneracy
   - Multiple lenses: DES J0408, HE0435, etc.

#### Likelihood Interface Pattern
```python
class LikelihoodBase:
    def __init__(self, data_path: str, **options):
        self._load_data()
        self._precompute_covariance()
    
    def log_likelihood(self, model: CosmologyBase) -> float:
        """Return log-likelihood for given cosmological model"""
        model_predictions = self._predict(model)
        return self._compute_log_likelihood(model_predictions)
```

### Layer 4: MCMC Sampling (`hicosmo/samplers/`)

**Backend**: NumPyro (thin wrapper, not abstraction layer)

#### Core Components

1. **MCMCSampler** (808 lines) - Lightweight NumPyro wrapper
   - Direct use of NUTS, HMC, SA samplers
   - No restrictions on user model definitions
   - Checkpoint save/restore
   - Rich progress display
   - Diagnostic output (R̂, ESS, Gelman-Rubin)

2. **MCMC** (943 lines) - High-level interface
   - Dictionary-driven configuration
   - Automatic parameter mapping
   - Intelligent initialization strategies
   - Comprehensive data persistence
   - Resume from checkpoint
   - Multi-chain support

3. **ParameterConfig** (584 lines) - Configuration system
   - Prior specification (uniform, normal, log-normal)
   - Reference values for initialization
   - Parameter bounds and validation
   - Free/fixed parameter control

4. **Persistence System** (799 lines)
   - `CheckpointManager`: Save intermediate states
   - `ResumeManager`: Continue from checkpoints
   - Complete metadata recording
   - GetDist-compatible output formats

#### MCMC Workflow
```python
# Simple dictionary-based configuration
config = {
    'parameters': {
        'H0': {'prior': {'dist': 'uniform', 'min': 50, 'max': 100}},
        'Omega_m': {'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}}
    },
    'likelihood': pantheon_likelihood
}

mcmc = MCMC(config)
mcmc.sample(num_samples=10000, num_chains=4)
results = mcmc.get_results()  # Returns chains, diagnostics, summary
```

### Layer 5: Fisher Matrix Analysis (`hicosmo/fisher/`)

**Key Strength**: Automatic differentiation-based exact Fisher matrices

#### Components

1. **FisherMatrix** (598 lines)
   - JAX automatic differentiation: `fisher = jax.hessian(log_likelihood)`
   - Parameter space transformations
   - Prior integration
   - Marginalization over nuisance parameters

2. **FiguresOfMerit** (657 lines)
   - Dark energy FoM calculation
   - Parameter constraint ellipses
   - Correlation analysis
   - Constraint degradation studies

3. **Forecasting** (643 lines)
   - Survey design optimization
   - Redshift bin optimization
   - Sample size/time trade-off analysis

4. **NumericalDerivatives** 
   - Fallback for non-differentiable operations
   - Finite difference gradients

### Layer 6: Visualization (`hicosmo/visualization/`)

**Major Refactoring**: Minimalist redesign (83% code reduction)
- **Original**: 3,818 lines across 10 files
- **Current**: ~800 lines across 3 files
- **Performance**: No functionality lost

#### Components

1. **Function Interface** (Recommended)
   ```python
   from hicosmo.visualization import plot_corner, plot_chains
   
   fig = plot_corner(chain_data, params=[1,2,3], filename='corner.pdf')
   fig = plot_chains(chain_data, params=['H0', 'Omega_m'])
   ```

2. **Class Interface** (Backward compatibility)
   ```python
   from hicosmo.visualization import HIcosmoViz
   
   viz = HIcosmoViz()
   fig = viz.corner(data, params=[1,2,3])
   ```

3. **Multi-Chain Management**
   - `MultiChain` class for comparing multiple runs
   - Automatic legend generation
   - Color consistency

4. **Features**
   - GetDist backend for professional plotting
   - Smart tick anti-overlap
   - LaTeX labels with auto-units
   - Professional styling (Modern/Classic)
   - High-quality PDF output (300 DPI)
   - Auto-save to results/ directory

---

## Data Organization (`hicosmo/data/`)

```
data/
├── bao_data/           # BAO measurements and covariances
├── sne/               # Pantheon+ supernovae data
├── h0licow/           # Strong lensing systems
├── tdcosmo/           # TDCOSMO lens systems
└── cmb/               # CMB power spectra
```

All data files are pre-processed and validated for consistency with literature values.

---

## Dependencies and External Interfaces

### Core Dependencies
- **JAX** (0.4.20+): Automatic differentiation, JIT compilation, GPU support
- **NumPyro** (0.13.0+): MCMC sampling (NUTS, HMC)
- **NumPy/SciPy**: Scientific computing
- **Astropy**: Astronomical utilities

### Optional Integrations
- **CAMB** (1.5.0+): CLASS power spectrum (optional)
- **Cobaya** (3.4+): External sampler compatibility (optional)
- **GetDist** (1.4.0+): Visualization backend
- **HealPy** (1.16.0+): CMB map tools

### No Longer Used
- **Diffrax**: Removed due to performance issues
  - Replaced by `FastIntegration` (3,400x faster)

---

## Integration Points Between Modules

### Model → Likelihood Flow
```
LCDM (model)
    ├── compute E_z(z)
    ├── calculate comoving_distance(z)
    └── provide luminosity_distance(z)
        ↓
    PantheonPlusLikelihood
        ├── compute distance modulus(z)
        ├── compare with SNe observations
        └── return log_likelihood
```

### Likelihood → MCMC Flow
```
Config (dict)
    ↓
ParameterMapper
    ├── parse parameters
    ├── create model factory
    └── wrap likelihood
        ↓
    MCMCSampler (NumPyro)
        ├── generate proposal
        ├── evaluate likelihood
        └── return acceptance decision
```

### MCMC → Visualization Flow
```
MCMCSampler (chains)
    ├── save to disk
    └── return chain arrays
        ↓
    HIcosmoViz
        ├── load chains
        ├── compute statistics
        └── generate corner/trace plots
```

### Fisher Matrix → Forecasting
```
Likelihood (function)
    ├── jax.hessian(log_likelihood)
    ├── evaluate at fiducial model
    └── compute Fisher matrix F_ij
        ↓
    FiguresOfMerit
        ├── marginalize parameters
        ├── compute constraint ellipses
        └── calculate FoM
```

---

## Testing Infrastructure

### Test Organization
```
tests/
├── test_models.py           # Model implementations
├── test_multicore_config.py # Multi-chain setup
├── analysis/
│   └── test_fast_integration_jax.py  # Integration engine
└── data/
    └── [various test data files]
```

### Test Coverage
- **Unit tests**: Model initialization, distance calculations
- **Integration tests**: MCMC → visualization pipeline
- **Performance tests**: Benchmarking against scipy/astropy

### Running Tests
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=hicosmo

# Specific markers
pytest -m "not slow"  # Skip slow tests
pytest -m gpu         # GPU-only tests
```

### CI/CD Pipeline
- **Workflow**: `.github/workflows/ci.yml`
- **Triggers**: Push to main/develop, Pull requests
- **Python Versions**: 3.9, 3.10, 3.11
- **Checks**: 
  - Linting (flake8)
  - Format (black)
  - Type checking (mypy)
  - Unit tests (pytest)
  - Coverage reporting

---

## Performance Characteristics

### Benchmark Results (vs. Competitors)

| Operation | qcosmc (scipy) | HIcosmo (JAX) | Speedup |
|-----------|---|---|---|
| Distance calculation (1000 points) | 0.15s | 0.02s | **7.5x** |
| MCMC sampling (10k samples) | 180s | 45s | **4.0x** |
| Fisher matrix | 2.1s | 0.5s | **4.2x** |
| Growth function (1000 points) | 0.08s | 0.01s | **8.0x** |

*Benchmarks: Intel i7-10700K, NVIDIA RTX 3080*

### JAX JIT Compilation Impact
- **First call**: ~2-5x slowdown (compilation)
- **Subsequent calls**: ~4-7x speedup (caching)
- **Break-even point**: ~2-3 function calls
- **Strategy**: Compile once at model initialization, reuse forever

### Memory Efficiency
- **Chain storage**: O(n_samples × n_chains × n_params)
- **Likelihood evaluation**: Vectorized batch operations
- **GPU ready**: Transparent CUDA/Metal support

---

## Recent Major Changes (Git History)

### Latest Commits (Last 30 Days)

1. **7e446c6**: 🚀 MAJOR: JAX JIT optimization fixes M_B performance issue
   - Critical fix for Pantheon+ M_B parameter handling
   - JAX compilation strategy optimization

2. **f9f0135**: 🔒 Backup before M_B JAX JIT optimization
   - Safety checkpoint before major refactoring

3. **d32a627**: ✨ Perfect MCplot visualization with intelligent tick optimization
   - Visualization refinement

4. **67e3d2b**: 🚀 MAJOR: Multi-Chain Visualization Support + Clean Architecture
   - Multi-chain comparison visualization
   - Architecture cleanup

5. **cc4572b**: 🚀 MAJOR: Minimalist Visualization System - 83% Code Reduction
   - Major refactor: 3,818 → 800 lines
   - Removed redundant code layers

6. **e58baa9**: 🚀 Major MCMC Interface Overhaul: Fix Sample Logic & Add Elegant Multi-core Init
   - MCMC interface improvements
   - Multi-core setup simplification

### Architectural Evolution
- **Phase 1** (Sept): Initial implementation with scipy.integrate
- **Phase 2** (Sept-Oct): Diffrax integration (performance issues)
- **Phase 3** (Oct): FastIntegration replacement (3,400x faster)
- **Phase 4** (Oct): Visualization refactoring (83% reduction)
- **Phase 5** (Current): JAX JIT optimization and multi-chain support

---

## Common Development Workflows

### Adding a New Cosmological Model
```python
# 1. Inherit from CosmologyBase
class NewModel(CosmologyBase):
    
    # 2. Implement core methods
    def E_z(self, z):
        """Custom Hubble parameter"""
        return jnp.sqrt(...)
    
    def comoving_distance(self, z):
        """Use FastIntegration"""
        return self.fast_integration.comoving_distance(z, self.E_z)
    
    # 3. Optional: override distance methods for efficiency
    def luminosity_distance(self, z):
        """Specialized calculation if possible"""
        D_M = self.comoving_distance(z)
        return D_M * (1 + z)

# 4. Register in models/__init__.py
from .newmodel import NewModel
__all__ = [..., 'NewModel']
```

### Adding a New Likelihood
```python
# 1. Inherit from likelihood base interface
class NewLikelihood:
    
    def __init__(self, data_path: str, **options):
        self._load_data()
        self._precompute_covariance()
    
    def log_likelihood(self, model: CosmologyBase) -> float:
        """Evaluate likelihood"""
        predictions = self._predict(model)
        return self._compute_log_likelihood(predictions)

# 2. Register in likelihoods/__init__.py
from .newlikelihood import NewLikelihood
__all__ = [..., 'NewLikelihood']
```

### Running MCMC Analysis
```python
# 1. Create configuration
config = {
    'parameters': {
        'H0': {'prior': {'dist': 'uniform', 'min': 50, 'max': 100}},
        'Omega_m': {'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}}
    },
    'likelihood': PantheonPlusLikelihood('path/to/data')
}

# 2. Run sampling
mcmc = MCMC(config)
mcmc.sample(num_samples=10000, num_chains=4)

# 3. Analyze results
results = mcmc.get_results()
fig = plot_corner(results.chains)
```

---

## Code Quality Standards (From CLAUDE.md)

### Absolute Rules
- ✅ **NO SLOW CODE**: Any calculation > 1ms must be optimized
- ✅ **NO REDUNDANT MODULES**: Reuse existing implementations
- ✅ **NO BLOATED BASE CLASSES**: Keep interfaces minimal
- ✅ **NO DEAD CODE**: Unused code removed immediately
- ✅ **NO CODE DUPLICATION**: Check existing implementations first

### Testing Rules
- ✅ **NO REIMPLEMENTATION**: Tests use existing modules
- ✅ **NO MOCK SERVICES**: Tests must use real data
- ✅ **VERBOSE DEBUGGING**: Tests show detailed output
- ✅ **COMPREHENSIVE COVERAGE**: Every function has tests

### Performance Rules
- ✅ **BENCHMARK EVERYTHING**: New features include performance tests
- ✅ **JAX FIRST**: Use JAX arrays and JIT compilation
- ✅ **VECTORIZE**: Use vmap instead of Python loops
- ✅ **PRECOMPUTE**: Cache tables for repeated calculations

---

## Future Development Priorities

### Phase 1: Core Models (Priority 1)
- [ ] wCDM: Constant dark energy equation of state
- [ ] CPL: Chevallier-Polarski-Linder parametrization ✓ (partial)
- [ ] JBP: Jassal-Bagla-Padmanabhan
- [ ] Chaplygin Gas: Multiple variants
- [ ] Modified Gravity: DGP, f(R), f(T)

### Phase 2: Observational Data (Priority 1)
- [x] Pantheon+ SNe: Complete implementation ✓
- [x] BAO: Multiple datasets ✓
- [ ] CMB: Full Planck likelihood (currently compressed)
- [ ] Growth: RSD measurements
- [ ] 21cm: Intensity mapping forecasts

### Phase 3: Advanced MCMC (Priority 2)
- [x] NUTS sampler: NumPyro wrapper ✓
- [x] Multi-chain support: Implemented ✓
- [ ] Convergence diagnostics: Real-time R̂ monitoring
- [ ] Adaptive tuning: Dynamic mass matrix
- [ ] Nested sampling: Alternative sampler

### Phase 4: Fisher Matrix (Priority 2)
- [x] Basic Fisher: JAX autodiff ✓
- [ ] Joint constraints: Multi-probe combinations
- [ ] Parameter transformations: Arbitrary reparametrizations
- [ ] Forecast optimization: Survey design

---

## Key Files Reference

### Must Read First
1. **hicosmo/core/base.py** - Understanding CosmologyBase interface
2. **hicosmo/models/lcdm.py** - LCDM as reference implementation
3. **hicosmo/samplers/inference.py** - MCMC high-level API
4. **hicosmo/likelihoods/__init__.py** - Available likelihoods

### Critical Performance Files
5. **hicosmo/core/fast_integration.py** - Performance engine
6. **hicosmo/samplers/core.py** - NumPyro wrapper
7. **hicosmo/fisher/fisher_matrix.py** - Automatic differentiation

### Configuration
8. **pyproject.toml** - Package configuration
9. **CLAUDE.md** - Development rules and principles
10. **.github/workflows/ci.yml** - CI/CD pipeline

---

## Common Debugging Scenarios

### "My model is slow"
→ Check: Is `E_z()` JIT compiled? Are you reusing `FastIntegration` object?

### "Likelihood evaluation fails"
→ Check: Does model implement required interface? Are parameter ranges valid?

### "MCMC chains not converging"
→ Check: Run MCMC with `verbose=True`, examine R̂ values. Check priors are reasonable.

### "Test failures in CI"
→ Check: `.github/workflows/ci.yml` for test runner command. Ensure all dependencies installed.

---

## Repository Structure

- **.git/**: Git history and configuration
- **.github/workflows/**: CI/CD pipeline (ci.yml)
- **hicosmo/**: Main package (source code)
- **tests/**: Test suite with pytest
- **docs/**: Documentation files
- **examples/**: Usage examples and tutorials
- **data/**: Observational datasets
- **results/**: Output from analyses (auto-created)
- **mcmc_chains/**: Saved MCMC chains (auto-created)

---

## Summary: Architecture at a Glance

```
User Code
    ↓
MCMC Interface (samplers/inference.py)
    ↓
ParameterConfig (samplers/config.py) → Model Factory
    ↓
CosmologyBase (core/base.py)
    ├── LCDM/wCDM/CPL (models/)
    └── FastIntegration (core/fast_integration.py)
    ↓
Likelihood (likelihoods/)
    ├── PantheonPlus (SNe)
    ├── BAO (BAO)
    ├── H0LiCOW (Lensing)
    └── Others
    ↓
NumPyro MCMC (samplers/core.py)
    ├── NUTS Sampler
    ├── Checkpoint (samplers/persistence.py)
    └── Diagnostics
    ↓
Results → Visualization (visualization/)
    ├── Corner plots
    ├── Trace plots
    └── Multi-chain comparison
```

---

**Document generated**: 2025-10-28
**Version**: 0.1.0
**Maintainer**: Jingzhao Qi
