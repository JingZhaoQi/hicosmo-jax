# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HIcosmo** (HI = neutral hydrogen, I = Roman numeral 1) is a high-performance JAX-based cosmological parameter estimation framework targeting 5-10x performance improvements over traditional scipy implementations. Built with modern software engineering practices and designed for both CPU and GPU acceleration.

**Code Scale**: ~22,900 lines across 66 Python files
**Tech Stack**: JAX + NumPyro + GetDist + Astropy
**Performance Goal**: Single calculation < 0.01ms, 5-10x faster than qcosmc

## Essential Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=hicosmo

# Run specific test file
pytest tests/test_lcdm.py -v

# Run performance benchmarks
pytest tests/test_*_benchmark.py -v

# Skip slow tests
pytest -m "not slow" -v
```

### Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Format code (auto-fixes)
black hicosmo/ tests/

# Sort imports
isort hicosmo/ tests/

# Type checking
mypy hicosmo/

# Linting
flake8 hicosmo/ tests/
```

### Running Examples
```bash
# Basic MCMC example
python example_sh0es_mcmc.py

# BAO analysis
python example_bao_analysis.py

# H0LiCOW constraints
python example_h0licow_mcmc.py
```

## Architecture Overview

### 6-Layer Hierarchical Design

#### Layer 1: Core Foundation (`hicosmo/core/`)
- **CosmologyBase** (base.py): Minimal abstract base class defining the model interface
  - Must implement: `E_z()`, distance calculations, growth functions
  - Pattern: Keep base class minimal, no conditionals
- **FastIntegration** (fast_integration.py:568 lines): Ultra-performance integration engine
  - 3,400x faster than original scipy.integrate.quad
  - Auto-selects method: Ultra-Fast, Vectorized, Interpolation, Optimized Batch
  - Modes: fast/balanced/precise
- **CosmologicalParameters** (unified_parameters.py): Centralized parameter management
  - Single source of truth for all parameters
  - Built-in validation and default values

#### Layer 2: Cosmological Models (`hicosmo/models/`)
- **LCDM** (lcdm.py:1,113 lines): Reference implementation, fully featured
  - Supports non-flat universes (Omega_k)
  - Sound horizon calculation (Eisenstein & Hu 1998)
  - Growth functions (Carroll, Press & Turner 1992)
- **wCDM**: Constant dark energy equation of state
- **CPL**: Chevallier-Polarski-Linder parameterization

**Pattern**: Each model inherits CosmologyBase and overrides E_z() with JIT-compiled implementation

#### Layer 3: Likelihood System (`hicosmo/likelihoods/`)
- **PantheonPlus** (pantheonplus.py): 1,701 SNe Ia with full covariance
- **BAO** (bao_datasets.py): Multiple BAO datasets (SDSS, DESI, 6dFGS)
- **Strong Lensing**: H0LiCOW (h0licow.py), TDCOSMO (tdcosmo.py)
- **CMB**: Planck 2018 distance priors (planck_distance.py)
- **H0**: SH0ES distance ladder (sh0es.py)

#### Layer 4: MCMC Sampling (`hicosmo/samplers/`)
- **MCMCSampler** (core.py:808 lines): NumPyro NUTS wrapper
- **MCMC** (inference.py:943 lines): High-level dict-driven interface
- **ParameterConfig** (config.py:584 lines): Parameter setup and validation
- **Persistence** (init.py:799 lines): Checkpoint save/restore system

#### Layer 5: Fisher Matrix (`hicosmo/fisher/`)
- **FisherMatrix** (fisher_matrix.py:598 lines): Autodiff-based exact Fisher matrix (4.2x faster)
- **FiguresOfMerit** (figures_of_merit.py:657 lines): Constraint calculations
- **Forecasting** (forecasting.py:643 lines): Survey optimization

#### Layer 6: Visualization (`hicosmo/visualization/`)
- **Minimalist Design**: Reduced from 3,818 to ~800 lines (83% reduction)
- **Function Interface**: `plot_corner()`, `plot_chains()`, `plot_traces()`
- **GetDist Backend**: Professional publication-quality plots

### Key Design Patterns

1. **Inheritance Over Conditionals**
   ```python
   # ✅ Good: Minimal base class
   class CosmologyBase:
       @abstractmethod
       def E_z(z, params): pass

   # ✅ Good: Specialized via inheritance
   class LCDM(CosmologyBase):
       @staticmethod
       @jit
       def E_z(z, params):
           return jnp.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)

   # ❌ Bad: Conditionals in base class
   class CosmologyBase:
       def E_z(z, params):
           if self.model == "LCDM":
               return ...
           elif self.model == "wCDM":
               return ...
   ```

2. **Pure Functional Design**
   ```python
   # ✅ Good: Pure function, JIT-able
   @jit
   def E_z(z, params):
       return jnp.sqrt(params['Omega_m'] * (1 + z)**3 + params['Omega_Lambda'])

   # ❌ Bad: Side effects
   def E_z(z, params):
       self.last_z = z  # Breaks JIT compilation!
       return jnp.sqrt(...)
   ```

3. **Unified Parameter System**
   ```python
   # ✅ Good: Use CosmologicalParameters
   from hicosmo.core import CosmologicalParameters
   params = CosmologicalParameters(H0=70.0, Omega_m=0.3)

   # ❌ Bad: Custom parameter dicts
   params = {"hubble": 70, "matter_density": 0.3}
   ```

## Critical Development Rules

### 🚨 Testing Rules (Most Important!)
- **NO REIMPLEMENTATION IN TESTS**: Tests must use existing modules, never reimplement!
- **TEST EXISTING CODE**: Purpose is to verify existing code works, not implement tested features!
- **USE EXISTING IMPORTS**: Import from hicosmo.models, hicosmo.likelihoods, hicosmo.samplers
- **NO TEST ADAPTERS**: Test production code directly, no wrapper layers!

When user asks to "write tests", they want to verify existing code functionality, not implement new features!

### Performance Standards
- **NO SLOW CODE**: Any calculation > 1ms must be optimized or rewritten
- **NO DIFFRAX**: Verified too slow, use FastIntegration or native JAX
- **NO NUMPY IN HOT PATHS**: Hot paths must use JAX for JIT compilation
- **BENCHMARK EVERYTHING**: New features require performance comparison tests

### Architecture Standards
- **NO DUPLICATE PARAMETERS**: Use unified CosmologicalParameters system
- **NO REDUNDANT MODULES**: Check for existing implementations before creating new ones
- **NO BLOATED BASE CLASSES**: Keep base classes minimal, specialize via inheritance
- **NO MIXED IMPORTS**: Import core components consistently from hicosmo.core

### Code Quality Standards
- **NO PARTIAL IMPLEMENTATION**: Either fully implement or don't do it
- **NO SIMPLIFICATION COMMENTS**: Don't write code marked as "simplified"
- **NO CODE DUPLICATION**: Check existing implementations first
- **NO DEAD CODE**: Remove unused code immediately
- **NO CHEATER TESTS**: Tests must reflect real usage scenarios
- **NO RESOURCE LEAKS**: Properly manage file handles, memory, etc.

## Performance Benchmarks

| Operation | qcosmc (scipy) | HIcosmo (JAX) | Speedup |
|-----------|----------------|---------------|---------|
| Distance calculation (1000 pts) | 0.15s | 0.02s | **7.5x** |
| MCMC sampling (10k samples) | 180s | 45s | **4.0x** |
| Fisher matrix | 2.1s | 0.5s | **4.2x** |
| Growth function (1000 pts) | 0.08s | 0.01s | **8.0x** |

## Typical Workflow Examples

### Running MCMC Analysis
```python
from hicosmo.models import LCDM
from hicosmo.likelihoods import PantheonPlusLikelihood
from hicosmo.samplers import MCMC

# Setup configuration
config = {
    'parameters': {
        'H0': {'prior': {'dist': 'uniform', 'min': 50, 'max': 100}},
        'Omega_m': {'prior': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}}
    },
    'likelihood': PantheonPlusLikelihood('data/pantheonplus/')
}

# Run MCMC
mcmc = MCMC(config)
mcmc.sample(num_samples=10000, num_chains=4)

# Visualize results
from hicosmo.visualization import plot_corner
fig = plot_corner(mcmc.chains)
```

### Adding a New Model
```python
from hicosmo.core import CosmologyBase
from jax import jit
import jax.numpy as jnp

class MyModel(CosmologyBase):
    """New cosmological model."""

    @staticmethod
    @jit
    def E_z(z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """Hubble parameter evolution E(z) = H(z)/H0."""
        # Your implementation here
        return jnp.sqrt(...)

    # Optionally override other methods for specialized behavior
```

## Important Files to Understand

1. **hicosmo/core/base.py** - CosmologyBase interface definition
2. **hicosmo/models/lcdm.py** - Reference LCDM implementation (1,113 lines)
3. **hicosmo/core/fast_integration.py** - High-performance integration engine
4. **hicosmo/samplers/inference.py** - MCMC high-level API
5. **hicosmo/core/unified_parameters.py** - Parameter management system

## Common Pitfalls

1. **Don't mix NumPy and JAX**
   ```python
   # ❌ Bad: Mixed NumPy/JAX
   import numpy as np
   result = jnp.sqrt(np.array([1, 2, 3]))

   # ✅ Good: Pure JAX
   import jax.numpy as jnp
   result = jnp.sqrt(jnp.array([1, 2, 3]))
   ```

2. **Don't modify arrays in-place**
   ```python
   # ❌ Bad: In-place modification (JAX arrays immutable)
   arr[0] = 5

   # ✅ Good: Create new array
   arr = arr.at[0].set(5)
   ```

3. **Don't use Python loops for array operations**
   ```python
   # ❌ Bad: Python loop
   results = [model.E_z(z, params) for z in z_array]

   # ✅ Good: Vectorized
   results = jax.vmap(lambda z: model.E_z(z, params))(z_array)
   ```

## Agent Usage Strategy

- **file-analyzer**: Always use for reading/analyzing files
- **code-analyzer**: Use for bug research, logic tracing
- **test-runner**: Use for running tests and analyzing results
- **Explore agent**: Use for codebase exploration (not needle queries)

## Module Status & Priority

### ✅ Production-Ready
- Core cosmology (CosmologyBase, LCDM)
- FastIntegration performance engine
- MCMC sampling framework
- Visualization system
- PantheonPlus, BAO, SH0ES likelihoods

### ⚠️ Needs Refactoring (High Priority)
1. **fisher/** - Outdated dependencies, inconsistent interface
2. **likelihoods/** - Some datasets need parameter system update

### 🚧 In Development
- Additional cosmological models (3/25+ complete)
- Advanced MCMC diagnostics
- 21cm intensity mapping module

## Success Criteria for New Code

- ✅ Performance test passes, beats competitors
- ✅ Unified architecture test passes
- ✅ Code is concise, no duplication
- ✅ Clean import structure
- ✅ Complete test coverage
- ✅ Type annotations on all functions
- ✅ NumPy-style docstrings

## Additional Resources

- **README.md**: Project overview, installation, quick start
- **AGENTS.md**: Contributor guide for multi-agent collaboration
- Root CLAUDE.md: Detailed architecture and roadmap
- **pyproject.toml**: Package configuration and dependencies

## Notes

- This is an active research codebase with frequent updates
- Recent major optimization: 83% code reduction in visualization module
- Performance is non-negotiable - benchmark everything
- When in doubt, check LCDM implementation as reference
