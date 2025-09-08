# HiCosmo Architecture

## Overview

HiCosmo is a high-performance cosmological parameter estimation framework built on JAX and NumPyro. The architecture follows functional programming principles with pure functions for all cosmological calculations, enabling automatic differentiation and JIT compilation.

## Core Design Principles

### 1. JAX-First Architecture
- **Pure Functions**: All cosmological calculations are pure functions without side effects
- **Automatic Differentiation**: Leverages JAX's autograd for gradient computation
- **JIT Compilation**: All performance-critical code is JIT-compiled for speed
- **GPU Compatible**: Transparent GPU acceleration through JAX

### 2. Modular Design
- **Single Responsibility**: Each module handles one specific aspect
- **Inheritance Over Conditionals**: New models extend base classes rather than adding conditionals
- **Plugin Architecture**: Easy to add new models and likelihoods without modifying core code

### 3. Scientific Rigor
- **Numerical Precision**: High-precision numerical integration and computation
- **Validation**: Extensive parameter validation and bounds checking
- **Reproducibility**: Fixed random seeds and checkpoint capabilities

## Module Structure

### Core (`hicosmo/core/`)
Foundation classes and utilities for cosmological calculations.

- `cosmology.py`: Abstract base class defining cosmology interface
- `distances.py`: Distance calculations (luminosity, angular diameter, comoving)
- `power_spectrum.py`: Power spectrum computations
- `perturbations.py`: Perturbation theory calculations

### Models (`hicosmo/models/`)
Concrete implementations of cosmological models.

- `base.py`: Base model class with common functionality
- `lcdm.py`: Standard ΛCDM model
- `wcdm.py`: Constant w dark energy model
- `w0wacdm.py`: Time-varying dark energy (CPL parameterization)

### Likelihoods (`hicosmo/likelihoods/`)
Observational data likelihoods, Cobaya-compatible.

- `base.py`: Abstract likelihood class and Gaussian likelihood
- `sne/`: Type Ia supernovae datasets
- `bao/`: Baryon Acoustic Oscillation measurements
- `cmb/`: Cosmic Microwave Background data
- `h0/`: Local H0 measurements
- `lensing/`: Strong lensing time delays
- `gw/`: Gravitational wave standard sirens
- `frb/`: Fast Radio Burst dispersion measures
- `hi21cm/`: 21cm intensity mapping

### Samplers (`hicosmo/samplers/`)
MCMC and other sampling algorithms.

- `mcmc.py`: NumPyro MCMC wrapper with NUTS sampler
- `initialization.py`: Smart initialization strategies
- `diagnostics.py`: Convergence diagnostics (R-hat, ESS)

### Parameters (`hicosmo/parameters/`)
Parameter management and prior specification.

- `manager.py`: Central parameter manager
- `priors.py`: Prior distribution definitions
- `transforms.py`: Parameter transformations (log, logit)

### Fisher (`hicosmo/fisher/`)
Fisher matrix analysis for forecasting.

- `matrix.py`: Fisher matrix computation
- `forecasts.py`: Parameter constraint forecasts
- `derivatives.py`: Numerical differentiation

## Data Flow

```
1. User Configuration
   ├── Model Selection (LCDM, wCDM, etc.)
   ├── Likelihood Selection (SNe, BAO, etc.)
   └── Parameter Specification

2. Initialization
   ├── Load observational data
   ├── Setup covariance matrices
   └── Initialize parameter manager

3. Sampling
   ├── Build NumPyro model
   ├── Run NUTS sampler
   └── Real-time diagnostics

4. Analysis
   ├── Convergence assessment
   ├── Parameter constraints
   └── Visualization
```

## Key Design Patterns

### 1. Strategy Pattern
Different cosmological models implement the same interface, allowing runtime model selection.

```python
class Cosmology(ABC):
    @abstractmethod
    def E(z, params):
        pass

class LCDM(Cosmology):
    def E(z, params):
        # LCDM implementation

class wCDM(Cosmology):
    def E(z, params):
        # wCDM implementation
```

### 2. Template Method Pattern
Base likelihood class defines the algorithm structure, derived classes implement specific steps.

```python
class Likelihood:
    def logp(self, **params):
        theory = self.theory(**params)
        chi2 = self.chi2(theory, self.data, self.inv_cov)
        return -0.5 * chi2
    
    @abstractmethod
    def theory(self, **params):
        pass
```

### 3. Factory Pattern
Parameter manager creates appropriate prior distributions based on configuration.

```python
def get_numpyro_prior(self, param_name):
    prior_config = self.params[param_name]['prior']
    if prior_config['dist'] == 'uniform':
        return dist.Uniform(prior_config['min'], prior_config['max'])
    # ... other distributions
```

## Performance Optimizations

### 1. JIT Compilation
All numerical functions are JIT-compiled using JAX:
```python
@jit
def luminosity_distance(z, params, cosmo_class):
    # Compiled distance calculation
```

### 2. Vectorization
Batch operations using `vmap` for multiple redshifts:
```python
distances = vmap(lambda z: distance_single(z, params))(z_array)
```

### 3. Memory Management
- Lazy loading of data files
- Efficient array operations using JAX
- Checkpoint system for long runs

## Extension Points

### Adding a New Model
1. Create new file in `models/`
2. Inherit from `Cosmology` base class
3. Implement required methods (`E`, `w`)
4. Register in model factory

### Adding a New Likelihood
1. Create new file in appropriate `likelihoods/` subdirectory
2. Inherit from `Likelihood` or `GaussianLikelihood`
3. Implement `_load_data`, `_setup_covariance`, `theory`
4. Define requirements and nuisance parameters

### Adding a New Sampler
1. Create new file in `samplers/`
2. Implement sampling algorithm
3. Ensure compatibility with parameter manager
4. Add diagnostics and checkpointing

## Configuration System

### YAML Configuration
All components support YAML configuration for reproducibility:

```yaml
model:
  type: LCDM
  
parameters:
  H0:
    prior: {min: 60, max: 80}
  Omega_m:
    prior: {min: 0.2, max: 0.4}
    
likelihoods:
  - type: PantheonPlus
    data_path: /path/to/data
```

### Environment Variables
- `HICOSMO_DATA`: Default data directory
- `HICOSMO_CACHE`: Cache directory for compiled functions
- `JAX_PLATFORM_NAME`: Select CPU/GPU backend

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Verify mathematical correctness
- Check edge cases and bounds

### Integration Tests
- Test full pipeline from data to constraints
- Verify compatibility between components
- Check convergence on known solutions

### Performance Benchmarks
- Monitor JIT compilation times
- Track sampling efficiency
- Compare with reference implementations

## Future Enhancements

1. **Power Spectrum Integration**: Full Boltzmann solver or CAMB/CLASS interface
2. **Nested Sampling**: Alternative to MCMC for evidence calculation
3. **Machine Learning**: Neural network emulators for expensive calculations
4. **Distributed Computing**: Multi-node MCMC chains
5. **Interactive Dashboard**: Real-time monitoring and analysis