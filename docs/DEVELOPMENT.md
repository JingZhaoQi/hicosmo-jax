# HIcosmo Development Guide

## Development Principles

### Core Rules

1. **Occam's Razor**: Do not add functionality unless necessary
2. **Clean Base Classes**: Never add conditionals to handle special cases in base classes
3. **Inheritance for Specialization**: Use inheritance and method overriding for special cases
4. **Single Responsibility**: Each class handles one thing, each method does one calculation
5. **Understand Data First**: Fully understand data formats before implementation
6. **Follow Established Patterns**: Reference mature implementations (e.g., Cobaya)

## Getting Started

### Environment Setup

```bash
# Clone repository
git clone https://github.com/hicosmo/hicosmo.git
cd hicosmo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `[dev]` extras include:
- `pytest`: Testing framework
- `black`: Code formatting
- `isort`: Import sorting
- `mypy`: Type checking
- `flake8`: Linting
- `pre-commit`: Git hooks

## Code Standards

### Python Style

We follow PEP 8 with these specifications:
- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings in NumPy style

Example:
```python
def luminosity_distance(
    z: Union[float, jnp.ndarray], 
    params: Dict[str, float]
) -> Union[float, jnp.ndarray]:
    """
    Calculate luminosity distance.
    
    Parameters
    ----------
    z : float or array_like
        Redshift(s)
    params : dict
        Cosmological parameters
        
    Returns
    -------
    float or array_like
        Luminosity distance in Mpc
    """
    pass
```

### JAX-Specific Guidelines

1. **Pure Functions Only**
```python
# Good - pure function
@jit
def compute_distance(z, params):
    return integrate(lambda x: 1/E(x, params), 0, z)

# Bad - has side effects
def compute_distance(z, params):
    self.last_z = z  # Side effect!
    return integrate(lambda x: 1/E(x, params), 0, z)
```

2. **Use Static Arguments for Non-Array Inputs**
```python
@partial(jit, static_argnums=(2,))
def distance(z, params, cosmo_class):
    # cosmo_class is static (not a JAX array)
    return cosmo_class.compute(z, params)
```

3. **Vectorize with vmap**
```python
# Single redshift function
def distance_single(z, params):
    return integrate(integrand, 0, z)

# Vectorized version
distance_vectorized = vmap(distance_single, in_axes=(0, None))
```

## Adding New Components

### Creating a New Cosmological Model

1. Create file in `hicosmo/models/`:
```python
# hicosmo/models/mymodel.py
from ..core.cosmology import Cosmology, DistanceCalculator

class MyModel(Cosmology, DistanceCalculator):
    @staticmethod
    @jit
    def E(z, params):
        # Implement E(z) for your model
        pass
    
    @staticmethod
    @jit
    def w(z, params):
        # Implement w(z) if non-standard
        pass
```

2. Add tests in `tests/test_models.py`:
```python
def test_mymodel_E():
    params = {'H0': 70, 'Omega_m': 0.3, ...}
    z = jnp.array([0.0, 0.5, 1.0])
    E_values = MyModel.E(z, params)
    # Assert expected values
```

### Creating a New Likelihood

1. Create file in appropriate subdirectory:
```python
# hicosmo/likelihoods/sne/mydataset.py
from ..base import GaussianLikelihood

class MyDataset(GaussianLikelihood):
    def _load_data(self):
        # Load your data
        self.z_data = ...
        self.mag_data = ...
        self.cov = ...
    
    def theory(self, luminosity_distance=None, **kwargs):
        # Compute theory vector
        return 5 * jnp.log10(luminosity_distance) + 25
    
    def get_requirements(self):
        return {'luminosity_distance': {'z': self.z_data}}
```

2. Create data loader if complex:
```python
# hicosmo/data/loaders/mydataset.py
def load_mydataset(path):
    # Complex data loading logic
    return data_dict
```

### Creating a New Prior Distribution

1. Add to parameter manager:
```python
# In hicosmo/parameters/manager.py
def get_numpyro_prior(self, param_name):
    # ... existing code ...
    elif dist_type == 'my_custom_dist':
        return MyCustomDistribution(prior['param1'], prior['param2'])
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=hicosmo --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run GPU tests (requires GPU)
pytest -m gpu
```

### Writing Tests

1. **Unit Tests**: Test individual functions
```python
def test_lcdm_E():
    """Test LCDM E(z) calculation."""
    params = {'Omega_m': 0.3}
    z = 0.5
    expected = np.sqrt(0.3 * 1.5**3 + 0.7)
    assert jnp.allclose(LCDM.E(z, params), expected)
```

2. **Integration Tests**: Test component interaction
```python
@pytest.mark.integration
def test_mcmc_lcdm():
    """Test MCMC sampling with LCDM model."""
    model = LCDM()
    likelihood = MockLikelihood()
    sampler = MCMCSampler(model, likelihood)
    samples = sampler.run(num_samples=100)
    assert 'H0' in samples
```

3. **Gradient Tests**: Verify automatic differentiation
```python
def test_likelihood_gradient():
    """Test likelihood gradient computation."""
    likelihood = MyLikelihood()
    params = {'H0': 70, 'Omega_m': 0.3}
    
    # Numerical gradient
    eps = 1e-6
    numerical_grad = (logp(H0+eps) - logp(H0-eps)) / (2*eps)
    
    # Automatic gradient
    _, grad = likelihood.logp_grad(**params)
    
    assert jnp.allclose(grad['H0'], numerical_grad, rtol=1e-5)
```

### Performance Benchmarks

```python
# tests/benchmarks/test_performance.py
import pytest

@pytest.mark.benchmark
def test_distance_performance(benchmark):
    """Benchmark distance calculations."""
    z = jnp.linspace(0, 2, 1000)
    params = {'H0': 70, 'Omega_m': 0.3}
    
    result = benchmark(LCDM.luminosity_distance, z, params, LCDM)
    assert len(result) == 1000
```

## Debugging

### JAX Debugging Tips

1. **Disable JIT for debugging**:
```python
from jax import config
config.update("jax_disable_jit", True)
```

2. **Check for NaN values**:
```python
from jax import config
config.update("jax_debug_nans", True)
```

3. **Print inside JIT functions**:
```python
from jax.debug import print as jprint

@jit
def my_function(x):
    jprint("x value: {}", x)
    return x ** 2
```

### Common Issues

1. **Shape Mismatches**: Use `jax.debug.print` to inspect shapes
2. **Type Issues**: Ensure all inputs are JAX arrays
3. **Static Argument Errors**: Mark non-array arguments as static
4. **Memory Issues**: Use smaller batch sizes or checkpointing

## Documentation

### Docstring Format

Use NumPy style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed, explaining the method,
    assumptions, and any important details.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2, by default value
        
    Returns
    -------
    type
        Description of return value
        
    Raises
    ------
    ValueError
        When invalid input provided
        
    Examples
    --------
    >>> function_name(1, 2)
    3
    
    Notes
    -----
    Additional notes about the implementation,
    references, or mathematical formulation.
    
    References
    ----------
    .. [1] Author et al., "Paper Title", Journal (Year)
    """
    pass
```

### Adding Examples

Create example scripts in `examples/`:

```python
# examples/basic_analysis.py
"""
Basic cosmological analysis example.

This script demonstrates:
1. Setting up a cosmological model
2. Loading observational data
3. Running MCMC sampling
4. Analyzing results
"""

from hicosmo.models import LCDM
from hicosmo.likelihoods.sne import PantheonPlus
from hicosmo.samplers import MCMCSampler

# ... example code ...
```

## Version Control

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements

Examples:
```
feat(models): add early dark energy model

fix(mcmc): correct convergence diagnostic calculation

docs(likelihood): update BAO likelihood documentation
```

### Branch Strategy

- `main`: Stable releases
- `develop`: Development branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Build and publish to PyPI

## Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Documentation: https://hicosmo.readthedocs.io
- Email: hicosmo@example.com