# Nuisance Parameters Rule

All likelihoods that register nuisance parameters **MUST** include LaTeX symbol information.

## Required Format

Use the `Parameter` class from `hicosmo.parameters`:

```python
from ..parameters import Parameter

def nuisance_parameters(self):
    """Return nuisance parameters with full metadata."""
    return [
        Parameter(
            name='param_name',           # Required: parameter identifier
            value=1.0,                   # Required: default value
            free=True,                   # Required: True for sampling
            prior={                      # Required: prior bounds
                'dist': 'uniform',
                'min': 0.5,
                'max': 1.5
            },
            latex_label=r'$\lambda$',    # **REQUIRED**: LaTeX for plotting
            description='Parameter description'  # Optional but recommended
        )
    ]
```

## Why LaTeX Labels are Required

1. **Plotting**: GetDist and corner plots need LaTeX labels
2. **Publication**: Scientific papers require proper notation
3. **Consistency**: All parameters should display correctly

## Bad Example (DO NOT DO THIS)

```python
# ❌ WRONG: Missing LaTeX label
def nuisance_parameters(self):
    return [('param_name', 1.0, 0.5, 1.5)]  # Tuple without LaTeX!
```

## Good Example

```python
# ✅ CORRECT: Full Parameter object with LaTeX
def nuisance_parameters(self):
    from ..parameters import Parameter
    return [
        Parameter(
            name='lambda_int_mean',
            value=1.0,
            free=True,
            prior={'dist': 'uniform', 'min': 0.5, 'max': 1.5},
            latex_label=r'$\langle\lambda_{\rm int}\rangle$',
            description='Internal MST population mean'
        )
    ]
```

## Standard LaTeX Labels

| Parameter | LaTeX |
|-----------|-------|
| H0 | `$H_0$` |
| Omega_m | `$\Omega_{\rm m}$` |
| Omega_b | `$\Omega_{\rm b}$` |
| M_B | `$\mathcal{M}_B$` |
| lambda_int_mean | `$\langle\lambda_{\rm int}\rangle$` |
| lambda_int_sigma | `$\sigma(\lambda_{\rm int})$` |
| a_ani_mean | `$\langle a_{\rm ani}\rangle$` |

## Affected Files

All files in `hicosmo/likelihoods/` that implement `nuisance_parameters()`:
- `pantheonplus.py` ✅ (uses Parameter with latex_label)
- `bao_datasets.py` - check
- `tdcosmo.py` - must use Parameter
- `planck_distance.py` - returns [] (no nuisance, OK)
- `h0licow.py` - returns [] (no nuisance, OK)
