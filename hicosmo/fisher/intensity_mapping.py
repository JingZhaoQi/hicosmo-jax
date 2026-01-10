"""General 21cm intensity mapping forecasting utilities.

The goal of this module is to expose a configuration-driven interface that
computes Fisher forecasts for 21 cm intensity mapping surveys, following the
approach of Bull (2016) but without hard-coding survey-specific constants.

Users can supply their own survey YAML files (see ``configs/surveys`` for
examples), pick a cosmological model implemented in :mod:`hicosmo.models`,
choose a set of cosmological parameters to constrain, and obtain Fisher
matrices, covariance estimates, and helper plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numbers

import jax
import jax.numpy as jnp
import yaml

from ..utils.constants import c_km_s
from ..utils.logging import get_logger
from ..utils.jax_tools import (
    integrate_simpson,
    integrate_logspace,
    trapezoid,
    gradient_1d,
)

# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

SURVEY_CONFIG_DIR = Path(__file__).resolve().parent / 'surveys'
logger = get_logger(__name__)


def _safe_inverse(
    matrix: jnp.ndarray,
    rcond: float = 1e-12,
    regularize: float = 0.0,
    name: str = "matrix",
    warn: bool = True
) -> jnp.ndarray:
    """Safely invert a matrix with optional regularization and pseudo-inverse fallback."""
    mat = jnp.asarray(matrix)
    if regularize > 0.0:
        mat = mat + regularize * jnp.eye(mat.shape[0], dtype=mat.dtype)
    if warn:
        cond = jnp.linalg.cond(mat)
        if cond > (1.0 / rcond):
            logger.warning("Ill-conditioned %s (cond=%s); using pseudo-inverse (rcond=%s).", name, cond, rcond)
    return jnp.linalg.pinv(mat, rcond=rcond)


@jax.jit
def _integrate_fisher_matrix(
    weights: jnp.ndarray,
    derivatives: jnp.ndarray,
    k_grid: jnp.ndarray,
    mu_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized Fisher integral for derivatives defined on (k, mu) grid."""
    integrand = weights[None, ...] * derivatives[:, None, ...] * derivatives[None, ...]
    inner = trapezoid(integrand, mu_grid, axis=-1)
    return trapezoid(inner, k_grid, axis=-1)


# -----------------------------------------------------------------------------
# Helper factories for redshift functions
# -----------------------------------------------------------------------------


def _build_redshift_function(cfg: Optional[Mapping[str, float]], default: float = 0.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return ``f(z)`` defined by *cfg*."""
    if cfg is None:
        return lambda z: jnp.full_like(jnp.asarray(z, dtype=float), default, dtype=float)

    kind = cfg.get('kind', 'constant').lower()
    if kind == 'constant':
        value = float(cfg.get('value', default))
        return lambda z: jnp.full_like(jnp.asarray(z, dtype=float), value, dtype=float)
    if kind == 'polynomial':
        coeffs = [float(c) for c in cfg.get('coefficients', [])]
        if not coeffs:
            raise ValueError('Polynomial model requires "coefficients" entries')

        def poly(z: jnp.ndarray) -> jnp.ndarray:
            z = jnp.asarray(z, dtype=float)
            acc = jnp.zeros_like(z)
            for power, coeff in enumerate(coeffs):
                acc += coeff * z**power
            return acc

        return poly
    if kind == 'table':
        table = cfg.get('values')
        if not table:
            raise ValueError('Tabulated model requires "values" list with {z, value}')
        z_nodes = jnp.array([float(entry['z']) for entry in table])
        v_nodes = jnp.array([float(entry['value']) for entry in table])

        def interp(z: jnp.ndarray) -> jnp.ndarray:
            z = jnp.asarray(z, dtype=float)
            return jnp.interp(z, z_nodes, v_nodes)

        return interp
    raise ValueError(f'Unknown redshift model kind: {kind}')


@dataclass
class SkyTemperatureModel:
    kind: str = 'none'
    value: float = 0.0
    T_ref: float = 60.0
    nu_ref_MHz: float = 300.0
    beta: float = 2.55

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, float]]) -> 'SkyTemperatureModel':
        if data is None:
            return cls(kind='none')
        return cls(
            kind=data.get('kind', 'power_law'),
            value=float(data.get('value', 0.0)),
            T_ref=float(data.get('T_ref_K', 60.0)),
            nu_ref_MHz=float(data.get('nu_ref_MHz', 300.0)),
            beta=float(data.get('beta', 2.55)),
        )

    def __call__(self, z: jnp.ndarray | float) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=float)
        if self.kind == 'none':
            return jnp.zeros_like(z)
        if self.kind == 'constant':
            return jnp.full_like(z, self.value)
        if self.kind == 'power_law':
            nu_MHz = (1420.405751 / (1.0 + z))
            return self.T_ref * (self.nu_ref_MHz / nu_MHz) ** self.beta
        raise ValueError(f'Unknown sky temperature model kind: {self.kind}')


# -----------------------------------------------------------------------------
# Survey configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class InstrumentConfig:
    ndish: int
    dish_diameter_m: float
    nbeam: int
    survey_area_deg2: float
    total_time_hours: float
    channel_width_hz: float
    system_temperature_K: float
    sky_temperature: SkyTemperatureModel

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> 'InstrumentConfig':
        required = ['ndish', 'dish_diameter_m', 'nbeam', 'survey_area_deg2', 'total_time_hours', 'channel_width_hz']
        for key in required:
            if key not in data:
                raise ValueError(f'Missing instrument field "{key}"')
        return cls(
            ndish=int(data['ndish']),
            dish_diameter_m=float(data['dish_diameter_m']),
            nbeam=int(data['nbeam']),
            survey_area_deg2=float(data['survey_area_deg2']),
            total_time_hours=float(data['total_time_hours']),
            channel_width_hz=float(data['channel_width_hz']),
            system_temperature_K=float(data.get('system_temperature_K', 0.0)),
            sky_temperature=SkyTemperatureModel.from_dict(data.get('sky_temperature')),
        )

    @property
    def solid_angle(self) -> float:
        return self.survey_area_deg2 * (jnp.pi / 180.0) ** 2

    @property
    def total_time_seconds(self) -> float:
        return self.total_time_hours * 3600.0


@dataclass
class RedshiftBin:
    z: float
    delta_z: float

    @classmethod
    def from_dict(cls, data: Mapping[str, float], default_delta_z: float) -> 'RedshiftBin':
        if 'z' not in data:
            raise ValueError('Each redshift bin requires a "z" entry')
        return cls(z=float(data['z']), delta_z=float(data.get('delta_z', default_delta_z)))


@dataclass
class IntensityMappingSurvey:
    name: str
    instrument: InstrumentConfig
    redshift_bins: List[RedshiftBin]
    bias_fn: Callable[[jnp.ndarray], jnp.ndarray]
    omega_hi_fn: Callable[[jnp.ndarray], jnp.ndarray]
    description: str = ''
    # Deprecated fields (for backward compatibility with Schema v1.0)
    model: Optional[str] = None
    reference: Optional[Dict[str, float]] = None
    priors: Optional[Dict[str, float]] = None

    @classmethod
    def from_file(cls, filename: Path | str) -> 'IntensityMappingSurvey':
        with open(filename, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        if 'name' not in data:
            raise ValueError('Survey configuration must contain "name" field')

        # Check schema version to determine how to parse
        metadata = data.get('metadata', {})
        schema_version = metadata.get('schema_version', '1.0')

        if schema_version == '2.0':
            # Schema v2.0: Nested structure (instrument, observing, noise separate)
            # Merge fields from different sections for InstrumentConfig
            instrument_data = {}

            # From instrument section
            if 'instrument' in data:
                inst = data['instrument']
                telescope_type = inst.get('telescope_type', 'single_dish')

                # Map different telescope types to unified parameters
                if telescope_type in ('single_dish', 'interferometer_autocorr'):
                    # Direct mapping (both use same parameter structure)
                    instrument_data.update({
                        'ndish': inst.get('ndish'),
                        'dish_diameter_m': inst.get('dish_diameter_m'),
                        'nbeam': inst.get('nbeam'),
                    })
                elif telescope_type == 'cylinder_array':
                    # Map cylinder parameters to equivalent single-dish parameters
                    # ncylinders → ndish
                    # cylinder_length_m → dish_diameter_m (effective aperture)
                    # nfeeds → total feeds (distribute to nbeam)
                    ncylinders = inst.get('ncylinders', 1)
                    nfeeds_total = inst.get('nfeeds', 1)
                    instrument_data.update({
                        'ndish': ncylinders,
                        'dish_diameter_m': inst.get('cylinder_length_m', inst.get('cylinder_width_m', 1.0)),
                        'nbeam': nfeeds_total // ncylinders if ncylinders > 0 else 1,
                    })
                else:
                    raise ValueError(f"Unknown telescope_type: {telescope_type}")

            # From observing section
            if 'observing' in data:
                observing = data['observing']
                instrument_data['survey_area_deg2'] = observing.get('survey_area_deg2')
                instrument_data['total_time_hours'] = observing.get('total_time_hours')

                # Handle channel_width - can be in MHz or Hz
                if 'channel_width_hz' in observing:
                    instrument_data['channel_width_hz'] = observing['channel_width_hz']
                elif 'channel_width_MHz' in observing:
                    instrument_data['channel_width_hz'] = observing['channel_width_MHz'] * 1e6

            # From noise section
            if 'noise' in data:
                noise = data['noise']
                instrument_data['system_temperature_K'] = noise.get('system_temperature_K', 0.0)
                instrument_data['sky_temperature'] = noise.get('sky_temperature')

            instrument_cfg = InstrumentConfig.from_dict(instrument_data)

            # Parse redshift bins (Schema v2.0 format)
            redshift_bins_data = data.get('redshift_bins', {})
            default_delta_z = float(redshift_bins_data.get('default_delta_z', 0.1))
            centers = redshift_bins_data.get('centers', [])
            bins = [RedshiftBin(z=float(z), delta_z=default_delta_z) for z in centers]

            # Parse HI tracers (Schema v2.0 format)
            hi_tracers = data.get('hi_tracers', {})
            bias_fn = _build_redshift_function(hi_tracers.get('bias'), default=1.0)
            omega_hi_fn = _build_redshift_function(hi_tracers.get('density'), default=4.8e-4)

            # Schema v2.0 should NOT have model/reference/priors
            reference = {}
            priors = {}
            model = 'lcdm'  # Default, should be overridden by explicit cosmology parameter

        else:
            # Schema v1.0: Flat structure (backward compatibility)
            instrument_cfg = InstrumentConfig.from_dict(data['instrument'])
            default_delta_z = float(data.get('default_delta_z', 0.1))
            bins = [RedshiftBin.from_dict(entry, default_delta_z) for entry in data['redshift_bins']]
            bias_fn = _build_redshift_function(data.get('hi_bias'), default=1.0)
            omega_hi_fn = _build_redshift_function(data.get('hi_density'), default=4.8e-4)
            reference = {k: float(v) for k, v in data.get('reference', {}).items()}
            priors = {k: float(v) for k, v in data.get('priors', {}).items()}
            model = data.get('model', 'lcdm')

        return cls(
            name=data['name'],
            model=model,
            reference=reference,
            instrument=instrument_cfg,
            redshift_bins=bins,
            bias_fn=bias_fn,
            omega_hi_fn=omega_hi_fn,
            priors=priors,
            description=data.get('description', ''),
        )


# -----------------------------------------------------------------------------
# Cosmology / growth wrappers
# -----------------------------------------------------------------------------


class GrowthScenario:
    """Configuration manager for Bull et al. (2016) Table 4 growth scenarios.

    Bull 2016 tests four phenomenological growth parameterizations:
    1. gamma0_free: γ₀ free, γ₁=0, η₀=0, η₁=0 (baseline MG test)
    2. gamma1_free: γ₁ free, γ₀=0.55, η₀=0, η₁=0 (evolution test)
    3. eta0_free: η₀ free, γ₀=0.55, γ₁=0, η₁=0 (MG offset test)
    4. eta1_free: η₁ free, γ₀=0.55, γ₁=0, η₀=0 (MG evolution test)

    The inactive parameters are frozen at their GR fiducial values.

    Parameters
    ----------
    scenario : str
        One of: 'gamma0_free', 'gamma1_free', 'eta0_free', 'eta1_free', 'gr'

    Attributes
    ----------
    free_params : list of str
        Names of free growth parameters for this scenario
    fixed_params : dict
        Fixed growth parameter values
    fiducial_values : dict
        Fiducial values for all growth parameters
    """

    # GR fiducial values
    GR_FIDUCIAL = {
        'gamma0': 0.55,
        'gamma1': 0.0,
        'eta0': 0.0,
        'eta1': 0.0,
    }

    # Scenario definitions (Bull 2016 Table 4)
    SCENARIOS = {
        'gr': {
            'free_params': [],
            'fixed_params': {'gamma0': 0.55, 'gamma1': 0.0, 'eta0': 0.0, 'eta1': 0.0},
            'description': 'General Relativity (no free growth parameters)',
        },
        'gamma0_free': {
            'free_params': ['gamma0'],
            'fixed_params': {'gamma1': 0.0, 'eta0': 0.0, 'eta1': 0.0},
            'description': 'γ₀ free, others fixed at GR values',
        },
        'gamma1_free': {
            'free_params': ['gamma1'],
            'fixed_params': {'gamma0': 0.55, 'eta0': 0.0, 'eta1': 0.0},
            'description': 'γ₁ free, others fixed at GR values',
        },
        'eta0_free': {
            'free_params': ['eta0'],
            'fixed_params': {'gamma0': 0.55, 'gamma1': 0.0, 'eta1': 0.0},
            'description': 'η₀ free, others fixed at GR values',
        },
        'eta1_free': {
            'free_params': ['eta1'],
            'fixed_params': {'gamma0': 0.55, 'gamma1': 0.0, 'eta0': 0.0},
            'description': 'η₁ free, others fixed at GR values',
        },
        'all_free': {
            'free_params': ['gamma0', 'gamma1', 'eta0', 'eta1'],
            'fixed_params': {},
            'description': 'All growth parameters free (most general)',
        },
    }

    def __init__(self, scenario: str = 'gamma0_free'):
        """Initialize growth scenario.

        Parameters
        ----------
        scenario : str, optional
            Scenario name (default: 'gamma0_free')
        """
        if scenario not in self.SCENARIOS:
            available = ', '.join(sorted(self.SCENARIOS.keys()))
            raise ValueError(
                f'Unknown growth scenario "{scenario}". Available: {available}'
            )

        self.scenario = scenario
        config = self.SCENARIOS[scenario]

        self.free_params = config['free_params']
        self.fixed_params = config['fixed_params']
        self.description = config['description']

        # Build fiducial values (combine fixed + GR defaults for free params)
        self.fiducial_values = self.GR_FIDUCIAL.copy()
        self.fiducial_values.update(self.fixed_params)

    def get_growth_params(self, **overrides) -> dict:
        """Get growth parameters for GrowthModel initialization.

        Parameters
        ----------
        **overrides
            Override values for free parameters

        Returns
        -------
        dict
            Complete parameter dictionary for GrowthModel(cosmology, **params)

        Examples
        --------
        >>> scenario = GrowthScenario('gamma0_free')
        >>> params = scenario.get_growth_params(gamma0=0.60)
        >>> growth = GrowthModel(cosmology, **params)
        """
        params = self.fiducial_values.copy()

        # Apply overrides only to free parameters
        for param_name, value in overrides.items():
            if param_name in self.free_params or param_name in self.fixed_params:
                params[param_name] = value
            else:
                raise ValueError(
                    f'Parameter "{param_name}" is not in the growth model. '
                    f'Valid: {list(self.fiducial_values.keys())}'
                )

        return params

    def is_free(self, param_name: str) -> bool:
        """Check if a parameter is free in this scenario."""
        return param_name in self.free_params

    def __repr__(self) -> str:
        """String representation."""
        free_str = ', '.join(self.free_params) if self.free_params else 'none'
        return (
            f'GrowthScenario(scenario="{self.scenario}", '
            f'free=[{free_str}], description="{self.description}")'
        )


class GrowthModel:
    """Growth model using Bull et al. (2016) growth parameterization.

    Implements the phenomenological growth ansatz:
        f(a) = Ω_m(a)^γ(a) · [1 + η(a)]

    where:
        γ(a) = γ₀ + γ₁(1-a)
        η(a) = η₀ + η₁(1-a)

    Parameters
    ----------
    cosmology : CosmologyBase
        Cosmological model instance
    gamma0 : float, optional
        Growth index baseline (default: 0.55, GR value)
    gamma1 : float, optional
        Growth index evolution (default: 0.0, GR value)
    eta0 : float, optional
        Modified gravity parameter baseline (default: 0.0, GR value)
    eta1 : float, optional
        Modified gravity parameter evolution (default: 0.0, GR value)

    References
    ----------
    Bull et al. (2016), ApJ 817:26, Section 2.2
    """

    def __init__(
        self,
        cosmology,
        gamma0: float = 0.55,
        gamma1: float = 0.0,
        eta0: float = 0.0,
        eta1: float = 0.0
    ) -> None:
        self.cosmology = cosmology
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.eta0 = eta0
        self.eta1 = eta1

        # Backward compatibility: if old 'gamma' parameter is used
        # it will be interpreted as gamma0
        self.gamma = gamma0  # For backward compatibility

    def gamma_of_a(self, a: jnp.ndarray | float) -> jnp.ndarray | float:
        """Growth index as function of scale factor.

        γ(a) = γ₀ + γ₁(1-a)
        """
        a = jnp.asarray(a, dtype=float)
        return self.gamma0 + self.gamma1 * (1.0 - a)

    def eta_of_a(self, a: jnp.ndarray | float) -> jnp.ndarray | float:
        """Modified gravity parameter as function of scale factor.

        η(a) = η₀ + η₁(1-a)
        """
        a = jnp.asarray(a, dtype=float)
        return self.eta0 + self.eta1 * (1.0 - a)

    def omega_m(self, z: jnp.ndarray | float) -> jnp.ndarray:
        """Matter density parameter Ω_m(z)."""
        z = jnp.asarray(z, dtype=float)
        numerator = jnp.asarray(self.cosmology.params['Omega_m'], dtype=float) * (1.0 + z) ** 3
        Ez = jnp.asarray(self.cosmology.E_z(z))
        return numerator / Ez**2

    def growth_rate(self, z: jnp.ndarray | float) -> jnp.ndarray:
        """Growth rate f(z) using Bull et al. (2016) parameterization.

        f(a) = Ω_m(a)^γ(a) · [1 + η(a)]

        In GR limit (γ₀=0.55, γ₁=0, η₀=0, η₁=0), reduces to:
        f = Ω_m(z)^0.55
        """
        z = jnp.asarray(z, dtype=float)
        a = 1.0 / (1.0 + z)

        omega_m_z = self.omega_m(z)
        gamma_a = self.gamma_of_a(a)
        eta_a = self.eta_of_a(a)

        return omega_m_z ** gamma_a * (1.0 + eta_a)

    def growth_factor(self, z: jnp.ndarray | float) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=float)

        def integrand(z_vals: jnp.ndarray) -> jnp.ndarray:
            return self.growth_rate(z_vals) / (1.0 + z_vals)

        integral = integrate_simpson(integrand, 0.0, z, num=512)
        result = jnp.exp(-integral)
        return jnp.where(z <= 0.0, jnp.asarray(1.0), result)

    def sigma8(self, z: float) -> jnp.ndarray:
        """Compute σ₈(z) = σ₈,0 × D(z).

        Uses Bull et al. (2016) prescription where σ₈,0 is the normalization
        parameter (z=0 value) that can be constrained by CMB priors.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            σ₈ at redshift z

        Notes
        -----
        The derivative is: ∂ln(σ₈)/∂σ₈,0 = 1/σ₈,0
        Or equivalently: ∂σ₈/∂σ₈,0 = D(z)
        """
        # Try sigma8_0 first (Bull 2016 convention), fall back to sigma8
        sigma8_0 = jnp.asarray(self.cosmology.params.get('sigma8_0',
                         self.cosmology.params.get('sigma8', 0.834)), dtype=float)
        return sigma8_0 * self.growth_factor(z)

    def f_sigma8(self, z: float) -> jnp.ndarray:
        """Growth rate times σ₈: f(z) × σ₈(z).

        This is the observable measured from RSD (redshift-space distortions).

        Notes
        -----
        Derivatives for Fisher matrix:
        - ∂ln(fσ₈)/∂σ₈,0 = ∂ln(σ₈)/∂σ₈,0 = 1/σ₈,0 ≈ 1 (if σ₈,0 ≈ 1)
        - ∂ln(fσ₈)/∂γ₀ includes both f and D contributions
        """
        return self.growth_rate(z) * self.sigma8(z)


class LinearPowerSpectrum:
    def __init__(self, cosmology) -> None:
        self.cosmology = cosmology
        self.h = jnp.asarray(cosmology.params['H0'], dtype=float) / 100.0
        self.n_s = jnp.asarray(cosmology.params.get('n_s', 0.962), dtype=float)
        self.sigma8 = jnp.asarray(cosmology.params.get('sigma8', 0.834), dtype=float)
        self.omega_m = jnp.asarray(cosmology.params['Omega_m'], dtype=float)
        self.omega_b = jnp.asarray(cosmology.params.get('Omega_b', 0.049), dtype=float)
        self.theta_cmb = jnp.asarray(cosmology.params.get('T_cmb', 2.7255), dtype=float) / 2.7
        self._normalisation = self._normalise()

    def _transfer(self, k: jnp.ndarray) -> jnp.ndarray:
        k = jnp.asarray(k)
        h = self.h
        omega_m_h2 = self.omega_m * h**2
        omega_b_h2 = self.omega_b * h**2
        theta_cmb = self.theta_cmb
        q = k / (13.41 * h) * theta_cmb**2
        beta_c = 1.0 / (1.0 + 0.944 / (1.0 + (458.0 * omega_m_h2) ** 0.708))
        L0 = jnp.log(jnp.e + 1.8 * beta_c * q)
        C0 = 14.4 + 325.0 / (1.0 + 60.5 * q ** 1.08)
        return L0 / (L0 + C0 * q**2)

    def _unnormalised(self, k: jnp.ndarray) -> jnp.ndarray:
        return (k ** self.n_s) * self._transfer(k) ** 2

    def _normalise(self) -> jnp.ndarray:
        R8 = 8.0  # h^-1 Mpc
        h = self.h
        n_s = self.n_s
        sigma8 = self.sigma8
        omega_m = self.omega_m
        omega_b = self.omega_b
        theta_cmb = self.theta_cmb

        def transfer(k: jnp.ndarray) -> jnp.ndarray:
            k = jnp.asarray(k)
            omega_m_h2 = omega_m * h**2
            omega_b_h2 = omega_b * h**2
            q = k / (13.41 * h) * theta_cmb**2
            beta_c = 1.0 / (1.0 + 0.944 / (1.0 + (458.0 * omega_m_h2) ** 0.708))
            L0 = jnp.log(jnp.e + 1.8 * beta_c * q)
            C0 = 14.4 + 325.0 / (1.0 + 60.5 * q ** 1.08)
            return L0 / (L0 + C0 * q**2)

        def unnormalised(k: jnp.ndarray) -> jnp.ndarray:
            return (k ** n_s) * transfer(k) ** 2

        def window(x: jnp.ndarray) -> jnp.ndarray:
            x = jnp.asarray(x)
            sinx = jnp.sin(x)
            cosx = jnp.cos(x)
            w = jnp.where(x < 1e-6, 1.0 - x**2 / 10.0, 3.0 * (sinx - x * cosx) / (x**3))
            return w

        def integrand(k_vals: jnp.ndarray) -> jnp.ndarray:
            p = unnormalised(k_vals / h)
            w = window(k_vals * R8)
            return k_vals**2 * p * w**2

        integral = integrate_logspace(integrand, 1e-4, 10.0, num=2048)
        return jnp.asarray(sigma8**2 * 2.0 * jnp.pi**2 / integral)

    def power(self, k: jnp.ndarray, growth_factor: float) -> jnp.ndarray:
        k = jnp.asarray(k)
        base = self._normalisation * self._unnormalised(k)
        return base * growth_factor**2


# -----------------------------------------------------------------------------
# Fisher calculator
# -----------------------------------------------------------------------------


def _beam_fwhm_m(z: float, instrument: InstrumentConfig) -> jnp.ndarray:
    wavelength = (299792458.0 / (1420.405751e6 / (1.0 + z)))
    return 1.22 * wavelength / instrument.dish_diameter_m


def _survey_volume_params(
    z_c: float,
    delta_z: float,
    instrument: InstrumentConfig,
    cosmology
) -> jnp.ndarray:
    z_min = jnp.maximum(0.0, z_c - delta_z / 2.0)
    z_max = z_c + delta_z / 2.0

    def integrand(z_vals: jnp.ndarray) -> jnp.ndarray:
        H = jnp.asarray(cosmology.H_z(z_vals))
        chi = jnp.asarray(cosmology.comoving_distance(z_vals))
        return (c_km_s / H) * chi**2

    shell = integrate_simpson(integrand, z_min, z_max, num=256)
    return instrument.solid_angle * shell


def _survey_volume(bin_cfg: RedshiftBin, instrument: InstrumentConfig, cosmology) -> jnp.ndarray:
    return _survey_volume_params(bin_cfg.z, bin_cfg.delta_z, instrument, cosmology)


def _pixel_volume(z: float, theta_b: float, instrument: InstrumentConfig, cosmology) -> jnp.ndarray:
    dz_pix = (1.0 + z) ** 2 * instrument.channel_width_hz / 1420.405751e6
    dchi_dz = c_km_s / jnp.asarray(cosmology.H_z(z))
    chi = jnp.asarray(cosmology.comoving_distance(z))
    omega_pix = 1.13 * theta_b**2
    return omega_pix * dchi_dz * chi**2 * dz_pix


def _noise_temperature(z: float, theta_b: float, instrument: InstrumentConfig) -> jnp.ndarray:
    t_sky = instrument.sky_temperature(z)
    t_sys = instrument.system_temperature_K + t_sky
    omega_pix = 1.13 * theta_b**2
    t_pix = instrument.total_time_seconds * (omega_pix / instrument.solid_angle) * instrument.ndish * instrument.nbeam
    return t_sys / jnp.sqrt(instrument.channel_width_hz * t_pix)


@dataclass
class FisherResult:
    """Low-level Fisher result for observable constraints (internal use)."""
    z: jnp.ndarray
    covariance_blocks: jnp.ndarray
    sigma_ln_fsigma8: jnp.ndarray
    sigma_ln_DA: jnp.ndarray
    sigma_ln_H: jnp.ndarray


@dataclass
class FisherForecastResult:
    """User-friendly Fisher forecast result for parameter constraints.

    Attributes
    ----------
    params : list of str
        Parameter names
    errors : dict of str to float
        1-sigma constraints: σ(param_name)
    fisher_matrix : jnp.ndarray
        Fisher information matrix
    covariance : jnp.ndarray
        Parameter covariance matrix
    correlation : jnp.ndarray
        Parameter correlation matrix
    survey_name : str
        Survey configuration name
    cosmology_summary : dict
        Fiducial cosmology parameters
    n_bins : int
        Number of redshift bins
    latex_labels : dict of str to str
        LaTeX labels for parameters (for plotting)
    """

    params: List[str]
    errors: Dict[str, float]
    fisher_matrix: jnp.ndarray
    covariance: jnp.ndarray
    correlation: jnp.ndarray
    survey_name: str
    cosmology_summary: Dict[str, float]
    n_bins: int
    latex_labels: Dict[str, str] = None

    def __str__(self) -> str:
        """Pretty-print forecast results."""
        lines = [
            "=" * 70,
            f"Fisher Forecast Results: {self.survey_name}",
            "=" * 70,
            f"Cosmology: {self.cosmology_summary.get('model', 'N/A')}",
            f"Redshift bins: {self.n_bins}",
            f"Parameters: {', '.join(self.params)}",
            "",
            "1σ Constraints:",
            "-" * 70,
        ]
        for param in self.params:
            lines.append(f"  σ({param:10s}) = {self.errors[param]:15.6e}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary (compatible with old API)."""
        return {
            'params': self.params,
            'errors': jnp.array([self.errors[p] for p in self.params]),
            'fisher': self.fisher_matrix,
            'covariance': self.covariance,
            'correlation': self.correlation,
        }

    def get_latex_labels(self) -> List[str]:
        """Get LaTeX labels in the order of params (for Plotter.from_fisher)."""
        if self.latex_labels is None:
            return self.params
        return [self.latex_labels.get(p, p) for p in self.params]

    def get_fiducial_values(self) -> List[float]:
        """Get fiducial parameter values in the order of params."""
        return [self.cosmology_summary.get(p, 0.0) for p in self.params]


class IntensityMappingFisher:
    """21cm Intensity Mapping Fisher forecast calculator.

    This class implements the correct separation of concerns:
    - Survey configuration contains ONLY hardware and observing strategy
    - Cosmological model is passed as a separate parameter
    - Parameters to constrain are specified by the user

    This allows testing multiple cosmological models with the same survey.

    Three Usage Patterns
    ---------------------

    1. **Quick API** (Recommended - 3 lines of code):
       >>> from hicosmo.models import ILCDM
       >>> from hicosmo.fisher import IntensityMappingFisher, load_survey
       >>>
       >>> ilcdm = ILCDM(beta=0.001)
       >>> result = IntensityMappingFisher.forecast('tianlai', ilcdm, ['beta', 'H0'])
       >>> print(result)  # User-friendly formatted output

    2. **Instance API** (When computing multiple forecasts):
       >>> survey = load_survey('tianlai')
       >>> fisher = IntensityMappingFisher(survey, ilcdm)
       >>> result1 = fisher.parameter_forecast(['beta', 'H0'])
       >>> result2 = fisher.parameter_forecast(['w0', 'wa'])

    3. **Low-Level API** (Advanced users):
       >>> fisher = IntensityMappingFisher(survey, cosmology)
       >>> obs_result = fisher.compute_observable_constraints()  # Observable Fisher
       >>> param_result = fisher.parameter_forecast(['beta', 'H0'])  # Parameter Fisher

    See Also
    --------
    FisherForecastResult : User-friendly result container
    load_survey : Load survey configuration from YAML file
    """

    @classmethod
    def forecast(
        cls,
        survey,
        cosmology,
        params: Sequence[str],
        gamma: float = 0.55,
        marginalize_over: Optional[Sequence[str]] = None,
    ) -> FisherForecastResult:
        """
        One-step Fisher forecast for 21cm intensity mapping (Recommended API).

        This is the simplest way to obtain Fisher constraints. It handles all
        internal steps automatically and returns a user-friendly result object.

        Parameters
        ----------
        survey : str or IntensityMappingSurvey
            Survey name (e.g., 'tianlai', 'ska1_mid_band2') or loaded survey object.
            If string, will load from hicosmo/fisher/surveys/{name}.yaml
        cosmology : CosmologyBase
            Cosmological model instance (e.g., LCDM, wCDM, ILCDM).
            Must have all required parameters defined.
        params : list of str
            Parameters to constrain (e.g., ['beta', 'H0', 'Omega_m', 'sigma8']).
            These are the parameters for which 1σ constraints will be computed.
        gamma : float, optional
            Growth index parameter (default: 0.55, GR value)
        marginalize_over : list of str, optional
            Additional nuisance parameters to marginalize over.
            If specified, constraints on `params` will account for uncertainties
            in these nuisance parameters.

        Returns
        -------
        FisherForecastResult
            User-friendly result object with:
            - `errors`: dict mapping param_name → σ(param)
            - `fisher_matrix`, `covariance`, `correlation`: JAX arrays
            - Pretty-print support via `print(result)`

        Examples
        --------
        Basic usage (3 lines):

        >>> from hicosmo.models import ILCDM
        >>> from hicosmo.fisher import IntensityMappingFisher
        >>>
        >>> ilcdm = ILCDM(H0=67.36, Omega_m=0.3153, beta=0.001)
        >>> result = IntensityMappingFisher.forecast('tianlai', ilcdm, ['beta', 'H0'])
        >>> print(result)
        ======================================================================
        Fisher Forecast Results: tianlai
        ======================================================================
        Redshift bins: 3
        Parameters: beta, H0
        <BLANKLINE>
        1σ Constraints:
        ----------------------------------------------------------------------
          σ(beta      ) =    1.340000e-02
          σ(H0        ) =    8.250000e-01
        ======================================================================

        With marginalization:

        >>> result = IntensityMappingFisher.forecast(
        ...     'ska1_mid_band2',
        ...     wcdm_model,
        ...     params=['w0', 'wa'],
        ...     marginalize_over=['H0', 'Omega_m', 'gamma']
        ... )
        >>> print(f"σ(w0) = {result.errors['w0']:.4f}")
        >>> print(f"σ(wa) = {result.errors['wa']:.4f}")

        See Also
        --------
        parameter_forecast : Instance method for multiple forecasts
        compute_observable_constraints : Low-level observable Fisher matrix
        """
        # Load survey if string provided
        if isinstance(survey, str):
            survey = load_survey(survey)

        # Create Fisher calculator instance
        fisher_calculator = cls(survey, cosmology, gamma=gamma)

        # Compute parameter forecast using existing method
        result_dict = fisher_calculator.parameter_forecast(
            params=params,
            marginalize_over=marginalize_over
        )

        # Build user-friendly result object
        errors_dict = {
            param: float(result_dict['errors'][i])
            for i, param in enumerate(result_dict['params'])
        }

        # Compute correlation matrix
        cov = result_dict['covariance']
        std_devs = jnp.sqrt(jnp.diag(cov))
        correlation = cov / jnp.outer(std_devs, std_devs)

        # Get cosmology summary
        cosmology_summary = {
            'model': type(cosmology).__name__,
            **{k: float(v) if isinstance(v, numbers.Real) else v
               for k, v in cosmology.params.items()}
        }

        # Extract LaTeX labels from cosmology model's get_parameters()
        latex_labels = {}
        if hasattr(cosmology, 'get_parameters'):
            for p in cosmology.get_parameters():
                if hasattr(p, 'latex_label') and p.latex_label:
                    latex_labels[p.name] = p.latex_label

        return FisherForecastResult(
            params=list(result_dict['params']),
            errors=errors_dict,
            fisher_matrix=result_dict['fisher'],
            covariance=result_dict['covariance'],
            correlation=correlation,
            survey_name=survey.name,
            cosmology_summary=cosmology_summary,
            n_bins=len(survey.redshift_bins),
            latex_labels=latex_labels,
        )

    def __init__(
        self,
        survey: IntensityMappingSurvey,
        cosmology,  # CosmologyBase instance (REQUIRED)
        gamma: float = 0.55,
    ) -> None:
        """
        Initialize Fisher matrix calculator for intensity mapping.

        Parameters
        ----------
        survey : IntensityMappingSurvey
            Survey configuration (hardware and observing strategy ONLY)
        cosmology : CosmologyBase
            Cosmological model instance (e.g., LCDM, wCDM, CPL)
            This parameter is REQUIRED.
        gamma : float, default=0.55
            Growth index parameter

        Examples
        --------
        >>> from hicosmo.models import CPL
        >>> from hicosmo.forecasts import load_survey, IntensityMappingFisher
        >>>
        >>> # Load survey (only hardware config)
        >>> survey = load_survey('ska1_mid_band2')
        >>>
        >>> # Define cosmology separately
        >>> cosmo = CPL(H0=67.36, Omega_m=0.3153, w0=-1.0, wa=0.0)
        >>>
        >>> # Create Fisher calculator with explicit cosmology
        >>> fisher = IntensityMappingFisher(survey, cosmo)
        """
        # Clean API: cosmology is required parameter
        self.cosmology = cosmology
        self.survey = survey
        self.gamma = gamma
        # Use new GrowthModel API with gamma0 parameter (backward compatible)
        self.growth = GrowthModel(self.cosmology, gamma0=gamma)
        self.power = LinearPowerSpectrum(self.cosmology)

    def _brightness_temperature(self, z: float) -> float:
        """Compute HI brightness temperature in Kelvin.

        Uses the standard 21cm cosmology formula:
        T_b = 27 mK × h × (Ω_HI / 10^-3) × (1+z)^2 / E(z)

        References:
        - Santos et al. (2015), Eq. 3
        - Bull et al. (2015), ApJ 803, 21
        """
        Ez = jnp.asarray(self.cosmology.E_z(z))
        h = jnp.asarray(self.cosmology.params['H0']) / 100.0
        omega_hi = jnp.asarray(self.survey.omega_hi_fn(z))

        # Standard 21cm formula: T_b in mK
        T_b_mK = 27.0 * h * (omega_hi / 1e-3) * (1.0 + z) ** 2 / Ez

        # Convert to Kelvin for consistency with power spectrum calculation
        return T_b_mK * 1e-3  # Kelvin

    def _fisher_per_bin_terms(
        self,
        z: jnp.ndarray,
        delta_z: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        theta_b = _beam_fwhm_m(z, self.survey.instrument)
        volume = _survey_volume_params(z, delta_z, self.survey.instrument, self.cosmology)
        sigma_pix = _noise_temperature(z, theta_b, self.survey.instrument)
        V_pix = _pixel_volume(z, theta_b, self.survey.instrument, self.cosmology)

        kmin = (2.0 * jnp.pi) / volume ** (1.0 / 3.0)
        ns = jnp.asarray(self.cosmology.params.get('n_s', 0.962))
        kmax = 0.14 * (1.0 + z) ** (2.0 / (2.0 + ns))
        k_grid = jnp.linspace(kmin, kmax, 256)
        mu_grid = jnp.linspace(-1.0, 1.0, 200)
        kk, mu = jnp.meshgrid(k_grid, mu_grid, indexing='ij')

        bias = jnp.asarray(self.survey.bias_fn(z))
        f_growth = jnp.asarray(self.growth.growth_rate(z))
        f_sigma8 = jnp.asarray(self.growth.f_sigma8(z))
        sigma8_z = jnp.asarray(self.growth.sigma8(z))
        beta = f_growth / bias

        T_b = self._brightness_temperature(z)
        growth_factor = self.growth.growth_factor(z)
        P_m = self.power.power(k_grid, growth_factor)
        dlogP_dlogk = gradient_1d(jnp.log(P_m), jnp.log(k_grid))

        # FIXED: P_m already includes σ8² normalization via growth_factor²
        # Do NOT multiply by sigma8_z² again (Bug Fix #1)
        P_signal = (T_b**2) * (bias ** 2) * (1.0 + beta * mu**2) ** 2 * P_m[:, None]
        chi = jnp.asarray(self.cosmology.comoving_distance(z))
        one_minus_mu2 = 1.0 - mu**2
        k_perp = kk * jnp.sqrt(one_minus_mu2)
        beam_window = jnp.exp(-(k_perp**2) * chi**2 * theta_b**2 / (8.0 * jnp.log(2.0)))
        beam_window = jnp.clip(beam_window, 1e-12, None)
        P_noise = (sigma_pix**2) * V_pix / beam_window
        power_tot = P_signal + P_noise
        V_eff = volume * (P_signal / power_tot) ** 2

        pref = 2.0 * mu**2 * beta / (1.0 + beta * mu**2)
        dlnP_dlnfs = pref
        # FIXED: Correct bias derivative formula (Bug Fix #2)
        # P ∝ b² × (1 + βμ²)², so ∂ln(P)/∂ln(b) = 2/(1 + βμ²)
        dlnP_dlnbs = 2.0 / (1.0 + beta * mu**2)
        dlnP_dlnDA = -2.0 + 4.0 * mu**2 * one_minus_mu2 * beta / (1.0 + beta * mu**2) - one_minus_mu2 * dlogP_dlogk[:, None]
        dlnP_dlnH = 1.0 + 4.0 * mu**2 * one_minus_mu2 * beta / (1.0 + beta * mu**2) + mu**2 * dlogP_dlogk[:, None]

        weights = (kk**2) * V_eff / (8.0 * jnp.pi**2)
        derivatives = jnp.stack([dlnP_dlnfs, dlnP_dlnbs, dlnP_dlnDA, dlnP_dlnH], axis=0)
        fisher = _integrate_fisher_matrix(weights, derivatives, k_grid, mu_grid)

        return fisher, bias, beta, f_sigma8, sigma8_z, volume, sigma_pix

    def fisher_per_bin(self, bin_cfg: RedshiftBin) -> Tuple[jnp.ndarray, Dict[str, float]]:
        fisher, bias, beta, f_sigma8, sigma8_z, volume, sigma_pix = self._fisher_per_bin_terms(
            jnp.asarray(bin_cfg.z, dtype=float),
            jnp.asarray(bin_cfg.delta_z, dtype=float),
        )
        cov = _safe_inverse(fisher, name="bin fisher")
        return cov, {
            'bias': bias,
            'beta': beta,
            'f_sigma8': f_sigma8,
            'sigma8_z': sigma8_z,
            'volume': volume,
            'sigma_pix': sigma_pix,
        }

    def compute_observable_constraints(self) -> FisherResult:
        """
        Compute per-redshift-bin observable constraints (low-level API).

        Returns covariance blocks of size 4×4 containing:
        [0] ln(fσ8), [1] ln(bσ8), [2] ln(DA), [3] ln(H)

        Following Bull et al. (2016), bias parameters will be marginalized
        in parameter_forecast() when projecting to cosmological parameters.

        Returns
        -------
        FisherResult
            Observable-level Fisher constraints per redshift bin

        Notes
        -----
        This is a low-level method for advanced users. Most users should use
        the high-level `forecast()` class method or `parameter_forecast()` instance method.
        """
        z = jnp.array([b.z for b in self.survey.redshift_bins], dtype=float)
        delta_z = jnp.array([b.delta_z for b in self.survey.redshift_bins], dtype=float)
        fisher_stack, _, _, _, _, _, _ = jax.vmap(self._fisher_per_bin_terms)(z, delta_z)
        covs_arr = jax.vmap(lambda m: jnp.linalg.pinv(m, rcond=1e-12))(fisher_stack)
        errors = jnp.sqrt(jnp.diagonal(covs_arr, axis1=1, axis2=2))
        return FisherResult(
            z=z,
            covariance_blocks=covs_arr,
            sigma_ln_fsigma8=errors[:, 0],
            sigma_ln_DA=errors[:, 2],  # Index 2 now (0=fσ8, 1=bσ8, 2=DA, 3=H)
            sigma_ln_H=errors[:, 3],   # Index 3 now
        )

    # ------------------------------------------------------------------
    # Parameter projection
    # ------------------------------------------------------------------

    def observable_vector(self) -> jnp.ndarray:
        """
        Compute observable vector for Fisher forecast.

        Following Bull et al. (2016) Section 3.6, we use per-redshift-bin observables:
        - ln(f σ8): Growth rate times σ8
        - ln(b σ8): Bias times σ8 (to be marginalized as nuisance parameter)
        - ln(D_A): Angular diameter distance
        - ln(H): Hubble parameter

        Returns 4*N_bins observables, where bias parameters will be
        marginalized over in parameter_forecast().
        """
        values = []

        for bin_cfg in self.survey.redshift_bins:
            z = bin_cfg.z

            # Observable 1: ln(f σ8)
            f_sigma8 = self.growth.f_sigma8(z)
            values.append(jnp.log(f_sigma8))

            # Observable 2: ln(b σ8) - HI bias as nuisance parameter
            bias = float(self.survey.bias_fn(z))
            # Use GrowthModel's sigma8(z) method
            sigma8_z = self.growth.sigma8(z)
            b_sigma8 = bias * sigma8_z
            values.append(jnp.log(b_sigma8))

            # Observable 3: ln(D_A)
            DA = jnp.asarray(self.cosmology.angular_diameter_distance(z))
            values.append(jnp.log(DA))

            # Observable 4: ln(H)
            H = jnp.asarray(self.cosmology.H_z(z))
            values.append(jnp.log(H))

        return jnp.array(values)

    def _clone(self, param_updates: Mapping[str, float]) -> 'IntensityMappingFisher':
        """Clone this Fisher calculator with updated cosmological parameters.

        Parameters
        ----------
        param_updates : Mapping[str, float]
            Parameter updates to apply

        Returns
        -------
        IntensityMappingFisher
            Cloned calculator with updated cosmology
        """
        # Get current cosmology parameters (base parameters only)
        params = {}
        if hasattr(self.cosmology, 'params'):
            current = self.cosmology.params
            if hasattr(current, '_given'):
                params = dict(current._given)
            elif hasattr(current, 'get_all_parameters'):
                params = dict(current.get_all_parameters())
            else:
                try:
                    params = dict(current)
                except TypeError:
                    params = {}
        params.update(param_updates)

        # Create new cosmology instance of the same type
        cosmology_class = type(self.cosmology)
        cosmology = cosmology_class(**params)

        # Create clone with new cosmology
        return IntensityMappingFisher(self.survey, cosmology, gamma=self.gamma)

    def derivative_matrix(
        self,
        params: Sequence[str],
        step_fraction: float = 1e-3,
        marginalize_bias: bool = True,
        verbose: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute derivative matrix for parameter forecast.

        Parameters
        ----------
        params : Sequence[str]
            Parameter names to compute derivatives for
        step_fraction : float, optional
            Relative step size for numerical derivatives
        marginalize_bias : bool, optional
            If True, only return derivatives for marginalized observables (fσ8, DA, H).
            If False, return all 4 observables including bias.

        Returns
        -------
        tuple
            (derivative_matrix, base_vector) where derivative_matrix has shape
            (3*N_bins, N_params) if marginalize_bias=True, else (4*N_bins, N_params)
        """
        base_vector = self.observable_vector()
        derivatives = []

        for name in params:
            if name == 'gamma':
                eps = step_fraction
                plus = IntensityMappingFisher(self.survey, self.cosmology, gamma=self.gamma + eps)
                minus = IntensityMappingFisher(self.survey, self.cosmology, gamma=self.gamma - eps)
                deriv = (plus.observable_vector() - minus.observable_vector()) / (2.0 * eps)
            else:
                # Get base value from cosmology parameters (NEW API)
                if name not in self.cosmology.params:
                    raise ValueError(f"Parameter '{name}' not found in cosmology.params. "
                                   f"Available: {list(self.cosmology.params.keys())}")
                base_value = float(self.cosmology.params[name])
                # For parameters near zero, use absolute step instead of relative
                eps = step_fraction * abs(base_value) if abs(base_value) > 1e-10 else step_fraction
                plus = self._clone({name: base_value + eps})
                minus = self._clone({name: base_value - eps})
                deriv = (plus.observable_vector() - minus.observable_vector()) / (2.0 * eps)
            derivatives.append(deriv)

        derivative_matrix = jnp.column_stack(derivatives)

        # Following Bull et al. (2016): marginalize over per-bin bias parameters
        # If observable_vector returns 4*N_bins (fσ8, bσ8, DA, H per bin),
        # extract only the derivatives for (fσ8, DA, H) by removing bias entries
        if marginalize_bias and len(base_vector) % 4 == 0:
            n_bins = len(base_vector) // 4
            # Extract indices [0, 2, 3] from each 4-element block (fσ8, DA, H)
            keep_indices = []
            for i in range(n_bins):
                base = i * 4
                keep_indices.extend([base + 0, base + 2, base + 3])  # Skip base+1 (bias)

            keep_idx = jnp.array(keep_indices)
            derivative_matrix = derivative_matrix[keep_idx, :]
            base_vector = base_vector[keep_idx]

            if verbose:
                logger.info("\n✓ Derivative matrix marginalization:")
                logger.info("  Removed bias derivatives, shape: %s", derivative_matrix.shape)

        return derivative_matrix, base_vector

    def parameter_forecast(
        self,
        params: Sequence[str],
        marginalize_over: Optional[Sequence[str]] = None,
        verbose: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute parameter forecast constraints from Fisher matrix.

        Parameters
        ----------
        params : Sequence[str]
            Parameters of interest for which to compute constraints
        marginalize_over : Sequence[str], optional
            Additional parameters to marginalize over. If specified, the Fisher
            matrix is computed for `params + marginalize_over`, then marginalized
            to obtain constraints only on `params`. This is the correct procedure
            for obtaining marginal constraints that account for uncertainties in
            nuisance parameters.
        verbose : bool, optional
            If True, print detailed debugging information. Default: False.

        Returns
        -------
        dict
            Dictionary with keys 'params', 'fisher', 'covariance', 'errors',
            'baseline_observables'

        Notes
        -----
        Following Bull et al. (2016), marginal constraints integrate over
        nuisance parameters and are always weaker (larger errors) than
        conditional constraints where nuisance parameters are fixed.
        """
        log = logger.info if verbose else (lambda *args, **kwargs: None)

        # Determine full parameter list
        if marginalize_over:
            all_params = list(params) + list(marginalize_over)
            log("\n%s", "=" * 60)
            log("Computing full Fisher matrix for: %s", all_params)
            log("Will marginalize over: %s", list(marginalize_over))
            log("Target parameters: %s", list(params))
            log("%s", "=" * 60)
        else:
            all_params = list(params)

        # Compute covariance matrix for observables (4*N_bins with bias)
        fisher_result = self.compute_observable_constraints()
        n_bins = len(fisher_result.z)
        cov_block = fisher_result.covariance_blocks
        block_dim = cov_block.shape[-1]  # Should be 4 (fσ8, bσ8, DA, H)

        if verbose:
            log("\n%s", "=" * 60)
            log("DEBUG: Per-bin Bias Marginalization (Bull 2016 Method)")
            log("%s", "=" * 60)
            log("Number of redshift bins: %s", n_bins)
            log("Block dimension (observables per bin): %s", block_dim)

        # Following Bull et al. (2016) and qcosmc reference code:
        # For each redshift bin, marginalize over HI bias parameter
        # Input: 4×4 covariance (fσ8, bσ8, DA, H)
        # Output: 3×3 covariance (fσ8, DA, H) with bias marginalized

        if block_dim == 4:
            if verbose:
                log("✓ Detected 4 observables per bin, performing bias marginalization")

            # Marginalized covariance will be 3*N_bins × 3*N_bins
            cov_marginalized = jnp.zeros((n_bins * 3, n_bins * 3))

            for idx in range(n_bins):
                # Get 4×4 covariance block for this redshift bin
                cov_4x4 = cov_block[idx]

                # Marginalize over bias (index 1 in the 4-element observable vector)
                # Keep indices [0, 2, 3] corresponding to (fσ8, DA, H)
                keep_indices = [0, 2, 3]
                cov_3x3 = cov_4x4[jnp.ix_(jnp.array(keep_indices), jnp.array(keep_indices))]

                # Place into marginalized covariance matrix
                start = idx * 3
                cov_marginalized = cov_marginalized.at[start:start + 3, start:start + 3].set(cov_3x3)

                if idx == 0 and verbose:
                    log("\nBin %s marginalization example:", idx)
                    log("  Input 4×4 cov diagonal: %s", jnp.diag(cov_4x4))
                    log("  Output 3×3 cov diagonal: %s", jnp.diag(cov_3x3))

            cov_total = cov_marginalized
            if verbose:
                log("\n✓ Bias marginalization complete for all %s bins", n_bins)
                log("  Final covariance shape: %s (3*%s)", cov_total.shape, n_bins)
        else:
            # Fallback: no marginalization if block_dim != 4
            if verbose:
                log("⚠ Block dimension is %s, not 4. Skipping bias marginalization.", block_dim)
            cov_total = jnp.zeros((n_bins * block_dim, n_bins * block_dim))
            for idx in range(n_bins):
                start = idx * block_dim
                cov_total = cov_total.at[start:start + block_dim, start:start + block_dim].set(cov_block[idx])

        # Print covariance properties (only if verbose)
        if verbose:
            log("\n%s", "=" * 60)
            log("DEBUG: Covariance Matrix Analysis")
            log("%s", "=" * 60)
            log("Covariance shape: %s", cov_total.shape)
            log("Covariance determinant: %.6e", jnp.linalg.det(cov_total))
            log("Covariance condition number: %.6e", jnp.linalg.cond(cov_total))
            log("Covariance diagonal (first 5): %s", jnp.diag(cov_total)[:5])

        cov_inv = _safe_inverse(cov_total, rcond=1e-12, name="covariance", warn=verbose)

        # Compute derivatives for ALL parameters (including those to marginalize)
        derivative_matrix, base_vector = self.derivative_matrix(all_params, verbose=verbose)

        # Print derivative matrix properties (only if verbose)
        if verbose:
            log("\n%s", "=" * 60)
            log("DEBUG: Derivative Matrix Analysis")
            log("%s", "=" * 60)
            log("Derivative matrix shape: %s", derivative_matrix.shape)
            log("Derivative matrix norm: %.6e", jnp.linalg.norm(derivative_matrix))
            log("Base observable vector (first 5): %s", base_vector[:5])
            log("Derivative matrix sample (first 3x3):\n%s", derivative_matrix[:3, :min(3, len(all_params))])

        # Compute full Fisher matrix
        fisher_full = derivative_matrix.T @ cov_inv @ derivative_matrix

        # Print Fisher matrix properties (only if verbose)
        if verbose:
            log("\n%s", "=" * 60)
            log("DEBUG: Full Fisher Matrix Analysis")
            log("%s", "=" * 60)
            log("Fisher matrix shape: %s", fisher_full.shape)
            log("Fisher matrix condition number: %.6e", jnp.linalg.cond(fisher_full))
            log("Fisher matrix diagonal: %s", jnp.diag(fisher_full))
            log("Fisher matrix (full):\n%s", fisher_full)

        # Marginalize if requested
        if marginalize_over:
            if verbose:
                log("\n%s", "=" * 60)
                log("Performing Marginalization")
                log("%s", "=" * 60)

            # Find indices of parameters to keep
            keep_indices = [i for i, p in enumerate(all_params) if p in params]

            # True marginalization: Fisher → Covariance → Extract → Fisher
            cov_full = _safe_inverse(fisher_full, rcond=1e-12, name="fisher", warn=verbose)
            cov_subset = cov_full[jnp.ix_(jnp.array(keep_indices), jnp.array(keep_indices))]
            fisher_params = _safe_inverse(cov_subset, rcond=1e-12, name="marginalized covariance", warn=verbose)

            if verbose:
                log("Marginalized Fisher matrix shape: %s", fisher_params.shape)
                log("Marginalized Fisher condition number: %.6e", jnp.linalg.cond(fisher_params))
                log("✓ Marginalization complete")
        else:
            fisher_params = fisher_full

        cov_params = _safe_inverse(fisher_params, rcond=1e-12, name="parameter fisher", warn=verbose)
        errors = jnp.sqrt(jnp.diag(cov_params))

        # Print final errors (only if verbose)
        if verbose:
            log("\n%s", "=" * 60)
            log("DEBUG: Final Parameter Constraints")
            log("%s", "=" * 60)
            log("Parameter covariance diagonal: %s", jnp.diag(cov_params))
            for i, param in enumerate(params):
                log("  σ(%s) = %.6e", param, errors[i])
            log("%s\n", "=" * 60)

        return {
            'params': list(params),
            'fisher': fisher_params,
            'covariance': cov_params,
            'errors': errors,
            'baseline_observables': base_vector,
        }


# -----------------------------------------------------------------------------
# Convenience API
# -----------------------------------------------------------------------------


def list_available_surveys(directory: Path | None = None) -> List[str]:
    directory = directory or SURVEY_CONFIG_DIR
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob('*.yaml'))


def load_survey(name_or_path: str) -> IntensityMappingSurvey:
    path = Path(name_or_path)
    if path.exists():
        return IntensityMappingSurvey.from_file(path)
    candidate = SURVEY_CONFIG_DIR / f'{name_or_path}.yaml'
    if candidate.exists():
        return IntensityMappingSurvey.from_file(candidate)
    available = ', '.join(list_available_surveys())
    raise FileNotFoundError(f'Survey configuration "{name_or_path}" not found. Available: {available}')


def run_forecast(
    survey_name: str,
    model_name: Optional[str] = None,
    parameters: Sequence[str] = ('H0', 'Omega_m', 'w0', 'wa', 'gamma'),
    gamma: float = 0.55,
    cosmology: Optional[object] = None,
    ) -> Dict[str, jnp.ndarray]:
    """Convenience wrapper for a one-shot Fisher forecast.

    Parameters
    ----------
    survey_name : str
        Survey configuration name (e.g., 'tianlai').
    model_name : str, optional
        Cosmology model name ('lcdm', 'wcdm', 'cpl', 'ilcdm'). Ignored if
        `cosmology` is provided.
    parameters : sequence of str
        Parameters to forecast.
    gamma : float
        Growth index parameter.
    cosmology : object, optional
        Explicit cosmology instance (e.g., LCDM(), wCDM()).
    """
    survey = load_survey(survey_name)

    if model_name is not None:
        survey = IntensityMappingSurvey(
            name=survey.name,
            model=model_name,
            reference=survey.reference,
            instrument=survey.instrument,
            redshift_bins=survey.redshift_bins,
            bias_fn=survey.bias_fn,
            omega_hi_fn=survey.omega_hi_fn,
            priors=survey.priors,
            description=survey.description,
        )

    if cosmology is None:
        from hicosmo.models import LCDM, wCDM, CPL, ILCDM

        model_key = (model_name or survey.model or "lcdm").lower()
        model_map = {
            "lcdm": LCDM,
            "wcdm": wCDM,
            "cpl": CPL,
            "ilcdm": ILCDM,
        }
        if model_key not in model_map:
            raise ValueError(f"Unknown cosmology model '{model_key}'.")
        cosmology = model_map[model_key]()

    calculator = IntensityMappingFisher(survey, cosmology, gamma=gamma)
    # Observable-space Fisher (for reference)
    fisher_result = calculator.compute_observable_constraints()
    # Parameter-space Fisher
    param_result = calculator.parameter_forecast(parameters)
    return {
        'survey': survey,
        'fisher_result': fisher_result,
        'parameter_result': param_result,
    }
