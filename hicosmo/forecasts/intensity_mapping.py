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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..models import LCDM, wCDM, CPL
from ..utils.constants import c_km_s
from ..utils.integration import integrate_simpson, integrate_logspace

# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_REGISTRY = {
    'lcdm': LCDM,
    'wcdm': wCDM,
    'cpl': CPL,
}

SURVEY_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'surveys'


# -----------------------------------------------------------------------------
# Helper factories for redshift functions
# -----------------------------------------------------------------------------


def _build_redshift_function(cfg: Optional[Mapping[str, float]], default: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(z)`` defined by *cfg*."""
    if cfg is None:
        return lambda z: np.full_like(np.asarray(z, dtype=float), default, dtype=float)

    kind = cfg.get('kind', 'constant').lower()
    if kind == 'constant':
        value = float(cfg.get('value', default))
        return lambda z: np.full_like(np.asarray(z, dtype=float), value, dtype=float)
    if kind == 'polynomial':
        coeffs = [float(c) for c in cfg.get('coefficients', [])]
        if not coeffs:
            raise ValueError('Polynomial model requires "coefficients" entries')

        def poly(z: np.ndarray) -> np.ndarray:
            z = np.asarray(z, dtype=float)
            acc = np.zeros_like(z)
            for power, coeff in enumerate(coeffs):
                acc += coeff * z**power
            return acc

        return poly
    if kind == 'table':
        table = cfg.get('values')
        if not table:
            raise ValueError('Tabulated model requires "values" list with {z, value}')
        z_nodes = np.array([float(entry['z']) for entry in table])
        v_nodes = np.array([float(entry['value']) for entry in table])

        def interp(z: np.ndarray) -> np.ndarray:
            z = np.asarray(z, dtype=float)
            return np.interp(z, z_nodes, v_nodes)

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

    def __call__(self, z: np.ndarray | float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.kind == 'none':
            return np.zeros_like(z)
        if self.kind == 'constant':
            return np.full_like(z, self.value)
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
        return (self.survey_area_deg2 * (np.pi / 180.0) ** 2)

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
    model: str
    reference: Dict[str, float]
    instrument: InstrumentConfig
    redshift_bins: List[RedshiftBin]
    bias_fn: Callable[[np.ndarray], np.ndarray]
    omega_hi_fn: Callable[[np.ndarray], np.ndarray]
    priors: Dict[str, float]
    description: str = ''

    @classmethod
    def from_file(cls, filename: Path | str) -> 'IntensityMappingSurvey':
        with open(filename, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
        if 'name' not in data:
            raise ValueError('Survey configuration must contain "name" field')
        instrument_cfg = InstrumentConfig.from_dict(data['instrument'])
        default_delta_z = float(data.get('default_delta_z', 0.1))
        bins = [RedshiftBin.from_dict(entry, default_delta_z) for entry in data['redshift_bins']]
        bias_fn = _build_redshift_function(data.get('hi_bias'), default=1.0)
        omega_hi_fn = _build_redshift_function(data.get('hi_density'), default=4.8e-4)
        reference = {k: float(v) for k, v in data.get('reference', {}).items()}
        priors = {k: float(v) for k, v in data.get('priors', {}).items()}
        return cls(
            name=data['name'],
            model=data.get('model', 'lcdm'),
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


def _instantiate_cosmology(model_name: str, params: Mapping[str, float]):
    model_cls = MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f'Unknown cosmology "{model_name}". Available: {available}')
    return model_cls(**params)


class GrowthModel:
    """Growth model using γ-parameterisation."""

    def __init__(self, cosmology, gamma: float = 0.55) -> None:
        self.cosmology = cosmology
        self.gamma = gamma

    def omega_m(self, z: np.ndarray | float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        numerator = float(self.cosmology.params['Omega_m']) * (1.0 + z) ** 3
        Ez = np.asarray(self.cosmology.E_z(z))
        return numerator / Ez**2

    def growth_rate(self, z: np.ndarray | float) -> np.ndarray:
        return self.omega_m(z) ** self.gamma

    def growth_factor(self, z: float) -> float:
        if z <= 0.0:
            return 1.0

        def integrand(z_vals: np.ndarray) -> np.ndarray:
            return self.growth_rate(z_vals) / (1.0 + z_vals)

        integral = integrate_simpson(integrand, 0.0, z, num=512)
        return float(np.exp(-integral))

    def sigma8(self, z: float) -> float:
        return float(self.cosmology.params.get('sigma8', 0.834)) * self.growth_factor(z)

    def f_sigma8(self, z: float) -> float:
        return self.growth_rate(z) * self.sigma8(z)


class LinearPowerSpectrum:
    def __init__(self, cosmology) -> None:
        self.cosmology = cosmology
        self.h = float(cosmology.params['H0']) / 100.0
        self.n_s = float(cosmology.params.get('n_s', 0.962))
        self.sigma8 = float(cosmology.params.get('sigma8', 0.834))
        self._normalisation = self._normalise()

    def _transfer(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k)
        h = self.h
        omega_m = float(self.cosmology.params['Omega_m'])
        omega_b = float(self.cosmology.params.get('Omega_b', 0.049))
        theta_cmb = float(self.cosmology.params.get('T_cmb', 2.7255)) / 2.7
        omega_m_h2 = omega_m * h**2
        omega_b_h2 = omega_b * h**2
        q = k / (13.41 * h) * theta_cmb**2
        beta_c = 1.0 / (1.0 + 0.944 / (1.0 + (458.0 * omega_m_h2) ** 0.708))
        L0 = np.log(np.e + 1.8 * beta_c * q)
        C0 = 14.4 + 325.0 / (1.0 + 60.5 * q ** 1.08)
        return L0 / (L0 + C0 * q**2)

    def _unnormalised(self, k: np.ndarray) -> np.ndarray:
        return (k ** self.n_s) * self._transfer(k) ** 2

    def _normalise(self) -> float:
        R8 = 8.0  # h^-1 Mpc

        def window(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            sinx = np.sin(x)
            cosx = np.cos(x)
            with np.errstate(divide='ignore', invalid='ignore'):
                w = np.where(x < 1e-6, 1.0 - x**2 / 10.0, 3.0 * (sinx - x * cosx) / (x**3))
            return w

        def integrand(k_vals: np.ndarray) -> np.ndarray:
            p = self._unnormalised(k_vals / self.h)
            w = window(k_vals * R8)
            return k_vals**2 * p * w**2

        integral = integrate_logspace(integrand, 1e-4, 10.0, num=2048)
        return self.sigma8**2 * 2.0 * np.pi**2 / integral

    def power(self, k: np.ndarray, growth_factor: float) -> np.ndarray:
        k = np.asarray(k)
        base = self._normalisation * self._unnormalised(k)
        return base * growth_factor**2


# -----------------------------------------------------------------------------
# Fisher calculator
# -----------------------------------------------------------------------------


def _beam_fwhm_m(z: float, instrument: InstrumentConfig) -> float:
    wavelength = (299792458.0 / (1420.405751e6 / (1.0 + z)))
    return 1.22 * wavelength / instrument.dish_diameter_m


def _survey_volume(bin_cfg: RedshiftBin, instrument: InstrumentConfig, cosmology) -> float:
    z_c = bin_cfg.z
    z_min = max(0.0, z_c - bin_cfg.delta_z / 2.0)
    z_max = z_c + bin_cfg.delta_z / 2.0

    def integrand(z_vals: np.ndarray) -> np.ndarray:
        H = np.asarray(cosmology.H_z(z_vals))
        chi = np.asarray(cosmology.comoving_distance(z_vals))
        return (c_km_s / H) * chi**2

    shell = integrate_simpson(integrand, z_min, z_max, num=256)
    return instrument.solid_angle * shell


def _pixel_volume(z: float, theta_b: float, bin_cfg: RedshiftBin, instrument: InstrumentConfig, cosmology) -> float:
    dz_pix = (1.0 + z) ** 2 * instrument.channel_width_hz / 1420.405751e6
    dchi_dz = c_km_s / float(np.asarray(cosmology.H_z(z)))
    chi = float(np.asarray(cosmology.comoving_distance(z)))
    omega_pix = 1.13 * theta_b**2
    return omega_pix * dchi_dz * chi**2 * dz_pix


def _noise_temperature(z: float, theta_b: float, instrument: InstrumentConfig) -> float:
    t_sky = instrument.sky_temperature(z)
    t_sys = instrument.system_temperature_K + t_sky
    omega_pix = 1.13 * theta_b**2
    t_pix = instrument.total_time_seconds * (omega_pix / instrument.solid_angle) * instrument.ndish * instrument.nbeam
    return t_sys / np.sqrt(instrument.channel_width_hz * t_pix)


@dataclass
class FisherResult:
    z: np.ndarray
    covariance_blocks: np.ndarray
    sigma_ln_fsigma8: np.ndarray
    sigma_ln_DA: np.ndarray
    sigma_ln_H: np.ndarray


class IntensityMappingFisher:
    """Compute Fisher information for an intensity mapping survey."""

    def __init__(
        self,
        survey: IntensityMappingSurvey,
        model_override: Optional[str] = None,
        gamma: float = 0.55,
    ) -> None:
        model_name = model_override or survey.model
        params = survey.reference.copy()
        self.cosmology = _instantiate_cosmology(model_name, params)
        self.survey = survey
        self.gamma = gamma
        self.growth = GrowthModel(self.cosmology, gamma=gamma)
        self.power = LinearPowerSpectrum(self.cosmology)

    def _brightness_temperature(self, z: float) -> float:
        Ez = float(np.asarray(self.cosmology.E_z(z)))
        h = float(self.cosmology.params['H0']) / 100.0
        omega_hi = float(self.survey.omega_hi_fn(z))
        return 0.180 * h * omega_hi * (1.0 + z) ** 2 / Ez  # Kelvin

    def fisher_per_bin(self, bin_cfg: RedshiftBin) -> Tuple[np.ndarray, Dict[str, float]]:
        z = bin_cfg.z
        theta_b = _beam_fwhm_m(z, self.survey.instrument)
        volume = _survey_volume(bin_cfg, self.survey.instrument, self.cosmology)
        sigma_pix = _noise_temperature(z, theta_b, self.survey.instrument)
        V_pix = _pixel_volume(z, theta_b, bin_cfg, self.survey.instrument, self.cosmology)

        kmin = (2.0 * np.pi) / volume ** (1.0 / 3.0)
        ns = float(self.cosmology.params.get('n_s', 0.962))
        kmax = 0.14 * (1.0 + z) ** (2.0 / (2.0 + ns))
        k_grid = np.linspace(kmin, kmax, 256)
        mu_grid = np.linspace(-1.0, 1.0, 200)
        kk, mu = np.meshgrid(k_grid, mu_grid, indexing='ij')

        bias = float(self.survey.bias_fn(z))
        f_growth = float(self.growth.growth_rate(z))
        f_sigma8 = float(self.growth.f_sigma8(z))
        sigma8_z = float(self.growth.sigma8(z))
        beta = f_growth / bias

        T_b = self._brightness_temperature(z)
        growth_factor = self.growth.growth_factor(z)
        P_m = self.power.power(k_grid, growth_factor)
        dlogP_dlogk = np.gradient(np.log(P_m), np.log(k_grid), edge_order=2)

        P_signal = (T_b**2) * (bias * sigma8_z) ** 2 * (1.0 + beta * mu**2) ** 2 * P_m[:, None]
        chi = float(np.asarray(self.cosmology.comoving_distance(z)))
        one_minus_mu2 = 1.0 - mu**2
        k_perp = kk * np.sqrt(one_minus_mu2)
        beam_window = np.exp(-(k_perp**2) * chi**2 * theta_b**2 / (8.0 * np.log(2.0)))
        beam_window = np.clip(beam_window, 1e-12, None)
        P_noise = (sigma_pix**2) * V_pix / beam_window
        power_tot = P_signal + P_noise
        V_eff = volume * (P_signal / power_tot) ** 2

        pref = 2.0 * mu**2 * beta / (1.0 + beta * mu**2)
        dlnP_dlnfs = pref
        dlnP_dlnbs = 2.0 * bias / (bias + f_growth * mu**2)
        dlnP_dlnDA = -2.0 + 4.0 * mu**2 * one_minus_mu2 * beta / (1.0 + beta * mu**2) - one_minus_mu2 * dlogP_dlogk[:, None]
        dlnP_dlnH = 1.0 + 4.0 * mu**2 * one_minus_mu2 * beta / (1.0 + beta * mu**2) + mu**2 * dlogP_dlogk[:, None]

        weights = (kk**2) * V_eff / (8.0 * np.pi**2)
        derivatives = [dlnP_dlnfs, dlnP_dlnbs, dlnP_dlnDA, dlnP_dlnH]
        fisher = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                integrand = weights * derivatives[i] * derivatives[j]
                inner = np.trapz(integrand, mu_grid, axis=1)
                fisher[i, j] = float(np.trapz(inner, k_grid))
                if i != j:
                    fisher[j, i] = fisher[i, j]

        cov = np.linalg.inv(fisher)
        cov_no_bias = cov[np.ix_([0, 2, 3], [0, 2, 3])]
        return cov_no_bias, {
            'bias': bias,
            'beta': beta,
            'f_sigma8': f_sigma8,
            'sigma8_z': sigma8_z,
            'volume': volume,
            'sigma_pix': sigma_pix,
        }

    def forecast(self) -> FisherResult:
        covs = []
        for bin_cfg in self.survey.redshift_bins:
            cov_block, _ = self.fisher_per_bin(bin_cfg)
            covs.append(cov_block)
        covs_arr = np.stack(covs)
        errors = np.sqrt(np.diagonal(covs_arr, axis1=1, axis2=2))
        return FisherResult(
            z=np.array([b.z for b in self.survey.redshift_bins]),
            covariance_blocks=covs_arr,
            sigma_ln_fsigma8=errors[:, 0],
            sigma_ln_DA=errors[:, 1],
            sigma_ln_H=errors[:, 2],
        )

    # ------------------------------------------------------------------
    # Parameter projection
    # ------------------------------------------------------------------

    def observable_vector(self) -> np.ndarray:
        values = []
        for bin_cfg in self.survey.redshift_bins:
            z = bin_cfg.z
            values.append(np.log(self.growth.f_sigma8(z)))
        for bin_cfg in self.survey.redshift_bins:
            z = bin_cfg.z
            values.append(np.log(float(np.asarray(self.cosmology.angular_diameter_distance(z)))))
        for bin_cfg in self.survey.redshift_bins:
            z = bin_cfg.z
            values.append(np.log(float(np.asarray(self.cosmology.H_z(z)))))
        return np.array(values)

    def _clone(self, param_updates: Mapping[str, float]) -> 'IntensityMappingFisher':
        params = self.survey.reference.copy()
        params.update(param_updates)
        cosmology = _instantiate_cosmology(self.survey.model, params)
        clone = IntensityMappingFisher(self.survey, model_override=self.survey.model, gamma=self.gamma)
        clone.cosmology = cosmology
        clone.growth = GrowthModel(cosmology, gamma=self.gamma)
        clone.power = LinearPowerSpectrum(cosmology)
        return clone

    def derivative_matrix(self, params: Sequence[str], step_fraction: float = 1e-3) -> np.ndarray:
        base_vector = self.observable_vector()
        derivatives = []
        for name in params:
            if name == 'gamma':
                eps = step_fraction
                plus = IntensityMappingFisher(self.survey, model_override=self.survey.model, gamma=self.gamma + eps)
                minus = IntensityMappingFisher(self.survey, model_override=self.survey.model, gamma=self.gamma - eps)
                deriv = (plus.observable_vector() - minus.observable_vector()) / (2.0 * eps)
            else:
                base_value = self.survey.reference[name]
                eps = step_fraction * abs(base_value) if base_value != 0 else step_fraction
                plus = self._clone({name: base_value + eps})
                minus = self._clone({name: base_value - eps})
                deriv = (plus.observable_vector() - minus.observable_vector()) / (2.0 * eps)
            derivatives.append(deriv)
        return np.column_stack(derivatives), base_vector

    def parameter_forecast(self, params: Sequence[str]) -> Dict[str, np.ndarray]:
        fisher_result = self.forecast()
        n_bins = len(fisher_result.z)
        cov_block = fisher_result.covariance_blocks
        block_dim = cov_block.shape[-1]
        cov_total = np.zeros((n_bins * block_dim, n_bins * block_dim))
        for idx in range(n_bins):
            start = idx * block_dim
            cov_total[start:start + block_dim, start:start + block_dim] = cov_block[idx]
        cov_inv = np.linalg.inv(cov_total)

        derivative_matrix, base_vector = self.derivative_matrix(params)
        fisher_params = derivative_matrix.T @ cov_inv @ derivative_matrix
        cov_params = np.linalg.inv(fisher_params)
        errors = np.sqrt(np.diag(cov_params))
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
) -> Dict[str, np.ndarray]:
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
    calculator = IntensityMappingFisher(survey, gamma=gamma)
    fisher_result = calculator.forecast()
    param_result = calculator.parameter_forecast(parameters)
    return {
        'survey': survey,
        'fisher_result': fisher_result,
        'parameter_result': param_result,
    }
