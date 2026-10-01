"""21cm intensity-mapping Fisher forecasts.

Implements the formalism of Bull et al. (2015, ApJ 803, 21):

1. In each redshift bin, the Fisher matrix of the observables
   p = (ln f sigma8, ln b sigma8, ln D_A, ln H, sigma_NL^2) is computed from the
   anisotropic HI power spectrum at the fiducial cosmology,

       C_S(k, mu) = T_b^2 (b sigma8 + f sigma8 mu^2)^2 exp(-k^2 mu^2 sigma_NL^2) P(k) / sigma8_0^2,

   using analytic derivatives of ln C_S (Alcock-Paczynski terms included) and
   the instrumental noise of a single-dish or cylinder-interferometer survey
   plus the residual foregrounds of Bull et al. (2015).
   b sigma8 and sigma_NL^2 are marginalised in each bin.
2. The covariance of (ln f sigma8, ln D_A, ln H) is projected onto cosmological
   parameters with the Jacobian obtained by ``jax.jacfwd`` through the
   background model, the linear-growth ODE and sigma8(z).

The fiducial linear P(k) is the Eisenstein & Hu (1998) spectrum with baryon
oscillations, normalised to sigma8; it enters only the per-bin covariance.

Survey configurations are YAML files in ``surveys/`` (schema_version 3.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ..utils.constants import c_km_s
from ..utils.logging import get_logger

logger = get_logger(__name__)

SURVEY_CONFIG_DIR = Path(__file__).resolve().parent / "surveys"
NU21_MHZ = 1420.405751
C_M_S = 299792458.0
DEG2_TO_SR = (np.pi / 180.0) ** 2
LN2 = np.log(2.0)

# Residual foregrounds after subtraction (Bull et al. 2015, Eq. 7 and Table 1,
# from Santos et al. 2005): extragalactic point sources, extragalactic
# free-free, Galactic synchrotron, Galactic free-free
FG_AMPLITUDE_MK2 = (57.0, 0.014, 700.0, 0.088)
FG_N = (1.1, 1.0, 2.4, 3.0)
FG_M = (-2.07, -2.10, -2.80, -2.15)
FG_L_PIVOT, FG_NU_PIVOT_MHZ = 1000.0, 130.0

# Observables kept after the per-bin marginalisation, in this order
OBSERVABLES = ("ln_fsigma8", "ln_DA", "ln_H")


# -----------------------------------------------------------------------------
# Survey configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RedshiftBin:
    z: float
    delta_z: float

    @property
    def z_min(self) -> float:
        return self.z - 0.5 * self.delta_z

    @property
    def z_max(self) -> float:
        return self.z + 0.5 * self.delta_z


@dataclass(frozen=True)
class IntensityMappingSurvey:
    """Hardware and observing strategy of an HI intensity-mapping survey.

    For ``telescope_type == "cylinder_array"``, ``n_receivers`` is the total
    number of feeds and ``dish_diameter_m`` the cylinder width.
    """

    name: str
    description: str
    telescope_type: str
    n_receivers: int
    dish_diameter_m: float
    n_beam: int
    aperture_efficiency: float
    t_inst_K: float
    survey_area_deg2: float
    total_time_hours: float
    nu_min_MHz: float
    nu_max_MHz: float
    channel_width_MHz: float
    n_pol: int
    bias_coefficients: Tuple[float, ...]
    omega_hi_coefficients: Tuple[float, ...]
    delta_z: float = 0.1
    n_cylinders: int = 0
    cylinder_length_m: float = 0.0
    min_baseline_m: float = 0.0

    @classmethod
    def from_file(cls, filename: Path | str) -> "IntensityMappingSurvey":
        with open(filename, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        version = str(data.get("metadata", {}).get("schema_version", ""))
        if version != "3.0":
            raise ValueError(
                f"{filename}: survey schema_version {version!r} is not supported (need 3.0)"
            )
        inst, obs, hi = data["instrument"], data["observing"], data["hi_tracers"]
        kind = inst["telescope_type"]
        if kind == "single_dish":
            extra = {"n_receivers": int(inst["ndish"]),
                     "dish_diameter_m": float(inst["dish_diameter_m"])}
        elif kind == "cylinder_array":
            width = float(inst["cylinder_width_m"])
            extra = {
                "n_receivers": int(inst["nfeeds"]),
                "dish_diameter_m": width,
                "n_cylinders": int(inst["ncylinders"]),
                "cylinder_length_m": float(inst["cylinder_length_m"]),
                "min_baseline_m": float(inst.get("min_baseline_m", width)),
            }
        else:
            raise ValueError(f"Unknown telescope_type: {kind}")
        nu_min, nu_max = (float(v) for v in obs["frequency_range_MHz"])
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            telescope_type=kind,
            n_beam=int(inst.get("nbeam", 1)),
            aperture_efficiency=float(inst.get("aperture_efficiency", 0.7)),
            t_inst_K=float(inst["receiver_temperature_K"]),
            survey_area_deg2=float(obs["survey_area_deg2"]),
            total_time_hours=float(obs["total_time_hours"]),
            nu_min_MHz=nu_min,
            nu_max_MHz=nu_max,
            channel_width_MHz=float(obs["channel_width_MHz"]),
            n_pol=int(obs.get("npol", 2)),
            bias_coefficients=tuple(float(c) for c in hi["bias_coefficients"]),
            omega_hi_coefficients=tuple(float(c) for c in hi["omega_hi_coefficients"]),
            delta_z=float(data.get("redshift_bins", {}).get("delta_z", 0.1)),
            **extra,
        )

    @property
    def z_min(self) -> float:
        return NU21_MHZ / self.nu_max_MHz - 1.0

    @property
    def z_max(self) -> float:
        return NU21_MHZ / self.nu_min_MHz - 1.0

    @property
    def redshift_bins(self) -> List[RedshiftBin]:
        """Equal-width bins covering [z_min, z_max] (as in Bull et al. 2015)."""
        n = max(1, int((self.z_max - self.z_min) / self.delta_z))
        edges = np.linspace(self.z_min, self.z_max, n + 1)
        width = edges[1] - edges[0]
        return [RedshiftBin(float(0.5 * (a + b)), float(width)) for a, b in zip(edges[:-1], edges[1:])]

    def bias(self, z):
        return sum(c * z**i for i, c in enumerate(self.bias_coefficients))

    def omega_hi(self, z):
        return sum(c * z**i for i, c in enumerate(self.omega_hi_coefficients))


def list_available_surveys(directory: Path | None = None) -> List[str]:
    directory = directory or SURVEY_CONFIG_DIR
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_survey(name_or_path: str) -> IntensityMappingSurvey:
    path = Path(name_or_path)
    if not path.exists():
        path = SURVEY_CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f'Survey "{name_or_path}" not found. Available: {", ".join(list_available_surveys())}'
        )
    return IntensityMappingSurvey.from_file(path)


def cylinder_baseline_density(survey: IntensityMappingSurvey, n_bins: int = 400):
    """Baseline density n_d(d) [per m^2] of a regular cylinder array.

    Feeds are placed on a regular grid: one column per cylinder (spaced by the
    cylinder width) and equally spaced along the cylinder. Baselines shorter
    than ``min_baseline_m`` are excluded, as in Bull et al. (2015). The density
    in the uv plane at wavelength lambda is n(u) = n_d(u lambda) lambda^2.
    """
    per_cyl = survey.n_receivers // survey.n_cylinders
    x = np.repeat(np.arange(survey.n_cylinders) * survey.dish_diameter_m, per_cyl)
    y = np.tile((np.arange(per_cyl) + 0.5) * survey.cylinder_length_m / per_cyl,
                survey.n_cylinders)
    i, j = np.triu_indices(x.size, 1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    d = d[d > survey.min_baseline_m]
    edges = np.linspace(0.0, d.max() * (1.0 + 1e-9), n_bins + 1)
    counts, _ = np.histogram(d, edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, counts / (2.0 * np.pi * centres * np.diff(edges))


# -----------------------------------------------------------------------------
# Linear matter power spectrum (Eisenstein & Hu 1998, with baryons)
# -----------------------------------------------------------------------------


def eh98_transfer(k, omega_m_h2, omega_b_h2, T_cmb=2.7255):
    """EH98 transfer function with baryon acoustic oscillations; k in Mpc^-1."""
    theta = T_cmb / 2.7
    f_b = omega_b_h2 / omega_m_h2
    f_c = 1.0 - f_b
    z_eq = 2.50e4 * omega_m_h2 * theta**-4
    k_eq = 7.46e-2 * omega_m_h2 * theta**-2
    b1 = 0.313 * omega_m_h2**-0.419 * (1.0 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    z_d = 1291.0 * omega_m_h2**0.251 / (1.0 + 0.659 * omega_m_h2**0.828) * (1.0 + b1 * omega_b_h2**b2)
    R = lambda z: 31.5 * omega_b_h2 * theta**-4 * (1000.0 / z)
    R_d, R_eq = R(z_d), R(z_eq)
    s = (2.0 / (3.0 * k_eq)) * jnp.sqrt(6.0 / R_eq) * jnp.log(
        (jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq))
    )
    k_silk = 1.6 * omega_b_h2**0.52 * omega_m_h2**0.73 * (1.0 + (10.4 * omega_m_h2) ** -0.95)
    a1 = (46.9 * omega_m_h2) ** 0.670 * (1.0 + (32.1 * omega_m_h2) ** -0.532)
    a2 = (12.0 * omega_m_h2) ** 0.424 * (1.0 + (45.0 * omega_m_h2) ** -0.582)
    alpha_c = a1**-f_b * a2 ** (-(f_b**3))
    bb1 = 0.944 / (1.0 + (458.0 * omega_m_h2) ** -0.708)
    bb2 = (0.395 * omega_m_h2) ** -0.0266
    beta_c = 1.0 / (1.0 + bb1 * (f_c**bb2 - 1.0))

    def t0(kk, alpha, beta):
        qq = kk / (13.41 * k_eq)
        C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * qq**1.08)
        L = jnp.log(jnp.e + 1.8 * beta * qq)
        return L / (L + C * qq**2)

    f = 1.0 / (1.0 + (k * s / 5.4) ** 4)
    T_c = f * t0(k, 1.0, beta_c) + (1.0 - f) * t0(k, alpha_c, beta_c)

    y = (1.0 + z_eq) / (1.0 + z_d)
    G = y * (-6.0 * jnp.sqrt(1.0 + y) + (2.0 + 3.0 * y) * jnp.log(
        (jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0)))
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** -0.75 * G
    beta_b = 0.5 + f_b + (3.0 - 2.0 * f_b) * jnp.sqrt((17.2 * omega_m_h2) ** 2 + 1.0)
    beta_node = 8.41 * omega_m_h2**0.435
    s_tilde = s / (1.0 + (beta_node / (k * s)) ** 3) ** (1.0 / 3.0)
    T_b = (t0(k, 1.0, 1.0) / (1.0 + (k * s / 5.2) ** 2)
           + alpha_b / (1.0 + (beta_b / (k * s)) ** 3) * jnp.exp(-((k / k_silk) ** 1.4))
           ) * jnp.sinc(k * s_tilde / jnp.pi)
    return f_b * T_b + f_c * T_c


def linear_power_z0(params: Dict[str, jnp.ndarray]):
    """Return P(k) [Mpc^3] at z = 0, normalised to sigma8, as a function of k [Mpc^-1]."""
    h = params["H0"] / 100.0
    om_h2 = params["Omega_m"] * h**2
    ob_h2 = params["Omega_b"] * h**2
    n_s, T_cmb = params["n_s"], params.get("T_cmb", 2.7255)

    def shape(k):
        return k**n_s * eh98_transfer(k, om_h2, ob_h2, T_cmb) ** 2

    lnk = jnp.linspace(jnp.log(1e-5), jnp.log(1e2), 4096)
    k = jnp.exp(lnk)
    x = k * 8.0 / h
    w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
    sigma2_unit = jnp.trapezoid(k**3 * shape(k) * w**2, lnk) / (2.0 * jnp.pi**2)
    amplitude = params["sigma8"] ** 2 / sigma2_unit
    return lambda kk: amplitude * shape(kk)


# -----------------------------------------------------------------------------
# Background and growth (traceable)
# -----------------------------------------------------------------------------


def growth(model_cls, params, z, n_steps: int = 1000, a_init: float = 1e-2):
    """Linear growth D(z) (D(0) = 1) and f(z) = dln D/dln a from the ODE

        D'' + (2 + dln E/dln a) D' - 3/2 Omega_m(a) D = 0,   ' = d/dln a,

    integrated with fixed-step RK4 from matter domination (D = D' = a)."""
    x = jnp.linspace(jnp.log(a_init), 0.0, 2 * n_steps + 1)
    zz = jnp.exp(-x) - 1.0
    lnE = jnp.log(model_cls._E_z_static(zz, params))
    dlnE = jnp.gradient(lnE, x)
    om_a = params["Omega_m"] * jnp.exp(-3.0 * x) / jnp.exp(2.0 * lnE)
    h = 2.0 * (x[1] - x[0])

    def rhs(y, i):
        D, dD = y
        return jnp.stack([dD, -(2.0 + dlnE[i]) * dD + 1.5 * om_a[i] * D])

    def step(y, i):
        k1 = rhs(y, 2 * i)
        k2 = rhs(y + 0.5 * h * k1, 2 * i + 1)
        k3 = rhs(y + 0.5 * h * k2, 2 * i + 1)
        k4 = rhs(y + h * k3, 2 * i + 2)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y, y

    y0 = jnp.array([a_init, a_init])
    _, ys = jax.lax.scan(step, y0, jnp.arange(n_steps))
    ys = jnp.vstack([y0, ys])
    lna = x[::2]
    D, dD = ys[:, 0], ys[:, 1]
    lna_z = -jnp.log1p(z)
    D_z = jnp.interp(lna_z, lna, D) / D[-1]
    f_z = jnp.interp(lna_z, lna, dD / D)
    return D_z, f_z


def observable_vector(model_cls, params, z) -> jnp.ndarray:
    """(ln f sigma8, ln D_A, ln H) at the bin centres z, flattened bin by bin."""
    params = model_cls.normalize_params(params)
    z_grid = jnp.linspace(0.0, jnp.max(z) + 0.1, 4096)
    grid = model_cls.compute_grid_traced(z_grid, params)
    D_A = jnp.interp(z, z_grid, grid["D_M"]) / (1.0 + z)
    H = params["H0"] * model_cls._E_z_static(z, params)
    D, f = growth(model_cls, params, z)
    fsigma8 = f * params["sigma8"] * D
    return jnp.stack([jnp.log(fsigma8), jnp.log(D_A), jnp.log(H)], axis=1).reshape(-1)


# -----------------------------------------------------------------------------
# Per-bin observable Fisher matrix
# -----------------------------------------------------------------------------


@dataclass
class FisherResult:
    """Per-bin covariance of (ln f sigma8, ln D_A, ln H), b sigma8 and sigma_NL marginalised."""

    z: np.ndarray
    covariance_blocks: np.ndarray
    sigma_ln_fsigma8: np.ndarray
    sigma_ln_DA: np.ndarray
    sigma_ln_H: np.ndarray


class IntensityMappingFisher:
    """Fisher forecast for an HI intensity-mapping survey.

    >>> survey = load_survey("ska1_wide_band1")
    >>> fisher = IntensityMappingFisher(survey, wCDM())
    >>> result = fisher.parameter_forecast(["w0"], marginalize_over=["H0", "Omega_m"])
    """

    K_MIN = 1e-4  # Mpc^-1; modes below the foreground cut carry no weight anyway
    N_K = 600
    N_MU = 600
    SIGMA_NL = 7.0  # Mpc, fiducial (marginalised)
    EPSILON_FG = 1e-6  # residual foreground amplitude (Bull et al. 2015)
    K_NL0 = 0.14  # Mpc^-1

    def __init__(self, survey: IntensityMappingSurvey, cosmology) -> None:
        self.survey = survey
        self.cosmology = cosmology
        self.model_cls = type(cosmology)
        self.fiducial = {k: float(v) for k, v in cosmology.params._given.items()}
        self._baselines = (
            cylinder_baseline_density(survey) if survey.telescope_type == "cylinder_array" else None
        )

    # --- signal and noise ----------------------------------------------------

    def _brightness_temperature_mK(self, z, E_z):
        """Mean HI brightness temperature, T_b = 188 h Omega_HI(z) (1+z)^2 / E(z) mK."""
        h = self.fiducial["H0"] / 100.0
        return 188.0 * h * self.survey.omega_hi(z) * (1.0 + z) ** 2 / E_z

    def _noise_power(self, z, r, r_nu, k_perp, k_par):
        """Noise power spectrum [mK^2 Mpc^3] on the (k_perp, k_par) grid."""
        s = self.survey
        nu = NU21_MHZ / (1.0 + z)
        lam = C_M_S / (nu * 1e6)
        t_sky = 60e3 * (300.0 / nu) ** 2.55
        t_sys = s.t_inst_K * 1e3 + t_sky
        t_tot = s.total_time_hours * 3600.0
        area = s.survey_area_deg2 * DEG2_TO_SR
        base = t_sys**2 * area / (s.n_pol * t_tot * NU21_MHZ * 1e6)
        if s.telescope_type == "single_dish":
            A_e = s.aperture_efficiency * 0.25 * np.pi * s.dish_diameter_m**2
            theta = lam / s.dish_diameter_m
            base *= lam**4 / (A_e**2 * theta**4) / (s.n_receivers * s.n_beam)
            inv_beam = np.exp(np.minimum((k_perp * r * theta) ** 2 / (8.0 * LN2), 700.0))
        else:
            A_e = (s.aperture_efficiency * s.n_cylinders * s.cylinder_length_m
                   * s.dish_diameter_m / s.n_receivers)
            fov = 0.5 * np.pi * lam / s.dish_diameter_m
            base *= lam**4 / (s.n_beam * A_e**2 * fov)
            d_centres, n_d = self._baselines
            n_u = np.interp(k_perp * r / (2.0 * np.pi) * lam, d_centres, n_d,
                            left=0.0, right=0.0) * lam**2
            inv_beam = np.where(n_u > 0.0, 1.0 / np.where(n_u > 0.0, n_u, 1.0), np.inf)
        inv_channel = np.exp(np.minimum(
            (k_par * r_nu * s.channel_width_MHz / NU21_MHZ) ** 2 / (8.0 * LN2), 700.0))
        return base * inv_beam * inv_channel * r**2 * r_nu

    def _foreground_power(self, z, r, r_nu, k_perp):
        """Residual foreground power [mK^2 Mpc^3], eps_fg^2 sum_X A_X (l_p/2 pi q)^n_X (nu/nu_p)^m_X."""
        nu = NU21_MHZ / (1.0 + z)
        q = np.maximum(k_perp * r, 1e-100)
        c_f = sum(a * (FG_L_PIVOT / (2.0 * np.pi * q)) ** n * (nu / FG_NU_PIVOT_MHZ) ** m
                  for a, n, m in zip(FG_AMPLITUDE_MK2, FG_N, FG_M))
        return self.EPSILON_FG**2 * c_f * r**2 * r_nu

    def _bin_fisher(self, bin_cfg: RedshiftBin, pk, dlnpk_dlnk, D_z, f_z, E_z, r):
        """5x5 Fisher matrix of (ln fs8, ln bs8, ln D_A, ln H, sigma_NL^2) in one bin."""
        s = self.survey
        z = bin_cfg.z
        h = self.fiducial["H0"] / 100.0
        r_nu = c_km_s * (1.0 + z) ** 2 / (self.fiducial["H0"] * E_z)
        dnu_bin = NU21_MHZ / (1.0 + bin_cfg.z_min) - NU21_MHZ / (1.0 + bin_cfg.z_max)
        volume = s.survey_area_deg2 * DEG2_TO_SR * dnu_bin / NU21_MHZ * r**2 * r_nu

        k_max = self.K_NL0 * (1.0 + z) ** (2.0 / (2.0 + self.fiducial["n_s"]))
        k_fg = 2.0 * np.pi * NU21_MHZ / ((s.nu_max_MHz - s.nu_min_MHz) * r_nu)
        k = np.logspace(np.log10(self.K_MIN), np.log10(k_max), self.N_K)
        mu = np.linspace(0.0, 1.0, self.N_MU)
        K, MU = np.meshgrid(k, mu, indexing="ij")
        mu2 = MU**2
        k_par, k_perp = K * MU, K * np.sqrt(1.0 - mu2)

        sigma8_z = self.fiducial["sigma8"] * D_z
        fs8 = f_z * sigma8_z
        bs8 = s.bias(z) * sigma8_z
        beta = fs8 / bs8
        sig2 = self.SIGMA_NL**2
        T_b = self._brightness_temperature_mK(z, E_z)
        P = np.asarray(pk(jnp.asarray(k)))[:, None]
        dlnP = np.asarray(dlnpk_dlnk(jnp.asarray(k)))[:, None]

        C_S = T_b**2 * (bs8 + fs8 * mu2) ** 2 * np.exp(-K**2 * mu2 * sig2) * P / self.fiducial["sigma8"] ** 2
        C_N = self._noise_power(z, r, r_nu, k_perp, k_par) + self._foreground_power(z, r, r_nu, k_perp)
        weight = (C_S / (C_S + C_N)) ** 2
        weight = np.where((k_par < k_fg) | ~np.isfinite(C_N), 0.0, weight)

        kaiser = 2.0 * beta / (1.0 + beta * mu2) - K**2 * sig2  # dln F / d mu^2
        dlnk_term = dlnP - 2.0 * K**2 * mu2 * sig2  # dln (F P) / dln k at fixed mu^2
        derivs = np.stack([
            2.0 * beta * mu2 / (1.0 + beta * mu2),
            2.0 / (1.0 + beta * mu2),
            -2.0 + kaiser * 2.0 * mu2 * (1.0 - mu2) - (1.0 - mu2) * dlnk_term,
            1.0 + kaiser * 2.0 * mu2 * (1.0 - mu2) + mu2 * dlnk_term,
            -K**2 * mu2,
        ])
        integrand = derivs[:, None] * derivs[None, :] * weight * K**2
        # mu in [0, 1] covers half of [-1, 1]: factor 2
        F = 2.0 * np.trapz(np.trapz(integrand, mu, axis=-1), k, axis=-1)
        return volume / (8.0 * np.pi**2) * F

    # --- public API ------------------------------------------------------------

    def compute_observable_constraints(self) -> FisherResult:
        bins = self.survey.redshift_bins
        z = np.array([b.z for b in bins])
        params = self.model_cls.normalize_params(self.fiducial)
        pk = jax.jit(jax.vmap(linear_power_z0(params)))
        lnpk = lambda lnk: jnp.log(linear_power_z0(params)(jnp.exp(lnk)))
        dlnpk = jax.jit(lambda k: jax.vmap(jax.grad(lnpk))(jnp.log(k)))
        D_z, f_z = (np.asarray(v) for v in growth(self.model_cls, params, jnp.asarray(z)))
        E_z = np.asarray(self.model_cls._E_z_static(jnp.asarray(z), params))
        z_grid = jnp.linspace(0.0, z.max() + 0.1, 4096)
        r = np.interp(z, np.asarray(z_grid),
                      np.asarray(self.model_cls.compute_grid_traced(z_grid, params)["D_M"]))

        blocks = []
        for i, b in enumerate(bins):
            F = self._bin_fisher(b, pk, dlnpk, D_z[i], f_z[i], E_z[i], r[i])
            cov = np.linalg.inv(F)
            keep = [0, 2, 3]  # (ln fs8, ln D_A, ln H); b sigma8 and sigma_NL marginalised
            blocks.append(cov[np.ix_(keep, keep)])
        blocks = np.array(blocks)
        err = np.sqrt(np.diagonal(blocks, axis1=1, axis2=2))
        return FisherResult(z, blocks, err[:, 0], err[:, 1], err[:, 2])

    def jacobian(self, params: Sequence[str]) -> np.ndarray:
        """d(observables)/d(params) by forward-mode automatic differentiation."""
        z = jnp.asarray([b.z for b in self.survey.redshift_bins])
        base = dict(self.fiducial)

        def obs(theta):
            p = {**base, **dict(zip(params, theta))}
            return observable_vector(self.model_cls, p, z)

        theta0 = jnp.asarray([self.fiducial[p] for p in params])
        return np.asarray(jax.jacfwd(obs)(theta0))

    def parameter_forecast(
        self,
        params: Sequence[str],
        marginalize_over: Optional[Sequence[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Marginalised constraints on ``params`` (additional ``marginalize_over`` free)."""
        all_params = list(params) + list(marginalize_over or [])
        missing = [p for p in all_params if p not in self.fiducial]
        if missing:
            raise ValueError(f"Parameters {missing} not in cosmology; available {list(self.fiducial)}")
        obs = self.compute_observable_constraints()
        n = len(obs.z)
        inv_cov = np.zeros((3 * n, 3 * n))
        for i, block in enumerate(obs.covariance_blocks):
            inv_cov[3 * i:3 * i + 3, 3 * i:3 * i + 3] = np.linalg.inv(block)
        J = self.jacobian(all_params)
        fisher_full = J.T @ inv_cov @ J
        cov_full = np.linalg.inv(fisher_full)
        m = len(params)
        cov = cov_full[:m, :m]
        if verbose:
            logger.info("%s: %s", self.survey.name, dict(zip(params, np.sqrt(np.diag(cov)))))
        return {
            "params": list(params),
            "fisher_full": fisher_full,
            "fisher": np.linalg.inv(cov),
            "covariance": cov,
            "errors": np.sqrt(np.diag(cov)),
            "observables": obs,
        }


def run_forecast(survey_name: str, cosmology, params: Sequence[str],
                 marginalize_over: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """One-shot forecast: load a survey and return ``parameter_forecast`` output."""
    return IntensityMappingFisher(load_survey(survey_name), cosmology).parameter_forecast(
        params, marginalize_over=marginalize_over
    )
