"""Combined-likelihood shared-grid regression tests."""

from __future__ import annotations

import jax.numpy as jnp

from hicosmo.likelihoods.combined import CombinedLikelihood


class CountingCosmology:
    calls = 0

    @staticmethod
    def _E_z_static(z_grid, params):
        return jnp.ones_like(z_grid)

    @staticmethod
    def normalize_params(params, dtype=None):
        return dict(params)

    @staticmethod
    def compute_grid_traced(z_grid, params):
        CountingCosmology.calls += 1
        return {"d_L": jnp.ones_like(z_grid), "E_z": jnp.ones_like(z_grid)}


class GridCapableLikelihood:
    _cosmology_class = CountingCosmology

    def __init__(self, value):
        self.value = value
        self._z_grid = jnp.linspace(0.0, 1.0, 8)
        self.call_count = 0
        self.grid_count = 0
        self._loglike_from_grid = self._from_grid

    def _prepare_params_dict(self, params):
        return dict(params)

    def _from_grid(self, cosmo_grid, z_grid, params):
        self.grid_count += 1
        return jnp.asarray(self.value)

    def __call__(self, **params):
        self.call_count += 1
        return jnp.asarray(self.value)


class PlainLowZLikelihood:
    _cosmology_class = CountingCosmology

    def __init__(self, value):
        self.value = value
        self._z_grid = jnp.linspace(0.0, 1.0, 8)
        self.call_count = 0

    def __call__(self, **params):
        self.call_count += 1
        return jnp.asarray(self.value)


class LightweightCosmology:
    full_calls = 0
    background_calls = 0

    @staticmethod
    def _E_z_static(z_grid, params):
        return jnp.ones_like(z_grid)

    @staticmethod
    def normalize_params(params, dtype=None):
        return dict(params)

    @staticmethod
    def compute_grid_traced(z_grid, params):
        LightweightCosmology.full_calls += 1
        return {
            "d_L": jnp.ones_like(z_grid),
            "D_M": jnp.ones_like(z_grid),
            "D_H": jnp.ones_like(z_grid),
            "E_z": jnp.ones_like(z_grid),
            "dVc_dz": jnp.ones_like(z_grid),
            "ddL_dz": jnp.ones_like(z_grid),
        }

    @staticmethod
    def compute_background_grid_traced(z_grid, params):
        LightweightCosmology.background_calls += 1
        return {
            "d_L": jnp.ones_like(z_grid),
            "D_M": jnp.ones_like(z_grid),
            "D_H": jnp.ones_like(z_grid),
            "E_z": jnp.ones_like(z_grid),
        }


class LightweightGridLikelihood(GridCapableLikelihood):
    _cosmology_class = LightweightCosmology


def test_shared_grid_computes_once_for_two_grid_capable_low_z_likelihoods():
    CountingCosmology.calls = 0
    first = GridCapableLikelihood(1.0)
    second = GridCapableLikelihood(2.0)

    combined = CombinedLikelihood([first, second])
    result = combined(H0=70.0, Omega_m=0.3)

    assert float(result) == 3.0
    assert CountingCosmology.calls == 1
    assert first.grid_count == 1
    assert second.grid_count == 1
    assert first.call_count == 0
    assert second.call_count == 0


def test_shared_grid_requires_two_grid_capable_low_z_likelihoods():
    CountingCosmology.calls = 0
    grid_capable = GridCapableLikelihood(1.0)
    plain = PlainLowZLikelihood(2.0)

    combined = CombinedLikelihood([grid_capable, plain])
    result = combined(H0=70.0, Omega_m=0.3)

    assert float(result) == 3.0
    assert CountingCosmology.calls == 0
    assert grid_capable.grid_count == 0
    assert grid_capable.call_count == 1
    assert plain.call_count == 1


def test_shared_grid_prefers_lightweight_background_grid():
    LightweightCosmology.full_calls = 0
    LightweightCosmology.background_calls = 0
    first = LightweightGridLikelihood(1.0)
    second = LightweightGridLikelihood(2.0)

    combined = CombinedLikelihood([first, second])
    result = combined(H0=70.0, Omega_m=0.3)

    assert float(result) == 3.0
    assert LightweightCosmology.background_calls == 1
    assert LightweightCosmology.full_calls == 0


def test_registered_models_provide_lightweight_background_grid():
    from hicosmo.models import CPL, LCDM

    z_grid = jnp.linspace(0.0, 2.0, 64)
    params = {"H0": 70.0, "Omega_m": 0.3, "Omega_b": 0.05, "w0": -1.0, "wa": 0.0}

    for model_class in (LCDM, CPL):
        grid = model_class.compute_background_grid_traced(z_grid, params)
        assert {"d_L", "D_M", "D_H", "E_z", "d_C"} <= set(grid)
        assert "dVc_dz" not in grid
        assert "ddL_dz" not in grid


def test_fixed_mb_wrapper_honored_on_shared_grid_path():
    """Regression: CombinedLikelihood fetched the base closure via __getattr__,
    silently replacing the user's fixed M_B with the closure default (-19.3)."""
    from hicosmo.likelihoods import BAO_likelihood, SN_likelihood
    from hicosmo.models import LCDM

    params = {"H0": 70.0, "Omega_m": 0.31}
    sn_fixed = SN_likelihood(LCDM, "pantheon+", M_B=-19.5)
    bao = BAO_likelihood(LCDM, "desi_dr2", verbose=False)
    combined = sn_fixed + bao
    assert combined._shared_grid_enabled

    v_combined = float(combined(**params, H0_rd=101.0))
    v_sum = float(sn_fixed(**params)) + float(bao(**params, H0_rd=101.0))
    v_old_bug = float(sn_fixed._base(M_B=-19.3, **params)) + float(
        bao(**params, H0_rd=101.0)
    )

    # Shared grid is denser than the SN-native grid; without analytic
    # marginalization the interpolation offset is ~3e-2 in logL, well below
    # the ~580 gap that the old silently-dropped-M_B bug produced.
    assert abs(v_combined - v_sum) < 0.1
    assert abs(v_combined - v_old_bug) > 10.0


# =====================================================================
# Shared-grid consistency matrix: for every likelihood combination and
# construction mode, the CombinedLikelihood shared-grid path must agree
# with the sum of independently evaluated likelihoods — in both logL
# and gradient (the gradient is what NUTS actually consumes).
# =====================================================================

import pytest


def _consistency_case(model_cls, lik_specs, params, logl_tol):
    import jax

    from hicosmo.api_registry import combine_likelihoods, resolve_likelihoods

    liks = resolve_likelihoods(model_cls, lik_specs)
    combined = combine_likelihoods(liks)
    assert combined._shared_grid_enabled

    v_combined = float(combined(**params))
    v_sum = sum(float(lik(**params)) for lik in liks)
    assert (
        abs(v_combined - v_sum) < logl_tol
    ), f"logL mismatch: combined={v_combined:.6f} sum={v_sum:.6f}"

    keys = list(params.keys())

    def loss(fn):
        return lambda th: -fn(**{k: th[i] for i, k in enumerate(keys)})

    th0 = jnp.array([params[k] for k in keys])
    g_combined = jax.grad(loss(combined))(th0)
    g_sum = jax.grad(lambda th: sum(loss(lik)(th) for lik in liks))(th0)
    rel = jnp.abs(g_combined - g_sum) / (jnp.abs(g_sum) + 1e-30)
    assert float(jnp.max(rel)) < 1e-4, f"gradient mismatch: rel={rel}"


def test_shared_grid_consistency_production_combo():
    """pantheon+(marginalized) + DESI DR2(h0rd) + Planck CMB on LCDM."""
    from hicosmo.models import LCDM

    _consistency_case(
        LCDM,
        [{"name": "sn"}, {"name": "bao", "dataset": "desi_dr2"}, "planck2018"],
        {"H0": 70.0, "Omega_m": 0.31, "H0_rd": 101.0},
        logl_tol=5e-3,
    )


def test_shared_grid_consistency_wcdm():
    """Same production combo on wCDM with w0 free."""
    from hicosmo.models.wcdm import wCDM

    _consistency_case(
        wCDM,
        [{"name": "sn"}, {"name": "bao", "dataset": "desi_dr2"}, "planck2018"],
        {"H0": 70.0, "Omega_m": 0.31, "w0": -1.05, "H0_rd": 101.0},
        logl_tol=5e-3,
    )


def test_shared_grid_consistency_union3_bbn_prior():
    """union3 + DESI DR2 in bbn_prior mode (the 0/1/2-count regression combo)."""
    from hicosmo.models import LCDM

    _consistency_case(
        LCDM,
        [
            {"name": "sn", "dataset": "union3"},
            {"name": "bao", "dataset": "desi_dr2", "omega_b_mode": "bbn_prior"},
        ],
        {"H0": 70.0, "Omega_m": 0.31, "Omega_b": 0.049},
        logl_tol=5e-3,
    )


def test_shared_grid_consistency_desy5():
    """DESY5(marginalized) + DESI DR2 on LCDM."""
    from hicosmo.models import LCDM

    _consistency_case(
        LCDM,
        [{"name": "sn", "dataset": "desy5"}, {"name": "bao", "dataset": "desi_dr2"}],
        {"H0": 70.0, "Omega_m": 0.31, "H0_rd": 101.0},
        logl_tol=5e-3,
    )
