"""Tests for DESI DR2 BAO likelihood implementation."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def _init_hicosmo():
    """Initialize HIcosmo once for all tests in this module."""
    import hicosmo as hc

    hc.init(1)


def test_desi_dr2_instantiation():
    """DESIDR2BAO loads 13 data points with 13x13 covariance."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    assert bao.dataset.name == "DESI DR2 BAO"
    assert len(bao.dataset.data_points) == 13
    assert bao.dataset.covariance.shape == (13, 13)
    assert bao.dataset.year == 2025
    assert bao.dataset.reference == "DESI Collaboration 2025"


def test_desi_dr2_redshift_bins():
    """DR2 has 7 effective redshift bins with correct z values."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    z_values = sorted(set(p.z for p in bao.dataset.data_points))
    expected_z = [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330]
    np.testing.assert_allclose(z_values, expected_z, atol=1e-3)


def test_desi_dr2_observables():
    """DR2 has DV/rd, DM/rd, and DH/rd observables."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    observables = set(p.observable for p in bao.dataset.data_points)
    assert observables == {"DV_over_rd", "DM_over_rd", "DH_over_rd"}


def test_desi_dr2_evaluation():
    """Likelihood evaluates to a finite number at a reasonable cosmology."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    log_L = bao(H0=67.36, Omega_m=0.3153, H0_rd=101.0)
    assert np.isfinite(float(log_L))
    assert float(log_L) < 0  # log-likelihood should be negative


def test_desi_dr2_aliases():
    """All registered aliases resolve to DESIDR2BAO."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    for alias in ["desi_dr2", "desidr2", "desi2025", "desi_y3"]:
        bao = BAO_likelihood(LCDM, alias)
        assert bao.dataset.name == "DESI DR2 BAO", f"Alias '{alias}' failed"


def test_desi_dr2_inherits_from_dr1():
    """DESIDR2BAO inherits from DESI2024BAO."""
    from hicosmo.likelihoods.bao.datasets import DESIDR2BAO, DESI2024BAO

    assert issubclass(DESIDR2BAO, DESI2024BAO)


def test_desi_dr2_omega_b_modes():
    """All omega_b_mode options work with DR2."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    # h0rd mode (default)
    bao = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode="h0rd")
    log_L = bao(H0=67.36, Omega_m=0.3, H0_rd=101.0)
    assert np.isfinite(float(log_L))

    # bbn_prior mode
    bao = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode="bbn_prior")
    log_L = bao(H0=67.36, Omega_m=0.3, Omega_b=0.0493)
    assert np.isfinite(float(log_L))

    # fixed mode
    bao = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode="fixed")
    log_L = bao(H0=67.36, Omega_m=0.3)
    assert np.isfinite(float(log_L))


def test_desi_dr2_fast_path_active():
    """JIT-compiled fast likelihood path is available."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    assert bao._loglike_fast is not None
    assert hasattr(bao, "_loglike_from_grid") and bao._loglike_from_grid is not None


def test_desi_dr2_fast_path_uses_background_grid():
    """BAO fast path should not require the heavier full GW-style grid."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    class BackgroundOnlyLCDM(LCDM):
        @staticmethod
        def compute_grid_traced(z_grid, params):
            raise AssertionError("full grid should not be used by BAO fast path")

        @staticmethod
        def compute_background_grid_traced(z_grid, params):
            return LCDM.compute_background_grid_traced(z_grid, params)

    bao = BAO_likelihood(BackgroundOnlyLCDM, "desi_dr2")
    assert bao._cosmology_class is BackgroundOnlyLCDM
    log_L = bao(H0=67.36, Omega_m=0.3, H0_rd=101.0)
    assert np.isfinite(float(log_L))


def test_desi_dr2_covariance_invertible():
    """DR2 covariance matrix is positive definite and well-conditioned."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    bao = BAO_likelihood(LCDM, "desi_dr2")
    cov = bao.dataset.covariance
    assert np.all(np.linalg.eigvalsh(cov) > 0), "Covariance not positive definite"
    cond = np.linalg.cond(cov)
    assert cond < 1e4, f"Covariance condition number too large: {cond}"


def test_desi_dr2_differs_from_dr1():
    """DR2 and DR1 produce different likelihood values."""
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM

    params = dict(H0=67.36, Omega_m=0.3, H0_rd=101.0)
    bao_dr1 = BAO_likelihood(LCDM, "desi2024")
    bao_dr2 = BAO_likelihood(LCDM, "desi_dr2")
    log_L_dr1 = float(bao_dr1(**params))
    log_L_dr2 = float(bao_dr2(**params))
    assert log_L_dr1 != log_L_dr2, "DR1 and DR2 should give different log_L"


def test_bbn_prior_counted_exactly_once_on_all_paths():
    """Regression: BBN prior was counted 0/1/2 times depending on call path."""
    import jax.numpy as jnp
    import numpy as np

    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM
    from hicosmo.models.base import compute_background_grid_for_model

    params = {"H0": 70.0, "Omega_m": 0.31, "Omega_b": 0.049}

    bbn = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode="bbn_prior", verbose=False)
    free = BAO_likelihood(LCDM, "desi_dr2", omega_b_mode="free", verbose=False)

    v_call = float(bbn(**params))
    v_params = float(bbn.log_likelihood_from_params(params))

    z_grid = jnp.linspace(0.0, bbn._z_grid_max, bbn._n_grid)
    pj = bbn._prepare_params_dict(params)
    grid = compute_background_grid_for_model(LCDM, z_grid, pj)
    v_grid = float(bbn._loglike_from_grid(grid, z_grid, pj))

    h = 0.70
    obh2 = 0.049 * h * h
    prior = -0.5 * ((obh2 - bbn.BBN_OMEGA_B_H2) / bbn.BBN_OMEGA_B_H2_ERR) ** 2 - np.log(
        bbn.BBN_OMEGA_B_H2_ERR * np.sqrt(2 * np.pi)
    )
    expected = float(free(**params)) + prior

    assert abs(v_call - v_params) < 1e-9
    assert abs(v_call - v_grid) < 1e-9
    assert abs(v_call - expected) < 1e-9
