"""
Tests for Union3 SN Ia Likelihood
=================================

Validates the Union3/UNITY1.5 compressed Gaussian likelihood implementation
against known results from Rubin et al. 2023 (arXiv:2311.12098).
"""

from __future__ import annotations

import pytest
import numpy as np


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
@pytest.fixture(scope="module")
def _init_hicosmo():
    """Initialize HIcosmo once for the module."""
    import hicosmo as hc

    hc.init(1)


@pytest.fixture
def union3(_init_hicosmo):
    """Create Union3 likelihood with LCDM."""
    from hicosmo.likelihoods.sn.union3 import Union3Likelihood
    from hicosmo.models import LCDM

    return Union3Likelihood(cosmology_class=LCDM)


@pytest.fixture
def union3_wcdm(_init_hicosmo):
    """Create Union3 likelihood with wCDM."""
    from hicosmo.likelihoods.sn.union3 import Union3Likelihood
    from hicosmo.models import wCDM

    return Union3Likelihood(cosmology_class=wCDM)


# ----------------------------------------------------------------
# Data loading and structure
# ----------------------------------------------------------------
class TestDataLoading:
    def test_data_shape(self, union3):
        assert union3.n_bins == 22

    def test_redshift_range(self, union3):
        z_min = float(union3.redshifts.min())
        z_max = float(union3.redshifts.max())
        assert z_min == pytest.approx(0.05, abs=0.01)
        assert z_max == pytest.approx(2.26, abs=0.01)

    def test_cov_inv_shape(self, union3):
        assert union3.cov_inv.shape == (22, 22)

    def test_cov_inv_symmetric(self, union3):
        cov_inv = np.array(union3.cov_inv)
        np.testing.assert_allclose(cov_inv, cov_inv.T, atol=1e-10)

    def test_mu_reasonable_range(self, union3):
        mu = np.array(union3.mu_obs)
        assert mu.min() > 35.0
        assert mu.max() < 50.0

    def test_info(self, union3):
        info = union3.get_info()
        assert info["name"] == "Union3"
        assert info["n_bins"] == 22
        assert info["n_sne"] == 2087
        assert info["standardization"] == "UNITY1.5"

    def test_repr(self, union3):
        r = repr(union3)
        assert "Union3" in r
        assert "2087" in r
        assert "22" in r


# ----------------------------------------------------------------
# Likelihood evaluation
# ----------------------------------------------------------------
class TestLikelihood:
    def test_callable_interface(self, union3):
        log_L = float(union3(H0=70.0, Omega_m=0.3))
        assert np.isfinite(log_L)
        assert log_L < 0  # log-likelihood should be negative

    def test_model_interface(self, union3):
        from hicosmo.models import LCDM

        model = LCDM(H0=70.0, Omega_m=0.3)
        log_L = float(union3.log_likelihood(model))
        assert np.isfinite(log_L)

    def test_chi2_consistency(self, union3):
        """chi2 should equal -2 * log_likelihood."""
        from hicosmo.models import LCDM

        model = LCDM(H0=70.0, Omega_m=0.3)
        log_L = float(union3.log_likelihood(model))
        chi2 = union3.chi2(model)
        assert chi2 == pytest.approx(-2 * log_L, abs=1e-6)

    def test_callable_matches_model_interface(self, union3):
        """__call__ and log_likelihood should agree."""
        from hicosmo.models import LCDM

        model = LCDM(H0=70.0, Omega_m=0.3)
        log_L_call = float(union3(H0=70.0, Omega_m=0.3))
        log_L_model = float(union3.log_likelihood(model))
        assert log_L_call == pytest.approx(log_L_model, abs=1e-6)

    def test_different_params_give_different_results(self, union3):
        log_L1 = float(union3(H0=70.0, Omega_m=0.3))
        log_L2 = float(union3(H0=70.0, Omega_m=0.5))
        assert log_L1 != log_L2

    def test_normalization_constant(self, union3):
        norm = union3.get_normalization_constant()
        assert np.isfinite(norm)

    def test_num_data(self, union3):
        assert union3.get_num_data() == 22


# ----------------------------------------------------------------
# No M_B nuisance
# ----------------------------------------------------------------
class TestNoMB:
    def test_no_nuisance_parameters(self, union3):
        assert len(union3.nuisance_parameters) == 0

    def test_m_b_not_required(self, union3):
        """Union3 should work without M_B."""
        log_L = float(union3(H0=70.0, Omega_m=0.3))
        assert np.isfinite(log_L)


# ----------------------------------------------------------------
# Factory and aliases
# ----------------------------------------------------------------
class TestFactory:
    def test_factory_union3(self, _init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        lik = SN_likelihood(LCDM, "union3")
        assert "Union3" in repr(lik)

    def test_factory_union31_alias(self, _init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        lik = SN_likelihood(LCDM, "union31")
        assert "Union3" in repr(lik)

    def test_factory_union3_dot1_alias(self, _init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        lik = SN_likelihood(LCDM, "union3.1")
        assert "Union3" in repr(lik)

    def test_factory_ignores_mb_for_union3(self, _init_hicosmo):
        """M_B parameter should be silently ignored for Union3."""
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        lik = SN_likelihood(LCDM, "union3", M_B="free")
        assert len(lik.nuisance_parameters) == 0

    def test_in_available_datasets(self, _init_hicosmo):
        from hicosmo.likelihoods import available_sn_datasets

        datasets = available_sn_datasets()
        assert "union3" in datasets


# ----------------------------------------------------------------
# Combination with other likelihoods
# ----------------------------------------------------------------
class TestCombination:
    def test_union3_plus_bao(self, _init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
        from hicosmo.models import LCDM

        u3 = SN_likelihood(LCDM, "union3")
        bao = BAO_likelihood(LCDM, "desi2024")
        joint = u3 + bao

        params = dict(H0=70.0, Omega_m=0.3)
        log_L_u3 = float(u3(**params))
        log_L_bao = float(bao(**params))
        log_L_joint = float(joint(**params))

        assert log_L_joint == pytest.approx(log_L_u3 + log_L_bao, abs=1e-4)

    def test_union3_plus_pantheon_different(self, _init_hicosmo):
        """Union3 and Pantheon+ should give different likelihoods."""
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        u3 = SN_likelihood(LCDM, "union3")
        pp = SN_likelihood(LCDM, "pantheon+")

        params = dict(H0=70.0, Omega_m=0.3)
        log_L_u3 = float(u3(**params))
        log_L_pp = float(pp(**params))

        assert log_L_u3 != log_L_pp


# ----------------------------------------------------------------
# wCDM model
# ----------------------------------------------------------------
class TestWCDM:
    def test_wcdm_w_minus1_equals_lcdm(self, _init_hicosmo):
        """wCDM with w=-1 should match LCDM."""
        from hicosmo.likelihoods.sn.union3 import Union3Likelihood
        from hicosmo.models import LCDM, wCDM

        u3_lcdm = Union3Likelihood(cosmology_class=LCDM)
        u3_wcdm = Union3Likelihood(cosmology_class=wCDM)

        log_L_lcdm = float(u3_lcdm(H0=70.0, Omega_m=0.3))
        log_L_wcdm = float(u3_wcdm(H0=70.0, Omega_m=0.3, w0=-1.0))

        assert log_L_wcdm == pytest.approx(log_L_lcdm, abs=0.01)


# ----------------------------------------------------------------
# Validation against published results
# ----------------------------------------------------------------
class TestValidation:
    def test_best_fit_omega_m_near_published(self, union3):
        """
        Best-fit Omega_m should be near 0.344 (Union3.1 flat LCDM).
        We use a coarse profile over H0 to check.
        """
        omega_m_grid = np.linspace(0.25, 0.45, 40)
        H0_grid = np.linspace(65.0, 78.0, 30)

        best_logL = -np.inf
        best_om = None

        for om in omega_m_grid:
            for H0 in H0_grid:
                logL = float(union3(H0=H0, Omega_m=om))
                if logL > best_logL:
                    best_logL = logL
                    best_om = om

        # Within ~0.02 of published value
        assert (
            abs(best_om - 0.344) < 0.025
        ), f"Best-fit Omega_m = {best_om:.3f}, expected ~0.344"

    def test_chi2_per_dof_reasonable(self, union3):
        """Chi2/dof should be around 1 at best fit."""
        # Use approximate best-fit parameters
        from hicosmo.models import LCDM

        model = LCDM(H0=72.0, Omega_m=0.35)
        chi2 = union3.chi2(model)
        dof = union3.n_bins - 2  # 22 bins - 2 params (H0, Omega_m)
        chi2_per_dof = chi2 / dof
        assert 0.5 < chi2_per_dof < 3.0, f"chi2/dof = {chi2_per_dof:.2f}, expected ~1"
