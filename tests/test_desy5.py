"""
Tests for DES-SN5YR (DESY5) supernova likelihood.

Validates data loading, likelihood computation, factory integration,
and compatibility with the combined likelihood framework.
"""

import pytest
import numpy as np


@pytest.fixture(scope="module")
def init_hicosmo():
    """Initialize HIcosmo once for all tests."""
    import hicosmo as hc

    hc.init(4)


@pytest.fixture(scope="module")
def desy5_likelihood(init_hicosmo):
    """Create a DESY5 likelihood instance."""
    from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

    return DESY5Likelihood()


@pytest.fixture(scope="module")
def lcdm_model(init_hicosmo):
    """Create a reference LCDM model."""
    from hicosmo.models import LCDM

    return LCDM(H0=70.0, Omega_m=0.352)


class TestDESY5DataLoading:
    """Tests for data loading and validation."""

    def test_instantiation(self, desy5_likelihood):
        assert desy5_likelihood is not None
        assert desy5_likelihood.n_sne == 1820

    def test_redshift_range(self, desy5_likelihood):
        z_min = float(desy5_likelihood.redshifts.min())
        z_max = float(desy5_likelihood.redshifts.max())
        # DES-SN5YR: 0.025 < z < 1.13 (with z_min cut at 0.025)
        assert z_min > 0.02, f"z_min = {z_min}"
        assert z_max < 1.2, f"z_max = {z_max}"
        assert z_max > 1.0, f"z_max = {z_max}"

    def test_distance_moduli_range(self, desy5_likelihood):
        mu_min = float(desy5_likelihood.mu_obs.min())
        mu_max = float(desy5_likelihood.mu_obs.max())
        # Distance moduli for z in [0.025, 1.14] should be roughly [34, 45]
        assert 33 < mu_min < 36, f"mu_min = {mu_min}"
        assert 43 < mu_max < 46, f"mu_max = {mu_max}"

    def test_inverse_covariance_shape(self, desy5_likelihood):
        assert desy5_likelihood.cov_inv.shape == (1820, 1820)

    def test_inverse_covariance_symmetric(self, desy5_likelihood):
        cov_inv = np.array(desy5_likelihood.cov_inv)
        assert np.allclose(cov_inv, cov_inv.T, atol=1e-8)

    def test_info(self, desy5_likelihood):
        info = desy5_likelihood.get_info()
        assert info["name"] == "DES-SN5YR"
        assert info["n_supernovae"] == 1820
        assert info["include_systematics"] is True

    def test_repr(self, desy5_likelihood):
        r = repr(desy5_likelihood)
        assert "DESY5" in r
        assert "1820" in r

    def test_stat_only_covariance(self, init_hicosmo):
        from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

        desy5_stat = DESY5Likelihood(include_systematics=False)
        assert desy5_stat.n_sne == 1820
        assert desy5_stat.cov_inv.shape == (1820, 1820)


class TestDESY5Likelihood:
    """Tests for likelihood computation."""

    def test_log_likelihood_returns_finite(self, desy5_likelihood, lcdm_model):
        log_L = desy5_likelihood.log_likelihood(lcdm_model)
        assert np.isfinite(float(log_L))

    def test_log_likelihood_negative(self, desy5_likelihood, lcdm_model):
        # Marginalized log_L should be negative (log of probability < 1)
        log_L = float(desy5_likelihood.log_likelihood(lcdm_model))
        assert log_L < 0

    def test_chi2_reasonable(self, desy5_likelihood, lcdm_model):
        best_M = desy5_likelihood.best_fit_M(lcdm_model)
        chi2 = desy5_likelihood.chi2(lcdm_model, best_M)
        chi2_per_dof = chi2 / (desy5_likelihood.n_sne - 1)
        # chi2/dof should be near 1 for a good fit
        assert 0.5 < chi2_per_dof < 2.0, f"chi2/dof = {chi2_per_dof}"

    def test_best_fit_M_reasonable(self, desy5_likelihood, lcdm_model):
        best_M = desy5_likelihood.best_fit_M(lcdm_model)
        # Since data gives MU with H0=70 and we evaluate with H0=70,
        # M should be close to 0
        assert abs(best_M) < 1.0, f"best_M = {best_M}"

    def test_callable_interface(self, desy5_likelihood):
        log_L = desy5_likelihood(H0=70.0, Omega_m=0.352)
        assert np.isfinite(float(log_L))

    def test_callable_matches_log_likelihood(self, desy5_likelihood, lcdm_model):
        log_L_method = float(desy5_likelihood.log_likelihood(lcdm_model))
        log_L_call = float(desy5_likelihood(H0=70.0, Omega_m=0.352))
        assert abs(log_L_method - log_L_call) < 1e-4

    def test_best_fit_omega_m_consistent(self, desy5_likelihood):
        # Scan over Omega_m to verify best-fit is near 0.352
        best_logL = -1e10
        best_Om = 0
        for Om in np.linspace(0.2, 0.5, 50):
            logL = float(desy5_likelihood(H0=70.0, Omega_m=float(Om)))
            if logL > best_logL:
                best_logL = logL
                best_Om = Om
        # At fixed H0=70, best Om should be within ~0.02 of official value
        assert abs(best_Om - 0.352) < 0.03, f"best_Om = {best_Om}"

    def test_marginalization_matches_official_formula(
        self, desy5_likelihood, lcdm_model
    ):
        """Verify our marginalization formula matches DES official code."""
        import jax.numpy as jnp

        theory_mu = np.array(desy5_likelihood._distance_modulus_from_model(lcdm_model))
        mu_obs = np.array(desy5_likelihood.mu_obs)
        inv_cov = np.array(desy5_likelihood.cov_inv)

        # Official DES formula from Dovekie_cosmosis_likelihood.py
        delta = theory_mu - mu_obs
        chit2 = delta @ inv_cov @ delta
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2_official = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
        log_L_official = -0.5 * chi2_official

        log_L_ours = float(
            desy5_likelihood._marginalized_likelihood(jnp.array(theory_mu))
        )

        assert (
            abs(log_L_official - log_L_ours) < 1e-8
        ), f"Official: {log_L_official}, Ours: {log_L_ours}"


class TestDESY5FreeMMode:
    """Tests for free M (non-marginalized) mode."""

    def test_free_M_instantiation(self, init_hicosmo):
        from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

        desy5 = DESY5Likelihood(marginalize_M=False)
        assert desy5.marginalize_M is False

    def test_free_M_requires_M_param(self, init_hicosmo):
        from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

        desy5 = DESY5Likelihood(marginalize_M=False)
        with pytest.raises(ValueError, match="M must be provided"):
            desy5(H0=70.0, Omega_m=0.3)

    def test_free_M_evaluation(self, init_hicosmo):
        from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

        desy5 = DESY5Likelihood(marginalize_M=False)
        log_L = desy5(H0=70.0, Omega_m=0.3, M=0.0)
        assert np.isfinite(float(log_L))

    def test_free_M_nuisance_parameters(self, init_hicosmo):
        from hicosmo.likelihoods.sn.desy5 import DESY5Likelihood

        desy5 = DESY5Likelihood(marginalize_M=False)
        params = desy5.nuisance_parameters
        assert len(params) == 1
        assert params[0].name == "M"

    def test_marginalized_no_nuisance(self, desy5_likelihood):
        params = desy5_likelihood.nuisance_parameters
        assert len(params) == 0


class TestDESY5Factory:
    """Tests for factory and alias integration."""

    def test_factory_default(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        desy5 = SN_likelihood(LCDM, "desy5")
        assert desy5.n_sne == 1820

    def test_factory_free_mode(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        desy5 = SN_likelihood(LCDM, "desy5", M_B="free")
        assert desy5.marginalize_M is False

    def test_factory_fixed_mode(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        desy5 = SN_likelihood(LCDM, "desy5", M_B=-19.3)
        # Should be wrapped in _FixedMLikelihood
        log_L = desy5(H0=70.0, Omega_m=0.3)
        assert np.isfinite(float(log_L))

    def test_aliases(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        for alias in ["desy5", "des5yr", "des_sn5yr"]:
            lik = SN_likelihood(LCDM, alias)
            assert lik.n_sne == 1820, f"Alias {alias} failed"

    def test_available_datasets_includes_desy5(self, init_hicosmo):
        from hicosmo.likelihoods import available_sn_datasets

        datasets = available_sn_datasets()
        assert "desy5" in datasets


class TestDESY5CombinedLikelihood:
    """Tests for combining DESY5 with other likelihoods."""

    def test_combine_with_bao(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
        from hicosmo.models import LCDM

        desy5 = SN_likelihood(LCDM, "desy5")
        bao = BAO_likelihood(LCDM, "desi2024")
        joint = desy5 + bao
        log_L = joint(H0=70.0, Omega_m=0.3)
        assert np.isfinite(float(log_L))

    def test_combine_with_pantheonplus(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        desy5 = SN_likelihood(LCDM, "desy5")
        pp = SN_likelihood(LCDM, "pantheon+")
        # Both SN likelihoods can be combined (even if not physically meaningful)
        joint = desy5 + pp
        log_L = joint(H0=70.0, Omega_m=0.3)
        assert np.isfinite(float(log_L))

    def test_normalization_constant(self, desy5_likelihood):
        norm = desy5_likelihood.get_normalization_constant()
        assert np.isfinite(norm)

    def test_num_data(self, desy5_likelihood):
        assert desy5_likelihood.get_num_data() == 1820


class TestDESY5PantheonPlusUnchanged:
    """Regression: ensure Pantheon+ is unaffected."""

    def test_pantheon_plus_still_works(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        pp = SN_likelihood(LCDM, "pantheon+")
        log_L = pp(H0=70.0, Omega_m=0.3)
        assert np.isfinite(float(log_L))
        # Rough regression value
        assert -700 < float(log_L) < -680

    def test_sn_alias_still_pantheon(self, init_hicosmo):
        from hicosmo.likelihoods import SN_likelihood
        from hicosmo.models import LCDM

        pp = SN_likelihood(LCDM, "sn")
        assert isinstance(pp, type(SN_likelihood(LCDM, "pantheon+")))
