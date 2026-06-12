"""Sampler backend behavior tests."""

from __future__ import annotations


def test_numpyro_sampler_uses_stable_default_initialization():
    from hicosmo.samplers.numpyro_backend import NumPyroSampler

    sampler = NumPyroSampler(
        log_probability=lambda params: 0.0,
        parameters={
            "H0": {
                "prior": {"dist": "uniform", "min": 50.0, "max": 100.0},
                "ref": 67.36,
            }
        },
        config={},
    )

    strategy_name, strategy = sampler._resolve_init_strategy()

    assert strategy_name == "median"
    assert callable(strategy)


def test_numpyro_value_initialization_can_use_parameter_refs():
    from hicosmo.samplers.numpyro_backend import NumPyroSampler

    sampler = NumPyroSampler(
        log_probability=lambda params: 0.0,
        parameters={
            "H0": {
                "prior": {"dist": "uniform", "min": 50.0, "max": 100.0},
                "ref": 67.36,
            }
        },
        config={"init_strategy": "value"},
    )

    strategy_name, strategy = sampler._resolve_init_strategy()

    assert strategy_name == "value"
    assert callable(strategy)


def test_numpyro_unknown_initialization_falls_back_to_median():
    from hicosmo.samplers.numpyro_backend import NumPyroSampler

    sampler = NumPyroSampler(
        log_probability=lambda params: 0.0,
        parameters={
            "H0": {
                "prior": {"dist": "uniform", "min": 50.0, "max": 100.0},
                "ref": 67.36,
            }
        },
        config={"init_strategy": "not-a-strategy"},
    )

    strategy_name, strategy = sampler._resolve_init_strategy()

    assert strategy_name == "median"
    assert callable(strategy)


def test_mcmc_registry_config_preserves_fixed_parameters_for_likelihood_calls():
    from hicosmo.parameters import ParameterRegistry
    from hicosmo.samplers.inference import MCMC

    registry = ParameterRegistry()
    registry.add(
        "H0",
        value=67.36,
        free=True,
        prior={"dist": "uniform", "min": 50.0, "max": 100.0},
    )
    registry.add("Omega_b", value=0.0493, free=False)

    seen = {}

    def likelihood(**params):
        seen.update(params)
        return 0.0

    mcmc = MCMC(
        registry,
        likelihood,
        chain_name=None,
        enable_checkpoints=False,
        chain_method="sequential",
    )

    mcmc.backend.log_probability({"H0": 68.0})

    assert set(mcmc.backend.parameters) == {"H0"}
    assert seen["H0"] == 68.0
    assert seen["Omega_b"] == 0.0493


def test_intelligent_defaults_do_not_redivide_per_chain_counts():
    from hicosmo.parameters import ParameterRegistry
    from hicosmo.samplers.inference import MCMC

    registry = ParameterRegistry()
    registry.add(
        "H0",
        value=67.36,
        free=True,
        prior={"dist": "uniform", "min": 50.0, "max": 100.0},
    )

    mcmc = MCMC(
        registry,
        lambda **params: 0.0,
        chain_name=None,
        enable_checkpoints=False,
        chain_method="sequential",
    )

    total_config = {
        "num_samples": 101,
        "num_warmup": 21,
        "num_chains": 4,
        "verbose": False,
    }
    first = mcmc._apply_intelligent_defaults(dict(total_config))
    second = mcmc._apply_intelligent_defaults(dict(first))

    assert first["num_samples"] == 26
    # num_warmup is PER-CHAIN: each chain needs its own full adaptation window,
    # so the user's value is used as-is (the old total/num_chains split gave
    # 4 chains only ~5 warmup steps each)
    assert first["num_warmup"] == 21
    assert second["num_samples"] == first["num_samples"]
    assert second["num_warmup"] == first["num_warmup"]


def test_derived_parameters_do_not_overwrite_sampled_parameters():
    from hicosmo.parameters import ParameterRegistry
    from hicosmo.samplers.inference import MCMC

    class DerivedCollisionLikelihood:
        def __call__(self, **params):
            return 0.0

        def derived_parameters(self, **params):
            return {"theta": 999.0, "phi": 2.0}

    registry = ParameterRegistry()
    registry.add(
        "theta",
        value=1.0,
        free=True,
        prior={"dist": "uniform", "min": 0.0, "max": 10.0},
    )
    mcmc = MCMC(
        registry,
        DerivedCollisionLikelihood(),
        likelihoods=[DerivedCollisionLikelihood()],
        chain_name=None,
        enable_checkpoints=False,
        chain_method="sequential",
    )

    samples = {"theta": [1.0, 1.5, 2.0]}
    updated = mcmc._compute_derived_parameters(samples)

    assert list(updated["theta"]) == [1.0, 1.5, 2.0]
    assert list(updated["phi"]) == [2.0, 2.0, 2.0]


def test_broken_likelihood_raises_instead_of_sampling_prior(tmp_path):
    """A likelihood that raises must fail loudly, not silently sample the prior.

    Regression for the trace-time try/except that returned -1e10 and made
    NUTS 'successfully' sample the prior with zero warnings.
    """
    import pytest

    import hicosmo as hc

    hc.set_output_dir(str(tmp_path))
    from hicosmo.samplers import MCMC

    def broken_likelihood(H0):
        raise ValueError("intentionally broken likelihood")

    params = {"H0": {"init": 70, "min": 60, "max": 80}}
    mcmc = MCMC(params, broken_likelihood, chain_name="test_broken_likelihood")
    with pytest.raises(Exception):
        mcmc.run(num_samples=50)


def test_emcee_respects_nonuniform_prior(tmp_path):
    """Regression: emcee treated every prior as uniform (log-prior = 0.0),
    so the two backends sampled different posteriors for the same config."""
    import numpy as np

    import hicosmo as hc

    hc.set_output_dir(str(tmp_path))
    from hicosmo.samplers import MCMC

    config = {
        "parameters": {
            "x": {"prior": {"dist": "normal", "loc": 3.0, "scale": 0.5}, "ref": 3.0},
        },
        "mcmc": {"num_samples": 4000, "num_chains": 2},
    }
    mcmc = MCMC(config, lambda x: 0.0, chain_name="test_emcee_prior", sampler="emcee")
    samples = mcmc.run()
    x = np.asarray(samples["x"])

    # Flat likelihood: posterior must reproduce the normal prior
    assert abs(float(x.mean()) - 3.0) < 0.15
    assert abs(float(x.std()) - 0.5) < 0.12  # uniform bug gave ~1.4
