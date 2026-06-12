"""Config runner smoke tests."""

from __future__ import annotations


def test_run_from_config_build_only(tmp_path):
    from hicosmo import run_from_config

    config = {
        "name": "smoke",
        "theory": "LCDM",
        "likelihood": [{"name": "shoes"}],
        "params": {
            "H0": {
                "prior": {"dist": "uniform", "min": 60.0, "max": 80.0},
                "ref": 70.0,
            }
        },
        "sampler": {
            "name": "numpyro",
            "num_samples": 16,
            "num_warmup": 4,
            "num_chains": 1,
        },
        "output": {"root": str(tmp_path), "chain_name": "smoke_chain"},
    }

    result = run_from_config(config, run=False)
    assert "runner" in result
    assert result["output_dir"].exists()
    assert result["config"]["theory"] == "LCDM"
    assert "samples" not in result


def test_run_from_config_registers_model_params_before_free_selection(tmp_path):
    from hicosmo import run_from_config

    config = {
        "name": "w0wa_smoke",
        "preset": "planck2018",
        "theory": "W0WACDM",
        "likelihood": [{"name": "bao"}],
        "free": ["H0", "Omega_m", "w0", "wa"],
        "sampler": {
            "name": "numpyro",
            "num_samples": 8,
            "num_warmup": 4,
            "num_chains": 1,
        },
        "output": {"root": str(tmp_path), "chain_name": "w0wa_chain"},
    }

    result = run_from_config(config, run=False)
    mcmc = result["runner"].mcmc

    assert mcmc is not None
    assert set(mcmc.backend.parameters) == {"H0", "Omega_m", "w0", "wa", "H0_rd"}
    assert mcmc.param_config.parameters["Omega_b"].free is False


def test_run_from_config_can_select_likelihood_nuisance_parameter(tmp_path):
    from hicosmo import run_from_config

    config = {
        "name": "bao_nuisance_smoke",
        "preset": "planck2018",
        "theory": "LCDM",
        "likelihood": [{"name": "bao"}],
        "free": ["H0", "Omega_m", "H0_rd"],
        "sampler": {
            "name": "numpyro",
            "num_samples": 8,
            "num_warmup": 4,
            "num_chains": 1,
        },
        "output": {"root": str(tmp_path), "chain_name": "bao_chain"},
    }

    result = run_from_config(config, run=False)
    mcmc = result["runner"].mcmc

    assert mcmc is not None
    assert set(mcmc.backend.parameters) == {"H0", "Omega_m", "H0_rd"}
    assert mcmc.param_config.parameters["Omega_b"].free is False
