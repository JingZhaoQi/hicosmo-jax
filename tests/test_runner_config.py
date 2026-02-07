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
