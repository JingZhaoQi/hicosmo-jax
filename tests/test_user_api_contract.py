"""Canonical user API contract — this exact interface shape must keep working.

Mirrors the project owner's standard workflow:

    import hicosmo as hc
    hc.init(8)
    from hicosmo.samplers import MCMC
    from hicosmo.models import LCDM
    from hicosmo.likelihoods import SN_likelihood, BAO_likelihood
    from hicosmo.visualization import Plotter

    sne = SN_likelihood(LCDM, "pantheon+")
    params = {
        'H0': {'init': 70, 'min': 60, 'max': 80},
        'Omega_m': {'init': 0.3, 'min': 0.1, 'max': 0.5},
    }
    mcmc = MCMC(params, sne, chain_name='test_sn')
    samples = mcmc.run(num_samples=20000)

Any refactor that breaks one of these steps must be rejected.
"""

from __future__ import annotations

import numpy as np


def test_user_canonical_api(tmp_path):
    import hicosmo as hc

    hc.init(2)  # idempotent with other tests' init calls
    hc.set_output_dir(str(tmp_path))

    from hicosmo.samplers import MCMC
    from hicosmo.models import LCDM
    from hicosmo.likelihoods import SN_likelihood, BAO_likelihood  # noqa: F401
    from hicosmo.visualization import Plotter  # noqa: F401

    sne = SN_likelihood(LCDM, "pantheon+")

    params = {
        "H0": {"init": 70, "min": 60, "max": 80},
        "Omega_m": {"init": 0.3, "min": 0.1, "max": 0.5},
    }

    mcmc = MCMC(params, sne, chain_name="test_user_api_contract")
    samples = mcmc.run(num_samples=200)

    assert "H0" in samples and "Omega_m" in samples
    assert len(np.asarray(samples["H0"])) == 200

    # Posterior must be physically sane (Pantheon+ alone)
    assert 60.0 < float(np.mean(samples["H0"])) < 80.0
    assert 0.20 < float(np.mean(samples["Omega_m"])) < 0.45


def test_user_param_dict_format_accepts_init_min_max(tmp_path):
    """The {'init': v, 'min': lo, 'max': hi} per-parameter dict must parse."""
    import hicosmo as hc

    hc.set_output_dir(str(tmp_path))
    from hicosmo.samplers import MCMC

    params = {
        "H0": {"init": 70, "min": 60, "max": 80},
        "Omega_m": {"init": 0.3, "min": 0.1, "max": 0.5},
    }
    mcmc = MCMC(params, lambda H0, Omega_m: 0.0, chain_name="test_param_format")

    names = set(mcmc.param_config.get_parameter_names())
    assert {"H0", "Omega_m"} <= names
    bounds = mcmc.param_config.get_bounds()
    assert bounds["H0"] == (60, 80)
    assert bounds["Omega_m"] == (0.1, 0.5)
    refs = mcmc.param_config.get_reference_values()
    assert refs["H0"] == 70
