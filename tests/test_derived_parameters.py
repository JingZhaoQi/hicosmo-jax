"""Derived-parameter post-processing tests."""

from __future__ import annotations

import numpy as np


def test_bulk_derived_parameters_preserve_sampled_names_and_avoid_scalar_loop():
    from hicosmo.samplers.derived import compute_derived_parameters

    class BulkLikelihood:
        def __init__(self):
            self.bulk_calls = 0
            self.scalar_calls = 0

        def derived_parameters_vectorized(self, samples):
            self.bulk_calls += 1
            theta = np.asarray(samples["theta"])
            return {"theta": theta + 100.0, "phi": theta * 2.0}

        def derived_parameters(self, **params):
            self.scalar_calls += 1
            return {"phi": -1.0}

    likelihood = BulkLikelihood()
    samples = {"theta": np.asarray([1.0, 1.5, 2.0])}

    updated = compute_derived_parameters(samples, [likelihood])

    assert list(updated["theta"]) == [1.0, 1.5, 2.0]
    assert list(updated["phi"]) == [2.0, 3.0, 4.0]
    assert likelihood.bulk_calls == 1
    assert likelihood.scalar_calls == 0


def test_scalar_derived_parameters_keep_first_non_sampled_name():
    from hicosmo.samplers.derived import compute_derived_parameters

    class FirstLikelihood:
        def derived_parameters(self, **params):
            return {"phi": 1.0}

    class SecondLikelihood:
        def derived_parameters(self, **params):
            return {"phi": 2.0}

    samples = {"theta": np.asarray([1.0, 1.5, 2.0])}
    updated = compute_derived_parameters(
        samples, [FirstLikelihood(), SecondLikelihood()]
    )

    assert list(updated["phi"]) == [1.0, 1.0, 1.0]


def test_bao_vectorized_derived_parameters_do_not_overwrite_h0rd():
    from hicosmo.likelihoods import BAO_likelihood
    from hicosmo.models import LCDM
    from hicosmo.samplers.derived import compute_derived_parameters

    bao = BAO_likelihood(LCDM, "desi2024")
    assert hasattr(bao, "derived_parameters_vectorized")
    samples = {
        "H0": np.asarray([70.0, 80.0]),
        "Omega_m": np.asarray([0.3, 0.31]),
        "H0_rd": np.asarray([102.0, 104.0]),
    }

    updated = compute_derived_parameters(samples, [bao])

    assert list(updated["H0_rd"]) == [102.0, 104.0]
    assert np.allclose(updated["rd"], [145.71428571428572, 130.0])
    assert np.allclose(updated["rd_h"], [102.0, 104.0])
