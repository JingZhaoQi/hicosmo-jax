"""Fisher matrix robustness regressions."""

from __future__ import annotations

import warnings

import jax.numpy as jnp


def test_get_fisher_summary_does_not_crash():
    """Regression: eigvals returned complex dtype and float(complex) raised."""
    from hicosmo.fisher.fisher_matrix import FisherMatrix

    fm = FisherMatrix()
    summary = fm.get_fisher_summary(jnp.array([[4.0, 0.5], [0.5, 1.0]]), ["a", "b"])
    assert summary["max_eigenvalue"] > summary["min_eigenvalue"] > 0
    assert summary["condition_number"] > 1.0


def test_singular_fisher_warns_instead_of_silent_inf():
    """Regression: 'except jnp.linalg.LinAlgError' never fired (JAX returns
    inf instead of raising), so singular matrices produced silent inf errors."""
    from hicosmo.fisher.fisher_matrix import FisherMatrix

    fm = FisherMatrix()
    singular = jnp.array([[1.0, 1.0], [1.0, 1.0]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        errors = fm.compute_parameter_errors(singular, ["a", "b"])

    assert any(
        "singular" in str(w.message).lower() or "pseudo-inverse" in str(w.message)
        for w in caught
    )
    # Pseudo-inverse fallback keeps the result finite
    assert all(jnp.isfinite(v) for v in errors.values())
