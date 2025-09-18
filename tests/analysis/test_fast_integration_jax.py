import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hicosmo.core.fast_integration import FastIntegration


def _make_engine(cache_size: int = 128) -> FastIntegration:
    params = {"H0": 70.0, "Omega_m": 0.3}
    return FastIntegration(
        params,
        precision_mode="balanced",
        cache_size=cache_size,
        z_max=5.0,
        auto_select=True,
    )


def test_precise_matches_numpy_scalar():
    engine = _make_engine(cache_size=0)
    expected = engine._ultra_precise_single_numpy(0.5)
    actual = engine.comoving_distance(0.5, method="precise")
    assert np.allclose(actual, expected, rtol=1e-6)


def test_comoving_distance_accepts_tracer_scalar():
    engine = _make_engine()
    compiled = jax.jit(lambda zz: engine.comoving_distance(zz))
    result = compiled(0.7)
    assert np.isfinite(np.asarray(result))


def test_comoving_distance_accepts_tracer_array():
    engine = _make_engine()
    zs = jnp.linspace(0.1, 1.0, 5)
    vmapped = jax.jit(jax.vmap(lambda zz: engine.comoving_distance(zz)))
    result = vmapped(zs)
    assert result.shape == zs.shape


def test_distance_modulus_accepts_tracer_scalar():
    engine = _make_engine()
    compiled = jax.jit(lambda zz: engine.distance_modulus(zz))
    value = compiled(1.2)
    assert np.isfinite(np.asarray(value))
