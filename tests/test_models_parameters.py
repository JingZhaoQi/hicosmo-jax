"""Model-layer parameter handling regressions."""

from __future__ import annotations


def test_omega_b_h2_parameter_basis_respected():
    """CMB-style parameter basis must not be shadowed by the Omega_b default.

    Regression: LCDM.__init__ unconditionally injected Omega_b=0.0493 into the
    parameter dict, which suppressed the reverse derivation from omega_b_h2
    (8% error, masked by the Planck-default numerical coincidence).
    """
    from hicosmo.models import LCDM

    m = LCDM(h=0.70, omega_b_h2=0.0224, omega_c_h2=0.120)
    assert abs(m.params["Omega_b"] - 0.0224 / 0.49) < 1e-12

    m2 = LCDM(omega_b=0.0224, H0=70.0)
    assert abs(m2.params["Omega_b"] - 0.0224 / 0.49) < 1e-12

    # Default path unchanged
    m3 = LCDM()
    assert abs(m3.params["Omega_b"] - 0.0493) < 1e-12


def test_sound_horizon_single_source_of_truth():
    """Instance rd must match the traced MCMC rd (regression: 2.7% fork)."""
    from hicosmo.models import LCDM
    from hicosmo.utils.jax_tools import sound_horizon_drag_eh98

    m = LCDM()
    rd_instance = float(m.sound_horizon_drag(use_camb=False))
    rd_traced = float(
        sound_horizon_drag_eh98(
            m.params["H0"], m.params["Omega_m"], m.params["Omega_b"], 2.7255
        )
    )
    assert abs(rd_instance - rd_traced) < 1e-9


def test_decorator_rejects_non_staticmethod_E_z():
    """Missing @staticmethod must fail loudly, not inherit parent physics."""
    import jax.numpy as jnp
    import pytest

    from hicosmo.models.base import register_cosmology_model
    from hicosmo.models.lcdm import LCDM

    with pytest.raises(TypeError, match="staticmethod"):

        @register_cosmology_model
        class BadModel(LCDM):
            def E_z(self, z, params):  # typo: not a staticmethod
                return jnp.ones_like(jnp.asarray(z))


def test_unregistered_subclass_with_own_physics_warns():
    """Forgetting the decorator must warn that MCMC would use parent physics."""
    import warnings

    import jax.numpy as jnp

    from hicosmo.models.lcdm import LCDM

    class ForgotDecorator(LCDM):
        @staticmethod
        def E_z(z, params):
            return jnp.ones_like(jnp.asarray(z))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ForgotDecorator()

    assert any("register_cosmology_model" in str(w.message) for w in caught)


def test_unknown_cosmology_parameter_warns_with_suggestion():
    """Regression: sn(w=-0.5) silently evaluated w0=-1 with no hint."""
    import warnings

    from hicosmo.likelihoods import SN_likelihood
    from hicosmo.models.wcdm import wCDM

    sn = SN_likelihood(wCDM, "pantheon+")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sn(H0=70.0, Omega_m=0.3, w=-0.5)
    msgs = [str(w.message) for w in caught if "unknown parameter" in str(w.message)]
    assert len(msgs) == 1
    assert "w0" in msgs[0]  # did-you-mean hint

    # Correct spelling stays silent, and the value actually matters
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v_correct = float(sn(H0=70.0, Omega_m=0.3, w0=-0.5))
        v_default = float(sn(H0=70.0, Omega_m=0.3))
    assert not [w for w in caught if "unknown parameter" in str(w.message)]
    assert abs(v_correct - v_default) > 1.0
