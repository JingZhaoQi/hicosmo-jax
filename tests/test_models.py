"""
Unit tests for cosmological models.

Tests LCDM, wCDM, and CPL models for:
- Correct initialization
- Parameter validation
- Distance calculations
- Model consistency
- Performance requirements
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit
import time

from hicosmo.models import LCDM, wCDM, CPL


class TestLCDM:
    """Test suite for LCDM model."""

    def test_initialization(self):
        """Test LCDM model initialization."""
        model = LCDM(H0=70.0, Omega_m=0.3)
        assert model.params['H0'] == 70.0
        assert model.params['Omega_m'] == 0.3
        assert hasattr(model, 'Omega_Lambda')

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            LCDM(H0=-10)  # Invalid H0

        with pytest.raises(ValueError):
            LCDM(Omega_m=2.0)  # Invalid Omega_m

    def test_hubble_parameter(self):
        """Test E(z) calculation."""
        model = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.0)

        # At z=0, E(z) should be 1
        assert np.abs(model.E_z(0.0) - 1.0) < 1e-10

        # E(z) should increase with redshift
        z = np.array([0.0, 0.5, 1.0, 2.0])
        E_z = model.E_z(z)
        assert np.all(np.diff(E_z) > 0)

    def test_distance_calculations(self):
        """Test distance measures."""
        model = LCDM(H0=70.0, Omega_m=0.3)

        z = np.array([0.1, 0.5, 1.0, 2.0])

        # Comoving distance
        D_M = model.comoving_distance(z)
        assert np.all(D_M > 0)
        assert np.all(np.diff(D_M) > 0)  # Should increase with z

        # Luminosity distance
        D_L = model.luminosity_distance(z)
        assert np.all(D_L > D_M)  # D_L = D_M * (1+z) for flat universe

        # Angular diameter distance
        D_A = model.angular_diameter_distance(z)
        assert np.all(D_A > 0)

        # Distance modulus
        mu = model.distance_modulus(z)
        assert np.all(mu > 0)
        assert np.all(np.diff(mu) > 0)

    def test_flat_universe_consistency(self):
        """Test flat universe relations."""
        model = LCDM(H0=70.0, Omega_m=0.3, Omega_k=0.0)

        z = 1.0
        D_M = model.comoving_distance(z)
        D_A = model.angular_diameter_distance(z)
        D_L = model.luminosity_distance(z)

        # For flat universe: D_L = D_M * (1+z) = D_A * (1+z)^2
        assert np.abs(D_L - D_M * (1 + z)) / D_L < 1e-6
        assert np.abs(D_L - D_A * (1 + z)**2) / D_L < 1e-6

    def test_performance(self):
        """Test performance requirements."""
        model = LCDM(H0=70.0, Omega_m=0.3)

        # Single calculation should be < 0.01ms
        z = 1.0
        start = time.time()
        for _ in range(1000):
            _ = model.luminosity_distance(z)
        elapsed = time.time() - start
        per_call = elapsed / 1000 * 1000  # Convert to ms
        assert per_call < 0.01, f"Single call took {per_call:.3f}ms"

        # Vectorized calculation
        z = np.linspace(0.1, 3.0, 100)
        start = time.time()
        _ = model.luminosity_distance(z)
        elapsed = time.time() - start
        assert elapsed < 0.1, f"Vectorized calculation took {elapsed:.3f}s"


class TestwCDM:
    """Test suite for wCDM model."""

    def test_initialization(self):
        """Test wCDM model initialization."""
        model = wCDM(H0=70.0, Omega_m=0.3, w=-0.9)
        assert model.params['H0'] == 70.0
        assert model.params['Omega_m'] == 0.3
        assert model.w == -0.9

    def test_reduces_to_lcdm(self):
        """Test that wCDM with w=-1 reduces to LCDM."""
        lcdm = LCDM(H0=70.0, Omega_m=0.3)
        wcdm = wCDM(H0=70.0, Omega_m=0.3, w=-1.0)

        z = np.linspace(0.1, 2.0, 10)

        # E(z) should be identical
        E_lcdm = lcdm.E_z(z)
        E_wcdm = wcdm.E_z(z)
        np.testing.assert_allclose(E_lcdm, E_wcdm, rtol=1e-10)

        # Distances should be identical
        D_L_lcdm = lcdm.luminosity_distance(z)
        D_L_wcdm = wcdm.luminosity_distance(z)
        np.testing.assert_allclose(D_L_lcdm, D_L_wcdm, rtol=1e-8)

    def test_equation_of_state(self):
        """Test equation of state."""
        model = wCDM(w=-0.8)
        z = np.array([0.0, 1.0, 2.0])
        w_z = model.w_z(z)

        # Should be constant for wCDM
        assert np.all(w_z == -0.8)

    def test_dark_energy_evolution(self):
        """Test dark energy density evolution."""
        model = wCDM(H0=70.0, Omega_m=0.3, w=-0.9)

        z = np.array([0.0, 0.5, 1.0, 2.0])
        Omega_DE = model.Omega_DE_z(z)

        # At z=0, should equal Omega_Lambda
        assert np.abs(Omega_DE[0] - model.Omega_Lambda) < 1e-10

        # Evolution depends on w
        if model.w > -1:  # Quintessence
            assert np.all(np.diff(Omega_DE) < 0)  # Decreases with z
        elif model.w < -1:  # Phantom
            pass  # More complex behavior

    def test_phantom_warning(self):
        """Test phantom energy warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = wCDM(w=-1.5)  # Phantom energy
            assert len(w) == 1
            assert "Phantom" in str(w[0].message)


class TestCPL:
    """Test suite for CPL model."""

    def test_initialization(self):
        """Test CPL model initialization."""
        model = CPL(H0=70.0, Omega_m=0.3, w0=-1.0, wa=0.3)
        assert model.params['H0'] == 70.0
        assert model.params['Omega_m'] == 0.3
        assert model.w0 == -1.0
        assert model.wa == 0.3

    def test_reduces_to_wcdm(self):
        """Test that CPL with wa=0 reduces to wCDM."""
        wcdm = wCDM(H0=70.0, Omega_m=0.3, w=-0.9)
        cpl = CPL(H0=70.0, Omega_m=0.3, w0=-0.9, wa=0.0)

        z = np.linspace(0.1, 2.0, 10)

        # E(z) should be identical
        E_wcdm = wcdm.E_z(z)
        E_cpl = cpl.E_z(z)
        np.testing.assert_allclose(E_wcdm, E_cpl, rtol=1e-10)

        # Distances should be identical
        D_L_wcdm = wcdm.luminosity_distance(z)
        D_L_cpl = cpl.luminosity_distance(z)
        np.testing.assert_allclose(D_L_wcdm, D_L_cpl, rtol=1e-8)

    def test_reduces_to_lcdm(self):
        """Test that CPL with w0=-1, wa=0 reduces to LCDM."""
        lcdm = LCDM(H0=70.0, Omega_m=0.3)
        cpl = CPL(H0=70.0, Omega_m=0.3, w0=-1.0, wa=0.0)

        z = np.linspace(0.1, 2.0, 10)

        # Distances should be identical
        D_L_lcdm = lcdm.luminosity_distance(z)
        D_L_cpl = cpl.luminosity_distance(z)
        np.testing.assert_allclose(D_L_lcdm, D_L_cpl, rtol=1e-8)

    def test_evolving_equation_of_state(self):
        """Test evolving equation of state."""
        model = CPL(w0=-0.9, wa=0.3)

        # Check w(z) evolution
        z = np.array([0.0, 1.0, 2.0, 10.0])
        w_z = model.w_z(z)

        # At z=0: w = w0
        assert np.abs(w_z[0] - model.w0) < 1e-10

        # At high z: w → w0 + wa
        assert np.abs(w_z[-1] - (model.w0 + model.wa * 10/11)) < 1e-8

        # Check CPL formula: w(z) = w0 + wa * z/(1+z)
        expected = model.w0 + model.wa * z / (1 + z)
        np.testing.assert_allclose(w_z, expected, rtol=1e-10)

    def test_phantom_crossing_warning(self):
        """Test phantom crossing warning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Model that crosses phantom divide
            model = CPL(w0=-0.8, wa=-0.5)
            assert len(w) == 1
            assert "phantom divide" in str(w[0].message)


class TestModelComparison:
    """Test consistency between different models."""

    def test_all_models_at_z0(self):
        """All models should give E(0) = 1."""
        models = [
            LCDM(H0=70.0, Omega_m=0.3),
            wCDM(H0=70.0, Omega_m=0.3, w=-0.9),
            CPL(H0=70.0, Omega_m=0.3, w0=-0.9, wa=0.2)
        ]

        for model in models:
            assert np.abs(model.E_z(0.0) - 1.0) < 1e-10

    def test_model_hierarchy(self):
        """Test model relationships at different parameter values."""
        H0, Omega_m = 70.0, 0.3
        z = np.linspace(0.1, 2.0, 20)

        # LCDM is special case of wCDM
        lcdm = LCDM(H0=H0, Omega_m=Omega_m)
        wcdm_as_lcdm = wCDM(H0=H0, Omega_m=Omega_m, w=-1.0)

        D_L_lcdm = lcdm.luminosity_distance(z)
        D_L_wcdm = wcdm_as_lcdm.luminosity_distance(z)
        np.testing.assert_allclose(D_L_lcdm, D_L_wcdm, rtol=1e-8)

        # wCDM is special case of CPL
        wcdm = wCDM(H0=H0, Omega_m=Omega_m, w=-0.9)
        cpl_as_wcdm = CPL(H0=H0, Omega_m=Omega_m, w0=-0.9, wa=0.0)

        D_L_wcdm = wcdm.luminosity_distance(z)
        D_L_cpl = cpl_as_wcdm.luminosity_distance(z)
        np.testing.assert_allclose(D_L_wcdm, D_L_cpl, rtol=1e-8)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])