"""CMB distance-prior pipeline must track CAMB across the sampled range.

Regression for three stacked errors that produced a ~0.5 sigma H0 shift
against Cobaya in joint fits: a rounded photon density (2.47e-5), a uniform
0..z* integration grid (0.09% D_M error), and a constant-only rs(z*)
calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

camb = pytest.importorskip("camb")

import jax.numpy as jnp

from hicosmo.likelihoods.cmb.planck_distance import (
    Planck2018DistancePriorsLikelihood as P,
)
from hicosmo.models import LCDM

C_KM_S = 299792.458
SIG_R = 1.0 / np.sqrt(9.43923971e04)
SIG_LA = 1.0 / np.sqrt(1.6143490e02)


def _camb_side(H0, wb, wm):
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=wb, omch2=wm - wb, TCMB=2.7255,
        num_massive_neutrinos=0, mnu=0.0,
    )
    res = camb.get_background(pars)
    d = res.get_derived_params()
    DM = res.angular_diameter_distance(d["zstar"]) * (1 + d["zstar"])
    h = H0 / 100.0
    return np.sqrt(wm / h**2) * (H0 / C_KM_S) * DM, np.pi * DM / d["rstar"]


def _hicosmo_traced(lik, H0, wb, wm):
    h = H0 / 100.0
    params = {
        "H0": jnp.float64(H0),
        "Omega_m": jnp.float64(wm / h**2),
        "Omega_b": jnp.float64(wb / h**2),
    }
    zs_raw = LCDM.recombination_redshift_traced(params)
    zs = zs_raw * P._camb_ratio(P._ZSTAR_CAL_COEF, wb, wm)
    rs = LCDM.sound_horizon_traced(zs_raw, params) * P._camb_ratio(
        P._RS_CAL_COEF, wb, wm
    )
    grid = LCDM.compute_grid_traced(zs * lik._z_base, params)
    DM = grid["D_M"][-1]
    R = jnp.sqrt(params["Omega_m"]) * (H0 / C_KM_S) * DM
    return float(R), float(jnp.pi * DM / rs)


@pytest.mark.parametrize(
    "H0,wb,wm",
    [
        (67.36, 0.02237, 0.1430),
        (70.0, 0.02237, 0.1430),
        (67.36, 0.0210, 0.1430),
        (67.36, 0.02237, 0.1600),
    ],
)
def test_traced_cmb_pipeline_matches_camb(H0, wb, wm):
    lik = P(cosmology_class=LCDM)
    Rh, lah = _hicosmo_traced(lik, H0, wb, wm)
    Rc, lac = _camb_side(H0, wb, wm)
    assert abs(Rh - Rc) / SIG_R < 0.05
    assert abs(lah - lac) / SIG_LA < 0.15
