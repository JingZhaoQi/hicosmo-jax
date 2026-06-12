"""Fit a parameter-dependent CAMB calibration for the EH98+HS rs(z*) pipeline.

The Planck distance-prior likelihood needs rs(z*) to ~0.01% for the l_a
residual to be subdominant to the Planck covariance. A constant factor
(current RS_ZSTAR_CALIBRATION=1.00157) only fixes the best-fit point; the
fitting-formula error drifts with (omega_b, omega_m), which is what produces
the ~0.5 sigma H0 shift against Cobaya in joint fits.

Strategy: ratio(omega_b, omega_m) = rs_CAMB(z*_CAMB) / rs_EH98(z*_HS),
fit a 2D quadratic in (omega_b, omega_m). Also verify H0-independence.
"""

import itertools

import numpy as np


import camb

from hicosmo.models import LCDM


def camb_rs_zstar(H0, omega_b, omega_c):
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=omega_b, omch2=omega_c, TCMB=2.7255,
        num_massive_neutrinos=0, mnu=0.0, nnu=3.046,
    )
    r = camb.get_background(pars)
    d = r.get_derived_params()
    return d["rstar"], d["zstar"]


def hicosmo_rs_zstar(H0, omega_b, omega_m):
    h = H0 / 100.0
    m = LCDM(H0=H0, Omega_m=omega_m / h**2, Omega_b=omega_b / h**2)
    z_star = float(m.recombination_redshift())
    rs = float(m.sound_horizon(z_star))
    return rs, z_star


# --- 1. H0-independence check at fixed physical densities ---
print("=== H0-independence of the ratio (fixed omega_b, omega_m) ===")
wb, wm = 0.02237, 0.1430
for H0 in (60.0, 67.36, 75.0):
    rs_c, zs_c = camb_rs_zstar(H0, wb, wm - wb)
    rs_h, zs_h = hicosmo_rs_zstar(H0, wb, wm)
    print(f"H0={H0:5.1f}: ratio_rs={rs_c/rs_h:.6f}  ratio_zstar={zs_c/zs_h:.6f}")

# --- 2. grid over (omega_b, omega_m) ---
print("\n=== building calibration grid ===")
wb_grid = np.linspace(0.018, 0.028, 7)
wm_grid = np.linspace(0.10, 0.22, 9)
H0_ref = 67.36

rows = []
for wb, wm in itertools.product(wb_grid, wm_grid):
    rs_c, zs_c = camb_rs_zstar(H0_ref, wb, wm - wb)
    rs_h, zs_h = hicosmo_rs_zstar(H0_ref, wb, wm)
    rows.append((wb, wm, rs_c / rs_h, zs_c / zs_h))
rows = np.array(rows)
print(f"grid: {len(rows)} points; rs ratio range [{rows[:,2].min():.5f}, {rows[:,2].max():.5f}]")
print(f"constant-calibration residual would be up to "
      f"{np.abs(rows[:,2]/1.00157 - 1).max()*100:.3f}% across the grid")

# --- 3. quadratic fit: ratio = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 ---
# with x = (omega_b - 0.02237)/0.01, y = (omega_m - 0.1430)/0.1 (centered, scaled)
x = (rows[:, 0] - 0.02237) / 0.01
y = (rows[:, 1] - 0.1430) / 0.1
A = np.column_stack([np.ones_like(x), x, y, x**2, x * y, y**2])

for name, col in (("rs", 2), ("zstar", 3)):
    coef, *_ = np.linalg.lstsq(A, rows[:, col], rcond=None)
    resid = A @ coef - rows[:, col]
    print(f"\n{name} ratio quadratic fit:")
    print("  coef =", ", ".join(f"{c:.8e}" for c in coef))
    print(f"  max |resid| = {np.abs(resid).max():.2e}  (relative)")

np.save("/tmp/hicosmo_bench/rs_calib_grid.npy", rows)
print("\ngrid saved")
