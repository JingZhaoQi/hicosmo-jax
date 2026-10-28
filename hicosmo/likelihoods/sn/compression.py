"""
Compressed form of the M-marginalised supernova chi^2.

The theoretical distance moduli depend on the cosmology only through the smooth
curve 5 log10 d_L(z) + 25. Writing it as a cubic (Catmull-Rom) interpolation
of its values g at G nodes uniform in ln z, mu = W g + c, with c holding the
heliocentric-redshift correction (or the Cepheid distances of calibrators,
whose rows of W vanish), the chi^2 marginalised over a constant M with a flat
prior is

    chi^2(g) = min_M |L^T (y - W g - M 1)|^2,   C^{-1} = L L^T,  y = m_obs - c.

With the thin QR decomposition L^T [1, W] = Q R (the M column first), the
first row of R is the only one containing M, so the minimisation removes it:

    chi^2(g) = c0 + |b - R' g|^2,   R' = R[1:, 1:],  b = (Q^T L^T y)[1:],

with c0 the squared norm of the part of L^T y outside the column space. This is
exactly the direct marginalised chi^2 for the given W, it is a sum of squares
(no cancellation between large terms, round-off ~1e-12 in ln L), and R', b, c0
are computed once, so each evaluation costs O(G^2) instead of O(N^2).
With G = 100 the difference from the direct evaluation is below 1.5e-3 in ln L
for Pantheon+, Pantheon+SH0ES and DES-SN5YR, the level at which the direct
evaluation itself agrees with Cobaya (scripts/revision_checks/check_sn_compression.py
in the paper repository).
"""

from typing import Tuple

import numpy as np
import jax.numpy as jnp

from ...utils.jax_tools import interp_linspace

DEFAULT_NODES = 100


def build_compression(
    z: np.ndarray,
    c: np.ndarray,
    y: np.ndarray,
    cov_inv: np.ndarray,
    on_curve: np.ndarray,
    n_nodes: int = DEFAULT_NODES,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (z_nodes, R', b, c0); rows with on_curve False do not depend on the cosmology."""
    z, y, cov_inv = np.asarray(z), np.asarray(y), np.asarray(cov_inv)
    on_curve = np.asarray(on_curve, dtype=bool)
    lnz = np.log(z[on_curve])
    lo, hi = lnz.min(), lnz.max()
    h = (hi - lo) / (n_nodes - 1)
    x_nodes = lo + h * np.arange(-1, n_nodes + 1)  # one extra node on each side

    rows = np.where(on_curve)[0]
    k = np.clip(np.floor((lnz - lo) / h).astype(int), 0, n_nodes - 2) + 1
    t = (lnz - x_nodes[k]) / h
    weights = np.stack(
        [(-t**3 + 2 * t**2 - t) / 2, (3 * t**3 - 5 * t**2 + 2) / 2,
         (-3 * t**3 + 4 * t**2 + t) / 2, (t**3 - t**2) / 2], axis=1)
    W = np.zeros((z.size, n_nodes + 2))
    for j in range(4):
        np.add.at(W, (rows, k - 1 + j), weights[:, j])

    L = np.linalg.cholesky(cov_inv)
    X = L.T @ np.column_stack([np.ones(z.size), W])  # whitened design matrix, M column first
    yw = L.T @ y
    Qm, R = np.linalg.qr(X)
    proj = Qm.T @ yw
    out = yw - Qm @ proj  # part of the data outside the column space
    return tuple(jnp.asarray(a) for a in (np.exp(x_nodes), R[1:, 1:], proj[1:], out @ out))


def compressed_chi2(g: jnp.ndarray, terms) -> jnp.ndarray:
    """chi^2 from the node moduli g and the terms returned by build_compression."""
    _, R, b, c0 = terms
    v = b - R @ g
    return c0 + v @ v


def node_moduli(d_L_grid: jnp.ndarray, grid_z: jnp.ndarray, z_nodes: jnp.ndarray) -> jnp.ndarray:
    """5 log10 d_L + 25 at the nodes, from d_L on a uniform redshift grid."""
    return 5.0 * jnp.log10(interp_linspace(z_nodes, grid_z, d_L_grid)) + 25.0
