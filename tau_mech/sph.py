"""Phase 3 - 2D weakly-compressible Smoothed Particle Hydrodynamics (SPH).

A self-contained, dependency-light (numpy + scipy) SPH engine written for this
study. It is a *CPU prototype* used to (a) validate the numerical machinery
against analytic limits and (b) produce an order-of-magnitude study of how
physiological shear deforms Tau condensate droplets. Production GPU runs
(Taichi/Warp, or 3D) are a documented follow-up (see README section 9).

Physics
-------
  * cubic-spline kernel (Monaghan 1992)
  * density summation
  * Tait equation of state (weakly compressible, gamma = 7)
  * Morris et al. (1997) viscous force (per-phase dynamic viscosity)
  * surface tension: Tartakovsky & Meakin (2005) inter-particle interaction
    force (short-range repulsion + medium-range attraction for droplet-droplet;
    repulsion only across the droplet/solvent interface)
  * XSPH velocity smoothing (Monaghan 1989) for stability
  * solid walls as frozen particles (Couette cell: periodic x, moving walls)

All quantities are dimensionless; the capillary number is the physically
meaningful reporting quantity (see :func:`droplet_shear_sweep`).

Validation
----------
  * :func:`validate_couette` - recovers the analytic linear Couette profile
  * :func:`validate_laplace` - Laplace law dP = sigma/R scaling across radius
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# SPH kernels
# ---------------------------------------------------------------------------


def cubic_spline(r: np.ndarray, h: float) -> np.ndarray:
    """Cubic spline kernel W(r, h) in 2D (Monaghan 1992), NORMALIZED.

    Published form: W(q) = c*(1 - 1.5 q^2 + 0.75 q^3), q<=1;
    c*(2-q)^3/4, 1<q<2, with c = 10/(7 pi h^2), so that int W dA = 1.
    (An early draft used a 2/3-scaled variant; the density mass-
    renormalization step makes the discretized system exactly invariant to a
    constant kernel scale - every force term is a product m*W or m*dW - but
    the code now matches the cited kernel.)
    """
    q = r / h
    w = np.zeros_like(q)
    c = 10.0 / (7.0 * np.pi * h * h)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = c * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    w[m2] = c * ((2.0 - q[m2]) ** 3) / 4.0
    return w


def cubic_spline_dwdr(r: np.ndarray, h: float) -> np.ndarray:
    """dW/dr of the cubic spline kernel (negative inside the support)."""
    q = r / h
    dw = np.zeros_like(q)
    c = 10.0 / (7.0 * np.pi * h * h * h)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    dw[m1] = c * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    dw[m2] = c * (-0.75 * (2.0 - q[m2]) ** 2)
    return dw


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class SPHState:
    """Particle state. phase: 0 = solvent, 1 = droplet, 2 = wall."""
    pos: np.ndarray                 # (N, 2)
    vel: np.ndarray                 # (N, 2)
    mass: np.ndarray                # (N,)
    phase: np.ndarray               # (N,) int
    rho: np.ndarray = None          # (N,)
    pressure: np.ndarray = None     # (N,)
    h: float = 1.0
    rho0: float = 1.0
    domain: Tuple[float, float, float, float] = (0.0, 0.0, 24.0, 16.0)
    wall_speed_top: float = 0.0
    wall_speed_bottom: float = 0.0

    @property
    def n(self) -> int:
        return len(self.pos)


@dataclass
class SPHParams:
    """Numerical parameters (dimensionless; tuned by validation)."""
    h: float = 1.0
    c_s: float = 10.0               # speed of sound (Tait); ~10x flow speeds
    gamma: float = 7.0
    mu_solvent: float = 0.05
    mu_droplet: float = 0.5
    A_surf: float = 10.0            # immiscibility repulsion strength
                                    # (mixed-phase pairs only; the CSF is the
                                    # cohesion/surface tension)
    B_surf: float = 0.0             # TM pair attraction DISABLED (replaced by
                                    # the CSF surface tension; see below)
    sigma_surf: float = 1.0         # CONTINUUM SURFACE FORCE (CSF) surface
                                    # tension coefficient (Brackbill 1992 /
                                    # Adami 2010): F_s = sigma * kappa * grad(c)
    # Model history (2026-08-05): the Tartakovsky-Meakin pair attraction was
    # abandoned after the CLUMPING discovery - the TM force is proportional
    # to W(r), which is nearly flat at small r, so it provides no short-range
    # barrier and particles collapsed into clumps (NN ~ 0.14 vs spacing 0.5;
    # 75% clumped), silently turning the "droplet" into a granular solid and
    # producing an artifact surface tension (anti-Laplace dP ~ R). The CSF
    # model decouples sigma from the pair forces: sigma is a direct parameter
    # and the Laplace law dP = sigma/R becomes a VERIFICATION, not a
    # calibration. (Full audit trail in PHASES_2_5_REPORT.md.)
    r_rep: float = 0.45             # repulsion cutoff (units of h): set BELOW
                                    # the lattice spacing (0.5h) so the
                                    # mixed-only immiscibility barrier acts as
                                    # an overlap barrier (no force at
                                    # first-neighbour distance). The earlier
                                    # 0.6h range pushed droplet rim particles
                                    # inward at first-neighbour distance,
                                    # crushing the interface shell
                                    # (NN < 0.35 for ~23% of the droplet) and
                                    # suppressing the CSF Laplace signal;
                                    # 0.45h with a steep ramp gives 0%
                                    # clumping (audit: PHASES_2_5_REPORT.md)
    switch_delta: float = 0.05      # smooth-switch half-width (units of h)
                                    # for the repulsion ramp
    r_att: float = 1.0              # attraction cutoff (units of h)
    xsph: float = 0.1              # XSPH smoothing; floored-density weighted
    eps: float = 0.01
    rho_floor: float = 0.5          # density floor (fraction of rho0) in the EOS
    p_cap: float = 0.3              # negative pressure cap (fraction of B)
    allow_neg_p: bool = True        # allow negative EOS pressure (tension).
                                    # The p >= 0 clamp corrupts the two-phase
                                    # pressure reference (masks solvent
                                    # tension, zeroes the Laplace signal);
                                    # stability is provided by the soft-core
                                    # repulsion + artificial viscosity + XSPH
    alpha_art: float = 0.1          # Monaghan artificial viscosity
    shepard: float = 1.0            # partition-of-unity density correction
                                    # (0 = off, 1 = one iteration); removes the
                                    # lattice-scale E0 imprint on rho -> P
    n_color_smooth: int = 2         # kernel-smoothing passes applied to the
                                    # CSF color field BEFORE computing grad(c)
                                    # and the curvature kappa = -div(n_hat).
                                    # The raw color field is discontinuous at
                                    # the interface; the divergence stencil is
                                    # then truncated there and kappa comes out
                                    # with the WRONG sign on the droplet rim
                                    # (measured -0.58 vs expected +1/R), which
                                    # reverses the surface force and EXPANDS
                                    # the droplet. Smoothing (Morris 2000;
                                    # Cummins & Rudman 1999) makes grad/div
                                    # well-defined; combined with the
                                    # renormalized divergence (Adami 2010) it
                                    # recovers kappa ~ +1/R at the interface
                                    # (audit trail: PHASES_2_5_REPORT.md).


def hexagonal_pack(x0, y0, x1, y1, spacing) -> np.ndarray:
    """Hexagonal-lattice points covering a rectangle [x0, x1) x [y0, y1].

    The x-interval is HALF-OPEN: the point at x = x1 is omitted because, in a
    periodic-x domain, x1 is the same physical point as x0 (a duplicate).
    Including it would (a) create a doubled particle at the seam and (b) make
    the periodic cKDTree reject the position as out-of-range. The y-interval
    is closed (y1 is a real boundary, not periodic - walls sit beyond it).
    """
    pts = []
    dy = spacing * np.sqrt(3.0) / 2.0
    row, y = 0, y0
    while y <= y1:
        x = x0 + (0.5 * spacing if row % 2 else 0.0)
        k = 0
        while x + k * spacing < x1:
            pts.append((x + k * spacing, y))
            k += 1
        y += dy
        row += 1
    return np.asarray(pts)


def particle_mass(spacing: float, rho0: float) -> float:
    """Mass per particle such that the density sum equals rho0 for a
    hexagonal lattice: m = rho0 * (cell area) = rho0 * spacing^2 * sqrt(3)/2."""
    return rho0 * spacing * spacing * np.sqrt(3.0) / 2.0


# ---------------------------------------------------------------------------
# Neighbor search + physics
# ---------------------------------------------------------------------------


def build_pairs(pos: np.ndarray, h: float,
                x_period: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All particle pairs within the kernel support: (pairs (K,2), d (K,), e (K,2)).

    ``x_period`` enables the periodic-x (Couette) topology. Without it, the
    two vertical domain edges lose ~half their neighbours (the raw cKDTree
    sees x=0 and x=Lx as 24 units apart), producing a ~20% density deficit
    and a large spurious pressure transient there that propagates to the
    droplet. That transient is the root cause of the sigma-independent
    "droplet shape oscillation" (it persisted with ALL surface forces off -
    scripts/diag_classify_osc.py, 2026-08-11).

    With ``x_period`` set, the neighbour search uses a periodic cKDTree and
    the pair displacement vector is wrapped to the minimum image in x, so the
    kernel weight and the force direction are evaluated at the correct
    (shortest) separation.
    """
    boxsize = [x_period, 0.0] if x_period is not None else None
    tree = cKDTree(pos, boxsize=boxsize)
    pairs = tree.query_pairs(r=2.0 * h, output_type="ndarray")
    if len(pairs) == 0:
        return (np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty((0, 2)))
    r12 = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    if x_period is not None:
        r12[:, 0] -= x_period * np.round(r12[:, 0] / x_period)  # minimum image
    d = np.linalg.norm(r12, axis=1)
    e = r12 / np.maximum(d, 1e-12)[:, None]
    return pairs, d, e


def compute_density(state: SPHState, params: SPHParams, pairs, d) -> None:
    n = state.n
    if len(pairs):
        w = cubic_spline(d, params.h)
        idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
        wts = np.concatenate([state.mass[pairs[:, 1]] * w,
                              state.mass[pairs[:, 0]] * w])
        rho = np.bincount(idx, weights=wts, minlength=n)
    else:
        rho = np.zeros(n)
    rho += state.mass * cubic_spline(np.zeros(1), params.h)[0]  # self term
    # Shepard (partition-of-unity) density correction, one iteration:
    #   rho_i = sum_j m_j W_ij / sum_j (m_j/rho_j) W_ij
    # The raw mass-sum has a lattice-scale E0 imprint (hex-lattice shells)
    # which the stiff EOS (gamma=7) amplifies into pressure swings of +/-2.5
    # that buried the Laplace signal (see PHASES_2_5_REPORT.md, Phase 3
    # failures: anti-Laplace dP). The correction divides by the same-hold
    # kernel-sum, canceling the imprint. Standard SPH practice (Shepard
    # filter / density renormalization).
    if params.shepard > 0:
        rho_eff0 = np.maximum(rho, params.rho_floor * state.rho0)
        wsum = np.zeros(n)
        if len(pairs):
            wts2 = np.concatenate([
                state.mass[pairs[:, 1]] / rho_eff0[pairs[:, 1]] * w,
                state.mass[pairs[:, 0]] / rho_eff0[pairs[:, 0]] * w])
            wsum = np.bincount(idx, weights=wts2, minlength=n)
        # self term of the partition of unity
        wsum += (state.mass / rho_eff0) * cubic_spline(np.zeros(1), params.h)[0]
        rho = rho / np.maximum(wsum, 1e-9)
    state.rho = rho
    # density floor prevents the p/rho^2 blow-up at under-resolved edges/walls
    rho_eff = np.maximum(rho, params.rho_floor * state.rho0)
    b = params.c_s ** 2 * state.rho0 / params.gamma
    p = b * ((rho_eff / state.rho0) ** params.gamma - 1.0)
    if not params.allow_neg_p:
        # legacy clamp: negative pressure is a tensile-instability source in
        # WCSPH, but clamping it masks solvent tension and zeroes the Laplace
        # signal. With the strong TM repulsive core the default now allows
        # tension; the clamp remains available as a switch.
        p = np.maximum(p, 0.0)
    state.pressure = p


def compute_acceleration(state: SPHState, params: SPHParams, pairs, d, e) -> np.ndarray:
    """Pairwise anti-symmetric forces, accumulated in ONE bincount pass.

    f is the force on particle i from particle j for every pair; the force on
    j is -f. This keeps the loop vectorized and ~10-20x faster than np.add.at.
    """
    vel, mass, phase = state.vel, state.mass, state.phase
    rho_raw = np.maximum(state.rho, params.rho_floor * state.rho0)  # floored
    pressure = state.pressure
    n = state.n
    acc = np.zeros((n, 2))
    if len(pairs) == 0:
        return acc
    i, j = pairs[:, 0], pairs[:, 1]
    dw = cubic_spline_dwdr(d, params.h)
    w = cubic_spline(d, params.h)
    mj = mass[j]

    # pressure gradient (symmetric form); e points j -> i, grad_i W = dw * e
    pi = pressure[i] / rho_raw[i] ** 2 + pressure[j] / rho_raw[j] ** 2
    f = -mj[:, None] * pi[:, None] * (dw[:, None] * e)

    # Morris viscosity (resists relative motion).
    # Canonical form (Morris et al. 1997; Monaghan 2005):
    #   dv_i/dt = sum_j m_j (mu_i + mu_j) v_ij / (rho_i rho_j)
    #             * (r_ij . grad W_ij) / (r_ij^2 + eta^2)
    # with v_ij = v_i - v_j. Here r_ij . grad W_ij = d * dw (grad W = dw * e,
    # r_ij = d * e), so (r_ij . grad W)/(r_ij^2) -> dw/d. The (mu_i + mu_j)
    # is the FULL sum, NOT the average: a 0.5 factor under-delivers the
    # viscosity by 2x (measured nu_eff/nu = 0.433 vs 0.866 after the fix,
    # scripts/diag_viscosity.py, 2026-08-10).
    mu = np.where(phase == 1, params.mu_droplet, params.mu_solvent)
    muij = mu[i] + mu[j]
    f += (mj[:, None] * (muij / (rho_raw[i] * rho_raw[j]))[:, None]
          * (dw / (d + params.eps * params.h))[:, None] * (vel[i] - vel[j]))

    # Monaghan artificial viscosity (damps approaching-particle noise)
    if params.alpha_art > 0:
        vij_dot_e = np.einsum("ij,ij->i", vel[i] - vel[j], e)
        approaching = vij_dot_e < 0
        if approaching.any():
            mu_a = params.h * vij_dot_e[approaching] / (d[approaching] + params.eps * params.h)
            rhobar = 0.5 * (rho_raw[i] + rho_raw[j])[approaching]
            Pi = -params.alpha_art * params.c_s * mu_a / np.maximum(rhobar, 1e-9)
            f[approaching] -= (mj[approaching, None] * Pi[:, None]
                               * (dw[approaching, None] * e[approaching]))

    # surface tension (Tartakovsky & Meakin 2005), with a SMOOTH
    # repulsion -> attraction transition across [r_rep-delta, r_rep+delta].
    # The sharp switch at r_rep caused particles to rattle at the force
    # discontinuity (vmax spikes, noisy pressure); a cosine ramp makes the
    # force continuous (documented in PHASES_2_5_REPORT.md, Phase 3).
    if params.A_surf > 0 or params.B_surf > 0:
        same = (phase[i] == 1) & (phase[j] == 1)          # droplet-droplet
        mixed = ((phase[i] == 1) & (phase[j] == 0)) | ((phase[i] == 0) & (phase[j] == 1))
        r_rep = params.r_rep * params.h
        r_att = params.r_att * params.h
        delta = params.switch_delta * params.h
        t = (d - (r_rep - delta)) / (2.0 * delta)
        t = np.clip(t, 0.0, 1.0)
        rep_w = 0.5 * (1.0 + np.cos(np.pi * t))  # 1 inside, 0 outside the core
        att_w = 1.0 - rep_w
        att_w[d >= r_att] = 0.0
        rep_w[d >= r_att] = 0.0
        we = w[:, None] * e
        if params.A_surf > 0:
            # immiscibility repulsion on MIXED-phase pairs ONLY. A same-phase
            # soft core is unphysical here: with B=0 (no pair cohesion; the
            # CSF is the surface tension) a same-phase repulsion rarefies the
            # droplet interior (measured dP(0) = -0.54 at sigma=0, droplet
            # rho 0.995 < solvent 1.000), which would corrupt the Laplace
            # reference pressure (see PHASES_2_5_REPORT.md, CSF audit).
            f[mixed] += (params.A_surf * mj[mixed, None] * we[mixed]
                         * rep_w[mixed, None])
        if params.B_surf > 0:
            f[same] -= (params.B_surf * mj[same, None] * we[same]
                        * att_w[same, None])

    # accumulate: acc[i] += f ; acc[j] -= f
    idx = np.concatenate([i, j])
    fw = np.concatenate([f, -f])
    acc[:, 0] = np.bincount(idx, weights=fw[:, 0], minlength=n)
    acc[:, 1] = np.bincount(idx, weights=fw[:, 1], minlength=n)
    return acc


def smooth_color_field(state: SPHState, params: SPHParams, pairs, d,
                       n_passes: int = 2) -> np.ndarray:
    """Shepard-normalized kernel smoothing of the CSF color field.

    c_tilde_i = sum_j V_j c_j W_ij / sum_j V_j W_ij, iterated n_passes times.
    The raw color field (1 droplet / 0 solvent) is discontinuous at the
    interface, which corrupts the curvature stencil there; smoothing spreads
    the transition over ~2h so grad(c) and kappa = -div(n_hat) are
    well-defined (Morris 2000; Cummins & Rudman 1999).
    """
    c = (state.phase == 1).astype(np.float64)
    vol = state.mass / np.maximum(state.rho, params.rho_floor * state.rho0)
    w = cubic_spline(d, params.h)
    w0 = cubic_spline(np.zeros(1), params.h)[0]
    for _ in range(max(1, n_passes)):
        idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
        num_w = np.concatenate([vol[pairs[:, 1]] * c[pairs[:, 1]] * w,
                                vol[pairs[:, 0]] * c[pairs[:, 0]] * w])
        den_w = np.concatenate([vol[pairs[:, 1]] * w, vol[pairs[:, 0]] * w])
        num = np.bincount(idx, weights=num_w, minlength=state.n)
        den = np.bincount(idx, weights=den_w, minlength=state.n)
        num += vol * c * w0                      # self terms
        den += vol * w0
        c = num / np.maximum(den, 1e-9)
    return c


def color_field_curvature(state: SPHState, params: SPHParams,
                          pairs, d, e):
    """Smoothed color field, its gradient, and the renormalized curvature.

    Single source of truth for the CSF interface operators: both
    :func:`compute_surface_force` and the audit diagnostic
    (scripts/diag_csf_symmetry.py) use THIS function, so the audit can never
    silently diverge from the solver.

    Returns (kappa, grad, nhat).

    The divergence stencils below are SYMMETRIC field evaluations (fixed
    2026-09-02; audit scripts/diag_csf_symmetry.py): like the gradient, the
    divergence of a FIELD accumulates on BOTH endpoints of every pair, each
    with its own volume. The original one-sided accumulation (bincount over
    the first pair endpoint only) saw query_pairs' (min, max) ordering and
    gave every particle a curvature stencil containing only its
    higher-indexed neighbours: measured 0.51 of its pairs, NOT
    permutation-invariant (kappa moved by up to 1.5 under a pure relabel),
    azimuthally biased on a circle (rim sector means 0.29-0.62 vs 1/R =
    0.333), and non-conservative (net internal surface force = 10.8% of its
    own magnitude) - the drift engine of the zero-shear control.

    Per pair (i, j) with e pointing j->i and s = (nhat_j - nhat_i).(dw*e):
      div_i += V_j * s   and   div_j += V_i * s
    (same scalar, the OTHER particle's volume; the sign closes because
    e_ij = -e_ji under the minimum-image wrap). The kernel-sum denominator
    accumulates the same way plus each particle's own self term.
    """
    n = state.n
    i, j = pairs[:, 0], pairs[:, 1]
    c = smooth_color_field(state, params, pairs, d,
                           n_passes=params.n_color_smooth)
    vol = state.mass / np.maximum(state.rho, params.rho_floor * state.rho0)
    dw = cubic_spline_dwdr(d, params.h)
    w = cubic_spline(d, params.h)
    w0 = cubic_spline(np.zeros(1), params.h)[0]
    # gradient of the smoothed color field:
    #   grad_i += V_j (c_j - c_i) dw e ;  grad_j += V_i (c_j - c_i) dw e
    # (a gradient is a FIELD evaluation - both endpoints accumulate the SAME
    # sign with their own volumes; the anti-symmetric +f/-f pattern used for
    # pairwise FORCES is wrong here. It flipped the normal at the outer rim
    # (n_hat.r_hat = +0.30 instead of -1) which corrupted kappa and EXPANDED
    # the droplet - caught by scripts/diag_csf_norm.py, 2026-08-05.)
    dc_i = (c[j] - c[i]) * vol[j] * dw      # weight for particle i
    dc_j = (c[j] - c[i]) * vol[i] * dw      # weight for particle j
    idx = np.concatenate([i, j])
    gx = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]),
                     minlength=n)
    gy = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]),
                     minlength=n)
    grad = np.stack([gx, gy], axis=1)
    ng = np.linalg.norm(grad, axis=1)
    nhat = grad / np.maximum(ng, 1e-12)[:, None]
    # RENORMALIZED divergence (Adami 2010): divide the raw stencil sum by the
    # kernel-sum on the same stencil so a truncated interface neighbourhood
    # does not bias the curvature sign/magnitude. SYMMETRIC accumulation
    # (both endpoints) as derived in the docstring above.
    s_pair = np.sum((nhat[j] - nhat[i]) * (dw[:, None] * e), axis=1)
    div_n_raw = np.bincount(idx, weights=np.concatenate([vol[j] * s_pair,
                                                         vol[i] * s_pair]),
                            minlength=n)
    den = np.bincount(idx, weights=np.concatenate([vol[j] * w, vol[i] * w]),
                      minlength=n) + vol * w0
    div_n = div_n_raw / np.maximum(den, 1e-9)
    kappa = -div_n
    return kappa, grad, nhat


def compute_surface_force(state: SPHState, params: SPHParams,
                          pairs, d, e) -> np.ndarray:
    """Continuum surface force (CSF) acceleration (Brackbill 1992; Adami 2010).

    a_s = +sigma * kappa * grad(c_tilde) / rho, where
    c_tilde = kernel-smoothed color field (see :func:`smooth_color_field`)
    n       = grad(c_tilde)  (interface normal * delta function)
    kappa   = -div(n_hat), RENORMALIZED by the kernel-sum (Adami 2010) so the
              truncated interface stencil does not bias the curvature
    sigma is a DIRECT parameter; the Laplace law dP = sigma/R is a
    VERIFICATION (see validate_laplace), not a calibration.

    Sign convention (verified empirically, 2026-08-05): with c = 1 inside
    the droplet, grad(c_tilde) points INWARD at the interface and kappa =
    +1/R for a circle, so +sigma*kappa*grad(c_tilde) points inward (surface
    tension pulls the interface toward its center of curvature) and the
    hydrostatic jump is dP = +sigma/R. Two failures were caught by
    scripts/diag_csf_sign.py and scripts/diag_csf_field.py: (1) an opposite
    prefactor sign, and (2) a truncated raw-color divergence stencil that
    gave kappa < 0 on the droplet rim and EXPANDED the droplet even with the
    correct prefactor. Both are documented in PHASES_2_5_REPORT.md.
    """
    n = state.n
    acc = np.zeros((n, 2))
    if params.sigma_surf <= 0 or len(pairs) == 0:
        return acc
    # interface operators (single source of truth; symmetric stencils)
    kappa, grad, _nhat = color_field_curvature(state, params, pairs, d, e)
    # surface acceleration: a_s = +sigma * kappa * grad(c_tilde) / rho
    acc[:, 0] = params.sigma_surf * kappa * grad[:, 0] / np.maximum(state.rho, 1e-9)
    acc[:, 1] = params.sigma_surf * kappa * grad[:, 1] / np.maximum(state.rho, 1e-9)
    return acc


def step(state: SPHState, params: SPHParams, dt: float) -> None:
    """One velocity-Verlet + XSPH step (single neighbor search)."""
    pairs, d, e = build_pairs(state.pos, params.h,
                              x_period=state.domain[2] - state.domain[0])
    a0 = compute_acceleration(state, params, pairs, d, e)
    # CSF surface tension is a full force term, so in velocity-Verlet it must
    # be evaluated in BOTH half-steps (0.5*dt each, total dt) exactly like the
    # pressure force. Evaluating it only in a1 halves its time weight, so the
    # equilibrium condition becomes grad P = 0.5 * F_CSF and the measured
    # Laplace jump is 0.5 * sigma/R instead of sigma/R (root cause of the
    # sigma_eff ~ 0.46-0.52 sigma_input under-delivery; audit 2026-08-14).
    a0 = a0 + compute_surface_force(state, params, pairs, d, e)
    state.vel = state.vel + 0.5 * dt * a0
    state.pos = state.pos + dt * state.vel
    _wrap_x(state)
    pairs, d, e = build_pairs(state.pos, params.h,
                              x_period=state.domain[2] - state.domain[0])
    compute_density(state, params, pairs, d)
    a1 = compute_acceleration(state, params, pairs, d, e)
    a1 = a1 + compute_surface_force(state, params, pairs, d, e)
    state.vel = state.vel + 0.5 * dt * a1

    # XSPH velocity smoothing (Monaghan 1989), free particles only.
    # Uses density-FLOORED symmetric weights: the raw m/rho diverges where rho
    # is under-resolved (near walls / at droplet rim), which drove the
    # exponential instability seen in early development (see README, SPH notes).
    if params.xsph > 0 and len(pairs):
        rho_floor = params.rho_floor * state.rho0
        rho_i = np.maximum(state.rho[pairs[:, 0]], rho_floor)
        rho_j = np.maximum(state.rho[pairs[:, 1]], rho_floor)
        w = cubic_spline(d, params.h)
        cij = params.xsph * state.mass[pairs[:, 1]] * w / (0.5 * (rho_i + rho_j))
        dv = (state.vel[pairs[:, 1]] - state.vel[pairs[:, 0]]) * cij[:, None]
        free_i = state.phase[pairs[:, 0]] != 2
        free_j = state.phase[pairs[:, 1]] != 2
        if free_i.any():
            idx = pairs[free_i, 0]
            state.vel[:, 0] += np.bincount(idx, weights=dv[free_i, 0], minlength=state.n)
            state.vel[:, 1] += np.bincount(idx, weights=dv[free_i, 1], minlength=state.n)
        if free_j.any():
            idx = pairs[free_j, 1]
            state.vel[:, 0] -= np.bincount(idx, weights=dv[free_j, 0], minlength=state.n)
            state.vel[:, 1] -= np.bincount(idx, weights=dv[free_j, 1], minlength=state.n)

    # walls: pin wall particles; top/bottom rows move with the wall
    wall = state.phase == 2
    state.vel[wall] = 0.0
    ymid = 0.5 * (state.domain[1] + state.domain[3])
    top = wall & (state.pos[:, 1] > ymid)
    bot = wall & (state.pos[:, 1] < ymid)
    state.vel[top, 0] = state.wall_speed_top
    state.vel[bot, 0] = state.wall_speed_bottom


def _wrap_x(state: SPHState) -> None:
    """Fold x into the periodic interval [x0, x1).

    Uses a modulo fold (robust to multi-period drift and to floating-point
    noise at the seam) rather than a single +/-L shift. The old
    ``pos < x0 -> pos + L`` test wrapped a particle sitting at x0 - 1e-18
    (pure rounding error at the seam) all the way to x1, which the periodic
    cKDTree then rejected as out of range (2026-08-11).
    """
    x0, x1 = state.domain[0], state.domain[2]
    L = x1 - x0
    state.pos[:, 0] = x0 + np.mod(state.pos[:, 0] - x0, L)
    # fold pure fp noise that landed exactly on the upper seam back to x0
    state.pos[:, 0] = np.where(np.isclose(state.pos[:, 0], x1, atol=1e-9),
                               x0, state.pos[:, 0])


def run(state: SPHState, params: SPHParams, n_steps: int, dt: float,
        callback: Optional[Callable[[int], None]] = None, every: int = 500) -> Dict:
    t0 = time.time()
    if state.rho is None:
        # density initialization with mass renormalization: the lattice density
        # sum is < rho0 (kernel truncation), so rescale masses so the fluid
        # starts at rho0 and the Tait EOS is at equilibrium (no initial shock).
        pairs, d, e = build_pairs(state.pos, params.h,
                                  x_period=state.domain[2] - state.domain[0])
        compute_density(state, params, pairs, d)
        free = state.phase != 2
        scale = state.rho0 / max(float(state.rho[free].mean()), 1e-9)
        state.mass *= scale
        compute_density(state, params, pairs, d)
    for s in range(n_steps):
        step(state, params, dt)
        if callback is not None and s % every == 0:
            callback(s)
    return {"steps": n_steps, "dt": dt, "wall_time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Setup: Couette cell with droplet
# ---------------------------------------------------------------------------


def _wall_lattice(x0, y0, x1, y1, spacing, n_wall_layers):
    """Continuous hex lattice for a periodic-x cell with frozen top/bottom walls.

    Generates ONE lattice covering the fluid rows [y0, y1] plus n_wall_layers
    ghost rows above and below, so the frozen wall rows are the EXACT lattice
    continuation of the fluid (same dy = spacing*sqrt(3)/2 and same alternating
    x-offset). The earlier construction placed walls on a 0.5-spacing grid
    misaligned with the 0.433 lattice, which left a spurious density layer at
    the boundary that launched an acoustic transient into the droplet.

    Returns (pos, fluid_mask, inner_bottom, inner_top):
      pos          - (N, 2) lattice positions
      fluid_mask   - (N,) bool, True for rows inside [y0, y1]
      inner_bottom - y of the innermost frozen row below the fluid
      inner_top    - y of the innermost frozen row above the fluid
    """
    dy = spacing * np.sqrt(3.0) / 2.0
    ylo = y0 - n_wall_layers * dy
    yhi = y1 + n_wall_layers * dy
    pos = hexagonal_pack(x0, ylo, x1, yhi, spacing)
    y = pos[:, 1]
    # The row positions come from an ACCUMULATING ``y += dy`` loop, so the row
    # that should sit exactly at y0 (a multiple of dy) lands at y0 - 1e-16.
    # A strict ``y >= y0`` test then mislabels that row as a WALL, giving the
    # bottom boundary one extra frozen row and breaking top/bottom symmetry
    # (the Couette profile showed a bulk drift and a large top-wall slip).
    # Classify with a tolerance far below the row spacing (dy) but far above
    # the ~1e-15 accumulation drift.
    tol = 1e-9
    fluid = (y >= y0 - tol) & (y <= y1 + tol)
    inner_bottom = y0 - dy
    top_fluid = float(y[fluid].max()) if fluid.any() else y1
    inner_top = top_fluid + dy
    return pos, fluid, inner_bottom, inner_top


def make_couette_droplet_state(params: SPHParams, domain=(0.0, 0.0, 24.0, 16.0),
                               spacing=0.5, droplet_center=None, droplet_radius=2.0,
                               n_wall_layers=4, rho0=1.0) -> SPHState:
    """Two-phase Couette cell: solvent + circular droplet, frozen walls.

    Walls are the lattice continuation of the fluid (see :func:`_wall_lattice`).
    """
    x0, y0, x1, y1 = domain
    m = particle_mass(spacing, rho0)
    pos, fluid, _, _ = _wall_lattice(x0, y0, x1, y1, spacing, n_wall_layers)
    phase = np.where(fluid, np.int8(0), np.int8(2))
    dc = np.asarray(droplet_center or ((x0 + x1) / 2, (y0 + y1) / 2))
    droplet = fluid & (np.linalg.norm(pos - dc, axis=1) <= droplet_radius)
    phase[droplet] = 1
    state = SPHState(pos=pos, vel=np.zeros_like(pos), mass=np.full(len(pos), m),
                     phase=phase, h=params.h, rho0=rho0, domain=domain)
    return state


def droplet_deformation(state: SPHState) -> Dict:
    """Inertia-tensor deformation descriptors of the droplet phase."""
    m = state.phase == 1
    n = int(m.sum())
    if n < 4:
        return {"aspect_ratio": np.nan, "taylor": np.nan, "angle_deg": np.nan,
                "n_droplet": n, "com": [np.nan, np.nan]}
    pts = state.pos[m]
    com = pts.mean(axis=0)
    cov = (pts - com).T @ (pts - com) / n
    evals, evecs = np.linalg.eigh(cov)
    a = np.sqrt(evals[1])
    b = np.sqrt(evals[0])
    D = (a - b) / (a + b)
    angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    return {"aspect_ratio": float(a / b), "taylor": float(D),
            "angle_deg": float(angle % 180.0), "n_droplet": n,
            "com": com.tolist()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_couette(params: SPHParams, n_steps: int = 6000, dt: float = 0.008,
                     U_wall: float = 2.0, domain=(0.0, 0.0, 24.0, 8.0),
                     spacing: float = 0.5, n_wall_layers: int = 4) -> Dict:
    """Steady-state Couette validation (planar Couette, periodic x).

    The no-slip planes are the innermost frozen wall rows (the lattice
    continuation of the fluid), so the wall-to-wall distance is
    H_wall = inner_top - inner_bottom and the expected shear rate is
    gamma_expected = 2*U_wall/H_wall. After running to steady state we report
    (i) the linearity of the measured profile (R^2 of a straight-line fit),
    (ii) the slope ratio (fitted shear rate / expected shear rate) - a direct
    check of the effective viscosity + wall coupling, and (iii) the deviation
    of the zero-velocity midpoint from the domain centre.

    The caller must pass a configuration whose viscosity reaches steady state
    within n_steps*dt (viscous time tau = H_wall**2/(nu*pi**2)).
    """
    x0, y0, x1, y1 = domain
    m = particle_mass(spacing, 1.0)
    pos, fluid, inner_bottom, inner_top = _wall_lattice(
        x0, y0, x1, y1, spacing, n_wall_layers)
    H_wall = inner_top - inner_bottom
    phase = np.where(fluid, np.int8(0), np.int8(2))
    state = SPHState(pos=pos, vel=np.zeros_like(pos), mass=np.full(len(pos), m),
                     phase=phase, h=params.h, rho0=1.0, domain=domain,
                     wall_speed_top=U_wall, wall_speed_bottom=-U_wall)
    run(state, params, n_steps, dt)

    free = state.phase == 0
    y = state.pos[free, 1]
    ux = state.vel[free, 0]
    bins = np.linspace(y0, y1, 25)
    idx = np.digitize(y, bins) - 1
    yb = (bins[:-1] + bins[1:]) / 2
    ub = np.array([ux[idx == k].mean() if (idx == k).any() else np.nan
                   for k in range(len(bins) - 1)])
    ok = ~np.isnan(ub)
    yy, uu = yb[ok], ub[ok]

    # (i) linearity: R^2 of a least-squares straight-line fit over the FULL
    # channel and over the CENTRAL region only. The frozen walls transmit
    # momentum through the kernel support (2h), so the fluid in a boundary
    # layer ~2h thick near each wall slips relative to the ideal no-slip
    # profile; a full-channel fit therefore under-measures the bulk shear
    # rate. The central fit (y in [2h, H-2h], away from the wall layers)
    # isolates the bulk Couette slope - the quantity the study droplet
    # actually experiences (it sits at the channel centre).
    A = np.vstack([yy, np.ones_like(yy)]).T
    coef, *_ = np.linalg.lstsq(A, uu, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((uu - pred) ** 2))
    ss_tot = float(np.sum((uu - uu.mean()) ** 2))
    r2_fit = 1.0 - ss_res / max(ss_tot, 1e-12)
    slope_fit = float(coef[0])

    z = 2.0 * params.h  # wall momentum-transmission zone
    cok = (yy > y0 + z) & (yy < y1 - z)
    if cok.sum() >= 4:
        Ac = np.vstack([yy[cok], np.ones(cok.sum())]).T
        cc, *_ = np.linalg.lstsq(Ac, uu[cok], rcond=None)
        predc = Ac @ cc
        ssr_c = float(np.sum((uu[cok] - predc) ** 2))
        sst_c = float(np.sum((uu[cok] - uu[cok].mean()) ** 2))
        r2_central = 1.0 - ssr_c / max(sst_c, 1e-12)
        slope_central = float(cc[0])
    else:
        r2_central, slope_central = float("nan"), float("nan")

    # (ii) slope ratio vs the geometry-derived expected shear rate
    gamma_expected = 2.0 * U_wall / H_wall
    slope_ratio = slope_fit / gamma_expected
    slope_ratio_central = slope_central / gamma_expected

    # analytic no-slip profile between the innermost wall rows (u=-U at the
    # inner bottom wall row, +U at the inner top wall row)
    analytic = -U_wall + gamma_expected * (yy - inner_bottom)
    ss_an = float(np.sum((uu - analytic) ** 2))
    ss_an_tot = float(np.sum((analytic - analytic.mean()) ** 2))
    r2_analytic = 1.0 - ss_an / max(ss_an_tot, 1e-12)

    # (iii) zero-velocity midpoint vs domain centre
    y_mid_fit = -coef[1] / max(abs(coef[0]), 1e-12)
    y_mid_dev = y_mid_fit - 0.5 * (y0 + y1)

    # (iv) wall slip: fluid velocity at the first interior bin vs U_wall
    # (the bottom wall moves at -U_wall, so compare magnitudes)
    wall_layer = (yy > y0) & (yy < y0 + spacing)
    u_wall_fluid = float(uu[wall_layer].mean()) if wall_layer.sum() else float("nan")
    slip_frac = 1.0 - abs(u_wall_fluid) / max(abs(U_wall), 1e-12)

    nu = params.mu_solvent / state.rho0
    tau = H_wall ** 2 / (nu * np.pi ** 2)  # slowest decaying mode
    return {"n_steps": n_steps, "dt": dt, "t_sim": n_steps * dt,
            "nu_used": nu, "tau_mode": tau, "t_over_tau": n_steps * dt / tau,
            "r2_fit": r2_fit, "r2_analytic": r2_analytic,
            "slope_fit": slope_fit, "gamma_expected": gamma_expected,
            "slope_ratio": slope_ratio,
            "r2_central": r2_central, "slope_central": slope_central,
            "slope_ratio_central": slope_ratio_central,
            "y_mid_dev": y_mid_dev, "u_wall_fluid": u_wall_fluid,
            "slip_frac": slip_frac, "n_bins": int(ok.sum()),
            "u_wall": U_wall, "H_wall": H_wall, "central_zone": z,
            "n_wall_layers": n_wall_layers,
            "note": ("steady-state linear-profile validation; no-slip planes "
                     "are the innermost frozen wall rows (lattice "
                     "continuation, H_wall = inner_top - inner_bottom); the "
                     "central-region fit excludes the ~2h wall momentum- "
                     "transmission layers and is the bulk shear rate")}


def _laplace_masks(state, R, h):
    """Band-aware core/annulus masks for the Laplace measurement.

    The CSF color transition band is ~2.5-3h wide (2 smoothing passes over a
    2h kernel): inside it the color field c~ is between 0 and 1 and the
    pressure is partway up the Laplace jump (p follows c~: p = p_out + sigma*
    kappa*c~). The measurement must therefore sample the c~=1 interior
    (r < R - 3h) and the c~=0 far solvent (r > R + 3h), NOT the band.
    At R <= 3h the band covers the whole droplet (no c~=1 interior exists)
    and the radii are unusable - the calibration uses R = 5, 6, 7.
    """
    in_d = state.phase == 1
    com = state.pos[in_d].mean(axis=0)
    r = np.linalg.norm(state.pos - com, axis=1)
    core = in_d & (r < R - 3.0 * h)
    x0, y0, x1, y1 = state.domain
    ylo, yhi = y0 + 2.5, y1 - 2.5
    far = ((state.phase == 0) & (r > R + 3.0 * h) & (r < R + 4.0 * h)
           & (state.pos[:, 1] > ylo) & (state.pos[:, 1] < yhi))
    return core, far


def validate_laplace(params: SPHParams, n_steps: int = 4500, dt: float = 0.008,
                     radii=(5.0, 6.0, 7.0), damp_mu: float = 0.5) -> Dict:
    """Laplace law: time-averaged dP = P_in - P_out across a static droplet.

    The hydrostatic pressure jump develops over the capillary relaxation time
    t_char = mu*R/sigma_eff, so the droplet phase must run well past t_char.
    With the study viscosity (damp_mu = 0.5, t_char ~ R units) and
    n_steps = 4500 (36 time units) every radius covers >= 5x t_char. The
    earlier choice damp_mu = 5 was COUNTERPRODUCTIVE: it slowed the viscous
    relaxation (t_char ~ 50-60 units) so the runs were trapped at ~0.3-0.4x
    t_char and under-measured dP with an R-dependent bias (measured sigma_eff
    ratio 1.16 across R=5/6 matched the t/t_char ratio 1.2 - documented in
    PHASES_2_5_REPORT.md). Pressure is averaged over the final half of the
    trajectory (sampled every 50 steps) to cancel residual surface-mode
    oscillation.

    Radii 5/6/7 are required for the CSF model: the color-transition band is
    ~2.5-3h wide, and at R <= 3h the band spans the whole droplet so no c~=1
    interior exists (see :func:`_laplace_masks`; the R=2 and R=3 failures
    are documented in PHASES_2_5_REPORT.md).
    """
    p = replace(params, mu_droplet=damp_mu)
    out = {}
    for R in radii:
        # domain sized so the far annulus (r up to R+4h) and the y-band are
        # both clear of the frozen walls
        W = 2.0 * (R + 4.0) + 4.0
        H = 2.0 * (R + 4.0) + 4.0
        domain = (0.0, 0.0, W, H)
        state = make_couette_droplet_state(p, domain=domain, spacing=0.5,
                                           droplet_radius=R, n_wall_layers=2)
        run(state, p, 0, dt)
        pin_acc, pout_acc = [], []
        # sample only after the surface-mode oscillation decays; the decay
        # time scales with the capillary time t_char = mu*R/sigma_eff (a
        # fixed 0.5*n_steps polluted the average with the early oscillation
        # at large R - audit: PHASES_2_5_REPORT.md)
        t_char = p.mu_droplet * R / max(params.sigma_surf, 1e-9)
        sample_from = int(max(0.6 * n_steps, 4.0 * t_char / dt))
        for s in range(n_steps):
            step(state, p, dt)
            if s >= sample_from and s % 50 == 0:
                core, far = _laplace_masks(state, R, params.h)
                if core.sum() > 0 and far.sum() > 0:
                    pin_acc.append(float(state.pressure[core].mean()))
                    pout_acc.append(float(state.pressure[far].mean()))
        out[R] = {"dP": float(np.mean(pin_acc)) - float(np.mean(pout_acc)),
                  "pin": float(np.mean(pin_acc)),
                  "pout": float(np.mean(pout_acc)),
                  "n_core": int(core.sum()), "n_far": int(far.sum()),
                  "n_samples": len(pin_acc)}
    xs = np.array([1.0 / R for R in radii])
    ys = np.array([out[R]["dP"] for R in radii])
    slope, _ = np.polyfit(xs, ys, 1)
    out["_sigma_fit"] = float(slope)
    out["_linearity"] = float(np.corrcoef(xs, ys)[0, 1])
    return out


# ---------------------------------------------------------------------------
# Droplet-in-shear study
# ---------------------------------------------------------------------------


def measure_shear_rate(state: SPHState, params: SPHParams,
                       com: np.ndarray, x_margin: float = 6.0,
                       y_half: float = 2.5) -> float:
    """Local bulk shear rate dux/dy at the droplet, from the solvent field.

    Uses solvent particles in a vertical window around the droplet COM but
    OUTSIDE its x-shadow (|x - x_com| > x_margin), so the droplet's own
    perturbation of the velocity field is excluded. This is the quantity the
    droplet actually experiences; it is measured rather than assumed, so the
    analysis is robust to (a) the Couette flow development transient and
    (b) finite wall slip (both documented in PHASES_2_5_REPORT.md).
    """
    m = ((state.phase == 0)
         & (np.abs(state.pos[:, 0] - com[0]) > x_margin)
         & (np.abs(state.pos[:, 1] - com[1]) <= y_half))
    if m.sum() < 10:
        return float("nan")
    y = state.pos[m, 1]
    ux = state.vel[m, 0]
    A = np.vstack([y, np.ones_like(y)]).T
    coef, *_ = np.linalg.lstsq(A, ux, rcond=None)
    return float(coef[0])


def droplet_shear_sweep(params: SPHParams, shear_rates: Sequence = None,
                        eq_steps: int = 4000, shear_steps: int = 60000,
                        dt: float = 0.008, spacing: float = 0.5,
                        droplet_radius: float = 3.0,
                        domain=(0.0, 0.0, 24.0, 16.0),
                        out_dir: str = "outputs/sph",
                        sigma: Optional[float] = None) -> List[Dict]:
    """Droplet-in-shear sweep with MEASURED local shear rate.

    Design (rewritten 2026-08-10 after a design audit, see
    PHASES_2_5_REPORT.md):

    * The Couette flow in a finite cell develops over t_flow = H_wall^2/nu,
      which is far longer than any feasible run at the study viscosity; the
      nominal wall speed therefore does NOT equal the shear rate at the
      droplet during the run. The sweep MEASURES the local shear rate
      gamma_dot(t) from the solvent velocity field at every trace sample
      (see :func:`measure_shear_rate`) and reports Ca(t) built on it.
    * The droplet deformation relaxes over the capillary time
      t_char = mu_d*R/sigma; the sweep runs for a fixed duration covering
      several t_char and several t_flow relaxation scales, and the plateau
      deformation D_inf plus the transient constant tau are extracted by a
      least-squares fit D(t) = D0 + A*(1 - exp(-t/tau)) on the measured
      trace (fit R^2 reported per case).
    * Ca is the physically meaningful quantity: Ca = mu_d*gamma_dot*R/sigma
      with the measured gamma_dot and the Laplace-verified sigma_eff.
      Dimensional mapping: a Tau condensate of R ~ 1 um, mu_d ~ 1e2 Pa s,
      sigma ~ 1e-4 N/m under CSF/perivascular shear stresses tau ~ 0.1-1 Pa
      has Ca ~ 1e-3..1e-2; the sweep covers Ca ~ 0.05-10 and the
      physiological point is reported via the analytic Taylor (1934) limit.
    """
    os.makedirs(out_dir, exist_ok=True)
    if sigma is None:
        # reuse the authoritative Laplace calibration record if present
        # (written by scripts/diag_surface_tension.py); only re-measure when
        # it is missing
        calib_path = os.path.join(out_dir, "laplace_calibration.json")
        lap = None
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                lap = json.load(f)
            sigma = float(lap.get("sigma_eff",
                                  lap.get("sigma_fit", float("nan"))))
            print(f"  [sph] reused Laplace calibration sigma_eff = {sigma:.4f} "
                  f"(linearity {lap.get('linearity_dP_vs_1R', float('nan')):.3f})")
        if sigma is None or not np.isfinite(sigma):
            lap = validate_laplace(params, n_steps=2500, dt=dt)
            sigma = lap.get("_sigma_fit", 1.0)
            print(f"  [sph] calibrated surface tension sigma = {sigma:.4f} "
                  f"(Laplace fit, linearity {lap.get('_linearity', float('nan')):.3f})")
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"invalid surface tension sigma = {sigma}; run "
                         "scripts/diag_surface_tension.py first")
    x0, y0, x1, y1 = domain
    _, _, inner_bottom, inner_top = _wall_lattice(x0, y0, x1, y1, spacing, 4)
    H_wall = inner_top - inner_bottom
    rows = []
    traces = {}
    shear_rates = list(shear_rates if shear_rates is not None else
                       [0.0, 0.001, 0.003, 0.01, 0.03, 0.1])
    # ---- resume from a prior (possibly interrupted) run -------------------
    # Each rate's result is written incrementally below, so the on-disk JSON
    # is the source of truth. Reusing finished rates makes the sweep robust to
    # a crash, machine sleep, or terminal close: re-running continues from the
    # first unfinished rate instead of restarting from scratch.
    resume_path = os.path.join(out_dir, "sph_shear_sweep.json")
    done_rates = set()
    if os.path.exists(resume_path):
        try:
            with open(resume_path) as _f:
                _prev = json.load(_f)
            _prev_rows = _prev.get("rows", [])
            _p0 = _prev.get("params", {}) or {}
            same_cfg = (
                float(_p0.get("sigma_surf", -1.0)) == float(params.sigma_surf)
                and float(_p0.get("mu_droplet", -1.0)) == float(params.mu_droplet)
                and float(_p0.get("mu_solvent", -1.0)) == float(params.mu_solvent)
                and list(_prev.get("domain", [])) == list(domain)
                and float(_prev.get("droplet_radius", -1.0)) == float(droplet_radius)
            )
            if same_cfg and _prev_rows:
                rows.extend(_prev_rows)
                done_rates = {float(r["shear_rate_nominal"]) for r in _prev_rows}
                traces_npz = os.path.join(out_dir, "sph_traces.npz")
                if os.path.exists(traces_npz):
                    try:
                        _t = np.load(traces_npz, allow_pickle=True)
                        _sr = [float(x) for x in _t["shear_rates"]]
                        _tr = _t["traces"]
                        if hasattr(_tr, "item") and getattr(_tr, "ndim", 1) == 0:
                            _tr = _tr.item()
                        for _r in _sr:
                            _key = str(_r)
                            if _key in _tr:
                                traces[_key] = _tr[_key]
                    except Exception:
                        pass
                print(f"  [sph] resume: reusing {len(rows)} completed rate(s) "
                      f"{sorted(done_rates)}")
        except Exception as _e:
            print(f"  [sph] resume: could not read prior sweep ({_e}); "
                  f"starting fresh")
    shear_rates = [g for g in shear_rates if float(g) not in done_rates]
    # flow development scale (effective nu may exceed the nominal nu; the
    # measured gamma_dot makes the analysis robust either way). The reference
    # density is 1.0 throughout (see make_couette_droplet_state, validate_couette).
    t_flow = H_wall ** 2 / max(params.mu_solvent / 1.0, 1e-9)
    t_char = params.mu_droplet * droplet_radius / max(abs(sigma), 1e-12)
    n_steps_here = int(min(shear_steps, max(3000, 1.5 * t_flow / dt)))
    n_samp = 25
    print(f"  [sph] t_flow~{t_flow:.0f}  t_char~{t_char:.0f}  "
          f"shear_phase={n_steps_here * dt:.0f} units ({n_steps_here} steps)")

    for gd in shear_rates:
        U_wall = gd * H_wall / 2.0
        state = make_couette_droplet_state(params, domain=domain, spacing=spacing,
                                           droplet_radius=droplet_radius, n_wall_layers=4)
        # 1) equilibrate the droplet at rest (walls stationary) so the initial
        #    wall-boundary acoustic transient decays before shear. The
        #    transient is small now that the periodic-x neighbour search and
        #    the wall lattice are correct; no artificial drag is needed.
        run(state, params, eq_steps, dt)
        d0 = droplet_deformation(state)
        # 2) apply shear and record the trace WITH the measured shear rate
        state.wall_speed_top = U_wall
        state.wall_speed_bottom = -U_wall
        trace = []
        every = max(1, n_steps_here // n_samp)
        for s in range(n_steps_here):
            step(state, params, dt)
            if s % every == 0:
                tr = droplet_deformation(state)
                gd_m = measure_shear_rate(state, params,
                                          np.asarray(tr["com"]))
                tr["gamma_dot_measured"] = gd_m
                tr["Ca_measured"] = (params.mu_droplet * gd_m
                                      * droplet_radius
                                      / max(abs(sigma), 1e-12))
                trace.append(tr)
        df = trace[-1]
        # 3) plateau fit on the deformation trace
        t_arr = np.arange(len(trace)) * (every * dt)
        d_arr = np.array([tr["taylor"] for tr in trace])
        fit = _fit_transient(d_arr, t_arr)
        # 4) Ca at the plateau: mean of the measured Ca over the final 25%
        n_late = max(3, len(trace) // 4)
        ca_late = np.array([tr["Ca_measured"] for tr in trace[-n_late:]])
        gd_late = np.array([tr["gamma_dot_measured"]
                            for tr in trace[-n_late:]])
        Ca = float(np.nanmean(ca_late))
        gd_final = float(np.nanmean(gd_late))
        Ca_nominal = params.mu_droplet * gd * droplet_radius \
            / max(abs(sigma), 1e-12)
        rows.append({
            "shear_rate_nominal": float(gd),
            "shear_rate_measured_final": gd_final,
            "U_wall": float(U_wall),
            "capillary_number_Ca": Ca,
            "capillary_number_nominal": Ca_nominal,
            "n_shear_steps": int(n_steps_here),
            "taylor_initial": d0["taylor"],
            "taylor_final": df["taylor"],
            "taylor_plateau_fit": fit["D_inf"],
            "taylor_coefficient_a": ((fit["D_inf"] - d0["taylor"])
                                      / max(abs(Ca), 1e-12)),
            "tau_transient": fit["tau"],
            "fit_r2": fit["r2"],
            "fit_converged": fit["converged"],
            "aspect_ratio_initial": d0["aspect_ratio"],
            "aspect_ratio_final": df["aspect_ratio"],
            "angle_deg_final": df["angle_deg"],
            "n_droplet": df["n_droplet"],
            "sigma_used": float(sigma),
            "trace_len": len(trace),
        })
        traces[str(gd)] = {
            "t": [k * (every * dt) for k in range(len(trace))],
            "taylor": [tr["taylor"] for tr in trace],
            "aspect_ratio": [tr["aspect_ratio"] for tr in trace],
            "angle_deg": [tr["angle_deg"] for tr in trace],
            "gamma_dot_measured": [tr["gamma_dot_measured"] for tr in trace],
            "Ca_measured": [tr["Ca_measured"] for tr in trace]}
        conv = "converged" if fit["converged"] else "NOT-converged"
        print(f"  [sph] gamma_dot={gd:.4f} (nominal), Ca_nom={Ca_nominal:.3f}  "
              f"Ca_meas(final)={Ca:.3f}  "
              f"D: {d0['taylor']:.3f} -> {df['taylor']:.3f} "
              f"(plateau fit {fit['D_inf']:.3f} [{conv}], "
              f"a={rows[-1]['taylor_coefficient_a']:.3f}, "
              f"tau={fit['tau']:.1f}, R2={fit['r2']:.3f})")
        # incremental persistence: a crash mid-sweep keeps finished cases
        with open(os.path.join(out_dir, "sph_shear_sweep.json"), "w") as f:
            json.dump({
                "params": params.__dict__,
                "domain": list(domain), "spacing": spacing,
                "droplet_radius": droplet_radius,
                "rows": rows,
                "note": ("2D dimensionless CPU prototype; Ca uses the "
                         "MEASURED local shear rate (flow development + wall "
                         "slip accounted for) and the Laplace-verified "
                         "sigma_eff. Validated against analytic Couette and "
                         "Laplace limits (validate_couette / "
                         "validate_laplace). Production 3D/GPU runs are a "
                         "follow-up."),
            }, f, indent=2)
        np.savez(os.path.join(out_dir, "sph_traces.npz"),
                 shear_rates=np.array([float(g) for g in traces]),
                 traces=traces)
    return rows


def _fit_transient(d: np.ndarray, t: np.ndarray) -> Dict:
    """Least-squares fit of the deformation transient D(t) = D0 + A*(1-e^-t/tau).

    Returns D_inf, tau, the fit R^2 and a `converged` flag. The exponential
    extrapolation is only trustworthy when the trace is long enough to
    approach its asymptote; if the fitted time constant tau is large relative
    to the trace duration T (tau > 2*T) the plateau is unconstrained and the
    least-squares fit can return a non-physical D_inf (the Taylor deformation
    D = (a-b)/(a+b) is bounded in [0, 1)). In that regime the final sample is
    reported as D_inf with converged=False instead.
    """
    if len(d) < 5:
        return {"D_inf": float(d[-1]), "tau": float("nan"), "r2": 0.0,
                "converged": False}
    y = np.asarray(d, dtype=float)
    x = np.asarray(t, dtype=float)
    D0 = y[0]
    yc = y - D0  # relative deformation from the initial value
    # linearized estimate for initialization: log(1 - yc/A) = -t/tau
    A_init = max(float(yc.max()) * 1.5, 1e-6)
    mask = (yc > 0) & (x > 0)
    if mask.sum() < 3:
        return {"D_inf": float(y[-1]), "tau": float("nan"), "r2": 0.0,
                "converged": False}
    tau_init = -np.mean(x[mask] / np.log1p(-yc[mask] / A_init))
    tau_init = float(np.clip(tau_init, 1e-6, 1e6))

    def resid(p):
        A, tau = p
        return (D0 + A * (1.0 - np.exp(-x / tau)) - y)

    from scipy.optimize import least_squares
    try:
        sol = least_squares(resid, x0=[A_init, tau_init],
                            bounds=([0, 1e-6], [np.inf, np.inf]))
        A, tau = sol.x
        pred = D0 + A * (1.0 - np.exp(-x / tau))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        D_inf = float(D0 + A)
        T = float(x[-1] - x[0]) if len(x) > 1 else 0.0
        # guard against an unconstrained plateau: (i) the trace is far shorter
        # than the fitted time constant, or (ii) D_inf exceeds the physical
        # bound for the Taylor deformation. Either means the asymptote is not
        # determined by the data and the final sample is the honest estimate.
        converged = (tau <= 2.0 * max(T, 1e-9)) and (0.0 <= D_inf < 1.0)
        if not converged:
            return {"D_inf": float(y[-1]), "tau": float(tau), "r2": r2,
                    "converged": False}
        return {"D_inf": D_inf, "tau": float(tau), "r2": r2,
                "converged": True}
    except Exception:
        return {"D_inf": float(y[-1]), "tau": float("nan"), "r2": 0.0,
                "converged": False}
