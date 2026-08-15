"""Tests for the SPH engine core (tau_mech.sph).

These are fast, focused unit tests: kernel normalization, neighbor-pair
correctness, lattice density consistency, deformation descriptors on known
shapes, and the transient-fit extractor. The expensive analytic validations
(Couette profile, Laplace law) live in scripts/sph_validate.py /
scripts/diag_surface_tension.py.
"""

import numpy as np
import pytest

from tau_mech.sph import (
    SPHParams,
    _fit_transient,
    build_pairs,
    compute_density,
    cubic_spline,
    cubic_spline_dwdr,
    droplet_deformation,
    hexagonal_pack,
    make_couette_droplet_state,
    particle_mass,
    run,
    step,
)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def test_kernel_normalized_2d():
    """int_0^{2h} 2*pi*r*W(r,h) dr = 1 (2D normalization)."""
    h = 1.0
    r = np.linspace(1e-6, 2.0 * h, 20000)
    integrand = 2.0 * np.pi * r * cubic_spline(r, h)
    integral = np.trapezoid(integrand, r)
    assert integral == pytest.approx(1.0, abs=0.02)


def test_kernel_compact_and_smooth():
    h = 1.0
    assert cubic_spline(np.array([0.0]), h)[0] > 0.0
    assert np.all(cubic_spline(np.array([2.0, 2.5, 10.0]), h) == 0.0)
    # dW/dr should be negative inside the support (attractive direction)
    assert np.all(cubic_spline_dwdr(np.linspace(0.1, 1.9, 50), h) <= 1e-12)


# ---------------------------------------------------------------------------
# Neighbor search
# ---------------------------------------------------------------------------


def test_build_pairs_within_support():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 10.0, size=(40, 2))
    h = 1.0
    pairs, d, e = build_pairs(pos, h)
    assert pairs.shape[1] == 2
    assert len(d) == len(pairs)
    # every reported pair is within the support
    assert np.all(d <= 2.0 * h + 1e-9)
    # unit vectors point from j toward i
    np.testing.assert_allclose(e, (pos[pairs[:, 0]] - pos[pairs[:, 1]]) / d[:, None],
                               atol=1e-9)
    # symmetric: (i, j) and (j, i) never both present (query_pairs returns
    # each unordered pair once), but each particle appears consistently
    assert pairs.dtype == np.int64


def test_build_pairs_periodic_x():
    """A particle near x=0 sees its periodic image near x=Lx (min-image).

    Regression test for the root cause of the sigma-independent "droplet
    shape oscillation": the neighbour search was NOT periodic in x, so the
    two vertical domain edges lost ~half their neighbours, produced a ~20%
    density deficit and a spurious pressure transient.
    """
    pos = np.array([[0.1, 0.5], [23.9, 0.5], [12.0, 0.5]])
    pairs, d, e = build_pairs(pos, 1.0, x_period=24.0)
    s = set(map(tuple, sorted(pairs.tolist())))
    assert (0, 1) in s  # cross-boundary pair found
    idx = int(np.where((pairs[:, 0] == 0) & (pairs[:, 1] == 1))[0][0])
    assert d[idx] == pytest.approx(0.2, abs=1e-6)   # min-image, not 23.8
    assert e[idx, 0] == pytest.approx(1.0, abs=1e-9)  # +x across the seam


def test_hexagonal_pack_half_open_x():
    """The x-interval is half-open: no point at x=x1 (the periodic seam)."""
    pts = hexagonal_pack(0.0, 0.0, 24.0, 2.0, 0.5)
    assert pts[:, 0].min() >= 0.0
    assert pts[:, 0].max() < 24.0  # x=24.0 is a duplicate of x=0.0


def test_wrap_x_folds_seam():
    """x=24.0 and fp noise at the seam both fold into [x0, x1)."""
    from tau_mech.sph import SPHState, _wrap_x

    pos = np.array([[24.0, 8.0], [-5e-19, 8.0], [0.5, 8.0]])
    st = SPHState(pos=pos.copy(), vel=np.zeros_like(pos),
                  mass=np.ones(3), phase=np.zeros(3, dtype=np.int8),
                  domain=(0.0, 0.0, 24.0, 16.0))
    _wrap_x(st)
    assert np.all((st.pos[:, 0] >= 0.0) & (st.pos[:, 0] < 24.0))
    assert st.pos[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_periodic_x_density_uniform():
    """A full periodic-x lattice is uniform across the seam."""
    p = SPHParams()
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 24.0, 8.0),
                                   spacing=0.5, droplet_radius=0.0,
                                   n_wall_layers=4)
    run(s, p, 0, 0.008)
    free = s.phase == 0
    ymid = 0.5 * (s.domain[1] + s.domain[3])
    band = free & (np.abs(s.pos[:, 1] - ymid) < 1.0)
    edge = band & (s.pos[:, 0] < 1.0)
    interior = band & (np.abs(s.pos[:, 0] - 12.0) < 1.0)
    assert edge.sum() > 0 and interior.sum() > 0
    assert s.rho[edge].mean() == pytest.approx(s.rho[interior].mean(), rel=0.02)


def test_morris_viscosity_delivers_nominal_nu():
    """The Morris viscosity term must deliver nu_eff ~ mu/rho (no 0.5 factor).

    Regression test for the factor-of-2 bug: the canonical Morris (1997) /
    Monaghan (2005) form uses (mu_i + mu_j), NOT 0.5*(mu_i + mu_j). With a
    sinusoidal velocity field u(y) = U sin(k y) on a lattice, the analytic
    viscous acceleration is a_x = -nu_eff k^2 u, so nu_eff is measured
    directly (k = 1 here). Pressure/surface forces/artificial viscosity/XSPH
    are all disabled, so only the Morris term contributes.
    """
    from tau_mech.sph import (SPHState, hexagonal_pack, particle_mass,
                              build_pairs, compute_density, compute_acceleration)

    mu = 1.0
    spacing = 0.25
    Ly = 2.0 * np.pi
    Lx = 2.0
    p = SPHParams(mu_solvent=mu, mu_droplet=mu, sigma_surf=0.0, A_surf=0.0,
                  B_surf=0.0, alpha_art=0.0, xsph=0.0)
    m = particle_mass(spacing, 1.0)
    pts = hexagonal_pack(0.0, 0.0, Lx, Ly, spacing)
    state = SPHState(pos=pts, vel=np.zeros_like(pts), mass=np.full(len(pts), m),
                     phase=np.zeros(len(pts), dtype=np.int8),
                     domain=(0.0, 0.0, Lx, Ly))
    pairs, d, e = build_pairs(state.pos, p.h)
    compute_density(state, p, pairs, d)
    free = state.phase != 2
    scale = state.rho0 / max(float(state.rho[free].mean()), 1e-9)
    state.mass *= scale
    compute_density(state, p, pairs, d)
    y = state.pos[:, 1]
    u = 0.1 * np.sin(y)  # k = 1
    state.vel[:, 0] = u
    ax = compute_acceleration(state, p, pairs, d, e)[:, 0]
    band = np.abs(y - Ly / 2.0) < Ly / 4.0
    slope, *_ = np.polyfit(u[band], ax[band], 1)
    nu_eff = -slope  # k = 1 -> a_x = -nu_eff * u
    assert nu_eff == pytest.approx(mu / 1.0, rel=0.15)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def test_density_lattice_consistency():
    """A hexagonal lattice at the reference density sums to ~rho0."""
    p = SPHParams()
    spacing = 0.5
    m = particle_mass(spacing, 1.0)
    pts = hexagonal_pack(0.0, 0.0, 10.0, 10.0, spacing)
    from tau_mech.sph import SPHState

    state = SPHState(pos=pts, vel=np.zeros_like(pts), mass=np.full(len(pts), m),
                     phase=np.zeros(len(pts), dtype=np.int8))
    # density init with mass renormalization (same path as run(..., 0, dt))
    run(state, p, 0, 0.008)
    assert state.rho.mean() == pytest.approx(1.0, rel=0.05)


def test_shepard_smooths_lattice_imprint():
    """The Shepard correction removes the lattice-scale density imprint.

    This was the root cause of the anti-Laplace dP scaling: the raw mass-sum
    density carries hex-lattice shell modulations (~1-2%) that the stiff EOS
    (gamma=7) amplifies into pressure swings of +/-2.5, burying the Laplace
    signal. The Shepard (partition-of-unity) normalization divides by the
    same-hold kernel sum, so the corrected density field is much smoother.
    """
    spacing = 0.5
    m = particle_mass(spacing, 1.0)
    pts = hexagonal_pack(0.0, 0.0, 14.0, 14.0, spacing)
    from tau_mech.sph import SPHState

    def rho_field(shepard: float):
        p = SPHParams(shepard=shepard)
        st = SPHState(pos=pts, vel=np.zeros_like(pts),
                      mass=np.full(len(pts), m),
                      phase=np.zeros(len(pts), dtype=np.int8))
        run(st, p, 0, 0.008)
        return st.rho

    raw = rho_field(0.0)
    corr = rho_field(1.0)
    # both keep the mean at ~rho0
    assert raw.mean() == pytest.approx(1.0, rel=0.05)
    assert corr.mean() == pytest.approx(1.0, rel=0.05)
    # the corrected field is smoother than the raw one (measured 0.75x on a
    # free lattice; the boundary truncation dominates the residual variance)
    assert corr.std() < 0.9 * raw.std()


def test_step_stability_no_nan():
    """A short droplet+solvent run stays finite (no NaN, no blow-up)."""
    p = SPHParams(mu_droplet=5.0)
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 18.0, 12.0),
                                   spacing=0.5, droplet_radius=2.0)
    run(s, p, 0, 0.008)
    for _ in range(60):
        step(s, p, 0.008)
    assert not np.isnan(s.pos).any()
    assert not np.isnan(s.vel).any()
    assert np.isfinite(s.pressure).all()
    # droplet phase intact
    assert s.phase.sum() > 0


def test_no_clumping_default_params():
    """Regression test for the clumping instability.

    With B >= A the TM smooth-switch region was net-attractive and - because
    the force is proportional to W(r), which is flat at small r - particles
    collapsed into clumps (NN ~ 0.14 vs spacing 0.5; 75% clumped), silently
    turning the "droplet" into a granular solid. The current model (CSF
    surface tension + mixed-only immiscibility repulsion, B=0) has no
    attraction mechanism at all: nearest-neighbour distances stay at the
    lattice scale and no sub-spacing clumps form.
    """
    p = SPHParams(mu_droplet=5.0)  # CSF model: A=10 (mixed only), B=0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 24.0, 16.0),
                                   spacing=0.5, droplet_radius=2.0)
    run(s, p, 0, 0.008)
    for _ in range(400):
        step(s, p, 0.008)
    pos = s.pos[s.phase == 1]
    pairs, d, _ = build_pairs(pos, p.h)
    i, j = pairs[:, 0], pairs[:, 1]
    nn = np.full(len(pos), np.inf)
    np.minimum.at(nn, i, d)
    np.minimum.at(nn, j, d)
    # fluid invariant: NN distances stay at the LATTICE scale. (The CSF
    # band spans the whole R=2 test droplet - the documented R=2 artifact -
    # so the surface tension slightly compresses it: measured median NN
    # ~0.44, MIN 0.40. True clumping collapses to NN ~ 0.14. The
    # discriminator is the sub-0.35 fraction, not the exact spacing.)
    assert np.median(nn) > 0.40
    assert (nn < 0.35).mean() < 0.05


def test_wall_pinning():
    p = SPHParams()
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, 18.0, 12.0),
                                   spacing=0.5, droplet_radius=2.0)
    s.wall_speed_top = 1.0
    s.wall_speed_bottom = -1.0
    run(s, p, 5, 0.008)
    wall = s.phase == 2
    ymid = 0.5 * (s.domain[1] + s.domain[3])
    top = wall & (s.pos[:, 1] > ymid)
    bot = wall & (s.pos[:, 1] < ymid)
    assert np.allclose(s.vel[top, 0], 1.0)
    assert np.allclose(s.vel[bot, 0], -1.0)
    assert np.allclose(s.vel[wall, 1], 0.0)


# ---------------------------------------------------------------------------
# Deformation descriptors
# ---------------------------------------------------------------------------


def test_droplet_deformation_circle():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    pts = np.stack([np.cos(th), np.sin(th)], axis=1) * 2.0
    from tau_mech.sph import SPHState

    state = SPHState(pos=pts, vel=np.zeros_like(pts), mass=np.ones(len(pts)),
                     phase=np.ones(len(pts), dtype=np.int8))
    d = droplet_deformation(state)
    assert d["aspect_ratio"] == pytest.approx(1.0, abs=0.02)
    assert d["taylor"] == pytest.approx(0.0, abs=0.02)
    assert d["n_droplet"] == len(pts)


def test_droplet_deformation_ellipse():
    """A 3:1 ellipse gives aspect_ratio 3, Taylor D = (3-1)/(3+1) = 0.5."""
    th = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    pts = np.stack([3.0 * np.cos(th), 1.0 * np.sin(th)], axis=1)
    from tau_mech.sph import SPHState

    state = SPHState(pos=pts, vel=np.zeros_like(pts), mass=np.ones(len(pts)),
                     phase=np.ones(len(pts), dtype=np.int8))
    d = droplet_deformation(state)
    assert d["aspect_ratio"] == pytest.approx(3.0, abs=0.05)
    assert d["taylor"] == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# Transient fit
# ---------------------------------------------------------------------------


def test_fit_transient_recovers_parameters():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40, 41)
    D0, A, tau = 0.02, 0.12, 6.0
    d = D0 + A * (1 - np.exp(-t / tau)) + rng.normal(0, 0.003, t.size)
    fit = _fit_transient(d, t)
    assert fit["D_inf"] == pytest.approx(D0 + A, abs=0.01)
    assert fit["tau"] == pytest.approx(tau, rel=0.3)
    assert fit["r2"] > 0.95


def test_fit_transient_flat_fallback():
    d = np.full(10, 0.05)
    t = np.arange(10) * 1.0
    fit = _fit_transient(d, t)
    assert fit["D_inf"] == pytest.approx(0.05)


def test_step_applies_csf_in_both_half_steps():
    """The CSF surface force must carry the full dt weight in velocity-Verlet.

    Regression test for the factor-of-2 integration bug: the surface force
    was added only to a1 (0.5*dt) while the pressure force is added to both
    a0 and a1 (dt total), so the effective equilibrium condition became
    grad P = 0.5 * F_CSF and the measured Laplace jump was 0.5 * sigma/R
    instead of sigma/R (sigma_eff ~ 0.46-0.52 vs the correct ~0.97; audit
    2026-08-14). Here we isolate the surface force (mu = 0, A = B = 0,
    alpha_art = 0, xsph = 0, high sigma so it dominates the ~O(1) lattice
    pressure noise) and verify that after ONE step the interface-particle
    velocity is v ~ dt * F_s (full weight), not 0.5 * dt * F_s.
    """
    from tau_mech.sph import compute_surface_force

    p = SPHParams(sigma_surf=100.0, mu_solvent=0.0, mu_droplet=0.0,
                  A_surf=0.0, B_surf=0.0, alpha_art=0.0, xsph=0.0)
    dt = 0.008
    Lx = 20.0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, Lx, 20.0),
                                   spacing=0.5, droplet_radius=4.0,
                                   n_wall_layers=2)
    run(s, p, 0, dt)
    pairs, d, e = build_pairs(s.pos, p.h, x_period=Lx)
    Fs = compute_surface_force(s, p, pairs, d, e)
    fmag = np.linalg.norm(Fs, axis=1)
    interface = fmag > 0.5 * fmag.max()
    assert int(interface.sum()) > 20, "interface band under-resolved"
    step(s, p, dt)
    vmag = np.linalg.norm(s.vel, axis=1)
    ratio = vmag[interface] / (dt * fmag[interface])
    # full-weight delivery -> ratio ~ 1; the old half-weight bug gives ~ 0.5
    assert float(np.median(ratio)) == pytest.approx(1.0, abs=0.3)
