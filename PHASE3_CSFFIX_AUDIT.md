# Phase 3 audit — CSF stencil symmetry defect (found, fixed, and gated revalidation)

**Status: Phase 3 remains BLOCKED** pending the full-duration zero-shear gate
and full revalidation of Couette + Laplace with the fixed stencil. Nothing in
this document should be read as "validated" until those runs complete.

---

## 1. Executive summary

The Phase 3 blocker (zero-shear droplet drift, D: 0.009 → 0.078 over ~406
time units, azimuth locked at ~171°) has been traced to a concrete,
demonstrated numerical defect in the CSF interface operator:

> **The curvature (normal-divergence) accumulation was one-sided: only the
> first endpoint of each pair received the divergence term.**

Because `scipy.spatial.cKDTree.query_pairs` returns each pair once as
(lower-index, higher-index), the defect made every particle's curvature
stencil see only its **higher-indexed neighbours** (~0.51 of its pairs,
measured). With row-major lattice indexing that is a position-dependent,
non-physical operator: **label-dependent, rotationally biased, and
non-momentum-conserving.** A 0.1%-level constant net force on the droplet,
applied every step for 400+ time units, is fully consistent with the observed
slow drift and its fixed azimuthal signature — and explains why the drift
survived every parameter variation (it was a stencil defect, not physics or
tuning).

## 2. Evidence (all probes in `scripts/diag_csf_symmetry.py`)

Baseline (one-sided stencil), R = 3 droplet, h = 1, spacing 0.5:

| probe | result | physical meaning |
|---|---|---|
| A. permutation invariance | **FAIL** — κ moved by up to 1.51 (κ ~ 0.33) under a pure particle relabeling | operator depends on particle numbering, not physics |
| B. azimuthal uniformity | **FAIL** — rim sector means 0.29–0.62 vs 1/R = 0.333 | rotational symmetry broken in a lattice-correlated way |
| C. internal force balance | **FAIL** — net internal force = **10.8%** of the surface-force magnitude; torque 0.6% | non-momentum-conserving; a constant spurious force every step |
| D. stencil census | **FAIL** — mean 0.51 of neighbour pairs in the divergence stencil | direct confirmation of the one-sided accumulation |

Post-fix (same probes, same state):

| probe | result |
|---|---|
| A | **INVARIANT** — max |Δκ| ~ 5e-15 (machine precision), 4 seeds |
| B | spread 0.33 → **0.106** (residual is finite-resolution bias, see §4) |
| C | **CONSERVATIVE** — net force 0.05% of magnitude, torque 0.006% |
| D | **SYMMETRIC** — census 1.0 for every droplet particle |

The fix is derived, not tuned: the divergence of a field is a field
evaluation, so **both** endpoints of every pair accumulate (each with the
other particle's volume, sign closing through `e_ij = −e_ji` under the
minimum-image wrap). This mirrors the color-gradient block, which was already
a correct both-endpoints evaluation.

## 3. Code changes

1. `tau_mech/sph.py`: new `color_field_curvature()` — single source of truth
   for the color field, gradient, and renormalized curvature with symmetric
   accumulation; `compute_surface_force()` now consumes it. Full derivation
   and defect history in the docstring.
2. `scripts/diag_csf_symmetry.py`: rewritten to import the shared helper (an
   earlier version duplicated the operator and therefore audited a stale
   copy — itself a lesson: **audits must consume the solver's own operators**).
3. `tests/test_csf_symmetry.py`: permanent regression tests (permutation
   invariance ×2 seeds, internal force/torque conservation, azimuthal spread
   bound). Suite: **58/58 pass**.
4. `scripts/sph_audit.py`: pre-registered gate harness (see §5) with
   checkpoint/resume (full state + trace persisted; config fingerprint guards
   against resuming into a changed configuration; smoke-tested
   end-to-end: pause at step 15 → resume → identical completion).

## 4. Convergence probe (`scripts/diag_csf_convergence.py`)

The post-fix rim curvature still over-predicts 1/R at coarse resolution
(κR = 1.50 at h/R = 1/3 → 1.15 at h/R = 1/7). Evidence that this is
**first-order-in-h/R discretization**, not a remaining defect:

- κR decreases monotonically with h/R along the radius axis.
- Halving the lattice spacing at FIXED h/R (R = 6: spacing 0.5 → 0.25)
  changed κR only 1.201 → 1.179 (weak spacing dependence) but reduced the
  azimuthal spread by 2.2× (0.085 → 0.039) — the anisotropy is a resolution
  effect, not a direction effect.

Consequence: the Laplace calibration must be **re-measured** with the fixed
stencil (the stored record predates the fix), and the calibration's operating
radii (R = 5–7) are already in the better-resolved regime.

## 5. Pre-registered gate (zero-shear control, sweep config)

Written here BEFORE the post-fix full-duration run is inspected. The gate
runs the exact sweep configuration (μ_solvent = 1.0, μ_droplet = 10.0,
R = 3, 24×16 domain, eq 4000 + 50 765 steps — the same measurement window in
which the defective solver drifted D 0.009 → 0.078):

- **G1** |dD/dt| (linear trend) < 5e-5 per time unit
- **G2** max |D(t) − D₀| < 0.02 over the whole window
- **G3** COM drift rate < 1e-3 per time unit
- **G4** ρ ∈ [0.98, 1.02] throughout (free particles)
- **G5** |p| < 0.5 throughout (free particles)
- **G6** zero NaNs at every sampled step

Rationale: thresholds are 3–4× stricter than the observed defect, and tied
to the physics requirement (a fake drift larger than the low-Ca signal would
corrupt the D-vs-Ca curve). G4/G5 follow from the Laplace records
(ρ 0.998–1.002, |p| < 0.18 in the measurement windows). Equilibration (4000
steps, walls at rest) precedes the gate window — the same protocol the sweep
uses — so the gate measures the post-equilibration behavior that the defect
actually polluted, not the documented initial acoustic transient.

The viscosity ratio λ = 10 (μ_solvent/μ_droplet = 1/10) is the study
configuration from the recorded physiological mapping (Ca = τR/σ, τ =
0.1–1 Pa, R = 1 μm, μ_d = 10² Pa·s, σ = 10⁻⁴ N/m → Ca ~ 1e-3–1e-2). The
Laplace pressure jump is hydrostatic and independent of viscosity; the
Couette validation is run at both nominal and study viscosities by design.

## 6. Revalidation protocol (after the gate passes)

1. **Laplace re-calibration** (`scripts/diag_surface_tension.py`): radii
   5/6/7, n_eq 4500, μ_droplet 0.5. Acceptance: linearity(dP vs 1/R) >
   0.999, σ_eff/σ_input in [0.93, 1.05], no NaNs, ρ ∈ [0.998, 1.002].
   Overwrites `outputs/sph/laplace_calibration.json` (the stored record was
   measured with the defective stencil and is retained in git history only).
2. **Couette re-validation** (`scripts/sph_validate.py --couette-only`): both
   viscosity configurations (ν = 0.5, H = 8; ν = 0.05, H = 4). Acceptance:
   central R² > 0.99, slope ratio in [0.9, 1.1].
3. **Full-duration zero-shear control** → gate §5.
4. Only then the physiological sweep (6 rates incl. the 0.0 control), with
   Re, Ca reported per case and an explicit Wi applicability statement (the
   model fluid is Newtonian; Wi is undefined/∞ for a solvated IDP chain —
   see final report).

## 7. Process lessons (recorded per the transparency requirement)

- **Audits must import the solver's operators**, not duplicate them: the
  first version of the symmetry audit carried a stale copy of the operator
  and briefly "proved" the fix had failed.
- **Background jobs under the tool wrapper are unreliable** on this machine
  (nondeterministic reaping; a flaky wmic query also produced one false
  "both jobs dead" reading). Mitigations: checkpoint/resume in the harness
  (verified), single-flight discipline (kill-verify-relaunch counts from
  process creation times, not `ps` line counts), wmic + creation dates as
  the source of truth for process identity.
- **Test permutations must map results back with the correct index
  direction** (`k2c[perm] = k2`, not `k2[perm]`) — the initial regression
  test compared different particles and produced a false failure across all
  seeds; the direction error was found by running three seeds and inspecting
  the argmax location.
