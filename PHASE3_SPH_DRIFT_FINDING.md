# Phase 3 (SPH) — Spurious zero-shear droplet drift: full finding record

> **RESOLUTION (2026-09-02, supersession notice):** root cause **identified and
> fixed** — a one-sided (higher-index-neighbour-only) accumulation in the CSF
> curvature divergence, making the operator label-dependent, azimuthally
> biased, and non-conservative (net internal surface force 10.8% of its own
> magnitude). Full diagnosis, operator-level evidence, the symmetric fix, the
> convergence analysis, and the pre-registered revalidation gate are in
> **`PHASE3_CSFFIX_AUDIT.md`**. The open status text below is retained as the
> historical record of the finding; the gate + full revalidation must PASS
> before any sweep result is reported.

**Status at time of finding: OPEN — the shear sweep results were not trustworthy as generated.**
**Date of finding: 2026-08-15.**
**Author: audited during finalization of `tau_mech`; written for external peer review.**

This document records, with the primary data, a numerical artifact in the
2D WCSPH (weakly-compressible smoothed-particle-hydrodynamics) shear-sweep
simulation that invalidates the no-shear control and contaminates the
low-capillary-number (`Ca`) points of the Phase 3 `D`-vs-`Ca` curve. It is
reported **without mitigation spin** so that reviewers can reproduce the
artifact and judge the Phase 3 results independently.

---

## 1. Executive summary

The no-shear control of the droplet-in-Couette sweep (shear rate = 0, both
walls stationary, measured local shear `γ̇ ≈ −2×10⁻⁴ ≈ 0`) shows the droplet
deformation parameter `D = (a−b)/(a+b)` **increasing monotonically** from
`D ≈ 0.009` to `D ≈ 0.078` (aspect ratio 1.02 → 1.17) over the ~406-time-unit
measurement window. A droplet under zero shear must remain circular
(`D ≈ 0`). The excess deformation `ΔD ≈ 0.069` is **larger than the expected
signal at the lowest swept capillary numbers**, so the sweep's low-`Ca`
points — the ones closest to the physiological regime — are unreliable.

The artifact is **not** specific to the surface-tension model: a control run
with *all* surface forces disabled reproduces comparable shape oscillation and
a non-zero center-of-mass drift over only ~24 time units. It is a solver-level
spurious-current / tensile-instability artifact of the WCSPH formulation, and
it was missed because prior validation only exercised short timescales
(≈8–48 time units) while the production sweep integrates ≈406 time units.

**Consequence for the manuscript:** the physiological conclusion
(`Ca ≈ 1×10⁻³ … 1×10⁻²` → negligible droplet deformation) rests on the
*analytic* Taylor (1934) small-`Ca` limit and does **not** depend on this
sweep. However, the numerical `D`-vs-`Ca` sweep, whose purpose was to validate
the mesoscale engine against that limit, cannot currently serve as clean
validation evidence and must either be (a) fixed at the solver level, or
(b) presented explicitly as a qualitative 2D prototype with this limitation
stated.

---

## 2. Primary evidence: the no-shear trace

Source: `outputs/sph/sph_traces.npz` (rate 0.0), produced by the sequential
sweep `scripts/sph_sweep.py` with the corrected surface tension
(`σ_eff = 0.968`, the Laplace-verified value after the velocity-Verlet
factor-of-2 fix of 2026-08-14). Parameters: `μ_solvent = 1.0`,
`μ_droplet = 10.0` (λ = 10), `σ_eff = 0.968`, `dt = 0.008`, domain 24×16,
droplet radius 3.0, `eq_steps = 4000` before the trace begins.

| t (units) | D (Taylor) | aspect ratio | angle (°) |
|---|---|---|---|
| 0 | 0.009 | 1.018 | 93.6 |
| 16 | 0.0047 | 1.009 | 103.8 |
| 32 | 0.0038 | 1.008 | 146.9 |
| 49 | 0.0095 | 1.019 | 164.0 |
| 65 | 0.0162 | 1.033 | 167.9 |
| 81 | 0.0222 | 1.045 | 169.4 |
| 97 | 0.0273 | 1.056 | 170.3 |
| 130 | 0.036 | 1.075 | 171.2 |
| 162 | 0.0429 | 1.090 | 171.3 |
| 244 | 0.0565 | 1.120 | 171.6 |
| 325 | 0.0679 | 1.146 | 171.4 |
| 406 | 0.0784 | 1.170 | 171.1 |

Key observations:

1. **Monotonic, not oscillatory.** After a brief dip to `D ≈ 0.004` at
   `t ≈ 32`, `D` rises monotonically for the remaining ~370 time units. This
   is a slow, cumulative drift, not a damped transient.
2. **Fixed elongation axis.** The inertia-tensor orientation angle locks onto
   ~171° (essentially the horizontal x-axis) by `t ≈ 65` and stays there.
   The elongation is systematic and directional, not random or rotating.
3. **Genuinely zero shear.** The measured local shear rate at the droplet is
   `γ̇ ≈ −2×10⁻⁴` throughout (walls stationary; the small residual is
   measurement noise). The drift therefore cannot be attributed to applied
   flow.

---

## 3. Root-cause evidence (why this is not a surface-tension bug)

The report `PHASES_2_5_REPORT.md` already contains a "root-cause audit" of a
droplet shape oscillation. That audit's decisive experiment was to disable
terms one at a time (`scripts/diag_classify_osc.py`). Its logged result
(`logs/classify.log`), reproduced here, is the key control:

```
config       D0     D_min   D_max   t_min  t_max   span   com_drift
baseline   0.0162  0.0003  0.0245   4.8    0.8   0.0242   0.0863   (σ=0, A=0, B=0)
csf_only   0.0162  0.0001  0.0245   4.8    0.8   0.0244   0.1428
```

- The `baseline` configuration has **all surface forces disabled**
  (`σ_surf = 0`, `A_surf = 0`, `B_surf = 0`), yet still shows a shape
  oscillation of span 0.024 **and** a center-of-mass drift of 0.086 over only
  ~24 time units.
- This proves the droplet is not at a stable mechanical equilibrium even
  without surface tension. The instability is in the WCSPH pressure / viscous /
  wall / periodic-boundary discretization itself (spurious interface currents
  and/or tensile instability), not in the Continuum Surface Force (CSF) term.

The prior audit concluded the droplet "settles to its discretization baseline
D₀ ≈ 0.016 within ~8 time units and stays stable". That conclusion is correct
**only over the ~8–48 time-unit window that was checked**. The production sweep
integrates ≈406 time units, and over that window the droplet does *not* stay
stable — it drifts to `D = 0.078`. The long-timescale stability was never
validated; this finding closes that gap.

---

## 4. A separate operational issue found during the same check

While verifying the sweep's progress on 2026-08-15, the sweep driver was found
running **twice** (two driver processes, each spawning 6 workers → 12 workers,
every rate duplicated). Cause: an earlier "kill" terminated only one driver
process; the other driver's worker children survived and were duplicated by a
subsequent relaunch. Two workers writing the same per-rate checkpoint race each
other and double memory pressure (which is consistent with the observed
kernel-time / memory-thrash slowdown).

Resolution: all 12 workers + 2 drivers + the `nohup` wrapper were terminated;
no rate had completed (`outputs/sph/workers/` was empty), so no checkpoint was
lost. This issue is operational (process management), **independent** of the
physics drift in §2–§3, and is recorded here for completeness.

---

## 5. Impact on each phase

| Phase | Impact of this finding |
|---|---|
| 1 Preprocessing | None — unaffected. |
| 2 EDA | None — unaffected. |
| 3 SPH sweep | **The D-vs-Ca curve is unreliable at low Ca** (spurious ΔD ≈ 0.069 at Ca = 0). High-Ca points (Ca ≳ 1) are less affected (drift ≈ 2–3 % of signal) but still biased. |
| 4 GNN | None — node features are sequence/hydropathy/position only (no SASA in the base features). |
| 5 ML evaluation | None — main results unaffected. |

---

## 6. Options under consideration (as presented to the author)

1. **Time-boxed root-cause hunt.** Re-run the existing force-isolation
   bisections on the now-idle cores to identify which term drives the
   *long-timescale* drift (candidates: artificial viscosity, XSPH, the
   velocity-Verlet time weight, the CSF, the wall/pressure coupling). If a
   clean, defensible fix emerges, re-validate against Laplace/Couette and
   relaunch. If not, fall back to option 2.
2. **Honest reframing.** State the WCSPH spurious-drift limitation explicitly,
   report the analytic Taylor-limit physiological result as the quantitative
   conclusion, and present the numerical sweep as a qualitative 2D prototype
   only.
3. **Empirical common-mode correction.** Run a full-length zero-shear control
   and subtract its time-matched drift from each rate. This assumes the drift
   is additive and rate-independent — an untested assumption, and an empirical
   correction rather than a fix. Not recommended as primary evidence.

---

## 7. Reproduction instructions

```bash
cd tau_mech
.venv/Scripts/python -m pip install -r requirements.txt

# (a) the no-shear drift over the full measurement window:
.venv/Scripts/python scripts/sph_sweep.py --shear-rates 0.0   # writes outputs/sph/
# inspect D(t) in outputs/sph/sph_traces.npz (rate 0.0)

# (b) the force-isolation control showing the instability is solver-level:
.venv/Scripts/python scripts/diag_classify_osc.py --steps 3000
```

Data provenance, exact protocol parameters, and the full audit trail are in
`PHASES_2_5_REPORT.md`, `outputs/provenance.json`, `outputs/config_used.json`,
and the per-phase reports.
