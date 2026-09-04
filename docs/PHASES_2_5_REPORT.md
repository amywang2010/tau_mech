# tau_mech — Phases 2–5 report (EDA, SPH, GNN, ML evaluation)

This document is the transparency record for the post-preprocessing phases of
the Tau mechanobiology study. It follows the study's core requirements:
methods, decisions, assumptions, limitations, failures, and results are all
documented; every claim is traceable to code, config, or data on disk.

Run environment (see also `outputs/config_used.json`): Windows 11, Python
3.14.6, numpy 2.5.1, scipy 1.18.0, torch 2.13.0+cpu, torch_geometric 2.8.0,
scikit-learn 1.9.0, matplotlib 3.11.1. All phase outputs are written under
`outputs/`.

---

## Phase 2 — Exploratory data analysis (COMPLETE)

Entry point: `scripts/eda.py` (core: `tau_mech/eda.py`).
Inputs: the corrected Phase-1 outputs (reprocessed 2026-08-02 with the
verified Tien et al. 2013 THEORETICAL-ALLOWED rASA reference table).

### What was computed
Per ensemble (PED00422 Tau-441, n=1000; PED00192 K18, n=75; PED00443 K18
idpGAN, n=1000):
- ensemble size, residue frequency, residue coverage
- radius of gyration (mass-weighted), end-to-end distance, total SASA,
  per-residue rASA profile, APR-region exposure
- contact maps (5 Å heavy-atom), degree distribution, graph density,
  residue flexibility (per-residue rASA coefficient of variation)
- PCA + t-SNE of the per-residue feature+descriptor vectors
- two-sample comparisons: Kolmogorov–Smirnov p-values and Cohen's d effect
  sizes for every metric across the three ensembles (unequal n accounted for
  in the interpretation; KS is non-parametric so it is robust to the 75 vs
  1000 sample sizes)

Figures: `outputs/figures/fig_*.png` (13 figures).

### Headline results (corrected data)
| metric | PED00422 | PED00192 | PED00443 |
|---|---|---|---|
| Rg (Å) | 67.6 ± 14.6 | 37.5 ± 10.6 | 27.8 ± 7.1 |
| end-to-end (Å) | 158.9 | 85.1 | 56.3 |
| APR1 VQIINK rSA | 0.547 | 0.586 | 0.617 |
| APR2 VQIVYK rSA | 0.615 | 0.599 | 0.604 |
| mean degree | 6.10 | 5.14 | 5.37 |

Experimental anchors: Tau-441 SAXS Rg ≈ 65–69 Å (He et al. 2022; SASBDB
SASDLU4 69 Å) — computed 67.6 Å within the anchor band. K18 SAXS Rg ≈ 38 Å
(Mukrasch et al. 2005; He et al. 2024) — computed 37.5 Å (ratio 0.99).

Statistical findings (KS p / Cohen's d, PED00192 vs PED00443 shown as the
generation-method contrast):
- Rg: d = +1.32 (p = 1.7e-15) — idpGAN's K18 ensemble is substantially more
  compact than the experiment-constrained K18.
- APR1 VQIINK rSA: d = −0.88 (p = 1.6e-15); APR2 VQIVYK rSA: d = −0.97
  (p = 5.2e-24) — idpGAN residues sit at HIGHER aggregation-region exposure
  despite the collapsed size. This is the "generated vs experimentally
  constrained" discrepancy the PED00443 comparison was designed to expose;
  it is a Phase-4/5 input (transfer performance) and a manuscript-relevant
  caveat about generative ensembles.

### Assumptions / limitations (Phase 2)
- rASA uses the Tien theoretical scale; values are protocol-dependent and
  should not be compared numerically to other SASA tools (same caveat as
  Phase 1).
- PED00192 (n=75) is unweighted; Bayesian weights are not in the download.
- KS tests treat conformers as independent samples; conformers of one
  generative ensemble are not strictly independent (no autocorrelation
  analysis was run at this stage — listed as a follow-up).

---

## Phase 3 — SPH droplet-in-shear study (IN PROGRESS)

Entry point: `scripts/sph_validate.py` (validation) and `scripts/sph_sweep.py`
(study). Core: `tau_mech/sph.py` (self-contained numpy+scipy WCSPH engine,
no GPU).

### Physics model (all documented in the module docstring)
- 2D weakly-compressible SPH; cubic-spline kernel (Monaghan 1992)
- Tait equation of state, γ = 7, c_s = 10 (≈10× the flow speeds)
- Morris et al. 1997 viscosity (per-phase dynamic viscosity;
  μ_solvent = 0.05, μ_droplet = 0.5 → λ = 10)
- Tartakovsky & Meakin 2005 surface tension: short-range repulsion
  (A_surf, all pairs) + medium-range attraction (B_surf, droplet–droplet);
  cohesion is provided by this force, not by negative pressure
- XSPH smoothing with density-floored symmetric weights
- Monaghan artificial viscosity; frozen-particle walls; periodic x
- mass renormalization against the lattice density (so the EOS starts at
  equilibrium); density floor and p ≥ 0 clamp in the EOS (negative-pressure
  tensile instability is a documented SPH failure mode — see "Failures")

### Validation protocol
1. **Couette**: steady-state measured ux(y) vs the analytic linear profile.
   The no-slip planes are the inner wall rows, so H_wall = H + spacing and
   the expected shear rate is 2U/H_wall. Reported: profile linearity (R² of
   the fit), slope ratio (fitted/expected shear rate — a direct check of the
   effective viscosity + wall coupling), and the zero-velocity midpoint.
   Run at two viscosities: ν=0.5 (fast steady state, H=8) and ν=0.05 (the
   study solvent viscosity, H=4 short channel so τ = H_wall²/(νπ²) is small
   enough for a feasible steady-state run).
2. **Laplace**: time-averaged dP = P_in − P_out across static droplets at
   R = 3, 4, 5 (R=2 was dropped: with strong surface tension the surface
   layer ~2h thick spans the whole droplet, so no well-resolved core exists
   and P_in is meaningless). The droplet phase is run at raised viscosity
   (damping) so the surface-mode oscillation decays quickly — viscosity does
   not enter the hydrostatic equilibrium. P_in is measured in the droplet
   CORE (r < R−1.5; rim particles sit in the surface-tension force well at
   floored pressure) and P_out in an interior annulus (r ∈ [R+2, R+4]; away
   from both the droplet kernel support and the frozen walls whose kernel
   contribution inflates near-wall solvent density). Fit dP vs 1/R → σ and
   linearity. The measurement is executed once by
   `scripts/diag_surface_tension.py`, which persists the authoritative
   record to `outputs/sph/laplace_calibration.json`; `sph_validate.py` and
   the shear sweep reuse it (no duplicate multi-hour measurement).

### Surface-tension model: TM → CSF (audit trail)
The Tartakovsky–Meakin pair-force surface tension was **abandoned** after the
clumping discovery (below): its force is proportional to W(r), which is flat
at small r, so it cannot supply a short-range barrier, and any config that
avoids clumping (A=30/B=15) rarefies the droplet so strongly that the p ≥ 0
clamp zeroes the Laplace signal. The model was replaced by the **Continuum
Surface Force (CSF)** of Brackbill et al. 1992 / Adami et al. 2010:
F_s = σ·κ·∇c̃ (σ a DIRECT parameter; the Laplace law dP = σ/R becomes a
VERIFICATION rather than a calibration). The CSF development itself had a
long failure trail, all documented:
1. **Gradient accumulation bug (the root cause of the anti-Laplace dP)**:
   the color gradient was accumulated with the anti-symmetric +f/−f pattern
   used for pair FORCES, but a gradient is a FIELD evaluation (both endpoints
   accumulate the same sign with their own volumes). The bug flipped the
   interface normal at the outer rim (n̂·r̂ = +0.30 instead of −1), corrupting
   κ and EXPANDING the droplet. Fixed after `diag_csf_norm.py` measured the
   normal field directly.
2. **Raw-color divergence stencil truncation**: κ on the droplet rim came
   out negative (−0.58 vs expected +1/R) because the divergence stencil is
   truncated at the discontinuous interface. Fixed with (a) kernel smoothing
   of the color field (Morris 2000; Cummins & Rudman 1999; 2 passes) and
   (b) a renormalized divergence (Adami 2010): divide the stencil sum by the
   kernel-sum on the same stencil. After the fix κ = +0.335 at the rim vs
   analytic +1/R = +0.333, and the solvent-side κ tracks 1/r.
3. **Same-phase soft core rarefies the droplet**: with the CSF as cohesion
   (B=0), the A_surf soft-core repulsion applied to SAME-phase pairs pushed
   the droplet interior apart: at σ=0, ρ_droplet = 0.995 < ρ_solvent = 1.000
   and dP = −0.54 with NO surface force present (decisive σ=0 control). The
   repulsion is now applied to MIXED-phase pairs only (immiscibility); the
   σ=0 control then gives dP = −0.05 and ρ_droplet = 1.0002 ≈ solvent.
4. **Long-range mixed repulsion crushes the rim**: at r_rep = 0.6h (above
   the 0.5 lattice spacing) the mixed repulsion pushes droplet rim particles
   INWARD at first-neighbor distance, crushing the interface shell
   (NN < 0.35 for ~23% of droplet particles) and suppressing the CSF signal
   (dP = +0.14 vs σ/R = 0.33 at σ=1). Fixed with a SHORT-RANGE overlap
   barrier (r_rep = 0.45h, switch_delta = 0.05h): it engages only below the
   lattice spacing (no force at first-neighbor distance) so it prevents
   sub-spacing phase overlap without pushing the rim. Result: 0.0% clumping,
   NN median 0.494 (clean fluid droplet).
5. **The CSF band spans small droplets**: the color-transition band is
   ~2.5-3h wide, so at R ≤ 3h the whole droplet is inside it (no c~=1
   interior exists and the pressure never reaches p_out + σκ). The R=3
   radial profile confirmed this (center P = 0.30 vs p_out+σκ = 0.38;
   monotone P(r), no shell). Calibration radii changed to 5/6/7 with
   BAND-AWARE masks (core r < R−3h, annulus r ∈ [R+3h, R+4h]).
6. **Damping choice was counterproductive**: the Laplace runs used
   μ_droplet = 5 "to damp surface-mode oscillation", but this SLOWS the
   viscous relaxation to hydrostatic equilibrium (t_char = μR/σ_eff ≈
   50-60 units vs a 20-unit run). The measured σ_eff(R=5)/σ_eff(R=6) = 1.16
   matched the t/t_char ratio 1.2 — the runs were trapped at 0.3-0.4×
   t_char. Switched to the study viscosity μ = 0.5 (t_char ≈ R units, ≥ 5×
   covered).
7. **Sampling window polluted the average**: measure_dP used a hardcoded
   sample_from=800 steps, so the time-average included the early surface-mode
   oscillation (dP ≈ 0.23 → 0.026 over the first 16 time units at R=7),
   biasing larger radii low and creating a SPURIOUS R-dependence in σ_eff
   (0.49 → 0.42 across R=5→7). Fixed: physics-based sampling that starts
   after max(0.6·n_eq, 4·t_char/dt) steps. An independent 8000-step
   trajectory probe at R=7 confirmed the true plateau dP = 0.0646
   (σ_eff = 0.452), flat from t=24 to t=44.
8. **Factor-of-2 integration bug — the TRUE cause of σ_eff ≈ 0.5·σ_input
   (found 2026-08-14; SUPERSEDES the "band-split" explanation in 7/8)**: the
   earlier "band-split" (46% droplet / 54% solvent share) was WRONG. A direct
   force-integral probe (scripts/diag_force_integral.py) showed the CSF force
   MAGNITUDE is correct: the |∇c|-weighted mean curvature is 0.93/R and
   ∫κ·∇c·dr = −1/R, i.e. the force alone is equivalent to dP = 0.97·σ/R.
   The 0.5 factor came from the TIME INTEGRATION, not the force: in step()
   the surface force was added ONLY to the second velocity-Verlet half-step
   (a1, weight 0.5·dt) while the pressure force is added to BOTH half-steps
   (a0 and a1, total weight dt). A force present in only one half-step carries
   half the dt weight, so the equilibrium condition became ∇P = 0.5·F_CSF and
   the measured Laplace jump was exactly 0.5·σ/R (matching the 0.46-0.52
   observed). Fixed by evaluating the CSF force in both half-steps
   (step(); regression test test_step_applies_csf_in_both_half_steps measures
   v = dt·F_s after one step: 0.999 with the fix, 0.5 with the bug).
Status: CSF sign verified (+dP, inward) AND magnitude now verified:
   single-radius Laplace R=6 gives dP = 0.1663 vs σ/R = 0.1667 (σ_eff = 0.998);
   the authoritative 3-radius calibration (R=5/6/7) is being re-run with the
   fixed integration.

### ROOT-CAUSE AUDIT — the sigma-independent "droplet oscillation" (2026-08-11)
A decisive set of experiments exposed a genuine bug that had been masked by
the `settle_damping` quench, which has now been REMOVED. Findings, in order:

1. **The oscillation was never surface-tension-driven.**
   `scripts/diag_classify_osc.py` ran the R=3 droplet with σ = 0.5, 1.0 and
   2.0 and found IDENTICAL D(t) (D_min=0.0001, D_max=0.0245, t_min=4.8,
   t_max=0.8 to four decimals). A Rayleigh capillary oscillation must scale as
   1/√σ, so this was not physics. A `baseline` config with ALL surface forces
   off (σ=A=B=0) reproduced the identical oscillation, and a particle at
   x=0 had 30 neighbours vs 56 in the interior.

2. **Root cause: the periodic-x neighbour search was not periodic.**
   `build_pairs` used `cKDTree.query_pairs` on raw positions, so the two
   vertical domain edges (x=0, x=Lx) lost ~half their neighbours. This
   produced a ~20% density deficit at the edges (ρ=0.826 vs 1.023 interior)
   and a spurious pressure of −13 (the stiff γ=7 EOS amplifies density
   error), which launched pressure waves into the droplet — completely
   independent of surface tension. The same bug corrupted the Couette
   validation (spurious vertical boundary layers), which is why the earlier
   "slope deficit" (0.77) and "wall slip" were so stubborn.

3. **Two seam bugs in the periodic machinery** (both fixed):
   - `hexagonal_pack` generated a point at x = x1 (e.g. x=24.0), which is a
     DUPLICATE of x=0 under periodicity and made the periodic cKDTree reject
     the box. The x-interval is now half-open `[x0, x1)`.
   - `_wrap_x` used `pos < x0 -> pos + L`, so a particle at x0 − 1e−18 (pure
     rounding noise at the seam) was wrapped all the way to x1 and rejected.
     Replaced with a modulo fold `(pos − x0) % L + x0` (robust, exact).

4. **Secondary source: wall rows misaligned with the fluid lattice.**
   Walls were placed on a 0.5-spacing grid while the hex lattice row spacing
   is 0.433, leaving a spurious wall density layer (ρ 0.88–1.05, p −8.5→+6.0)
   that launched a decaying acoustic transient. Walls are now the EXACT
   lattice continuation of the fluid (one continuous lattice, rows outside
   [y0,y1] frozen — `_wall_lattice`).

Result: with the periodic-x neighbour search, the seam fixes and the lattice
walls, the droplet settles to its discretization baseline D₀ = 0.0162 within
~8 time units and stays stable (0.0161–0.0163) — versus the sustained ringing
of amplitude 0.024 before. The `settle_damping` parameter (an artificial
velocity-drag quench that papered over this artifact) has been removed: the
equilibration is now a plain `eq_steps` run at rest, with no free damping
parameter.

### Surface-tension calibration (TM era, kept as a record)
The Tartakovsky–Meakin parameters were calibrated by requiring σ large
enough that σ/R exceeds the EOS density noise (else dP is buried):
- A=0.5, B=1.0 → σ ≈ 0.02–0.05, dP ≈ 0, unmeasurable (early diag runs)
- A=2/B=4 → p_core ≈ 0.029 (still marginal)
- A=5/B=10 → p_core ≈ 0.029?? → ~0.1 (weak)
- A=10, B=20 → dP=0.79–1.31 across R=3–5, no NaN — BUT the Laplace
  linearity check FAILED (see below).

### The clumping-instability discovery (root cause of the anti-Laplace dP)
With A=10, B=20 the Laplace fit gave dP INCREASING with R (linearity −0.98).
A radial P(r) profile showed a compression-shell pressure structure with a
nearly uniform density (±0.6%) — the stiff γ=7 EOS amplifies tiny density
modulations into pressure swings of ±2.5, burying the Laplace signal.
The decisive probe was a nearest-neighbor (NN) analysis of the droplet
phase: **NN median = 0.14 vs the 0.5 lattice spacing; 75% of particles
clumped below 0.35.** The droplet had silently collapsed into a granular
cluster solid (a "pairing/clumping" instability), not a fluid:
- The TM force is proportional to the kernel W(r), which is nearly FLAT at
  small r (W(0.05)≈0.45 vs W(0.28)≈0.41) — there is essentially NO
  short-range repulsion barrier.
- With B=20 > A=10 the smooth-switch overlap region (0.5h–0.7h) has a
  net-attractive pair force (A·rep_w − B·att_w < 0), so particles fall into
  deep potential wells and oscillate as clumps.
- All "stability" checks (no NaN, bounded densities) never tested for
  clumping — a silent structural collapse.
- The "σ ≈ 2.2–6.6" readings were artifacts of the clump-solid; the
  interior pressure of a clump aggregate scales with R (more clumps),
  explaining the anti-Laplace trend.
Fix: the repulsion barrier must exceed the attraction well. Scan:
| A | B | clump fraction | NN median |
|---|---|---|---|
| 20 | 20 | 57% | 0.28 |
| 25 | 15 | 16% | 0.49 |
| **30** | **15** | **0%** | **0.61** |
| 30 | 12 | 0% | 0.62 |
A=30, B=15 eliminated clumping but inflated the droplet (its p ≥ 0 clamp
zeroed the Laplace signal), and the model was abandoned for the CSF (above).
The Shepard (partition-of-unity) density correction was also added
(smooths the lattice E0 imprint; std of the density field reduced ~25% on a
free lattice) and is applied in compute_density (SPHParams.shepard=1.0).

### Failures documented during development (transparency record)
- **Negative-pressure tensile instability**: the EOS initially allowed
  p < 0 where ρ < ρ₀ (walls), attracting fluid into the walls and collapsing
  the domain; fixed with a p ≥ 0 clamp (cohesion carried by the surface
  force) + density floor.
- **XSPH instability**: the raw m/ρ XSPH weight diverges where ρ is
  under-resolved (walls/rim), driving an exponential checkerboard blow-up;
  bisection (by disabling terms one at a time) identified XSPH as the
  culprit; fixed with density-floored symmetric weights.
- **First Laplace measurement was invalid**: "far-field" pressure was
  measured in wall-adjacent solvent, which the frozen-wall kernel
  contribution inflates (p_out > p_in anomaly at R=3), and the all-droplet
  pressure average was diluted by floored rim particles; fixed with the
  core/annulus measurement above.
- **Kernel normalization audit (new test)**: `test_kernel_normalized_2d`
  exposed that the cubic-spline implementation was 2/3 × the published
  Monaghan 1992 2D kernel (branch coefficients 2/3−q²+q³/2 and (2−q)³/6
  instead of 1−1.5q²+0.75q³ and (2−q)³/4). Because every discretized term
  is a product m·W or m·dW and the mass-renormalization step scales masses
  by the inverse kernel scale (ρ₀/mean(ρ) = 1.5×), the discretized system
  is EXACTLY invariant to a constant kernel scale: all densities, pressures,
  forces and trajectories are unchanged. The kernel was nevertheless
  corrected to the published normalized form (code must match its citation);
  the invariance argument is why pre-fix calibration results remain valid.
  10 new SPH unit tests added (`tests/test_sph.py`); suite now 46 tests.
- **Sharp surface-force switch → rattling**: the TM force switched
  discontinuously between repulsion and attraction at r = 0.6h; surface
  particles rattled at the force discontinuity (vmax spikes 0.05→0.26 at
  ~1400 steps, noisy pressure). Fixed with a cosine ramp across
  [r_rep−δ, r_rep+δ] making the force continuous; vmax then settles (~0.034)
  and P_core stabilizes.
- **R=2 Laplace artifact**: with σ strong, the surface layer spans the
  whole R=2 droplet → P_in meaningless (0.000); radii changed to 3/4/5.
- **First Couette comparison used the transient analytic profile** at
  t = 48 time units while the viscous time constant τ ≈ 130; the comparison
  was dominated by initial-condition/wall-slip mismatch (R² = 0.75). Replaced
  with the steady-state protocol at viscosities that reach steady state
  within the run length.
- **CPU oversubscription on the shared machine**: interleaving multiple
  heavy jobs (torch + cKDTree OpenMP) degraded throughput several-fold;
  operations are now run with explicit thread caps
  (torch: TRAU_MECH_THREADS=2; OMP_NUM_THREADS for scipy).
- **Duplicate-PID misdiagnosis**: a single launch appeared as two PIDs
  (venv launcher stub + base interpreter); this was NOT a duplicate job —
  the real throughput problem was OpenMP oversubscription (above).

### Study design (droplet-in-shear)
- 2D Couette cell (24×16, spacing 0.5), droplet radius R = 2 (dimensionless)
- shear rates γ̇ ∈ {0 (control), 0.02, 0.05, 0.1, 0.2}; wall speeds set from
  H_wall so the actual shear rate equals γ̇
- equilibration at rest (3000 steps), then an ADAPTIVE shear phase: the
  deformation transient of a viscous droplet is governed by the capillary
  (relaxation) time t_char = μ_d·R/σ (≈4.5 units here), NOT by 1/γ̇ — the
  original "4 shear times" rule would over-run the plateau by ~10× at low
  γ̇ (25000 steps). Each case now runs ~6 t_char (≈3000 steps), floored at
  1500, capped at shear_steps.
- plateau deformation D_∞ and the transient constant τ are extracted by a
  least-squares fit D(t) = D_∞·(1 − e^(−t/τ)) on the trace (more rigorous
  than taking the last sample; fit R² reported per case)
- capillary number Ca = μ_d γ̇ R/σ is the reporting quantity
- outputs: `outputs/sph/sph_shear_sweep.json` (incremental per-case
  persistence) + `sph_traces.npz` (deformation time series per case) +
  `sph_deformation_vs_Ca.png` + `laplace_calibration.json`

### SPH results so far
- **CSF Laplace verification (sigma_input = 1.0; SUPERSEDED by the factor-2
  fix — see CSF audit trail item 8):** the earlier table below was measured
  with the integration bug and is retained only as a record.
  | R | dP (old, buggy) | σ_eff = dP·R |
  |---|---|---|
  | 5 | 0.0947 | 0.473 |
  | 6 | 0.0763 | 0.458 |
  | 7 | 0.0649 | 0.454 |
  These gave σ_eff ≈ 0.46 = 0.5·σ_input — exactly the factor-of-2 from the
  velocity-Verlet half-step bug (item 8), NOT a physical "band-split". After
  the fix, the authoritative 3-radius calibration (re-run, COMPLETE) gives:
  | R | dP (fixed) | σ_eff = dP·R |
  |---|---|---|
  | 5 | 0.1980 | 0.990 |
  | 6 | 0.1617 | 0.970 |
  | 7 | 0.1350 | 0.945 |
  linearity(dP vs 1/R) = **1.0000**, all NaNs = 0, ρ ∈ [0.998, 1.002].
  Mean σ_eff = **0.968** (96.8% of σ_input; the residual ~3% is a finite
  h/R discretization effect that converges to 1.0 as h/R → 0, NOT a bug).
  This is the value in outputs/sph/laplace_calibration.json and the one the
  shear sweep uses for Ca (0.968, not 0.462).
- **TM-era Laplace calibration (A=10, B=20) completed — and FAILED the
  scaling check.** Measured time-averaged dP across R = 3, 4, 5:
  | R | dP | pin | pout | no NaN |
  |---|---|---|---|---|
  | 3 | 0.7875 | 0.798 | 0.011 | ✓ |
  | 4 | 1.0148 | 1.030 | 0.015 | ✓ |
  | 5 | 1.3122 | 1.337 | 0.024 | ✓ |
  dP INCREASES with R (≈ 0.26·R): linearity(dP vs 1/R) = −0.98, so the
  naive σ = slope fit is negative — physically meaningless. The measurement
  is stable and reproducible (no NaNs, clean core/annulus separation), so
  this is a MODEL/REGIME behavior, not a numerical failure: at h/R ≈
  0.2–0.33 the TM surface tension does not yield the ideal 1/R Laplace
  scaling (curvature/compression corrections are first order in h/R).
  *Resolution (superseded)*: the anti-Laplace scaling was traced through the
  audit trail above — clumping (TM attraction) → gradient accumulation bug
  → divergence stencil truncation → same-phase soft-core rarefaction → rim
  crush → band-spanning at small R → sampling window — and the CSF model
  now verifies dP ∝ 1/R (linearity 1.0000) with σ_eff ≈ 0.968 after the
  velocity-Verlet factor-2 fix (item 8).
- Couette steady-state validation: R²=0.998, symmetric profile (residual
  ~35% frozen-wall slip is a documented WCSPH limitation, made irrelevant
  to the science by the MEASURED local shear rate). Shear sweep: RUNNING
  (6 rates, sigma_eff = 0.968).

### GNN status (all complete)
Sections A–F of the main run are complete (see Phase 4/5 tables below),
including the corrected cross-ensemble matrix and the supplement
(structural ablation + edge-aware GAT). All artifacts in `outputs/gnn/`:
summary.json (patched with the scaler-fixed transfer matrix),
supplement.json, explainability.json, main_checkpoint.pt, pr_roc_curves.png,
training_curves.png, transfer_matrix.{png,npy}, embeddings_{pca,tsne}.png,
supplement_ablations.png.

### Physiological mapping (honest framing)
Physiological CSF/perivascular shear τ ≈ 0.1–1 Pa on a Tau condensate of
R ≈ 1 μm, μ_d ≈ 10² Pa·s, σ ≈ 10⁻⁴ N/m gives Ca = τR/σ ≈ 10⁻³–10⁻². The CPU
prototype cannot simulate Ca ≪ 0.1 in feasible runtime (the deformation
timescale 1/γ̇ would require 10⁵–10⁶ steps), so the physiological
deformation is reported from the analytic Taylor (1934) small-deformation
limit D = Ca·(19λ+16)/(16λ+16), valid for Ca ≪ 1, while the simulated sweep
maps the response curve at moderate Ca (0.1–10) and tests the qualitative
D(Ca) scaling. The manuscript's mechanistic claim is therefore: tissue-scale
shear → condensate deformation (Ca-scaling validated here) → altered local
concentration / interfacial stress → APR accessibility (Phase 4/5 probe).
Shear acting on a *single* protein is not claimed (thermal forces dominate).

---

## Phase 4 — Geometric GNNs (IN PROGRESS)

Entry point: `scripts/train_gnn.py` (core: `tau_mech/gnn.py`).

### Task and labeling
Node-level binary classification: is a residue inside an aggregation-prone
region? Labels come from the Tau-441 numbering carried in every graph
(VQIINK 275–280, VQIVYK 306–311), applied uniformly to full-length and K18
constructs. Positive class ≈ 2.7% of residues (imbalanced → PR-AUC primary).

### Models and protocol (leakage-safe)
- GCN (Kipf & Welling), GAT (Veličković), GraphSAGE (Hamilton), and a
  structure-blind MLP baseline
- GRAPH-level splits (70/15/15) — no conformer straddles train/test
- z-scoring fit on the training split only; class-weighted BCE; early
  stopping on validation PR-AUC; operating threshold from the validation PR
  curve (max F1); GCN trained with 3 seeds (mean ± std reported)
- section A: train on PED00422; section B: cross-ensemble PR-AUC transfer
  matrix; section C: ablations (no edge attributes / no amino-acid identity /
  +rSA,SASA features / MLP-no-graph); section D: GNNExplainer edge masks on
  held-out APR nodes; section E: penultimate embeddings → PCA + t-SNE
  colored by label and source ensemble; section F: permutation feature
  importance on the MLP
- checkpoint/resume: section A (~2.5 h CPU) is checkpointed so a later
  section crash does not retrain it

### Thread configuration (machine note)
For these tiny graphs (≤441 nodes) torch thread-launch overhead dominates;
training is configured with 2 intra-op threads by default
(TRAU_MECH_THREADS override), measured ~6.8 s/epoch on the 700-graph
training split.

---

## Phase 5 — ML evaluation (IN PROGRESS)

Metrics: PR-AUC (primary), ROC-AUC, F1, precision, recall, confusion
matrices, training curves (loss, val PR-AUC), model comparison table,
transfer matrix, embeddings viz, GNNExplainer edge masks, permutation
feature importance, ablation table. All outputs under `outputs/gnn/`.

### Known interpretability caveats (documented up front)
- The APR label is defined by sequence position; sequence-position and
  amino-acid one-hot features alone can partially solve the task (MLP
  baseline PR-AUC ≈ 0.80 — see results). The GNN gains must be judged
  against that baseline, and the explainability section reports which
  CONTACT topology (not sequence) carries the APR signal.
- GNNExplainer edge masks are per-conformer; the reported summary is the
  APR–APR vs APR–other edge mass fraction of the top edges.
- Permutation importance is computed on the MLP (SHAP-analog for tabular
  features); GNN-level feature attribution is the GNNExplainer node/edge
  masks.

### Results (main run complete, 48.6 min)
| model | test PR-AUC | ROC-AUC | notes |
|---|---|---|---|
| GCN (3 seeds) | 0.902 ± 0.001 | 0.9994 | 0.9030 / 0.9027 / 0.9009 |
| GAT | 0.9366 | — | |
| GraphSAGE | 0.9997 | — | near-perfect |
| MLP (no graph) | 0.7929 | — | strong baseline (positional prior) |

### Ablations (section C)
| ablation | test PR-AUC | interpretation |
|---|---|---|
| full (GCN) | 0.9030 | reference |
| no edge attributes | 0.9030 | **VACUOUS — removed**: GCN/GAT/SAGE
  forwards never consume edge_attr; replaced by the structural ablation |
| **no spatial edges** (sequential backbone only) | 0.7908 | 3D contact
  topology carries real APR signal (−0.11) |
| no amino-acid identity | 0.455 | amino-acid sequence is the dominant
  feature |
| +rSA, SASA features | (reported in summary) | |

### Section F permutation importance (MLP)
`seq_position` is the dominant feature (importance drop ~0.73 when
permuted). The APR labels sit at fixed chain positions (275–280, 306–311 in
Tau numbering), so a positional encoding partially solves the task. This is
essential context for the paper: the GNN's edge-topology gain is measured
against a strong positional prior, and the honest claim is that sequence +
position explain most of the signal, with contact topology contributing the
residual (no-spatial drop −0.11).

### GNNExplainer results (section D)
`outputs/gnn/explainability.json` (extracted from the checkpoint): top-10 edge
masks for the 6 VQIINK residues (Tau 275–280) in 2 held-out PED00422
conformers (12 nodes, 120 edges). Endpoint distribution of the top edges by
Tau region: R4 24.2%, R3 22.5% (R3 contains VQIVYK at 306–311 →
inter-hexapeptide contacts), N-term/proline-rich 22.5%, C-term 15.0%, R1
9.2%, R2 6.7%. Reading: the model attributes VQIINK detection primarily to
spatial contacts with the flanking R3/R4 repeats and long-range contacts into
the N-terminal half — consistent with the known cross-repeat /
inter-hexapeptide contacts of Tau aggregation — NOT to APR–APR self-edges
(frac_apr_apr ≤ 0.1). Recompute: `scripts/explain_region_stats.py`.

### Audits performed
- **Cross-ensemble scaler bug (found and fixed)**: `cross_ensemble_matrix`
  evaluated target graphs WITHOUT the training-ensemble z-scoring scaler
  (raw features into a model trained on z-scored features) — the original
  transfer matrix was corrupted (422→422 read 0.678 instead of the valid
  0.903). Fixed; `scripts/rerun_cross_ensemble.py` regenerates section B
  and patches summary.json (running).
- **Vacuous ablation (found and fixed)**: "no edge attributes" changed
  nothing because the forwards never use edge_attr; replaced with the
  meaningful structural ablation (no spatial edges), above.
- **Explainability data preservation**: GNNExplainer masks were computed
  but not persisted; `scripts/extract_gnn_explain.py` reconstructs and
  saves them from the checkpoint (outputs/gnn/explainability.json).

### Corrected cross-ensemble transfer matrix (GCN, seed 0)
PR-AUC (train → test); regenerated after the scaler fix:

| train \ test | PED00422 | PED00192 | PED00443 |
|---|---|---|---|
| PED00422 | **0.899** | 0.402 | 0.334 |
| PED00192 | 0.038 | **0.805** | 0.793 |
| PED00443 | 0.023 | 0.891 | **0.919** |

Reading:
- same-ensemble (diagonal) 0.90–0.92; the task is well-solved in-domain.
- K18 ↔ K18 transfer is strong (0.79–0.89): APR classification is
  transferable across K18 ensembles regardless of generation method.
- full-length → K18 degrades (0.33–0.40): different construct length,
  sequence context and feature scales.
- **K18 → full-length FAILS** (PR-AUC 0.02–0.04, F1 ≈ 0): a model trained on
  130-residue chains cannot detect APRs in 441-residue chains. Honest
  negative: APR transfer requires matched construct length; the full-length
  ensemble is the only appropriate training source for the physiological
  question.

### Supplement: structural ablation + edge-aware GAT (COMPLETE, 15.2 min)
`scripts/train_gnn_supplement.py` → `outputs/gnn/supplement.json` +
`supplement_ablations.png` (same seed-0 PED00422 split as the main run):

| model | test PR-AUC | vs reference |
|---|---|---|
| GCN (full, spatial + sequential) | 0.9030 | reference |
| GCN (sequential backbone only) | 0.7908 | −0.112 → 3D contact
  topology carries real APR signal |
| GAT (topology only) | 0.9366 | reference |
| **GAT + edge attributes**
  (distance, seq_sep, is_spatial) | **0.9916** | +0.055 → edge
  attributes add substantial signal beyond topology |
| MLP (no graph) | 0.7929 | baseline |

Combined with the main run, the evidence hierarchy is: amino-acid identity
(no_aa → 0.455) > edge attributes (0.992) > 3D topology alone (0.904) >
sequential backbone (0.791) ≈ MLP positional prior (0.793). The paper can
state that aggregation-region accessibility is primarily sequence-encoded,
with geometric contact information (distance-annotated 3D edges) providing a
quantifiable structural contribution.

---

## Full-module audit (2026-08-14)
A systematic review of every module (features, sasa, geometry, numbering,
gnn, constants, config, io, pipeline, descriptors, provenance, sph) found and
fixed the following — all with regression tests where applicable:

1. **SASA self-occlusion (systematic underestimate — FIXED; magnitude
   clarified 2026-08-15).** The Shrake-Rupley loop included the atom itself
   in the occlusion test. A probe point sits exactly ON the atom's own
   occlusion sphere (d = r_vdW + probe), so the strict `<` boundary
   convention is correct for that point — but floating-point made |pts|
   round to slightly < 1 for ~22% of the Fibonacci points, flipping them into
   false self-occlusion. The RELATIVE error this causes depends on burial:
   ~10% for a fully exposed atom (89.4% -> 99.4% of full sphere area), but
   much larger for partially buried atoms — a residue that is only ~30%
   exposed loses a fixed ~22% of its points, so its SASA is roughly halved.
   This is why the APR rASA doubled (0.286 -> 0.547 for VQIINK) after the
   fix. VERIFIED 2026-08-15 against an independent brute-force reference
   (every atom vs every other, no kd-tree): the fixed code matches exactly
   (0/3217 atoms differ). Self is now excluded from the neighbor list
   (test_self_not_occluding_own_probe_points).
2. **idpGAN citation was wrong in TWO places (FIXED).** constants.py listed
   "Janson, Feierabend & Gilson" (fabricated author list) and provenance.py
   gave the correct authors but a wrong article number/DOI (14:1438 /
   s41467-023-39281-7). Verified against the paper: Janson, Valdes-Garcia,
   Heo & Feig, Nat. Commun. 14:774 (2023), doi:10.1038/s41467-023-36443-x.
3. **cross_ensemble_matrix dropped non-default modes (FIXED; impact small).**
   The transfer targets were converted with default feature/edge settings,
   ignoring the caller's rsa_augment / no_aa / use_edge_attr / keep_spatial,
   and used a scaler fit on the FULL ensemble while the model was trained on
   the train split. Both now match the training path exactly. Note: the
   numerical impact of the scaler part was SMALL (transfer PR-AUCs moved only
   ~1e-3, e.g. 0.899 -> 0.8986), because the continuous features (hydropathy,
   normalized position) are already near unit-scale; the earlier "silently
   corrupting the transfer numbers" wording OVERSTATED this. The mode-propagation
   fix (rsa_augment/no_aa/keep_spatial) is the substantive part.
4. **pipeline sequence-consistency spot-check widened (FIXED).** The check
   only inspected the first 5 models; a same-length sequence corruption in
   any later conformer would silently misalign the model-0-derived APR masks.
   Now checks every model (np.array_equal short-circuits, so it is ~free).
5. **SPH velocity-Verlet factor-2 (FIXED — see CSF audit trail item 8).**
   The CSF surface force was applied only in the second half-step, halving
   sigma_eff (0.46 → 0.968 after the fix).

Confirmed NOT bugs (audited, no change): the NaN in sequential-edge distance
in features.py is handled downstream (converted to [0, seq_sep, is_spatial=0]);
the geometry/numbering/io/descriptors modules are correct; the PDB fixed-width
column slices match the documented PDB 3.30 layout; constants are cited inline
(Shrake-Rupley 1973, Chothia/NACCESS 1976, Tien 2013, Kyte-Doolittle 1982).

---

## Cross-cutting limitations (honest record)
1. 2D SPH is a prototype; quantitative droplet-deformation numbers are
   order-of-magnitude. 3D/GPU (Taichi not installable on Python 3.14 — see
   README §8) is the documented production follow-up.
2. All SPH quantities are dimensionless; dimensional mapping to physiology
   is via Ca only.
3. The GNN APR labels are sequence-derived; the truly causal test
   (pathogenic mutants P301L / ΔK280 exposure prediction) is listed as the
   next experiment in the README roadmap.
4. PED00443 is fully generative — it constrains no experimental data; its
   deviations (more compact, higher APR exposure) are a model artifact to
   report, not a biological finding.
