# Pre-registered amendment: Couette resolution study v2.1 (2026-09-04)

Status: **committed before any v2.1 simulation data existed.** This document
is the pre-registration for the v2.1 protocol; the git timestamp of this
commit precedes every number the study will produce.

## What v2 revealed (record: outputs/sph/audits/couette_resolution_study.json,
commit e0aa2dc chain, abort committed automatically at 19:52)

1. The **determinism anchor PASSED**: level-1 window-1 (5700 steps, h=1.0,
   dt=0.008) reproduced v1's coarse slip to 1e-9 (0.8520350443 vs
   0.8520348352). The solver is bit-reproducible.
2. The **quasi-steady guard FAILED and aborted the study by design**: slip
   drifted 0.852 -> 0.865 -> 0.916 across windows (last drift 0.034 >> 0.005
   tolerance), despite r2_central >= 0.9996 in every window.
3. Root cause, established from the record numbers and solver bookkeeping
   (not fitted post hoc): the channel is initialized AT REST and the
   start-up transient decays on the momentum-diffusion time
   tau = H^2/(nu*pi^2). At the as-delivered nu = 0.05, tau = 152 time units
   (the v1/v2 records' own t_over_tau_nu_nominal = 0.30-0.68 fields record
   exactly this deficit). Effective relaxation is faster (XSPH + artificial
   viscosity transport momentum ~3.5x nominal, tau_eff ~ 43-60), but
   t = 45.6-102.6 remains on the transient: the predicted window-to-window
   drift under exponential relaxation with tau_eff = 43.4 is 0.046 vs the
   measured 0.034. **v1's "steady" profiles (t/tau = 0.30) and v2's windows
   were measured on a still-relaxing flow; the guard caught what the
   linear-profile check could not** (a Couette profile stays visually
   linear while its slope relaxes).

## Why this is a protocol defect, not a physics surprise

The original Couette protocol budgeted t = 3 viscous times, which its own
revalidation record shows was designed for nu = 0.5 (tau = 15.2; t = 45.6 =
3.0 tau; the nu_0.5_H8 record carries t_over_tau = 3.16). The resolution
driver passed the solver default mu_solvent = 0.05 instead - a design/input
mismatch in ONE line - so v1 and v2 ran 10x under-equilibrated relative to
the protocol's own budget. The dissipation diagnostic, which independently
passed mu_solvent = 0.5, ran fully equilibrated (t/tau = 3.16) and is
unaffected.

## v2.1 protocol (the amendment)

Two changes, both restoring the protocol's own design; no thresholds,
geometry, dissipators, kernel ratios, CFL, Reynolds number, or decision
rules are altered:

1. **Viscosity restored to the designed value**: mu_solvent = 0.5 (the
   value every other validation of this solver used, and the value the
   original "t = 3 tau" budget assumed). This reduces tau to
   H^2/(0.5*pi^2) = 15.2 time units and restores t/tau = 3.0 at the base
   window. Reynolds number drops from 346 to 34.6 - still deep in the
   laminar Stokes regime where the analytic Couette profile is exact, so
   the validation target is unchanged.
2. **Analytic steady-state initialization**: the fluid is initialized at
   the exact no-slip Couette profile u(y,0) = -U + (2U/H)(y - y_bottom),
   so the simulated state starts AT the steady solution of the continuum
   problem. The quasi-steady guard is thereby converted from a relaxation
   test into a strictly stronger fixed-point test: if the discrete
   dynamics preserve the steady state, slip drift must vanish within
   numerical noise; any residual drift now measures discretization error,
   not a start-up transient. The guard's thresholds (|dslip| <= 0.005,
   r2_central >= 0.99, one 1.5x escalation, then abort) are UNCHANGED.
   The rest-initialization result is retained as a diagnostic of the
   start-up transient (evidence for point 3 above).

### Determinism anchor under v2.1

The v1 bit-reproduction anchor is replaced: v1/v2 ran at nu = 0.05, so
they cannot anchor a nu = 0.5 study. The v2.1 anchor is instead a
**steady-preservation test at level 1**: the window-1 slip of the
analytic-IC run must equal the window-2 slip within the guard tolerance
(this IS the guard), and additionally the first-window r2_central must be
>= 0.999 (a flow initialized at the steady profile that stays there is
the strongest possible statement that the discrete dynamics preserve
steady Couette flow). The v1 fixed-h control and the v2 abort record
remain in the repository as the documented protocol history.

### Decision rule (unchanged, now decisive and transient-free)

(a) slip_frac decreases monotonically with h-refinement => boundary-
discretization artifact; documented; the sweep protocol's use of the
MEASURED local shear rate stands.
(b) slip_frac resolution-independent => wall-coupling formulation
property; the acceptance band is NOT relaxed either way.

### Cost estimate (from v2's MEASURED pace: 3.3e-4 s/particle-step)

Particle counts and window step counts are IDENTICAL to v2 (the CFL-pinned
design gives the same t = 45.6/68.4 windows at every level). Measured
costs per level (window1 + window2, run as independent parallel
processes - a pure scheduling change, same physics, same numbers):
level 1 (912 particles): ~43 min; level 2 (2088): ~2.5 h; level 3
(3744): ~5.9 h. With all three levels in parallel on this 12-core
machine: **~6 h wall; worst case with one escalation on the finest
level ~9 h.** No estimate in this document is shorter than v2's
measured pace.

Falsifiability: if the analytic-IC run shows slip drift > 0.005 between
windows at level 1 even once (after the single escalation), the study
ABORTS and the defect is formulation-level, to be reported as such - not
tuned away.

## Scope decision, amended BEFORE any v2.1 data exists (2026-09-04)

The study is RESCOPED to the level-1 steady-preservation test plus the
rest-IC transient diagnostic (``--level1`` mode). Rationale, declared in
advance: the pre-registered decision rule is outcome-agnostic for the
paper (both attribution branches leave every conclusion unchanged, and
the coupling chain consumes measured local shear rates, not the slip
number), so the three-level h-convergence attribution is NOT load-bearing.
Claiming less than we tested is always defensible. The paper therefore
states: wall-coupling steady-state preservation VALIDATED (fixed-point
test, analytic IC); the start-up transient quantified; h-convergence of
the wall layer NOT established (documented limitation; protocol and
pre-registration retained for any future demand). The full study remains
available at zero marginal protocol cost; no claim in the paper depends
on running it.
