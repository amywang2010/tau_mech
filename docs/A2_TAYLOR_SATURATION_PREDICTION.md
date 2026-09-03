# Pre-registered prediction: A2 (monotonicity) outcome and interpretation

**Written: 2026-09-03 ~11:40 local. Status of data at time of writing:**
- rate_0.03 extension (`ext_0.03`): **running** (launched 11:13, ~43,500 steps,
  no shear-phase record yet). No extension trace exists at this moment.
- All other rates: complete (short windows, 28,000 shear steps).
- Wave-3 rates 0.003/0.01: complete and TRUSTED (T/tau = 3.15, 2.61).

Commit hash of this file fixes the timestamp; it is committed **before any
extension data exists**.

## The prediction (falsifiable, derived from committed short-window fits)

From the already-committed per-rate records:

| rate | plateau fit | tau | T/tau | trusted? |
|---|---|---|---|---|
| 0.003 | 0.0471 | 71.1 | 3.15 | yes |
| 0.01  | 0.1987 | 85.9 | 2.61 | yes |
| 0.03  | 0.7433 | 116  | 1.93 | **no -> extension running** |
| 0.1   | 0.6887 | 17.9 | 12.52 | yes |

The extension protocol (v1.4, commit `4500298`) sets T_ext = 3*tau_fit =
3*116 = 348 units (43,500 steps), so **if the fit is stable, the extended
run will be TRUSTED (T/tau = 3.0)**.

**Prediction P1:** the extended rate-0.03 plateau will remain ABOVE rate
0.1's trusted plateau (0.6887). Basis: the short-window fit asymptote
(0.7433) and the last measured sample (0.6401) BOTH exceed 0.6887, so the
extension's plateau would have to fall through 0.689 — below where the
trajectory already is at T/tau ~ 1.9 and still rising — to falsify P1.
Under the pre-registered interval-feasibility A2, a trusted 0.03 plateau
> 0.6887 + 1e-3 makes the monotone selection **infeasible** at the
(0.03, 0.1) pair: **A2 will FAIL.**

If instead the extended plateau lands <= 0.689 (or the extension's
short-window-prefix check A5 fails, or the fit is unstable), P1 is
falsified and A2 may pass — also an acceptable, pre-declared outcome.

## Pre-declared interpretations (fixed BEFORE the data exists)

**Interpretation I (if P1 holds, A2 fails): physical Taylor saturation.**
D_inf(Ca) is monotone increasing only in the small-Ca regime; near
Ca ~ 0.5 the droplet deformation approaches a saturation/turnover (cf.
Taylor 1934; Grace 1982: deformation grows, then tips/streaming or
breakup intervene at high Ca). Our two largest rates sit near Ca_meas
0.54 and 1.78: the turnover is physically plausible. In this case:
  - A2 as pre-registered **fails and is reported as failed** — no
    post-hoc redefinition, no removal of the 0.03 point.
  - The merged record's verdict will read FAIL(A2) and the report will
    state the saturation finding explicitly, with the D_inf(Ca) curve
    shown in full, including the non-monotone pair.
  - Follow-up (scientifically the RIGHT response): characterize the
    turnover with 1-2 additional rates (e.g. 0.05, 0.07) under the same
    protocol — only if compute budget allows; never by altering the
    pre-registered checks.
**Interpretation II (if P1 fails): the window-limited interval was
conservative and the true 0.03 plateau <= 0.689** — A2 passes, monotone
D_inf(Ca) over the measured range, standard Taylor analysis.

Either outcome is scientifically publishable and was declared here in
advance. No other interpretation will be introduced after seeing the
extension data.

## Note on A5 (integrity)

A5 compares the extension trace against the short-window trace on the
shared span within 2N (control-derived noise floor, N = 0.0096). Both
runs start from the same deterministic equilibrated state with identical
equilibration (4,000 steps, identical seed path) — this is a
determinism/integrity check, not a physical test. A5 failure would mean
nondeterminism between reruns and would invalidate the comparison; it is
pre-declared as a data-integrity stop, not a physics result.
