"""SPH -> APR/GNN coupling: quantitative transfer-function analysis.

This is the item-7 deliverable: connect the validated mechanical response
(Phase 3) to the APR exposure / GNN analysis (Phases 4/5) WITHOUT overclaiming
what was simulated. The documented causal chain is

    tissue-scale shear -> condensate deformation (Ca-scaling)
        -> altered local concentration / interfacial stress
        -> APR accessibility (Phase 4/5 probe).

Only the first arrow is simulated (SPH droplet under shear; the physiological
point is reported via the analytic Taylor 1934 limit). This script makes the
SECOND-TO-LAST arrow quantitative and falsifiable:

  1. EMPIRICAL SENSITIVITY: across the 1000-conformer PED00422 ensemble,
     regress per-conformer APR rASA on the global extension descriptor
     (Rg, end-to-end). Slope beta = d(APR rASA)/d(Rg) with bootstrap CI.
     This is the conformational coupling the protein itself exhibits.
  2. MECHANICAL INPUT: capillary number Ca = tau*R/sigma for physiological
     wall shear tau in {0.1, 1.0} Pa on a condensate R = 1 um with
     sigma = 1e-4 N/m (recorded mapping -> Ca ~ 1e-3..1e-2). The Taylor
     small-deformation limit with lambda = mu_d/mu_s = 10 gives
     D(Ca) = Ca (19 lambda + 16)/(16 lambda + 16) = 1.1702 Ca.
     Both physiological Ca fall BELOW the measured SPH range: interpolation
     is refused there and D is BOUNDED from above via the pre-registered
     monotonicity of D(Ca), using the tightest classified sweep point
     (censored bound / interval hi / trusted plateau). The bound is
     parameter-free: every input is read from the canonical merged record.
     A merged verdict of FAIL is accepted ONLY as the pre-declared A2
     saturation outcome (all integrity checks passing); the anchor is then
     restricted to the monotone-feasible prefix and the excluded turnover
     points are reported separately.
  3. STRAIN TRANSFER (bounded): a conformation embedded in an affinely
     deformed condensate experiences at most the condensate strain:
     dRg/Rg <= D (compliance factor k in (0, 1]; k = 1 is the affine UPPER
     bound). So dRg = k * D * Rg.
  4. PREDICTION: dAPR_rASA = beta * k * D(Ca) * Rg, compared against the
     NATIVE heterogeneity (the ensemble SD of APR rASA) and against the
     rASA feature scale the GNN actually consumes (the rsa_augment ablation,
     PR-AUC 0.9163, is the exposure-detection reference).

Every assumption is explicit and carried into the output JSON:
  - linear response (small D); Taylor limit valid for Ca << 1
  - affine upper bound k <= 1; reported for k = 1 and k = 0.1
  - ensemble sensitivity (equilibrium fluctuations) proxies the driven
    response (fluctuation-dissipation-style assumption; stated, not proven)
  - static-ensemble analysis; no re-equilibration under shear

Outputs:
  outputs/coupling/coupling_sph_apr.json
  outputs/coupling/coupling_transfer_curve.png

Run:  python scripts/coupling_sph_apr.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = Path("outputs/coupling")
SPH_SWEEP_JSON = Path("outputs/sph/sph_shear_sweep.json")
SPH_SUMMARY_JSON = Path("outputs/sph/sph_shear_sweep_summary.json")

# physiological mapping (recorded in sph_study_summary.json / report)
TAU_SHEAR_PA = (0.1, 1.0)     # wall shear stress on the condensate (Pa)
R_CONDENSATE_M = 1e-6         # condensate radius (m)
SIGMA_CONDENSATE = 1e-4       # interfacial tension (N/m)
LAMBDA = 10.0                 # viscosity ratio mu_droplet/mu_solvent
K_AFFINE = 1.0                # affine upper bound
K_COMPLIANT = 0.1             # illustrative compliant chain

TAYLOR_COEF = (19.0 * LAMBDA + 16.0) / (16.0 * LAMBDA + 16.0)  # = 1.1702


def taylor_D(Ca):
    """Taylor (1934) small-deformation limit for a viscous droplet."""
    return TAYLOR_COEF * np.asarray(Ca, dtype=float)


# ---------------------------------------------------------------------------
# Measured SPH anchor (v2): the coupling's mechanical input is interpolated
# from the CANONICAL merged sweep record instead of the analytic Taylor law.
# Justification of the dimensionless transfer: the SPH prototype and the
# physiological condensate share the viscosity ratio lambda = 10, so the
# measured dimensionless response D(Ca; lambda) maps onto the condensate via
# Ca = tau*R/sigma. Interpolation is piecewise-linear in log(Ca) over TRUSTED
# points only; queries outside the trusted Ca range are REFUSED (no
# extrapolation). The analytic Taylor law is retained solely as the small-Ca
# reference curve and is labeled as such everywhere.
# ---------------------------------------------------------------------------
def load_sph_anchor(sweep_json=SPH_SWEEP_JSON, summary_json=SPH_SUMMARY_JSON):
    """Extract the measured D_inf(Ca) curve from the canonical sweep record.

    Returns (anchor, reason): anchor is a dict with 'trusted' points
    [(Ca, D_inf, rate)], 'window_limited' intervals, 'below_floor' bounds,
    the merged record's acceptance_version and verdict_class; reason is a
    string when the anchor is unavailable.
    """
    if not sweep_json.exists():
        return None, f"canonical sweep record missing: {sweep_json}"
    rec = json.loads(sweep_json.read_text())
    # The summary record owns the acceptance data (verdict_class,
    # acceptance_notes.classification, acceptance_pre_registered); the
    # merged record owns the rows. Schema verified against the real
    # outputs on 2026-09-03.
    summary = None
    if summary_json.exists():
        summary = json.loads(summary_json.read_text())
    else:
        return None, f"acceptance summary record missing: {summary_json}"
    verdict_class = (summary or {}).get("verdict_class")
    notes = ((summary or {}).get("acceptance_notes", {}) or {})
    cls_map = {}
    for key, c in (notes.get("classification", {}) or {}).items():
        cls_map.setdefault(str(float(key)), c.get("class"))
    trusted, window_limited, below_floor = [], [], []
    for r in rec.get("rows", []):
        rate = r.get("shear_rate_nominal")
        if rate is None or rate <= 0:
            continue
        ca = r.get("capillary_number_Ca")
        if ca is None or ca <= 0:
            continue
        cls = cls_map.get(str(float(rate)), "unknown")
        if cls == "signal_distinguishable":
            trusted.append((float(ca), float(r["taylor_plateau_fit"]),
                            float(rate)))
        elif cls == "window_limited":
            iv = ((notes.get("classification", {}) or {})
                  .get(str(float(rate)), {}).get("D_inf_interval"))
            if iv is None:
                iv = [r.get("taylor_final"), r.get("taylor_plateau_fit")]
            window_limited.append((float(ca), [float(iv[0]), float(iv[1])],
                                   float(rate)))
        elif cls == "below_noise_floor":
            ub = ((notes.get("classification", {}) or {})
                  .get(str(float(rate)), {}).get("D_inf_censored_upper"))
            below_floor.append((float(ca), float(ub), float(rate)))
    if len(trusted) < 2:
        return None, (f"fewer than 2 trusted points in the merged record "
                      f"(found {len(trusted)}); merge verdict: {verdict_class}")
    trusted.sort()

    # Pre-registered failure handling. A merged verdict of FAIL triggers the
    # analytic-only fallback UNLESS the single failing pre-registered check
    # is A2 (monotonicity) AND every integrity check passes - the exact
    # outcome declared in advance as "Interpretation I: physical Taylor
    # saturation" (docs/A2_TAYLOR_SATURATION_PREDICTION.md, committed
    # before extension data existed). In that case the anchor is restricted
    # to the monotone-feasible prefix: the turnover region is excluded from
    # interpolation AND from monotonicity bounding (monotonicity is used
    # only where the data itself supports it), and the excluded points are
    # carried separately as 'turnover_points' so the full curve is still
    # reported, as the pre-declared interpretation requires.
    pre = (summary or {}).get("acceptance_pre_registered", {}) or {}
    failed = {k for k, v in pre.items()
              if isinstance(v, dict) and v.get("pass") is False}
    a2_key = "A2_monotone_D_inf_in_Ca_distinguishable"
    a2_only = failed == {a2_key}
    if verdict_class == "FAIL" and not a2_only:
        others = sorted(failed) if failed else "unspecified"
        return None, (f"merged sweep verdict_class is FAIL on non-A2 "
                      f"check(s): {others} - the measured anchor is not used")
    turnover = []
    a2_status = "not_failed"
    if a2_only:
        per_case = pre[a2_key].get("per_case", []) or []
        prefix_hi = None
        for case in per_case:  # Ca-sorted by the merge script
            if case.get("feasible"):
                prefix_hi = float(case["Ca"])
            else:
                break
        if prefix_hi is None:
            return None, "A2 failed with no feasible prefix - anchor unusable"
        # per_case Ca values are display-rounded (4 decimals) in the summary;
        # match the exact row Ca by nearest log-space distance. Unambiguous:
        # rounding error ~1e-5 dex vs inter-rate spacing ~0.48 dex.
        prefix_hi_exact = max(
            (p[0] for p in trusted
             if abs(np.log10(p[0]) - np.log10(prefix_hi)) < 0.01),
            default=None)
        if prefix_hi_exact is None:
            return None, ("A2 feasible prefix does not match any measured "
                          "point - anchor unusable")
        turnover = [p for p in trusted if p[0] > prefix_hi_exact]
        trusted = [p for p in trusted if p[0] <= prefix_hi_exact]
        window_limited = [p for p in window_limited
                          if p[0] <= prefix_hi_exact]
        below_floor = [p for p in below_floor if p[0] <= prefix_hi_exact]
        a2_status = ("fail_pre_declared_saturation; anchor restricted to "
                     f"the monotone-feasible prefix (Ca <= {prefix_hi_exact:g})")
        if len(trusted) < 2:
            return None, ("fewer than 2 trusted points in the A2-feasible "
                          f"prefix (found {len(trusted)})")
    return {
        "trusted_points": trusted,
        "window_limited": window_limited,
        "below_noise_floor": below_floor,
        "turnover_points": turnover,
        "a2_status": a2_status,
        "a2_only_pre_declared": bool(a2_only),
        "verdict_class": verdict_class,
        "acceptance_version": rec.get("acceptance_version"),
        "ca_trusted_range": [trusted[0][0], trusted[-1][0]],
    }, None


def D_SPH(ca, anchor):
    """Piecewise-linear interpolation in log(Ca) over trusted points.

    Raises ValueError outside the trusted range (never extrapolates).
    """
    ca = float(ca)
    lo, hi = anchor["ca_trusted_range"]
    if not (lo <= ca <= hi):
        raise ValueError(
            f"Ca={ca:g} outside the trusted SPH range [{lo:g}, {hi:g}] - "
            "refusing to extrapolate; add a sweep rate or use the analytic "
            "reference")
    xs = np.log10([p[0] for p in anchor["trusted_points"]])
    ys = [p[1] for p in anchor["trusted_points"]]
    return float(np.interp(np.log10(ca), xs, ys))


def bound_below_range(ca, anchor):
    """Upper bound on D(Ca) for Ca BELOW the trusted range (never a point
    estimate).

    Uses the pre-registered monotonicity of D(Ca) (small-deformation
    droplet physics; consistent with every measured point): for any
    classified sweep point p with Ca_p >= ca, D(ca) <= upper(p), where
    upper(p) is the censored bound (below_noise_floor), the interval hi
    (window_limited), or the trusted plateau (signal_distinguishable).
    The tightest such bound is returned. Raises ValueError if no point
    bounds ca from above (cannot happen for ca < ca_trusted_range[0],
    because the smallest classified point then qualifies).
    """
    ca = float(ca)
    cands = []
    for c, d, _rate in anchor.get("trusted_points", []) or []:
        if c >= ca:
            cands.append(float(d))
    for c, iv, _rate in anchor.get("window_limited", []) or []:
        if c >= ca and iv is not None:
            cands.append(float(iv[1]))
    for c, ub, _rate in anchor.get("below_noise_floor", []) or []:
        if c >= ca and ub is not None:
            cands.append(float(ub))
    if not cands:
        raise ValueError(
            f"no classified sweep point bounds Ca={ca:g} from above")
    return float(min(cands))


def selftest():
    """Offline verification of the anchor loader/interpolator (no repo IO)."""
    anchor = {"trusted_points": [(0.0543, 0.0471, 0.003),
                                 (0.1945, 0.1987, 0.01),
                                 (1.7794, 0.6887, 0.1)],
              "window_limited": [(0.5374, [0.6401, 0.7433], 0.03)],
              "below_noise_floor": [], "verdict_class": "PASS_WITH_LIMITS",
              "acceptance_version": "selftest",
              "ca_trusted_range": [0.0543, 1.7794]}
    # interpolation inside range: monotone, exact at nodes
    assert abs(D_SPH(0.0543, anchor) - 0.0471) < 1e-12
    assert abs(D_SPH(1.7794, anchor) - 0.6887) < 1e-12
    assert D_SPH(0.1, anchor) > 0.0471 and D_SPH(0.1, anchor) < 0.1987
    assert D_SPH(1.0, anchor) > 0.1987 and D_SPH(1.0, anchor) < 0.6887
    # physiological operating points: tau*R/sigma with tau in (0.1, 1.0) Pa,
    # R = 1e-6 m, sigma = 1e-4 N/m  ->  Ca in {0.001, 0.01}: BOTH below the
    # trusted range -> interpolation MUST refuse, and the monotonicity bound
    # must supply a finite upper bound instead (never an extrapolated point).
    ca_phys = [t * R_CONDENSATE_M / SIGMA_CONDENSATE for t in TAU_SHEAR_PA]
    assert max(ca_phys) < anchor["ca_trusted_range"][0]
    for c in ca_phys:
        try:
            D_SPH(c, anchor)
            raise AssertionError("interpolation below range not refused")
        except ValueError:
            pass
    b_hi = bound_below_range(max(ca_phys), anchor)
    b_lo = bound_below_range(min(ca_phys), anchor)
    assert 0.0 < b_hi <= 0.7433 + 1e-12   # bounded by a classified point
    assert b_lo <= b_hi + 1e-15           # smaller Ca -> tighter-or-equal

    # pre-declared A2-saturation restriction (as produced by load_sph_anchor
    # on a FAIL(A2-only) record): the turnover point is excluded from
    # interpolation AND from monotonicity bounding
    sat = {"trusted_points": [(0.0543, 0.0471, 0.003),
                              (0.1945, 0.1987, 0.01),
                              (0.5208, 0.8382, 0.03)],
           "window_limited": [(0.0215, [0.0148, 0.0148], 0.001)],
           "below_noise_floor": [],
           "turnover_points": [(1.7794, 0.6887, 0.1)],
           "verdict_class": "FAIL",
           "a2_status": "fail_pre_declared_saturation",
           "a2_only_pre_declared": True,
           "acceptance_version": "selftest",
           "ca_trusted_range": [0.0543, 0.5208]}
    assert D_SPH(0.5208, sat) == 0.8382        # prefix node still exact
    try:
        D_SPH(1.7794, sat)
        raise AssertionError("turnover point not excluded from interpolation")
    except ValueError:
        pass
    assert bound_below_range(0.01, sat) == 0.0148   # window-limited hi binds
    assert bound_below_range(0.001, sat) == 0.0148
    # monotonicity bounding does NOT see turnover points: for a Ca in the
    # gap between the prefix end (0.5208) and the turnover (1.7794), the
    # only higher-Ca point is the turnover itself - excluded because the
    # data refutes monotonicity there - so NO bound exists and the query
    # must refuse
    try:
        bound_below_range(0.6, sat)
        raise AssertionError("gap query not refused")
    except ValueError:
        pass

    # refusal outside the range
    for bad in (0.01, 5.0):
        try:
            D_SPH(bad, anchor)
            raise AssertionError("extrapolation not refused")
        except ValueError:
            pass
    print("selftest OK: interpolation exact at nodes, monotone inside "
          "range, physiological Ca REFUSED below range and bounded by "
          "monotonicity, A2-saturation restriction enforced, "
          "extrapolation refused")


def sensitivity(df, x_col, y_col, n_boot=10000, seed=20260902):
    """OLS slope of y on x with bootstrap CI and Pearson r."""
    from scipy import stats
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = np.polyfit(x[idx], y[idx], 1)[0]
    return {
        "x": x_col, "y": y_col, "n": int(n),
        "slope_per_unit": float(slope),
        "slope_ci95": [float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5))],
        "intercept": float(intercept),
        "pearson_r": float(r), "p_value": float(p),
        "y_sd": float(y.std()),
        "x_sd": float(x.std()),
    }


def load_ensemble(csv="outputs/PED00422/summary.csv"):
    import pandas as pd
    df = pd.read_csv(csv)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_ensemble()
    anchor, anchor_reason = load_sph_anchor()

    # ---- 1. empirical sensitivity (full-length ensemble) ------------------
    sens = {
        "apr1_vs_rg": sensitivity(df, "rg_equal_weight", "apr1_mean_rsa"),
        "apr2_vs_rg": sensitivity(df, "rg_equal_weight", "apr2_mean_rsa"),
        "apr1_vs_e2e": sensitivity(df, "end_to_end", "apr1_mean_rsa"),
        "apr2_vs_e2e": sensitivity(df, "end_to_end", "apr2_mean_rsa"),
    }

    # ---- 2. mechanical input ---------------------------------------------
    Ca = {f"tau_{t}_Pa": t * R_CONDENSATE_M / SIGMA_CONDENSATE
          for t in TAU_SHEAR_PA}
    D = {k: float(taylor_D(v)) for k, v in Ca.items()}

    # ---- 2b. measured SPH anchor (supersedes the analytic input when the
    #      canonical merged sweep record is available and not FAIL) ---------
    sph_anchor_block = {"used": anchor is not None,
                        "reason_unavailable": anchor_reason,
                        "note": ("D_inf(Ca) interpolated in log-Ca over "
                                 "TRUSTED points of the canonical merged "
                                 "record; no extrapolation beyond the "
                                 "trusted range; similarity mapping justified "
                                 "by identical lambda = 10")}
    if anchor is not None:
        if (anchor.get("verdict_class") == "FAIL"
                and not anchor.get("a2_only_pre_declared", False)):
            # defensive: the loader already refuses this case; kept so a
            # future loader change cannot silently use an invalid anchor
            sph_anchor_block["used"] = False
            sph_anchor_block["reason_unavailable"] = (
                "merged sweep verdict_class is FAIL - the measured anchor "
                "is not used for predictions (analytic reference only)")
        else:
            sph_anchor_block.update({
                k: anchor[k] for k in ("trusted_points", "window_limited",
                                       "below_noise_floor", "verdict_class",
                                       "acceptance_version",
                                       "ca_trusted_range",
                                       "turnover_points", "a2_status",
                                       "a2_only_pre_declared")})
            d_meas = {}
            for k, v in Ca.items():
                try:
                    d_meas[k] = {"D": D_SPH(v, anchor),
                                 "type": "interpolated_trusted"}
                except ValueError:
                    ub = bound_below_range(v, anchor)
                    d_meas[k] = {
                        "D_upper": ub, "D_lower": 0.0,
                        "type": "monotonicity_bounded",
                        "note": ("Ca below the measured SPH range; "
                                 "bounded above by the pre-registered "
                                 "monotonicity of D(Ca) using the "
                                 "tightest classified sweep point")}
            sph_anchor_block["D_measured"] = d_meas
            sph_anchor_block["D_conservative_scalar"] = {
                k: (e["D"] if "D" in e else e["D_upper"])
                for k, e in d_meas.items()}
            if anchor.get("a2_only_pre_declared"):
                sph_anchor_block["note_fail_a2"] = (
                    "verdict FAIL(A2 only): the pre-declared Taylor-"
                    "saturation outcome (docs/"
                    "A2_TAYLOR_SATURATION_PREDICTION.md); the anchor is "
                    "restricted to the monotone-feasible prefix and the "
                    "turnover points are reported separately")

    # ---- 3+4. prediction with bounded strain transfer ---------------------
    rg_mean = float(df["rg_equal_weight"].mean())

    def predict(D_map, label):
        preds = {}
        for sens_key, _ in (("apr1_vs_rg", None), ("apr2_vs_rg", None)):
            s = sens[sens_key]
            for tau_key, d_val in D_map.items():
                dRg = K_AFFINE * d_val * rg_mean if label.startswith("affine") \
                    else K_COMPLIANT * d_val * rg_mean
                dAPR = s["slope_per_unit"] * dRg
                preds[f"{sens_key}|{tau_key}"] = {
                    "D_input": d_val,
                    "dRg_A": float(dRg),
                    "dAPR_rASA": float(dAPR),
                    "fraction_of_native_sd": float(abs(dAPR) / s["y_sd"]),
                }
        return preds

    predictions = {"affine_upper_bound_k1": predict(D, "affine_upper_bound_k1"),
                   "compliant_k0.1": predict(D, "compliant_k0.1")}
    if sph_anchor_block.get("used"):
        # conservative scalar input: the interpolated value where trusted,
        # otherwise the monotonicity UPPER bound (predictions at a bounded
        # input are upper-bound predictions, labeled as such)
        D_cons = sph_anchor_block["D_conservative_scalar"]
        predictions["sph_measured_affine_k1"] = predict(D_cons, "affine_meas")
        predictions["sph_measured_compliant_k0.1"] = predict(D_cons,
                                                             "compliant_meas")
        # provenance: label every prediction entry with whether its D input
        # was interpolated (trusted) or a monotonicity upper bound
        for block in ("sph_measured_affine_k1", "sph_measured_compliant_k0.1"):
            for k, entry in predictions[block].items():
                tau_key = k.split("|", 1)[1]
                entry["input_type"] = d_meas[tau_key]["type"]

    # Ca sweep for the transfer curve figure. The analytic Taylor law is the
    # labeled small-Ca reference; the measured SPH anchor is overlaid where
    # the trusted measured range permits (nan = refusal zone, no extrapolation).
    ca_axis = np.logspace(-4, -1, 60)
    d_apr1 = (sens["apr1_vs_rg"]["slope_per_unit"] * K_AFFINE
              * taylor_D(ca_axis) * rg_mean)
    d_axis = d_apr1_meas = refusal_note = None
    if anchor is not None and sph_anchor_block.get("used"):
        lo, hi = anchor["ca_trusted_range"]
        d_axis = np.array([D_SPH(c, anchor) if lo <= c <= hi else np.nan
                           for c in ca_axis])
        d_apr1_meas = (sens["apr1_vs_rg"]["slope_per_unit"] * K_AFFINE
                       * d_axis * rg_mean)
        refusal_note = (f"Ca < {lo:.4g}: below the measured range - D = 0 "
                        f"within the control noise floor and bounded above "
                        f"by monotonicity (dotted); Ca > {hi:.4g}: outside "
                        f"the measured range - no extrapolation")
        # monotonicity upper bounds below the trusted range (NaN elsewhere);
        # computed here so the JSON record carries exactly what the figure draws
        bound_axis = np.array([bound_below_range(c, anchor) if c < lo
                               else np.nan for c in ca_axis])
        bound_apr = (sens["apr1_vs_rg"]["slope_per_unit"] * K_AFFINE
                     * bound_axis * rg_mean)

    # ---- GNN context -------------------------------------------------------
    gnn = {}
    try:
        s = json.load(open("outputs/gnn/summary.json"))
        ab = s.get("ablations", {})
        gnn = {
            "rsa_augment_PR-AUC": ab.get("rsa_augment", {}).get("pr_auc"),
            "full_PR-AUC": ab.get("full", {}).get("pr_auc"),
            "note": "rsa_augment is the exposure-feature reference; the "
                    "mechanical dAPR below is compared to the rASA scale "
                    "this detector consumes",
        }
    except Exception as e:
        gnn = {"error": str(e)}

    chain_text = ("tissue shear -> condensate deformation (MEASURED SPH "
                  "D_inf(Ca); analytic Taylor reference) -> bounded affine "
                  "strain transfer -> APR rASA change (ensemble sensitivity)")
    if not sph_anchor_block.get("used"):
        chain_text = ("tissue shear -> condensate deformation (analytic "
                      "Taylor/Ca reference; measured anchor unavailable: "
                      + str(anchor_reason) + ") -> bounded affine strain "
                      "transfer -> APR rASA change (ensemble sensitivity)")
    result = {
        "chain": chain_text,
        "physiology": {
            "tau_shear_Pa": list(TAU_SHEAR_PA),
            "R_condensate_m": R_CONDENSATE_M,
            "sigma_N_per_m": SIGMA_CONDENSATE,
            "lambda_viscosity_ratio": LAMBDA,
            "capillary_number": Ca,
            "D_taylor": D,
            "D_SPH_measured": (sph_anchor_block.get("D_measured")
                               if sph_anchor_block.get("used") else None),
            "D_SPH_note": ("interpolated on the measured D_inf(Ca) curve "
                           "where the trusted range permits; where the "
                           "physiological Ca falls BELOW the measured "
                           "range, D is reported as a monotonicity "
                           "upper bound (type=monotonicity_bounded), "
                           "never as an extrapolated point"
                           if sph_anchor_block.get("used")
                           else "unavailable - see sph_anchor block"),
            "taylor_coefficient": TAYLOR_COEF,
        },
        "ensemble_sensitivity": sens,
        "rg_mean_A": rg_mean,
        "predictions": predictions,
        "sph_anchor": sph_anchor_block,
        "native_scale": {
            "apr1_rASA_sd": sens["apr1_vs_rg"]["y_sd"],
            "apr2_rASA_sd": sens["apr2_vs_rg"]["y_sd"],
        },
        "gnn_context": gnn,
        "assumptions": [
            "mechanical input: MEASURED D_inf(Ca) interpolated in log-Ca over "
            "trusted SPH points; the dimensionless transfer is justified by "
            "the identical viscosity ratio lambda = 10 between the SPH "
            "prototype and the physiological condensate; no extrapolation "
            "beyond the trusted Ca range",
            "where the physiological Ca (1e-3..1e-2) falls below the measured "
            "range, D is bounded from above via the pre-registered "
            "monotonicity of D(Ca) using the tightest classified sweep "
            "point (censored bound / interval hi / trusted plateau); the "
            "bound is parameter-free and labeled, never an extrapolated "
            "point estimate",
            "a merged verdict FAIL is accepted ONLY as the pre-declared A2 "
            "saturation outcome (every integrity check passing); in that "
            "case the anchor is restricted to the monotone-feasible prefix "
            "and the excluded turnover points are reported separately "
            "(docs/A2_TAYLOR_SATURATION_PREDICTION.md)",
            "the analytic Taylor law (valid Ca << 1, Newtonian) is retained "
            "only as the labeled reference curve",
            "affine strain transfer is an UPPER bound (k <= 1); k = 0.1 "
            "shown as a compliant-chain illustration",
            "ensemble (equilibrium) sensitivity proxies the driven response "
            "(fluctuation-dissipation-style assumption; stated, not proven)",
            "static-ensemble analysis; no re-equilibration under shear",
            "physiological mapping uses the recorded condensate parameters "
            "(R = 1 um, sigma = 1e-4 N/m, mu_d = 1e2 Pa s)",
        ],
        "transfer_curve": {
            "Ca": ca_axis.tolist(),
            "dAPR1_rASA_affine": d_apr1.tolist(),
            "dAPR1_rASA_affine_measured": (d_apr1_meas.tolist()
                                           if d_apr1_meas is not None
                                           else None),
            "dAPR1_rASA_affine_bound_below": (
                [None if np.isnan(v) else float(v) for v in bound_apr]
                if d_apr1_meas is not None else None),
            "sph_D_on_axis": (d_axis.tolist() if d_axis is not None
                              else None),
            "sph_D_bound_below": (
                [None if np.isnan(v) else float(v) for v in bound_axis]
                if (anchor is not None and sph_anchor_block.get("used"))
                else None),
            "sph_trusted_Ca_nodes": ([p[0] for p in anchor["trusted_points"]]
                                     if anchor is not None else None),
            "refusal": refusal_note,
        },
    }
    (OUT / "coupling_sph_apr.json").write_text(json.dumps(result, indent=2))

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(ca_axis, taylor_D(ca_axis), "k--", lw=1.2,
                label=f"Taylor reference D = {TAYLOR_COEF:.3f} Ca (Ca << 1)")
    if anchor is not None and sph_anchor_block.get("used"):
        pts = anchor["trusted_points"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "o-", color="tab:blue", lw=1.8, ms=5,
                label="MEASURED SPH $D_\\infty$(Ca) (trusted fits)")
        wl = anchor.get("window_limited", []) or []
        if wl:
            mid = [(p[1][0] + p[1][1]) / 2 for p in wl]
            err = [[mid_i - p[1][0] for mid_i, p in zip(mid, wl)],
                   [p[1][1] - mid_i for mid_i, p in zip(mid, wl)]]
            ax.errorbar([p[0] for p in wl], mid, yerr=err, fmt="s",
                        color="tab:cyan", ms=5, capsize=3,
                        label="window-limited (bounded interval)")
        bf = anchor.get("below_noise_floor", []) or []
        if bf:
            ax.plot([p[0] for p in bf], [p[1] for p in bf],
                    "v", color="tab:gray", ms=6,
                    label="below noise floor (censored upper bound)")
        tp = anchor.get("turnover_points", []) or []
        if tp:
            ax.plot([p[0] for p in tp], [p[1] for p in tp],
                    "o", mfc="none", color="crimson", ms=8, mew=1.6,
                    label="beyond turnover (A2 infeasible;\n"
                          "pre-declared saturation - excluded)")
        ax.semilogx(ca_axis, d_axis, "-", color="tab:blue", lw=1.0,
                    alpha=0.85)
        alo, ahi = anchor["ca_trusted_range"]
        ax.axvspan(ca_axis.min(), alo, color="0.92", zorder=0)
        ax.axvspan(ahi, ca_axis.max(), color="0.92", zorder=0)
        ax.semilogx(ca_axis, bound_axis, ":", color="tab:blue", lw=1.2,
                    drawstyle="steps-pre",
                    label="monotonicity upper bound (below range)")
        ax.text(np.sqrt(alo * ca_axis.min()), 0.55,
                "no extrapolation\n(unforced)", fontsize=7, ha="center",
                color="0.35")
        ax.text(np.sqrt(ahi * ca_axis.max()), 0.55,
                "no extrapolation\n(beyond trusted range)", fontsize=7,
                ha="center", color="0.35")
    for (t, c) in zip(TAU_SHEAR_PA, Ca.values()):
        ax.axvline(c, color="crimson", ls="--", alpha=0.6)
        ax.text(c, 1e-4 * 2, f"{t} Pa", color="crimson", rotation=90,
                va="bottom", fontsize=8)
    ax.set_xlabel("capillary number Ca")
    ax.set_ylabel("condensate deformation D")
    ax.set_title("mechanical input (measured SPH + analytic reference)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.semilogx(ca_axis, d_apr1, "--", color="0.6", lw=1.2,
                label=r"$\Delta$rASA (Taylor reference, affine bound)")
    if d_apr1_meas is not None:
        ax.semilogx(ca_axis, d_apr1_meas, "-", color="tab:blue", lw=2.0,
                    label=r"$\Delta$rASA (MEASURED SPH anchor, affine)")
        lo, hi = anchor["ca_trusted_range"]
        ax.axvspan(ca_axis.min(), lo, color="0.92", zorder=0)
        ax.axvspan(hi, ca_axis.max(), color="0.92", zorder=0)
        ax.semilogx(ca_axis, bound_apr, ":", color="tab:blue", lw=1.2,
                    drawstyle="steps-pre",
                    label=r"$\Delta$rASA monotonicity bound (below range)")
    else:
        ax.text(0.04, 0.96, "measured anchor unavailable:\n"
                + str(anchor_reason), transform=ax.transAxes, fontsize=7,
                color="0.35", va="top")
    sd = sens["apr1_vs_rg"]["y_sd"]
    ax.axhline(sd, color="gray", ls=":", label=f"native heterogeneity SD = {sd:.3f}")
    ax.axhline(0.0, color="k", lw=0.5)
    for (t, c) in zip(TAU_SHEAR_PA, Ca.values()):
        ax.axvline(c, color="crimson", ls="--", alpha=0.6)
    ax.set_xlabel("capillary number Ca")
    ax.set_ylabel(r"predicted $\Delta$ APR1 rASA")
    ax.set_title("coupled response vs native variability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "coupling_transfer_curve.png", dpi=150)
    print(f"saved -> {OUT/'coupling_sph_apr.json'}")
    print(f"saved -> {OUT/'coupling_transfer_curve.png'}")
    print(json.dumps({"sensitivity_slope_apr1_vs_rg_per_A":
                      sens["apr1_vs_rg"]["slope_per_unit"],
                      "Ca_phys": Ca, "D_phys": D,
                      "dAPR1_affine_1Pa": predictions["affine_upper_bound_k1"]
                      [f"apr1_vs_rg|tau_{TAU_SHEAR_PA[1]}_Pa"]["dAPR_rASA"],
                      "native_sd_apr1": sd}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        selftest()
    else:
        main()
