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


def load_ensemble(csv="outputs/PED00422/summary.csv"):
    import pandas as pd
    df = pd.read_csv(csv)
    return df


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_ensemble()

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

    # ---- 3+4. prediction with bounded strain transfer ---------------------
    rg_mean = float(df["rg_equal_weight"].mean())
    predictions = {}
    for label, k in (("affine_upper_bound_k1", K_AFFINE),
                     ("compliant_k0.1", K_COMPLIANT)):
        preds = {}
        for sens_key, d_key in (("apr1_vs_rg", None), ("apr2_vs_rg", None)):
            s = sens[sens_key]
            for tau_key, d_val in D.items():
                dRg = k * d_val * rg_mean
                dAPR = s["slope_per_unit"] * dRg
                preds[f"{sens_key}|{tau_key}"] = {
                    "D_taylor": d_val,
                    "dRg_A": float(dRg),
                    "dAPR_rASA": float(dAPR),
                    "fraction_of_native_sd": float(abs(dAPR) / s["y_sd"]),
                }
        predictions[label] = preds

    # Ca sweep for the transfer curve figure (analytic regime, Ca << 1)
    ca_axis = np.logspace(-4, -1, 60)
    d_apr1 = (sens["apr1_vs_rg"]["slope_per_unit"] * K_AFFINE
              * taylor_D(ca_axis) * rg_mean)

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

    result = {
        "chain": ("tissue shear -> condensate deformation (Taylor/Ca) -> "
                  "bounded affine strain transfer -> APR rASA change "
                  "(ensemble sensitivity)"),
        "physiology": {
            "tau_shear_Pa": list(TAU_SHEAR_PA),
            "R_condensate_m": R_CONDENSATE_M,
            "sigma_N_per_m": SIGMA_CONDENSATE,
            "lambda_viscosity_ratio": LAMBDA,
            "capillary_number": Ca,
            "D_taylor": D,
            "taylor_coefficient": TAYLOR_COEF,
        },
        "ensemble_sensitivity": sens,
        "rg_mean_A": rg_mean,
        "predictions": predictions,
        "native_scale": {
            "apr1_rASA_sd": sens["apr1_vs_rg"]["y_sd"],
            "apr2_rASA_sd": sens["apr2_vs_rg"]["y_sd"],
        },
        "gnn_context": gnn,
        "assumptions": [
            "linear response; Taylor limit valid for Ca << 1",
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
        },
    }
    (OUT / "coupling_sph_apr.json").write_text(json.dumps(result, indent=2))

    # ---- figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(ca_axis, taylor_D(ca_axis), "k-",
                label=f"Taylor limit D = {TAYLOR_COEF:.3f} Ca")
    for (t, c) in zip(TAU_SHEAR_PA, Ca.values()):
        ax.axvline(c, color="crimson", ls="--", alpha=0.6)
        ax.text(c, 1e-4 * 2, f"{t} Pa", color="crimson", rotation=90,
                va="bottom", fontsize=8)
    ax.set_xlabel("capillary number Ca")
    ax.set_ylabel("condensate deformation D")
    ax.set_title("mechanical input (analytic regime)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(ca_axis, d_apr1, "b-",
                label=r"$\Delta$rASA (APR1, affine bound)")
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
    main()
