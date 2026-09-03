"""Publication-grade figure for the final physiological shear sweep (2026-09-02).

Reads ONLY the canonical merged record (outputs/sph/sph_shear_sweep.json +
sph_traces.npz) produced by scripts/merge_final_sweep.py — no solver runs,
no free parameters. Panels:

  (a) Deformation D vs capillary number: measured D_inf per rate (filled),
      censored cases (< LOD) as open markers at their upper bound with
      downward arrows, Taylor small-deformation prediction
      D = (9/50) * Ca_meas (no fit — analytic curve, zero parameters).
  (b) Raw deformation traces D(t) per rate with the control envelope
      (D_ctrl_sustained ± N) shaded.
  (c) Ca consistency: Ca_measured / Ca_nominal per rate (the wall-slip
      attenuation factor), log-x.

Usage:
    python scripts/plot_final_sweep.py            # real canonical record
    python scripts/plot_final_sweep.py --sbx      # sandbox dry-run mode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def resolve(sweep_dir: Path) -> tuple[Path, Path]:
    """Locate (merged_record, traces) under either layout.

    Canonical:   outputs/sph/{sph_shear_sweep.json, sph_traces.npz}
                 outputs/sph/sweep/rate_*/...
    Sandbox:     <dir>/rate_*/...   (merged record built on the fly)
    """
    for base in (sweep_dir, sweep_dir / "sweep"):
        rec_p, tr_p = base / "sph_shear_sweep.json", base / "sph_traces.npz"
        if rec_p.exists() and tr_p.exists():
            return rec_p, tr_p
    # on-the-fly merge from rate_* dirs (sandbox mode)
    rows, traces = [], {}
    for rd in sorted(sweep_dir.glob("rate_*")):
        rp = rd / "sph_shear_sweep.json"
        if not rp.exists():
            continue
        rec = json.loads(rp.read_text())
        if rec.get("rows"):
            rows.append(rec["rows"][0])
        tp = rd / "sph_traces.npz"
        if tp.exists():
            t = np.load(tp, allow_pickle=True)
            tr = t["traces"].item() if getattr(t["traces"], "ndim", 1) == 0 \
                else t["traces"]
            key = str(rec["rows"][0]["shear_rate_nominal"])
            if key in tr:
                traces[key] = tr[key]
    rows.sort(key=lambda r: r["shear_rate_nominal"])
    merged = {"rows": rows}
    tmp = sweep_dir / "_merged_for_plot.json"
    tmp.write_text(json.dumps(merged, indent=2))
    np.savez(sweep_dir / "_merged_for_plot.npz", traces=traces)
    return tmp, sweep_dir / "_merged_for_plot.npz"


def load(sweep_dir: Path):
    rec_p, tr_p = resolve(sweep_dir)
    rec = json.loads(rec_p.read_text())
    t = np.load(tr_p, allow_pickle=True)
    tr = t["traces"].item() if getattr(t["traces"], "ndim", 1) == 0 else t["traces"]
    return rec, tr


def control_envelope(tr: dict) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(D_ctrl_sustained, noise_floor_N, t_ctrl, D_ctrl) from the control trace."""
    d_c = np.asarray(tr["0.0"]["taylor"], dtype=float)
    t_c = np.asarray(tr["0.0"]["t"], dtype=float)
    n_tail = max(1, len(d_c) // 5)
    sust = float(np.mean(d_c[-n_tail:]))
    return sust, float(np.abs(d_c - sust).max()), t_c, d_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=None,
                    help="override record dir (sandbox dry-runs)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sbx", action="store_true",
                    help="write to /tmp-style sandbox paths instead of canonical")
    args = ap.parse_args()
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir \
        else ROOT / "outputs" / "sph"
    out_dir = Path(args.out_dir) if args.out_dir else sweep_dir
    if args.sbx:
        out_dir = Path(os.environ.get("SBX_OUT", str(out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    rec, tr = load(sweep_dir)
    sust, N, t_c, d_c = control_envelope(tr)
    rows = sorted(rec["rows"], key=lambda r: r["shear_rate_nominal"])
    sheared = [r for r in rows if r["shear_rate_nominal"] > 0.0]

    # classification identical to merge_final_sweep.py (v1.2 rule)
    def is_below_floor(r) -> bool:
        plateau = r.get("taylor_plateau_fit")
        if plateau is None:
            return False
        return abs(float(plateau) - sust) < N

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    taylor = 9.0 / 50.0

    # ---- (a) D_inf vs Ca ---------------------------------------------------
    ax = axes[0]
    ca_theory = np.logspace(-3, 0.5, 100)
    ax.plot(ca_theory, taylor * ca_theory, "k--", lw=1.2,
            label=r"Taylor $D=\frac{9}{50}\,Ca$ (analytic)")
    for r in sheared:
        ca, plateau = r["capillary_number_Ca"], r.get("taylor_plateau_fit")
        if plateau is None:
            continue
        below = is_below_floor(r)
        # No invented error bars: no measurement-uncertainty record exists
        # for D_inf in the sweep; real variability is shown by the traces
        # in panel (b) and the control envelope.
        ax.plot(ca, plateau,
                marker="o" if not below else "v", ms=7,
                mfc="C0" if not below else "none", mec="C0",
                ls="none",
                label=(f"rate {r['shear_rate_nominal']:g}"
                       + (" (< LOD)" if below else "")))
        if below:
            ax.annotate("", xy=(ca, plateau * 0.55), xytext=(ca, plateau),
                        arrowprops=dict(arrowstyle="->", color="C0", lw=0.9))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"measured $Ca = \mu\dot\gamma R/\sigma_{eff}$")
    ax.set_ylabel(r"deformation $D = (L-B)/(L+B)$")
    ax.set_title("(a) steady deformation vs Ca")
    ax.legend(fontsize=7, framealpha=0.9)

    # ---- (b) traces --------------------------------------------------------
    ax = axes[1]
    ax.axhspan(sust - N, sust + N, color="gray", alpha=0.25,
               label="control envelope $\\pm N$")
    ax.plot(t_c, d_c, color="k", lw=1.4, label="control (0.0)")
    for r in sheared:
        key = str(r["shear_rate_nominal"])
        if key not in tr:
            continue
        tt = np.asarray(tr[key]["t"], dtype=float)
        dd = np.asarray(tr[key]["taylor"], dtype=float)
        below = is_below_floor(r)
        ax.plot(tt, dd, lw=1.2,
                label=f"rate {r['shear_rate_nominal']:g}" + (" (< LOD)" if below else ""))
    ax.set_xlabel("time (SPH units)")
    ax.set_ylabel(r"$D(t)$")
    ax.set_title("(b) deformation traces")
    ax.legend(fontsize=7)

    # ---- (c) Ca consistency ------------------------------------------------
    ax = axes[2]
    for r in sheared:
        if r["capillary_number_Ca"] and r["capillary_number_nominal"]:
            ax.plot(r["shear_rate_nominal"],
                    r["capillary_number_Ca"] / r["capillary_number_nominal"],
                    "o", color="C1", ms=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"nominal shear rate $\dot\gamma_{nom}$")
    ax.set_ylabel(r"$Ca_{meas}/Ca_{nom}$")
    ax.set_title("(c) wall-slip attenuation of Ca")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylim(0, 1.1)

    fig.suptitle("Tau-droplet SPH response to physiological shear "
                 "(validated solver, CSF-symmetric stencil, "
                 r"$\sigma_{eff}$=1.064)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "sph_deformation_vs_Ca.png"
    fig.savefig(out, dpi=300)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
