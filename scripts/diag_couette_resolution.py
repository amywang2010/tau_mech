"""Couette wall-slip resolution study (2026-09-02).

Context: after the CSF symmetric-stencil fix, the steady Couette profile is
linear (R2_central ~ 0.998) but the bulk slope is ~14% below the nominal
expectation for the as-delivered config, quantitatively consistent with wall
slip (slope_ratio 0.807 -> slip_frac 0.193 in the old metric; the dissipation
diagnostic showed the dissipative stabilizers account for only part of it and
the residual tracks the slip fraction 1 - slip = 0.857 vs slope_ratio 0.884).

Open question (pre-registered decision rule):
  (a) if slip_frac decreases with resolution (h = 2*spacing -> 0 at fixed
      physical geometry), the slip is a BOUNDARY-DISCRETIZATION artifact of
      the frozen-lattice wall; it is documented as such, and the study
      protocol's use of the MEASURED local shear rate stands;
  (b) if slip_frac is resolution-independent, the slip is a wall-coupling
      FORMULATION property and must be addressed at the formulation level -
      the acceptance band is NOT relaxed either way.

Design: three resolutions with EXACT lattice-row domains (the H=8 domain is
not an integer row multiple of dy = spacing*sqrt(3)/2; exact-row domains
remove the resulting top/bottom wall-gap asymmetry). Same physical channel
(N_rows fluid rows), same run length t = 3 viscous times tau (slowest mode
residual e^-3 ~ 5%, and profile linearity R2 is monitored as the transient
guard), same U_wall. Only the discretization varies.

Run:  python scripts/diag_couette_resolution.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import SPHParams, validate_couette  # noqa: E402

OUT = Path("outputs/sph/audits/couette_resolution_study.json")
SPACING_ROWS = {0.5: 19, 1.0 / 3.0: 29, 0.25: 39}  # exact fluid rows per level


def domain_for(spacing: float, n_rows: int):
    """Exact-row Couette domain: fluid occupies exactly n_rows lattice rows.

    Rows sit at y = k*dy (the lattice generator anchors the first fluid row at
    y0 = 0 up to the documented ~1e-16 accumulation drift). Choosing
    y1 = (n_rows - 0.5)*dy places the top boundary strictly between rows
    n_rows-1 and n_rows, so exactly n_rows rows are fluid for any spacing.
    H_wall = (n_rows + 1)*dy (bottom/top no-slip planes one row outside).
    """
    dy = spacing * np.sqrt(3.0) / 2.0
    y1 = (n_rows - 0.5) * dy
    return (0.0, 0.0, 24.0, float(y1))


def main() -> None:
    smoke = "--smoke" in sys.argv
    params = SPHParams()  # as-delivered: alpha_art=0.1, xsph=0.1, mu=0.05/0.5
    mu = 0.5  # study-droplet viscosity config (nu_0.5_H8 protocol)
    rows = {}
    for spacing, n_rows in sorted(SPACING_ROWS.items(), reverse=True):
        domain = domain_for(spacing, n_rows)
        H_wall = (n_rows + 1) * spacing * np.sqrt(3.0) / 2.0
        tau = H_wall**2 / ((mu / 1.0) * np.pi**2)
        dt = 0.008
        cfl = dt * params.c_s / (2.0 * spacing)  # sound-speed margin
        n_steps = 300 if smoke else int(np.ceil(3.0 * tau / dt))
        r = validate_couette(params=params, n_steps=n_steps, dt=dt,
                             domain=domain, spacing=spacing)
        r.update({"spacing": spacing, "n_rows": n_rows, "H_wall": H_wall,
                  "tau": tau, "cfl_dt_ratio": cfl,
                  "viscous_dt_ratio": dt / (0.125 * (2 * spacing) ** 2 / mu)})
        rows[f"s_{spacing:.4f}_rows_{n_rows}"] = r
        print(f"s={spacing:.4f} rows={n_rows}: slip_frac={r['slip_frac']:.4f} "
              f"u_wall_fluid={r['u_wall_fluid']:.4f} u_top={r['u_top_fluid']:.4f} "
              f"slope_ratio_central={r['slope_ratio_central']:.4f} "
              f"R2c={r['r2_central']:.4f} y_mid_dev={r['y_mid_dev']:.4f} "
              f"CFL={cfl:.3f} [{n_steps} steps]")

    if smoke:
        print("SMOKE OK (no record written)")
        return

    # Pre-registered attribution analysis: slip vs h = 2*spacing.
    h = np.array([2.0 * r["spacing"] for r in rows.values()])
    slip = np.array([r["slip_frac"] for r in rows.values()])
    order = np.argsort(-h)  # coarse -> fine
    monotone = bool(np.all(np.diff(slip[order]) < 0))
    if len(h) >= 3 and slip[order][-1] < slip[order][0]:
        coef = np.polyfit(h, slip, 1)
        slip_extrapolated = float(max(np.polyval(coef, 0.0), 0.0))
    else:
        coef, slip_extrapolated = None, None
    attribution = {
        "monotone_decreasing_with_refinement": monotone,
        "slip_vs_h_fit": None if coef is None else
                        {"slope_per_h": float(coef[0]),
                         "intercept": float(coef[1]),
                         "slip_at_h_zero": slip_extrapolated},
        "decision_rule": ("(a) monotone decrease with refinement => boundary-"
                          "discretization artifact, documented; measured-"
                          "local-shear-rate protocol stands. (b) resolution-"
                          "independent => wall-formulation property; band NOT "
                          "relaxed; formulation-level response required."),
    }
    record = {
        "solver_state": "post CSF-symmetric-stencil fix + lattice-row profile "
                        "binning (2026-09-02)",
        "protocol": ("exact-row domains; t = 3 viscous times tau; U_wall = 2; "
                     "as-delivered dissipators (alpha=0.1, xsph=0.1); "
                     "mu_solvent = 0.5 config; slip = linear extrapolation of "
                     "the bulk fit to both no-slip planes"),
        "configs": rows,
        "attribution": attribution,
    }
    OUT.write_text(json.dumps(record, indent=2))
    print(f"saved -> {OUT}")
    print("ATTRIBUTION:", json.dumps(attribution, indent=1))


if __name__ == "__main__":
    main()
