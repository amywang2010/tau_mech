"""Static CSF band analysis on one equilibrated droplet.

Answers two questions raised by the Laplace calibration (which showed
sigma_eff = dP*R ~ 0.42-0.49 x sigma_input, decreasing with R):

  1. Where does the smoothed color field c~ reach its plateau (c~=1) and does
     the Laplace core mask (r < R - 3h) actually sit at c~=1?
  2. What fraction of the color-gradient integral (which sets the pressure
     jump) lies inside the DROPLET phase vs the solvent? (The "band-split"
     hypothesis says ~half, giving sigma_eff ~ 0.5 sigma_input.)

The pressure field follows p = p_out + sigma*kappa*c~ across the band, so
c~(core) directly sets the measured dP.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.sph import (  # noqa: E402
    SPHParams,
    build_pairs,
    cubic_spline,
    cubic_spline_dwdr,
    make_couette_droplet_state,
    run,
    smooth_color_field,
    step,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    p = SPHParams(sigma_surf=args.sigma, mu_droplet=0.5)
    R = args.R
    W = 2.0 * (R + 4.0) + 4.0
    H = 2.0 * (R + 4.0) + 4.0
    s = make_couette_droplet_state(p, domain=(0.0, 0.0, W, H), spacing=0.5,
                                   droplet_radius=R, n_wall_layers=2)
    run(s, p, 0, 0.008)
    for _ in range(args.steps):
        step(s, p, 0.008)
    pairs, d, e = build_pairs(s.pos, p.h)
    c = smooth_color_field(s, p, pairs, d, n_passes=p.n_color_smooth)
    vol = s.mass / np.maximum(s.rho, p.rho_floor * s.rho0)
    in_d = s.phase == 1
    com = s.pos[in_d].mean(axis=0)
    r = np.linalg.norm(s.pos - com, axis=1)
    rhat = (s.pos - com) / np.maximum(r, 1e-9)[:, None]

    # --- radial profiles of c~ and P ---
    nb = int((R + 4.0) * 4) + 8
    bins = np.minimum((r * 4).astype(int), nb - 1)
    cm = np.zeros(nb)
    pm = np.zeros(nb)
    rm = np.zeros(nb)
    cnt = np.zeros(nb)
    np.add.at(cm, bins, c)
    np.add.at(pm, bins, s.pressure)
    np.add.at(rm, bins, s.rho)
    np.add.at(cnt, bins, 1.0)
    ok = cnt > 0
    rb = (np.arange(nb) + 0.5) / 4.0
    print(f"R={R}  sigma={p.sigma_surf}  steps={args.steps}  N={s.n}")
    print("  r       c~      P(r)     rho(r)")
    for i in np.where(ok)[0]:
        print(f"  {rb[i]:5.2f}  {cm[i] / cnt[i]:5.3f}  {pm[i] / cnt[i]:+7.4f}  "
              f"{rm[i] / cnt[i]:6.4f}")

    # c~=1 plateau boundary: outermost radius where c~ >= 0.99
    c_avg = np.full(nb, np.nan)
    c_avg[ok] = cm[ok] / cnt[ok]
    cc = np.where(ok & (c_avg >= 0.99))[0]
    if len(cc):
        print(f"  c~>=0.99 plateau starts at r <= {rb[cc[-1]]:.2f} "
              f"(droplet side; R-{R - rb[cc[-1]]:.2f}h from interface)")
    else:
        print("  c~ NEVER reaches 0.99 inside the domain!")

    # --- color-gradient shares (the band-split) ---
    # grad(c~)_i = sum_j V_j (c_j - c_i) dw e (field evaluation, both ends)
    i, j = pairs[:, 0], pairs[:, 1]
    dw = cubic_spline_dwdr(d, p.h)
    dc_i = (c[j] - c[i]) * vol[j] * dw
    dc_j = (c[j] - c[i]) * vol[i] * dw
    idx = np.concatenate([i, j])
    gx = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 0], dc_j * e[:, 0]]),
                     minlength=s.n)
    gy = np.bincount(idx, weights=np.concatenate([dc_i * e[:, 1], dc_j * e[:, 1]]),
                     minlength=s.n)
    grad = np.stack([gx, gy], axis=1)
    # radial component of V*grad(c~): the part that balances the pressure jump
    vg = vol * np.einsum("ij,ij->i", grad, rhat)
    droplet_share = float(vg[in_d].sum())
    solvent_share = float(vg[~in_d & (s.phase != 2)].sum())
    total = droplet_share + solvent_share
    print(f"  band-split: droplet share {droplet_share:+.3f} "
          f"({droplet_share / total * 100:.1f}%), solvent share "
          f"{solvent_share:+.3f} ({solvent_share / total * 100:.1f}%), "
          f"total {total:+.3f}")
    print(f"  (a 50/50 split => dP ~ 0.5*sigma/R; a split shifted inward "
          f"=> dP closer to sigma/R)")


if __name__ == "__main__":
    main()
