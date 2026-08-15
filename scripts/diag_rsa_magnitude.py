"""Diagnose the SASA self-occlusion bug magnitude and the rASA change.

Question: the pre-fix APR1 (VQIINK) mean rASA was 0.286 and the post-fix
value is 0.547 (1.91x). The documented self-occlusion bug is claimed to be
~10% (exposed atoms) or ~22% (fraction of Fibonacci points with |pts|<1).
Neither explains a 1.9x change. This script measures, from first principles:

  1. The fraction of Fibonacci points with |pts| strictly < 1.0.
  2. Old-style (self-included) vs new-style (self-excluded) per-atom SASA
     ratio on a REAL Tau model, for exposed vs buried atoms.
  3. The resulting APR1/APR2 rASA under both conventions.

This distinguishes "the rASA change is fully explained by the self-occlusion
fix" from "there is an unexplained discrepancy that needs further audit".
"""
import json
import sys

import numpy as np

from tau_mech.sasa import fibonacci_sphere, vdw_radii_for_elements
from tau_mech.constants import REFERENCE_SASA, VDW_RADII, PROBE_RADIUS, N_PROBE_POINTS
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
# load a real preprocessed model npz (post-fix, has coords + res_rsa + apr)
MODEL_NPZ = "outputs/PED00422/models/model_0000.npz"  # may not exist; try others


def main() -> None:
    # ---- 1. fraction of |pts| < 1 ---------------------------------------
    for n in (480, 2000):
        pts = fibonacci_sphere(n)
        norms = np.linalg.norm(pts, axis=1)
        frac_lt = float((norms < 1.0).mean())
        frac_gt = float((norms > 1.0).mean())
        frac_eq = float((norms == 1.0).mean())
        print(f"[1] n={n}: |pts|<1: {frac_lt:.4f}  |pts|>1: {frac_gt:.4f}  ==1: {frac_eq:.4f}")

    # ---- 2. old vs new SASA on a real model ------------------------------
    # find a model npz
    import glob
    cands = sorted(glob.glob("outputs/PED00422/models/*.npz"))
    print(f"[2] found {len(cands)} model npz files")
    if not cands:
        print("    no model npz -> skipping real-model test")
        return
    d = np.load(cands[0], allow_pickle=True)
    print(f"    keys: {list(d.keys())}")
    coords = d["coords"].astype(np.float64)
    elements = [str(e) for e in d["element"]]
    radii = vdw_radii_for_elements(elements)
    res_rsa = d["res_rsa"]
    print(f"    model shape: coords={coords.shape}  res_rsa mean={res_rsa.mean():.4f}")

    n = len(coords)
    sphere_radii = radii + PROBE_RADIUS
    max_sphere = float(sphere_radii.max())
    tree = cKDTree(coords)
    pts = fibonacci_sphere(N_PROBE_POINTS)
    sphere_area = 4.0 * np.pi

    sasa_new = np.zeros(n)
    sasa_old = np.zeros(n)
    for i in range(n):
        nbr_all = tree.query_ball_point(coords[i], r=sphere_radii[i] + max_sphere)
        # NEW: exclude self
        nbr_new = np.asarray([j for j in nbr_all if j != i], dtype=np.int64)
        sphere = coords[i] + pts * sphere_radii[i]
        if len(nbr_new) == 0:
            sasa_new[i] = sphere_area * sphere_radii[i] ** 2
        else:
            d = np.linalg.norm(sphere[:, None, :] - coords[nbr_new][None, :, :], axis=2)
            occ = (d < (radii[nbr_new] + PROBE_RADIUS)).any(axis=1)
            sasa_new[i] = sphere_area * sphere_radii[i] ** 2 * int((~occ).sum()) / N_PROBE_POINTS
        # OLD: self included
        nbr_old = np.asarray(nbr_all, dtype=np.int64)
        d = np.linalg.norm(sphere[:, None, :] - coords[nbr_old][None, :, :], axis=2)
        occ = (d < (radii[nbr_old] + PROBE_RADIUS)).any(axis=1)
        sasa_old[i] = sphere_area * sphere_radii[i] ** 2 * int((~occ).sum()) / N_PROBE_POINTS

    ratio = sasa_new / np.where(sasa_old > 0, sasa_old, 1.0)
    # exposed = high old SASA, buried = low old SASA
    full = sphere_area * sphere_radii ** 2
    exposed_frac_old = sasa_old / full
    # buckets
    for lo, hi, label in [(0.0, 0.2, "buried(0-20%)"), (0.2, 0.6, "partial(20-60%)"),
                          (0.6, 1.0, "exposed(60-100%)")]:
        m = (exposed_frac_old >= lo) & (exposed_frac_old < hi)
        if m.sum():
            print(f"    {label}: n={m.sum():5d}  new/old ratio mean={ratio[m].mean():.4f} "
                  f"median={np.median(ratio[m]):.4f}")

    # ---- 3. rASA under old vs new convention -----------------------------
    one_letter = [str(e) for e in d["one_letter"]] if "one_letter" in d else None
    # reconstruct res rsa from atom sasa using res idx if present
    if "res_idx" in d:
        ridx = d["res_idx"].astype(np.int64)
        n_res = int(ridx.max()) + 1
        res_sasa_old = np.zeros(n_res); np.add.at(res_sasa_old, ridx, sasa_old)
        res_sasa_new = np.zeros(n_res); np.add.at(res_sasa_new, ridx, sasa_new)
        seq = [str(e) for e in d["sequence"]] if "sequence" in d else None
        print(f"    has res_idx, n_res={n_res}, has sequence={seq is not None}")
        if seq is not None:
            refs = np.asarray([REFERENCE_SASA.get(a.upper(), REFERENCE_SASA['X']) for a in seq])
            rsa_old = res_sasa_old / np.where(refs > 0, refs, 1.0)
            rsa_new = res_sasa_new / np.where(refs > 0, refs, 1.0)
            # APR1 VQIINK = 275-280 (0-based 274..279 for full-length tau offset 0)
            m1 = np.arange(n_res)
            apr1 = (m1 >= 274) & (m1 <= 279)
            apr2 = (m1 >= 305) & (m1 <= 310)
            print(f"    APR1 VQIINK old rASA mean={rsa_old[apr1].mean():.4f}  new={rsa_new[apr1].mean():.4f}  ratio={rsa_new[apr1].mean()/max(rsa_old[apr1].mean(),1e-9):.3f}")
            print(f"    APR2 VQIVYK old rASA mean={rsa_old[apr2].mean():.4f}  new={rsa_new[apr2].mean():.4f}  ratio={rsa_new[apr2].mean()/max(rsa_old[apr2].mean(),1e-9):.3f}")
            print(f"    whole-model mean rASA old={rsa_old.mean():.4f}  new={rsa_new.mean():.4f}  ratio={rsa_new.mean()/max(rsa_old.mean(),1e-9):.3f}")


if __name__ == "__main__":
    main()
