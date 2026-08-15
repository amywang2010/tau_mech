"""Definitive SASA magnitude audit on a real PED model.

Resolves the discrepancy: pre-fix APR1 (VQIINK) mean rASA was 0.286, post-fix
is 0.547 (1.91x). The documented self-occlusion bug removes ~22% of probe
points, predicting ~1.28x, NOT 1.91x. This script, on the FIRST real model of
PED00422, computes:

  A. OLD code path  : self atom INCLUDED in the occlusion test (reconstructed)
  B. NEW code path  : self excluded (current shipped code)
  C. BRUTE-FORCE    : independent reference: for each atom, for each probe
                      point, distance to EVERY other atom (no kd-tree, no
                      early return), strict `<` boundary.

It reports per-atom SASA and the APR1/APR2 rASA under each convention, so we
can see (1) whether NEW == BRUTE-FORCE (correctness) and (2) what the OLD code
actually produced (to explain the 1.91x).
"""
import tarfile
import io
import numpy as np
from scipy.spatial import cKDTree

from tau_mech.io import parse_models
from tau_mech.sasa import fibonacci_sphere, vdw_radii_for_elements
from tau_mech.constants import REFERENCE_SASA, PROBE_RADIUS, N_PROBE_POINTS
from tau_mech.numbering import residue_index_from_atoms

DATA = "../PED00422_ensembles.tar.gz"
MEMBER = "PED00422e002.tar.gz"
INNER = "pdbfile.pdb"


def load_first_model():
    outer = tarfile.open(DATA, "r:gz")
    inner_bytes = outer.extractfile(MEMBER).read()
    outer.close()
    inner = tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz")
    f = inner.extractfile(INNER)
    text = f.read().decode("ascii")
    inner.close()
    model = next(parse_models(io.StringIO(text), heavy_only=True))
    return model


def sasa_old(coords, radii, probe, n_points):
    """Reconstructed OLD code: self INCLUDED in neighbor list."""
    n = len(coords)
    sphere_radii = radii + probe
    max_sphere = float(sphere_radii.max())
    tree = cKDTree(coords)
    pts = fibonacci_sphere(n_points)
    sasa = np.zeros(n)
    sphere_area = 4.0 * np.pi
    for i in range(n):
        nbr = tree.query_ball_point(coords[i], r=sphere_radii[i] + max_sphere)
        nbr = np.asarray(nbr, dtype=np.int64)  # self included (OLD)
        if len(nbr) == 0:
            sasa[i] = sphere_area * sphere_radii[i] ** 2
            continue
        sphere = coords[i] + pts * sphere_radii[i]
        d = np.linalg.norm(sphere[:, None, :] - coords[nbr][None, :, :], axis=2)
        occluded = (d < (radii[nbr] + probe)).any(axis=1)
        sasa[i] = sphere_area * sphere_radii[i] ** 2 * int((~occluded).sum()) / n_points
    return sasa


def sasa_new(coords, radii, probe, n_points):
    """Current shipped code: self excluded."""
    n = len(coords)
    sphere_radii = radii + probe
    max_sphere = float(sphere_radii.max())
    tree = cKDTree(coords)
    pts = fibonacci_sphere(n_points)
    sasa = np.zeros(n)
    sphere_area = 4.0 * np.pi
    for i in range(n):
        nbr = tree.query_ball_point(coords[i], r=sphere_radii[i] + max_sphere)
        nbr = np.asarray([j for j in nbr if j != i], dtype=np.int64)
        if len(nbr) == 0:
            sasa[i] = sphere_area * sphere_radii[i] ** 2
            continue
        sphere = coords[i] + pts * sphere_radii[i]
        d = np.linalg.norm(sphere[:, None, :] - coords[nbr][None, :, :], axis=2)
        occluded = (d < (radii[nbr] + probe)).any(axis=1)
        sasa[i] = sphere_area * sphere_radii[i] ** 2 * int((~occluded).sum()) / n_points
    return sasa


def sasa_brute(coords, radii, probe, n_points):
    """Independent reference: every atom vs every OTHER atom, no kd-tree."""
    n = len(coords)
    sphere_radii = radii + probe
    pts = fibonacci_sphere(n_points)
    sasa = np.zeros(n)
    sphere_area = 4.0 * np.pi
    for i in range(n):
        sphere = coords[i] + pts * sphere_radii[i]
        exposed = np.ones(n_points, dtype=bool)
        for j in range(n):
            if j == i:
                continue
            # occlusion possible only if the spheres could overlap
            if np.linalg.norm(coords[i] - coords[j]) > sphere_radii[i] + sphere_radii[j]:
                continue
            d = np.linalg.norm(sphere - coords[j][None, :], axis=1)
            exposed &= (d >= (radii[j] + probe))
        sasa[i] = sphere_area * sphere_radii[i] ** 2 * int(exposed.sum()) / n_points
    return sasa


def main():
    model = load_first_model()
    coords = model["coords"].astype(np.float64)
    elements = [str(e) for e in model["element"]]
    radii = vdw_radii_for_elements(elements)
    print(f"model: {len(coords)} heavy atoms")

    # use a coarser point count for brute force to keep it fast
    NP = 240
    print("computing OLD ...")
    s_old = sasa_old(coords, radii, PROBE_RADIUS, NP)
    print("computing NEW ...")
    s_new = sasa_new(coords, radii, PROBE_RADIUS, NP)
    print("computing BRUTE ...")
    s_brute = sasa_brute(coords, radii, PROBE_RADIUS, NP)

    # per-atom comparison
    rel_new_old = s_new / np.where(s_old > 0, s_old, 1.0)
    rel_new_brute = s_new / np.where(s_brute > 0, s_brute, 1.0)
    print("\nper-atom NEW/OLD ratio: mean=%.4f median=%.4f" % (rel_new_old.mean(), np.median(rel_new_old)))
    print("per-atom NEW/BRUTE ratio: mean=%.4f median=%.4f" % (rel_new_brute.mean(), np.median(rel_new_brute)))
    # which atoms have NEW < BRUTE (potential kd-tree neighbor miss)?
    bad = np.where(s_new < s_brute - 1e-9)[0]
    print(f"atoms with NEW < BRUTE: {len(bad)} / {len(coords)}")
    if len(bad):
        diffs = (s_brute[bad] - s_new[bad])
        print(f"  max abs diff = {diffs.max():.4f} A^2  at atom {bad[diffs.argmax()]}")
        # fraction of full sphere each diff represents
        sphere_radii = radii + PROBE_RADIUS
        full = 4 * np.pi * sphere_radii ** 2
        frac = diffs / full[bad]
        print(f"  diffs as fraction of full sphere: max={frac.max():.4f}  n>1%={int((frac > 0.01).sum())}  n>5%={int((frac > 0.05).sum())}")

    # residue-level rASA
    ridx = residue_index_from_atoms(model["resseq"], model["chain"])
    n_res = int(ridx.max()) + 1
    from tau_mech.constants import THREE_TO_ONE
    resname = np.asarray([str(r) for r in model["resname"]])
    # map residue index -> one-letter (take first atom's resname per residue)
    one_letter = [
        THREE_TO_ONE.get(resname[int(np.where(ridx == r)[0][0])], "X")
        for r in range(n_res)
    ]

    def res_rsa(atom_sasa):
        rs = np.zeros(n_res)
        np.add.at(rs, ridx, atom_sasa)
        refs = np.asarray([REFERENCE_SASA.get(a, REFERENCE_SASA["X"]) for a in one_letter])
        return rs / np.where(refs > 0, refs, 1.0)

    for name, s in [("OLD", s_old), ("NEW", s_new), ("BRUTE", s_brute)]:
        r = res_rsa(s)
        # PED00422 is full-length Tau: PDB numbering == Tau numbering, APR1=275-280
        apr1 = r[274:280].mean()
        apr2 = r[305:311].mean()
        print(f"{name:6s}: APR1(VQIINK) rASA={apr1:.4f}  APR2(VQIVYK) rASA={apr2:.4f}  mean_rASA={r.mean():.4f}")


if __name__ == "__main__":
    main()
