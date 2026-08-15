"""Solvent-accessible surface area (SASA) via the Shrake-Rupley algorithm.

Implementation follows Shrake A. & Rupley J.A. (1973), J. Mol. Biol.
79:351-371: a set of probe points is distributed on a sphere of radius
r_vdW + r_probe around each atom; the fraction of points not inside any
neighboring atom's sphere (r_vdW + r_probe) is the atom's exposed fraction.

Exact protocol (recorded in the pipeline config for reproducibility):
  * probe radius            : 1.4 A (water)
  * van der Waals radii     : Chothia/NACCESS convention
                             (C 1.80, N 1.65, O 1.40, S 1.85)
  * probe points per atom   : 480 (Fibonacci sphere distribution)
  * only heavy atoms        : hydrogens excluded
  * neighbors               : via scipy cKDTree (query radius r_i + r_max)

Relative accessibility (rASA) uses the Tien et al. (2013) per-residue
reference values (see constants.REFERENCE_SASA).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .constants import (
    N_PROBE_POINTS,
    PROBE_RADIUS,
    REFERENCE_SASA,
    VDW_RADII,
)


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Uniform distribution of ``n_points`` unit vectors (Fibonacci sphere)."""
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    i = np.arange(n_points, dtype=np.float64)
    y = 1.0 - 2.0 * (i / (n_points - 1.0))
    r = np.sqrt(np.maximum(0.0, 1.0 - y ** 2))
    theta = ga * i
    pts = np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])
    return pts


def vdw_radii_for_elements(elements: Sequence[str]) -> np.ndarray:
    """vdW radii per atom element (unknown element -> carbon radius)."""
    return np.asarray([VDW_RADII.get(str(e).upper(), VDW_RADII["C"]) for e in elements],
                      dtype=np.float64)


def compute_sasa(
    coords: np.ndarray,
    radii: np.ndarray,
    probe: float = PROBE_RADIUS,
    n_points: int = N_PROBE_POINTS,
    progress: bool = False,
) -> np.ndarray:
    """Per-atom solvent-accessible surface area (Angstrom^2).

    Parameters
    ----------
    coords : (N, 3) float array of heavy-atom coordinates
    radii : (N,) float array of vdW radii per atom
    probe : probe radius in Angstrom
    n_points : probe points per atom sphere
    progress : print per-atom progress (mainly for debugging)

    Returns (N,) array of per-atom SASA values.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(coords)
    sphere_radii = radii + probe
    max_sphere = float(sphere_radii.max())
    tree = cKDTree(coords)
    pts = fibonacci_sphere(n_points)
    sasa = np.zeros(n, dtype=np.float64)
    sphere_area = 4.0 * np.pi

    for i in range(n):
        # Candidate neighbors: any OTHER atom whose sphere could cover a probe
        # point of atom i. A probe point of i lies at distance sphere_radii[i]
        # from center i; atom j can occlude it only if dist(ci, cj) <
        # sphere_radii[i] + sphere_radii[j] <= sphere_radii[i] + max_sphere.
        # SELF is explicitly excluded: atom i's own occlusion sphere coincides
        # exactly with its probe sphere (radius radii[i] + probe), so a probe
        # point sits ON that sphere (d == radii[i] + probe). The strict `<`
        # occlusion test is correct for that boundary (a point on the surface
        # is accessible), BUT floating-point makes |pts| round to slightly < 1
        # for ~22% of the Fibonacci points, turning d into a strict `<` hit and
        # falsely self-occluding those points. The RELATIVE underestimate this
        # causes depends on burial: ~10% for a fully exposed atom (89.4% ->
        # 99.4% of full sphere area), but much larger for partially buried
        # atoms -- a residue that is only ~30% exposed loses a fixed ~22% of
        # its points, so its SASA is roughly halved. That is why the
        # aggregation-prone-region rASA roughly doubled (0.286 -> 0.547) after
        # this fix; verified against an independent brute-force reference
        # (audit 2026-08-15: NEW == BRUTE, 0/3217 atoms differ). Excluding
        # self removes the bug while keeping the correct strict-`<` boundary
        # convention for genuine neighbors.
        nbr = tree.query_ball_point(coords[i], r=sphere_radii[i] + max_sphere)
        nbr = np.asarray([j for j in nbr if j != i], dtype=np.int64)
        if len(nbr) == 0:  # only itself
            sasa[i] = sphere_area * sphere_radii[i] ** 2
            continue
        sphere = coords[i] + pts * sphere_radii[i]      # (P, 3)
        # distances from each probe point to each neighbor center
        d = np.linalg.norm(sphere[:, None, :] - coords[nbr][None, :, :], axis=2)
        occluded = (d < (radii[nbr] + probe)).any(axis=1)
        exposed = int((~occluded).sum())
        sasa[i] = sphere_area * sphere_radii[i] ** 2 * exposed / n_points
        if progress and (i % 100 == 0):
            print(f"  atom {i}/{n} SASA={sasa[i]:.1f}")
    return sasa


def residue_sasa(atom_sasa: np.ndarray, atom_res_idx: np.ndarray, n_res: int) -> np.ndarray:
    """Sum per-atom SASA into per-residue SASA."""
    out = np.zeros(n_res, dtype=np.float64)
    np.add.at(out, atom_res_idx, atom_sasa)
    return out


def relative_sasa(res_sasa: np.ndarray, one_letter: Sequence[str]) -> np.ndarray:
    """Relative solvent accessibility rASA = SASA / reference max ASA.

    Reference values: Tien et al. 2013, THEORETICAL ALLOWED scale
    (see constants.REFERENCE_SASA). Values are NOT clipped: because the
    reference is the maximum SASA in a Gly-X-Gly tripeptide (whose context
    itself occludes the side chain), a residue in a less-crowded conformation
    can legitimately exceed 1.0. In the actual ensembles the maximum rASA is
    ~0.94, so no real residue approaches the theoretical upper bound.
    """
    refs = np.asarray([REFERENCE_SASA.get(str(a).upper(), REFERENCE_SASA["X"])
                       for a in one_letter], dtype=np.float64)
    return res_sasa / np.where(refs > 0, refs, 1.0)


def residue_rsa_stats(res_rsa: np.ndarray, motif_spans, tau_resseq: np.ndarray
                      ) -> dict:
    """Mean rASA over each aggregation-prone region (in Tau numbering).

    Returns {motif: mean_rSA} and {motif: nan} if the region is not present.
    """
    out = {}
    for motif, (start, end) in motif_spans.items():
        mask = (tau_resseq >= start) & (tau_resseq <= end)
        out[motif] = float(res_rsa[mask].mean()) if mask.any() else float("nan")
    return out
