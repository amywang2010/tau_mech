"""In-silico mutation validation of APR exposure (P301L, dK280).

Purpose
-------
The repo's documented next experiment: does the pipeline predict higher APR
exposure for aggregation-promoting mutations? This script measures, per
conformer, the relative SASA (Tien et al. 2013 reference, via the Phase-1
`relative_sasa` on the validated Shrake-Rupley SASA) of the two APR
hexapeptides in (a) the WT structure and (b) the mutated structure, holding
the conformation fixed.

Scientific scope (explicit, honest):
  * What it CAN say: given the WT conformational ensembles, does altering
    side-chain volume at the mutation site measurably change the exposure of
    the aggregation-prone hexapeptides in the EXISTING conformations?
  * What it CANNOT say: whether the mutant ADOPTS a different conformational
    distribution (requires mutant MD or mutant-specific ensembles). This
    script complements, does not replace, that follow-up.

Mutations:
  * P301L: proline side chain at Tau 301 (PDB 59 in the K18 constructs)
    replaced by leucine geometry built on the residue's own N/CA/C backbone
    frame; backbone (N, CA, C, O) preserved; all other residues untouched.
  * dK280 (phi): lysine at Tau 280 (PDB 38) deleted from the atom list.
    PDB numbering of other residues is unchanged (numbering is the identity
    for SASA; the Tau mapping is applied afterwards).

Both mutations reuse the SAME validated Shrake-Rupley code path as Phase 1
(brute-force-verified), so WT values here reproduce Phase-1/2 numbers by
construction and each mutant is a single, audited change.

Run:  python scripts/mutation_sasa_validation.py --n-models 20
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tau_mech.constants import (  # noqa: E402
    AGGREGATION_PRONE_MOTIFS,
    THREE_TO_ONE,
)
from tau_mech.io import is_hydrogen, open_ensemble_pdb, parse_models  # noqa: E402
from tau_mech.sasa import (  # noqa: E402
    compute_sasa,
    relative_sasa,
    residue_rsa_stats,
    residue_sasa,
    vdw_radii_for_elements,
)

# Leucine heavy-atom side-chain geometry relative to Cal (angstrom; extended
# rotamer). Built on the residue's N/CA/C backbone frame.
LEU_SIDECHAIN_LOCAL = {
    "CB": (1.53, -0.30, 0.05),
    "CG": (2.02, 1.04, 0.20),
    "CD1": (3.06, 1.13, -0.87),
    "CD2": (2.06, 1.02, 1.70),
}

K18_TAU_OFFSET = 242          # PDB resseq + 242 = Tau numbering (verified in Phase 1)
P301_PDB = 301 - K18_TAU_OFFSET   # 59
K280_PDB = 280 - K18_TAU_OFFSET   # 38


def parse_model_atoms(model):
    """Heavy-atom arrays from one parse_models() model dict."""
    name = np.asarray(model["name"], dtype=object)
    resname = np.asarray(model["resname"], dtype=object)
    resseq = np.asarray(model["resseq"], dtype=int)
    element = np.asarray(model["element"], dtype=object)
    coords = np.asarray(model["coords"], dtype=float)
    keep = ~np.array([is_hydrogen(str(n), str(e) if e else None)
                      for n, e in zip(name, element)])
    return (name[keep], resname[keep], resseq[keep], element[keep],
            coords[keep])


def one_letter_per_residue(resnames, resseqs):
    """One-letter sequence in ascending PDB-residue order."""
    uniq = np.unique(resseqs)
    first = {}
    for r, s in zip(resseqs, resnames):
        if r not in first:
            first[r] = s
    return uniq, [THREE_TO_ONE.get(first[r], "X") for r in uniq]


def sasa_and_apr(names, resnames, resseqs, elems, coords,
                 return_residue_level: bool = False):
    """Per-residue SASA/rASA + APR means under the Phase-1 protocol."""
    uniq, seq1 = one_letter_per_residue(resnames, resseqs)
    res_index = {int(r): i for i, r in enumerate(uniq)}
    atom_res_idx = np.array([res_index[int(r)] for r in resseqs], dtype=np.int64)
    radii = vdw_radii_for_elements(elems)
    atom_sasa = compute_sasa(coords, radii)
    n_res = len(uniq)
    rs = residue_sasa(atom_sasa, atom_res_idx, n_res)
    rsa = relative_sasa(rs, seq1)
    tau = uniq + K18_TAU_OFFSET
    stats = residue_rsa_stats(rsa, AGGREGATION_PRONE_MOTIFS, tau)
    if return_residue_level:
        return stats, n_res, {"resseq_pdb": uniq, "res_sasa": rs,
                               "res_rsa": rsa, "seq1": seq1}
    return stats, n_res


def min_residue_distance(coords, atom_res_idx, resseqs, res_a, res_b):
    """Min heavy-atom distance between two PDB-numbered residues."""
    ia = coords[np.asarray(resseqs) == res_a]
    ib = coords[np.asarray(resseqs) == res_b]
    if len(ia) == 0 or len(ib) == 0:
        return float("nan")
    d = np.linalg.norm(ia[:, None, :] - ib[None, :, :], axis=2)
    return float(d.min())


def mutate_p301l(names, resnames, resseqs, elems, coords):
    """Replace proline P301 (PDB numbering) side chain with leucine."""
    is_target = resseqs == P301_PDB
    if not is_target.any() or not (resnames[is_target] == "PRO").all():
        raise ValueError("P301 (PDB numbering) not found as PRO")
    bb = np.array(["N", "CA", "C", "O"])
    keep = ~(is_target & ~np.isin(names, bb))
    ca_idx = np.where(is_target & (names == "CA"))[0][0]
    n_idx = np.where(is_target & (names == "N"))[0][0]
    c_idx = np.where(is_target & (names == "C"))[0][0]
    ca, n_pos, c_pos = coords[ca_idx], coords[n_idx], coords[c_idx]
    # orthonormal backbone frame at Cal
    v1 = n_pos - ca
    v1 /= np.linalg.norm(v1)
    v2 = c_pos - ca
    v2 = v2 - (v2 @ v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    frame = np.column_stack([v1, v2, v3])  # local -> world
    leu = [(nm, ca + frame @ np.asarray(local))
           for nm, local in LEU_SIDECHAIN_LOCAL.items()]
    keep_idx = np.where(keep)[0]
    new_names = list(names[keep_idx]) + [nm for nm, _ in leu]
    new_res = list(resnames[keep_idx]) + ["LEU"] * len(leu)
    new_seq = list(resseqs[keep_idx]) + [P301_PDB] * len(leu)
    new_elems = list(elems[keep_idx]) + ["C"] * len(leu)
    new_xyz = [coords[a] for a in keep_idx] + [xyz for _, xyz in leu]
    order = np.argsort(np.asarray(new_seq), kind="stable")
    arr = lambda lst, dt: np.asarray(lst, dtype=dt)[order]
    return (arr(new_names, object), arr(new_res, object),
            arr(new_seq, int), arr(new_elems, object),
            np.asarray(new_xyz, dtype=float)[order])


def mutate_dk280(names, resnames, resseqs, elems, coords):
    """Delete lysine K280 (PDB numbering) entirely."""
    is_target = resseqs == K280_PDB
    if not is_target.any() or not (resnames[is_target] == "LYS").all():
        raise ValueError("K280 (PDB numbering) not found as LYS")
    keep = ~is_target
    return (names[keep], resnames[keep], resseqs[keep], elems[keep],
            coords[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/PED00192_ensembles.tar.gz")
    ap.add_argument("--n-models", type=int, default=20)
    ap.add_argument("--out", default="outputs/mutation/mutation_sasa.json")
    args = ap.parse_args()

    # enumerate inner PDB members (outer tar.gz -> inner tar.gz -> PDB)
    with tarfile.open(args.archive, "r:gz") as tf:
        outer = [m.name for m in tf.getmembers() if m.name.endswith(".tar.gz")]
    if len(outer) != 1:
        raise ValueError(f"expected one nested archive, got {outer}")
    with tarfile.open(args.archive, "r:gz") as tf:
        itf = tarfile.open(fileobj=tf.extractfile(outer[0]), mode="r:gz")
        pdbs = [m.name for m in itf.getmembers() if m.name.endswith(".pdb")]
    if len(pdbs) != 1:
        raise ValueError(f"expected one inner PDB, got {pdbs}")

    results = {"archive": args.archive, "n_models": 0,
               "mutations": {}, "per_conformer": [],
               "scope_note": (
                   "Conformationally-fixed (no re-sampling) SASA analysis of "
                   "the WT ensemble under single-site mutation; does NOT "
                   "measure mutant conformational redistribution. Complements "
                   "mutant-MD follow-up.")}
    import io as _io
    with open_ensemble_pdb(args.archive, outer[0], pdbs[0]) as raw:
        text_stream = _io.TextIOWrapper(raw, encoding="ascii")
        models = list(parse_models(text_stream, heavy_only=True,
                                   max_models=args.n_models))
    for model in models:
        name, resname, resseq, element, coords = parse_model_atoms(model)
        wt, n_res, wt_res = sasa_and_apr(name, resname, resseq, element,
                                          coords, return_residue_level=True)
        p301l, _, pl_res = sasa_and_apr(
            *mutate_p301l(name, resname, resseq, element, coords),
            return_residue_level=True)
        dk, _, dk_res = sasa_and_apr(
            *mutate_dk280(name, resname, resseq, element, coords),
            return_residue_level=True)

        # --- audit 1: the P301L substitution MUST change residue 59's own
        # SASA (proline ring -> leucine branch), else the mutation silently
        # no-opped.
        i59 = int(np.where(wt_res["resseq_pdb"] == P301_PDB)[0][0])
        sasa59_wt = float(wt_res["res_sasa"][i59])
        sasa59_pl = float(pl_res["res_sasa"][i59])

        # --- audit 2: WHY P301L can leave the APRs bit-identical: measure
        # the actual contact distance from residue 59 to the APR spans.
        contact_59_apr1 = min(min_residue_distance(
            coords, None, resseq, P301_PDB, r) for r in range(275 - K18_TAU_OFFSET,
                                                              281 - K18_TAU_OFFSET))
        contact_59_apr2 = min(min_residue_distance(
            coords, None, resseq, P301_PDB, r) for r in range(306 - K18_TAU_OFFSET,
                                                              312 - K18_TAU_OFFSET))

        # --- audit 3: dK280 composition confound. The span mean over
        # 275-280 changes partly because K280 LEAVES the average. Report the
        # MATCHED-residue comparison (PDB 33-37 = Tau 275-279) separately;
        # each mask is built from its OWN structure's residue array (the
        # deletion removes one residue, so the arrays differ in length).
        def matched_mean(res, lo, hi):
            m = (res["resseq_pdb"] >= lo) & (res["resseq_pdb"] <= hi)
            return float(np.mean(res["res_rsa"][m]))
        rsa_matched_wt = matched_mean(wt_res, 275 - K18_TAU_OFFSET,
                                       279 - K18_TAU_OFFSET)
        rsa_matched_dk = matched_mean(dk_res, 275 - K18_TAU_OFFSET,
                                       279 - K18_TAU_OFFSET)

        results["per_conformer"].append({
            "conformer": len(results["per_conformer"]),
            "n_res": n_res, "wt": wt, "P301L": p301l, "dK280": dk,
            "audit": {
                "res59_pdb_sasa_wt": sasa59_wt,
                "res59_pdb_sasa_P301L": sasa59_pl,
                "res59_min_dist_to_VQIINK": contact_59_apr1,
                "res59_min_dist_to_VQIVYK": contact_59_apr2,
                "VQIINK_matched_275_279_rsa_wt": rsa_matched_wt,
                "VQIINK_matched_275_279_rsa_dK280": rsa_matched_dk,
                "VQIINK_matched_delta": rsa_matched_dk - rsa_matched_wt,
            }})
    results["n_models"] = len(results["per_conformer"])
    if not results["per_conformer"]:
        raise SystemExit("no conformers parsed")

    def summ(key):
        rows = [pc[key] for pc in results["per_conformer"]]
        return {m: {"mean": float(np.mean([r[m] for r in rows])),
                    "sd": float(np.std([r[m] for r in rows]))}
                for m in rows[0]}
    results["mutations"] = {k: summ(k) for k in ("wt", "P301L", "dK280")}
    results["mutations"]["delta_P301L_minus_wt"] = {
        m: results["mutations"]["P301L"][m]["mean"]
           - results["mutations"]["wt"][m]["mean"]
        for m in results["mutations"]["wt"]}
    results["mutations"]["delta_dK280_minus_wt"] = {
        m: results["mutations"]["dK280"][m]["mean"]
           - results["mutations"]["wt"][m]["mean"]
        for m in results["mutations"]["wt"]}
    # aggregate the audit channels
    aud = [pc["audit"] for pc in results["per_conformer"]]
    results["audit_summary"] = {
        "res59_sasa_changed_P301L": all(
            abs(a["res59_pdb_sasa_P301L"] - a["res59_pdb_sasa_wt"]) > 1.0
            for a in aud),
        "res59_min_dist_to_VQIINK_median": float(np.median(
            [a["res59_min_dist_to_VQIINK"] for a in aud])),
        "res59_min_dist_to_VQIVYK_median": float(np.median(
            [a["res59_min_dist_to_VQIVYK"] for a in aud])),
        "VQIINK_matched_275_279_delta_mean": float(np.mean(
            [a["VQIINK_matched_delta"] for a in aud])),
        "note": ("matched delta isolates the exposure change of the "
                 "SURVIVING VQIINK residues (275-279) from the composition "
                 "effect of deleting K280 from the 275-280 average"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["mutations"], indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
