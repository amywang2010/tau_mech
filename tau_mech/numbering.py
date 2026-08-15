"""Residue-name normalization, sequence extraction and numbering mapping.

Handles two integrity-critical tasks:

1. Normalize PDB residue names (e.g. HIP -> HIS) to canonical three-letter
   codes and derive the one-letter sequence.
2. Map the residue numbering used in each downloaded ensemble file onto the
   canonical Tau-441 (2N4R, UniProt P10636-8) numbering, and locate the
   aggregation-prone hexapeptide motifs VQIINK (275-280) and VQIVYK (306-311).

All mapping is *data-driven*: motif positions are detected directly in the
actual sequences of the downloaded files, and the configured offsets are
validated against those detections at runtime (a mismatch raises a warning
so assumptions are never silently wrong).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    AA_TO_INDEX,
    AGGREGATION_PRONE_MOTIFS,
    AMINO_ACIDS,
    THREE_TO_ONE,
    UNKNOWN_INDEX,
)


def normalize_resnames(resnames: Sequence[str]) -> np.ndarray:
    """Map a sequence of PDB three-letter residue names to canonical one-letter
    codes (unknown residues -> 'X')."""
    return np.asarray([THREE_TO_ONE.get(str(r).strip().upper(), "X") for r in resnames])


def onehot_aa(one_letter: Sequence[str]) -> np.ndarray:
    """One-hot encode one-letter codes into a (N, 21) array (index 20 = unknown)."""
    n = len(one_letter)
    out = np.zeros((n, len(AMINO_ACIDS) + 1), dtype=np.float32)
    for i, aa in enumerate(one_letter):
        out[i, AA_TO_INDEX.get(str(aa).upper(), UNKNOWN_INDEX)] = 1.0
    return out


def sequence_from_resnames(resnames: Sequence[str]) -> str:
    """One-letter sequence from three-letter residue names."""
    return "".join(normalize_resnames(resnames).tolist())


def find_motif(motif: str, sequence: str) -> List[Tuple[int, int]]:
    """Locate all occurrences of a peptide motif in a one-letter sequence.

    Returns a list of 1-based inclusive (start, end) residue positions.
    """
    hits = []
    pos = 0
    while True:
        i = sequence.find(motif, pos)
        if i == -1:
            break
        hits.append((i + 1, i + len(motif)))
        pos = i + 1
    return hits


def resolve_numbering(
    ensemble_id: str,
    pdb_resseq: Sequence[int],
    one_letter: Sequence[str],
    tau_offset: int,
    expected_motifs: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """Map PDB residue numbers to Tau-441 numbering and locate APR motifs.

    Parameters
    ----------
    ensemble_id : str
        PED identifier (for logging).
    pdb_resseq : sequence of int
        Residue numbers as they appear in the downloaded PDB file.
    one_letter : sequence of str
        One-letter codes in the same order as ``pdb_resseq``.
    tau_offset : int
        tau_number = pdb_number + tau_offset (0 for PED00422, 242 for K18).
    expected_motifs : dict, optional
        {motif: (start, end)} in Tau-441 numbering (1-based, inclusive), for
        runtime validation against the detected positions.

    Returns
    -------
    tau_resseq : (N,) int array of Tau-441 numbering
    motif_spans : {motif: (start, end)} in Tau numbering (1-based inclusive),
        detected directly in the sequence.
    """
    seq = "".join(str(a) for a in one_letter)
    motif_spans: Dict[str, Tuple[int, int]] = {}
    for motif in AGGREGATION_PRONE_MOTIFS:
        hits = find_motif(motif, seq)
        if not hits:
            continue
        # Map detected positions (PDB numbering) into Tau numbering.
        start_pdb, end_pdb = hits[0]
        tau_start = start_pdb + tau_offset
        tau_end = end_pdb + tau_offset
        motif_spans[motif] = (tau_start, tau_end)

    if expected_motifs:
        for motif, (es, ee) in expected_motifs.items():
            got = motif_spans.get(motif)
            if got is None:
                print(
                    f"[WARN] {ensemble_id}: motif {motif} NOT FOUND in sequence "
                    f"(expected Tau {es}-{ee}); APR exposure will be NaN."
                )
            elif got != (es, ee):
                print(
                    f"[WARN] {ensemble_id}: motif {motif} found at Tau {got[0]}-{got[1]} "
                    f"but expected {es}-{ee}. Using detected positions."
                )

    # Vectorized numbering map; handles gaps/non-contiguous numbering by
    # mapping each unique residue number independently.
    pdb_arr = np.asarray(pdb_resseq, dtype=np.int64)
    tau_arr = pdb_arr + tau_offset
    return tau_arr, motif_spans


def residue_index_from_atoms(resseq: Sequence[int], chain: Optional[Sequence[str]] = None
                             ) -> np.ndarray:
    """Assign a consecutive residue index (0, 1, 2, ...) to each atom.

    Residues are identified by unique (chain, resseq) pairs in order of first
    appearance, so numbering gaps do not create spurious residues.
    """
    seen = {}
    out = np.empty(len(resseq), dtype=np.int32)
    idx = 0
    for i, r in enumerate(resseq):
        c = chain[i] if chain is not None else ""
        key = (c, int(r))
        if key not in seen:
            seen[key] = idx
            idx += 1
        out[i] = seen[key]
    return out


def check_sequence_consistency(models, ensemble_id: str, tolerance_lines: int = 5):
    """Verify the residue sequence is identical across all models (spot check).

    Returns the canonical one-letter sequence string.
    """
    from .constants import THREE_TO_ONE

    first_seq = None
    mismatches = 0
    for m in models:
        seq = "".join(THREE_TO_ONE.get(str(r).strip().upper(), "X") for r in m["resname"])
        if first_seq is None:
            first_seq = seq
        elif seq != first_seq:
            mismatches += 1
            if mismatches > tolerance_lines:
                break
    if mismatches:
        print(
            f"[WARN] {ensemble_id}: {mismatches} models have a residue sequence "
            f"different from the first model (up to {tolerance_lines} checked)."
        )
    return first_seq
