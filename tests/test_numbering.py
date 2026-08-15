"""Tests for residue normalization and numbering (tau_mech.numbering)."""

import numpy as np
import pytest

from tau_mech.constants import AGGREGATION_PRONE_MOTIFS, THREE_TO_ONE
from tau_mech.numbering import (
    check_sequence_consistency,
    find_motif,
    normalize_resnames,
    onehot_aa,
    residue_index_from_atoms,
    resolve_numbering,
    sequence_from_resnames,
)

# 2N4R Tau-441 reference sequence (UniProt P10636-8), first ~320 residues
# suffice to locate the APR motifs. Full sequence not needed for these tests.
TAU441_SEQ = (
    "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTA"
    "EDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGA"
    "DGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTP"
    "PTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKC"
    "GSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVP"
    "GGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLA"
    "KQGL"
)


def test_motif_positions_in_tau441():
    """The APR motifs must sit at the canonical Tau positions."""
    assert find_motif("VQIINK", TAU441_SEQ) == [(275, 280)]
    assert find_motif("VQIVYK", TAU441_SEQ) == [(306, 311)]
    assert AGGREGATION_PRONE_MOTIFS == {"VQIINK": (275, 280), "VQIVYK": (306, 311)}


def test_normalize_resnames_hip():
    rn = np.asarray(["HIP", "ALA", "MSE", "GLY", "UNK"], dtype=object)
    one = normalize_resnames(rn)
    assert one.tolist() == ["H", "A", "M", "G", "X"]


def test_onehot_aa_dims():
    oh = onehot_aa(["A", "X", "G"])
    assert oh.shape == (3, 21)
    assert oh[0, 0] == 1.0       # A -> index 0
    assert oh[1, 20] == 1.0      # X -> unknown index 20
    assert oh[2].sum() == 1.0


def test_sequence_from_resnames():
    seq = sequence_from_resnames(["MET", "ALA", "GLY"])
    assert seq == "MAG"


def test_resolve_numbering_k18():
    """K18 file numbering 1..130 with offset 242 must map VQIINK to 275-280
    and VQIVYK to 306-311 in Tau numbering."""
    # K18 = initiator Met + Tau 244..372, i.e. "M" + TAU441_SEQ[244-1:372]
    # (130 residues). Residue 1 (M) is the expression tag; file residue k>=2
    # maps to Tau residue k+242 (Q244 at file position 2).
    k18_seq = "M" + TAU441_SEQ[244 - 1:372]  # M + Tau 244..372
    assert len(k18_seq) == 130
    assert k18_seq[0] == "M" and k18_seq[1] == "Q"  # initiator M, Tau 244 = Q
    assert k18_seq.find("VQIINK") == 32  # 0-based -> file residue 33 (Tau 275)
    assert k18_seq.find("VQIVYK") == 63  # 0-based -> file residue 64 (Tau 306)

    pdb_resseq = np.arange(1, len(k18_seq) + 1, dtype=np.int64)
    one = np.asarray(list(k18_seq))
    tau, spans = resolve_numbering(
        "PED00192", pdb_resseq, one, tau_offset=242,
        expected_motifs=dict(AGGREGATION_PRONE_MOTIFS),
    )
    assert spans == {"VQIINK": (275, 280), "VQIVYK": (306, 311)}
    # residue 1 (initiator M) maps to Tau 243
    assert tau[0] == 243
    assert tau[1] == 244


def test_resolve_numbering_tau441():
    pdb_resseq = np.arange(1, 442, dtype=np.int64)
    one = np.asarray(list(TAU441_SEQ))
    tau, spans = resolve_numbering("PED00422", pdb_resseq, one, tau_offset=0,
                                   expected_motifs=dict(AGGREGATION_PRONE_MOTIFS))
    assert spans == AGGREGATION_PRONE_MOTIFS
    assert tau[274] == 275
    assert tau[-1] == 441


def test_residue_index_from_atoms():
    resseq = np.asarray([1, 1, 2, 2, 5, 5, 5])
    idx = residue_index_from_atoms(resseq)
    assert idx.tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_check_sequence_consistency_ok():
    models = [
        {"resname": np.asarray(["MET", "GLY"], dtype=object)},
        {"resname": np.asarray(["MET", "GLY"], dtype=object)},
    ]
    assert check_sequence_consistency(models, "TEST") == "MG"
