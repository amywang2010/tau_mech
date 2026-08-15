"""Tests for the streaming PDB parser (tau_mech.io)."""

import gzip
import io
import os
import tarfile

import numpy as np
import pytest

from tau_mech.io import (
    count_models,
    element_from_atom_name,
    extract_remark_conformer_ids,
    is_hydrogen,
    open_ensemble_pdb,
    parse_models,
)

# A tiny synthetic multi-model PDB with known values (2 models x 2 residues).
SYNTHETIC_PDB = """\
MODEL        1
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MET A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      3  HA  MET A   1       2.500   2.500   3.000  1.00  0.00           H
ATOM      4  C   GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      5  O   GLY A   2       5.000   5.000   6.000  1.00  0.00           O
ENDMDL
MODEL        2
ATOM      6  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      7  CA  MET A   1       2.000   2.000   3.000  1.00  0.00           C
ATOM      8  HA  MET A   1       2.500   2.500   3.000  1.00  0.00           H
ATOM      9  C   GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM     10  O   GLY A   2       5.000   5.000   6.000  1.00  0.00           O
ENDMDL
END
"""


@pytest.fixture
def pdb_file(tmp_path):
    path = tmp_path / "test.pdb"
    path.write_text(SYNTHETIC_PDB)
    return str(path)


def test_count_models(pdb_file):
    assert count_models(pdb_file) == 2


def test_parse_models_all_atoms(pdb_file):
    models = list(parse_models(pdb_file, heavy_only=False))
    assert len(models) == 2
    m = models[0]
    assert m["coords"].shape == (5, 3)
    assert m["resseq"].tolist() == [1, 1, 1, 2, 2]
    assert m["resname"].tolist() == ["MET", "MET", "MET", "GLY", "GLY"]
    # coordinates of first atom
    np.testing.assert_allclose(m["coords"][0], [1.0, 2.0, 3.0], atol=1e-4)


def test_parse_models_heavy_only(pdb_file):
    models = list(parse_models(pdb_file, heavy_only=True))
    m = models[0]
    assert m["coords"].shape == (4, 3)  # HA dropped
    assert "HA" not in m["name"]


def test_parse_models_max_and_start(pdb_file):
    models = list(parse_models(pdb_file, heavy_only=False, max_models=1))
    assert len(models) == 1
    models = list(parse_models(pdb_file, heavy_only=False, start_at=1))
    assert len(models) == 1
    assert models[0]["resseq"].tolist() == [1, 1, 1, 2, 2]


def test_element_heuristic():
    assert element_from_atom_name("CA") == "C"
    assert element_from_atom_name("N") == "N"
    assert element_from_atom_name("OXT") == "O"
    assert element_from_atom_name("SD") == "S"
    assert element_from_atom_name("HT1") == "H"
    assert is_hydrogen("HA") is True
    assert is_hydrogen("CA") is False


def _make_nested_archive(tmp_path):
    """Build outer.tar.gz -> inner.tar.gz -> pdbfile.pdb in tmp_path."""
    pdb_bytes = SYNTHETIC_PDB.encode("ascii")
    inner_tar_path = os.path.join(str(tmp_path), "inner.tar.gz")
    with tarfile.open(inner_tar_path, "w:gz") as tf:
        data = io.BytesIO(pdb_bytes)
        info = tarfile.TarInfo("pdbfile.pdb")
        info.size = len(pdb_bytes)
        tf.addfile(info, data)
    outer_path = os.path.join(str(tmp_path), "outer.tar.gz")
    with tarfile.open(outer_path, "w:gz") as tf:
        with open(inner_tar_path, "rb") as f:
            inner_bytes = f.read()
        info = tarfile.TarInfo("ens.tar.gz")
        info.size = len(inner_bytes)
        tf.addfile(info, io.BytesIO(inner_bytes))
    return outer_path


def test_open_ensemble_pdb_streams(tmp_path):
    outer = _make_nested_archive(tmp_path)
    with open_ensemble_pdb(outer, "ens.tar.gz", "pdbfile.pdb") as raw:
        text = io.TextIOWrapper(raw, encoding="ascii")
        models = list(parse_models(text, heavy_only=False))
    assert len(models) == 2
    assert models[0]["coords"].shape == (5, 3)


def test_extract_remark_conformer_ids():
    text = (
        "REMARK     MODEL 1 FROM conformer_10005_mcsce.pdb\n"
        "REMARK     MODEL 2 FROM conformer_10010_mcsce.pdb\n"
        "MODEL        1\n"
    )
    ids = extract_remark_conformer_ids(io.StringIO(text))
    assert ids == {1: "10005", 2: "10010"}


def test_extract_remark_conformer_ids_empty():
    assert extract_remark_conformer_ids(io.StringIO("MODEL 1\n")) == {}
