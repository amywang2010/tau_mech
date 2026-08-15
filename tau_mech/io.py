"""Fast, streaming PDB parser for multi-model conformational ensembles.

The three PED ensembles used in this project are distributed as nested gzip
tarballs (outer tar.gz -> inner tar.gz -> PDB file). Rather than extracting
hundreds of MB to disk, the pipeline streams the PDB directly from the
compressed archive. This module provides:

  * ``open_ensemble_pdb(...)``  - context manager yielding a text stream of the
    PDB file inside the nested archives (no disk extraction).
  * ``parse_models(...)``       - generator yielding one dict of atom arrays
    per MODEL/ENDMDL block (also handles files without MODEL records, and a
    missing trailing ENDMDL).
  * ``count_models(...)``       - fast model counter.

Fixed-width PDB column layout used (PDB 3.30 convention, 1-based columns):
  7-11 serial, 13-16 atom name, 17 altLoc, 18-20 resName, 22 chainID,
  23-26 resSeq, 27 iCode, 31-38 x, 39-46 y, 47-54 z, 55-60 occupancy,
  61-66 tempFactor, 77-78 element.
"""

from __future__ import annotations

import io
import tarfile
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import numpy as np

# Fixed-width slice indices (0-based Python slices into each line).
_SLICE = dict(
    name=slice(12, 16),
    resname=slice(17, 20),
    chain=21,
    resseq=slice(22, 26),
    x=slice(30, 38),
    y=slice(38, 46),
    z=slice(46, 54),
    occ=slice(54, 60),
    bfac=slice(60, 66),
    element=slice(76, 78),
)


def element_from_atom_name(name: str) -> str:
    """Infer the chemical element from a PDB atom name.

    PDB files without an explicit element column (e.g. PED00192) require this
    heuristic: the element is the leading uppercase letter of the atom name
    (in the protein context "CA" is the alpha carbon, not calcium).
    """
    if not name:
        return "?"
    first = name[0]
    if first in "CNOSH":
        return first
    return "?"


def is_hydrogen(name: str, element: Optional[str] = None) -> bool:
    """True if the atom is a hydrogen (by name or by explicit element).

    All three PED files name hydrogens with a leading 'H' (H, HA, HB2, HT1,
    HZ1, ...), so the name test is sufficient; the element column is used
    when present.
    """
    if element == "H":
        return True
    return name.startswith("H")


@contextmanager
def open_ensemble_pdb(archive_path: str, member: str, inner: str):
    """Context manager yielding a binary file-like object for the PDB file
    nested inside ``archive_path`` -> ``member`` (a tar.gz) -> ``inner``.

    Streams from the compressed archives directly; nothing is written to disk.
    """
    outer = tarfile.open(archive_path, "r:gz")
    try:
        member_file = outer.extractfile(member)
        if member_file is None:
            raise FileNotFoundError(f"member {member!r} not found in {archive_path}")
        inner_tar = tarfile.open(fileobj=member_file, mode="r:gz")
        try:
            pdb_file = inner_tar.extractfile(inner)
            if pdb_file is None:
                raise FileNotFoundError(f"{inner!r} not found inside {member}")
            yield pdb_file
        finally:
            inner_tar.close()
    finally:
        outer.close()


def count_models(path_or_stream) -> int:
    """Count MODEL records in an already-open text stream or file path."""
    close = False
    if isinstance(path_or_stream, (str, bytes)):
        path_or_stream = io.open(path_or_stream, "r", encoding="ascii")
        close = True
    try:
        n = 0
        for line in path_or_stream:
            if line.startswith("MODEL"):
                n += 1
        return n
    finally:
        if close:
            path_or_stream.close()


def parse_models(
    path_or_stream,
    heavy_only: bool = True,
    with_occupancy: bool = False,
    max_models: Optional[int] = None,
    start_at: int = 0,
) -> Iterator[Dict]:
    """Yield one dict per MODEL block of a multi-model PDB file.

    Parameters
    ----------
    path_or_stream : str or file-like
        Path to a plain PDB file, or a text-mode stream (e.g. produced by
        wrapping :func:`open_ensemble_pdb` output with ``io.TextIOWrapper``).
    heavy_only : bool
        Drop hydrogen atoms (named H*, element H). Heavy-atom analyses are
        the standard for SASA / graph construction.
    max_models : int, optional
        Stop after this many models (0-based offset applied first).
    start_at : int
        Skip this many models at the start (mostly for debugging).

    Handles files without MODEL/ENDMDL records (a single implicit model) and
    files missing a trailing ENDMDL. Each yielded dict contains numpy arrays:
        name, resname, chain, element : str arrays
        resseq : int32 array
        coords : float32 array of shape (N, 3)
        occupancy, bfactor : float32 arrays (if with_occupancy)
    """
    close = False
    if isinstance(path_or_stream, (str, bytes)):
        path_or_stream = io.open(path_or_stream, "r", encoding="ascii")
        close = True
    try:
        names, resnames, chains, resseqs, els = [], [], [], [], []
        xs, ys, zs = [], [], []
        occs, bfacs = [], []
        model_idx = -1
        yielded = 0

        def emit():
            nonlocal yielded, names, resnames, chains, resseqs, els, xs, ys, zs, occs, bfacs
            model = _finalize(
                names, resnames, chains, resseqs, els, xs, ys, zs, occs, bfacs,
                with_occupancy,
            )
            names, resnames, chains, resseqs, els = [], [], [], [], []
            xs, ys, zs = [], [], []
            occs, bfacs = [], []
            yielded += 1
            return model

        for line in path_or_stream:
            if line.startswith("MODEL"):
                if model_idx >= start_at and model_idx >= 0 and names and (
                    max_models is None or yielded < max_models
                ):
                    yield emit()
                    if max_models is not None and yielded >= max_models:
                        return
                model_idx += 1
                names, resnames, chains, resseqs, els = [], [], [], [], []
                xs, ys, zs = [], [], []
                occs, bfacs = [], []
            elif line.startswith("ATOM"):
                if model_idx == -1:
                    model_idx = 0  # implicit single model (no MODEL records)
                if model_idx < start_at:
                    continue
                name = line[_SLICE["name"]].strip()
                element = line[_SLICE["element"]].strip() or element_from_atom_name(name)
                if heavy_only and is_hydrogen(name, element):
                    continue
                names.append(name)
                resnames.append(line[_SLICE["resname"]].strip())
                chains.append(line[_SLICE["chain"]])
                resseqs.append(int(line[_SLICE["resseq"]]))
                els.append(element)
                xs.append(float(line[_SLICE["x"]]))
                ys.append(float(line[_SLICE["y"]]))
                zs.append(float(line[_SLICE["z"]]))
                if with_occupancy:
                    occs.append(float(line[_SLICE["occ"]]))
                    bfacs.append(float(line[_SLICE["bfac"]]))
            elif line.startswith("ENDMDL"):
                if model_idx >= start_at and names:
                    yield emit()
                    if max_models is not None and yielded >= max_models:
                        return

        # trailing: an implicit single model or a model without a final ENDMDL
        if model_idx >= start_at and names and (max_models is None or yielded < max_models):
            yield emit()
    finally:
        if close:
            path_or_stream.close()


def _finalize(names, resnames, chains, resseqs, els, xs, ys, zs, occs, bfacs,
              with_occupancy) -> Dict:
    model = {
        "name": np.asarray(names, dtype=object),
        "resname": np.asarray(resnames, dtype=object),
        "chain": np.asarray(chains, dtype=object),
        "resseq": np.asarray(resseqs, dtype=np.int32),
        "element": np.asarray(els, dtype=object),
        "coords": np.stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)], axis=1).astype(
            np.float32
        ),
    }
    if with_occupancy:
        model["occupancy"] = np.asarray(occs, dtype=np.float32)
        model["bfactor"] = np.asarray(bfacs, dtype=np.float32)
    return model


def extract_remark_conformer_ids(stream_or_path) -> Dict[int, str]:
    """Parse ``REMARK MODEL <n> FROM conformer_<id>.pdb`` lines (PED00422).

    Returns a dict {model_number: conformer_id}. Returns {} if absent.
    The REMARK block precedes the first MODEL record in PED files, so the
    scan stops at the first MODEL line (avoids decompressing the whole
    ensemble just for metadata).
    """
    import re

    close = False
    if isinstance(stream_or_path, (str, bytes)):
        stream_or_path = io.open(stream_or_path, "r", encoding="ascii")
        close = True
    out: Dict[int, str] = {}
    try:
        for line in stream_or_path:
            if line.startswith("MODEL"):
                break
            if line.startswith("REMARK") and "FROM conformer_" in line:
                parts = line.split()
                # e.g. ['REMARK', 'MODEL', '1', 'FROM', 'conformer_10005_mcsce.pdb']
                if len(parts) >= 5:
                    try:
                        num = int(parts[2])
                        m = re.search(r"\d+", parts[4])
                        if m:
                            out[num] = m.group()
                    except (ValueError, IndexError):
                        continue
        return out
    finally:
        if close:
            stream_or_path.close()
