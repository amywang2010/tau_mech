"""Data provenance and integrity.

Preserves complete data provenance so every claim in the manuscript is
traceable to a specific, verifiable source file:

  * SHA-256 digest of each raw downloaded archive (integrity check)
  * file size and modification date of the archives
  * PED identifiers, construct/method metadata and citations

Run :func:`write_provenance` once after downloading; the resulting
``outputs/provenance.json`` becomes part of the repository record.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

from .constants import ENSEMBLES

# Citation strings for the methods / data used in this study.
CITATIONS = {
    "PED": "Lazar T. et al., 'PED in 2021: a major update of the protein "
           "ensemble database for intrinsically disordered proteins', "
           "Nucleic Acids Research 2021; doi:10.1093/nar/gkaa1051",
    "PED00422_method": "Teixeira J.M.C. et al., 'IDPConformerGenerator: a "
                       "flexible software suite for sampling the conformational "
                       "space of disordered protein states', J. Phys. Chem. A "
                       "2022, 126(35):5985-6003; doi:10.1021/acs.jpca.2c03726",
    "PED00192_method": "Fisher C.K., Huang A., Stultz C.M., 'Modeling "
                       "intrinsically disordered proteins with Bayesian "
                       "statistics', PLoS Comput. Biol. 2010, 6(2):e1000692; "
                       "doi:10.1371/journal.pcbi.1000692",
    "PED00443_method": "Janson G., Valdes-Garcia G., Heo L., Feig M., "
                       "'Direct generation of protein conformational ensembles "
                       "via machine learning' (idpGAN), Nat. Commun. 2023, "
                       "14:774; doi:10.1038/s41467-023-36443-x",
    "sasa": "Shrake A., Rupley J.A., 'Environment and exposure to solvent of "
            "protein atoms', J. Mol. Biol. 1973, 79:351-371; "
            "doi:10.1016/0022-2836(73)90011-9",
    "vdw": "Chothia C., 'The nature of the accessible and buried surfaces in "
           "proteins', J. Mol. Biol. 1976, 105:1-14; "
           "doi:10.1016/0022-2836(76)90191-1",
    "rsa_reference": "Tien M.Z. et al., 'Maximum allowed solvent accessibilities "
                     "of residues in proteins', PLoS ONE 2013, 8(11):e80635; "
                     "doi:10.1371/journal.pone.0080635. Used: Table 1 THEORETICAL "
                     "scale, ALLOWED Ramachandran region (Gly-X-Gly tripeptides, "
                     "Lee-Richards/DSSP definition, probe 1.4 A); values verified "
                     "against the published table 2026-08-02.",
    "hydrophobicity": "Kyte J., Doolittle R.F., 'A simple method for displaying "
                      "the hydropathic character of a protein', J. Mol. Biol. "
                      "1982, 157:105-132",
}

# PED download URLs (proteinensemble.org entry pages; the archive download
# links follow the PED_<id> pattern).
PED_URLS = {
    "PED00422": "https://proteinensemble.org/entries/PED00422/",
    "PED00192": "https://proteinensemble.org/entries/PED00192/",
    "PED00443": "https://proteinensemble.org/entries/PED00443/",
}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (streamed, memory-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect_archive_metadata(data_dir: str) -> Dict[str, dict]:
    """SHA-256, size and mtime of each raw archive."""
    out = {}
    for eid, ec in ENSEMBLES.items():
        path = os.path.join(data_dir, ec["archive"])
        if not os.path.exists(path):
            out[eid] = {"error": f"archive not found: {path}"}
            continue
        st = os.stat(path)
        out[eid] = {
            "archive": ec["archive"],
            "member": ec["member"],
            "inner": ec["inner"],
            "size_bytes": st.st_size,
            "mtime": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(),
            "sha256": sha256_file(path),
        }
    return out


def write_provenance(data_dir: str, output_dir: str) -> str:
    """Write outputs/provenance.json; returns the JSON path."""
    os.makedirs(output_dir, exist_ok=True)
    doc = {
        "description": "Data provenance for the Tau mechanobiology study "
                       "(IEEE submission). Raw archives are the canonical "
                       "sources; processed outputs are derived artifacts.",
        "ensembles": collect_archive_metadata(data_dir),
        "ped_urls": PED_URLS,
        "citations": CITATIONS,
        "note": ("PED00192 Bayesian statistical weights are not included in "
                 "the downloaded bundle; ensemble averages using weights "
                 "require a separate download from the PED entry."),
    }
    path = os.path.join(output_dir, "provenance.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path
