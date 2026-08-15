"""Constants and reference tables for the Tau mechanobiology pipeline.

Every numeric table used by the pipeline is defined here so that all methods
are transparent and reproducible. Sources for each table are cited inline.
"""

# ---------------------------------------------------------------------------
# Amino acids
# ---------------------------------------------------------------------------
# The 20 standard amino acids (one-letter codes, alphabetical order so the
# one-hot encoding index is stable and unambiguous).
AMINO_ACIDS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_INDEX = len(AMINO_ACIDS)  # index 20: any non-standard residue
N_AA = len(AMINO_ACIDS) + 1       # 21 including the unknown bucket

# Three-letter -> one-letter mapping, including common non-standard /
# protonation-state variants which are mapped onto their canonical residue.
# (PDB residue nomenclature; HIP/HID/HIE are protonation states of histidine,
# CYX/CYM of cysteine, GLH/ASH protonated Glu/Asp, LYN/ARN deprotonated
# Lys/Arg, MSE = selenomethionine treated as Met, SEP/TPO/PTR phosphorylated
# Ser/Thr/Tyr treated as the unmodified residue.)
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HIP": "H", "HID": "H", "HIE": "H", "CYX": "C", "CYM": "C",
    "GLH": "E", "ASH": "D", "LYN": "K", "ARN": "R",
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

# Kyte-Doolittle hydropathy index (Kyte J. & Doolittle R.F., 1982,
# "A simple method for displaying the hydropathic character of a protein",
# J. Mol. Biol. 157:105-132). X = unknown residue -> 0.0 (neutral).
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    "X": 0.0,
}

# Average atomic masses (g/mol) of the heavy elements found in proteins
# (IUPAC standard atomic weights, rounded). Used for mass-weighted radius
# of gyration. Hydrogens are excluded from heavy-atom analyses.
ATOMIC_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}

# Backbone atom names (excluded from the side-chain centroid).
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT", "OT1", "OT2"}

# ---------------------------------------------------------------------------
# Solvent-accessible surface area (SASA)
# ---------------------------------------------------------------------------
# Probe radius (Angstrom) for solvent-accessible surface area, the radius of
# a water molecule (Shrake A. & Rupley J.A., 1973, J. Mol. Biol. 79:351-371).
PROBE_RADIUS = 1.4

# Van der Waals radii (Angstrom) per element following the NACCESS/Chothia
# convention (Chothia C., 1976, "The nature of the accessible and buried
# surfaces in proteins", J. Mol. Biol. 105:1-14). These are the radii adopted
# by the NACCESS program and are the de-facto standard for SASA calculations.
VDW_RADII = {"C": 1.80, "N": 1.65, "O": 1.40, "S": 1.85}

# Default number of probe points per atom sphere for the Shrake-Rupley
# numerical integration. 480 points is a documented, well-tested compromise
# between accuracy and speed (FreeSASA uses a comparable point count by
# default). The exact value is recorded in the pipeline config so results are
# reproducible.
N_PROBE_POINTS = 480

# Reference (maximal) per-residue solvent-accessible surface area in Angstrom^2,
# used to compute RELATIVE solvent accessibility (rASA = SASA / reference).
#
# Values: THEORETICAL scale, ALLOWED Ramachandran region, from Table 1 of
# Tien M.Z., Meyer A.G., Sydykova D.K., Spielman S.J., Wilke C.O., 2013,
# "Maximum allowed solvent accessibilities of residues in proteins",
# PLOS ONE 8(11):e80635; doi:10.1371/journal.pone.0080635. The scale was
# derived from Gly-X-Gly tripeptides exhaustively sampled over biophysically
# allowed backbone/rotamer conformations, for the Lee-Richards/DSSP solvent
# definition (probe radius 1.4 A) -- the same geometric definition as the
# Shrake-Rupley implementation used here.
#
# VERIFIED 2026-08-02 directly against the published Table 1 (PLOS manuscript
# XML). These are the values the authors recommend for computing relative
# solvent accessibility.
REFERENCE_SASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
    "X": 202.0,  # fallback for unknown residues (mean of the theoretical 20)
}

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
# Default distance cutoff (Angstrom) for a spatial residue-residue edge:
# an edge exists between two residues if any pair of their heavy atoms is
# within this distance. 5.0 A captures direct van der Waals / hydrogen-bond
# contacts between side chains while remaining sparse for IDPs.
DEFAULT_EDGE_CUTOFF = 5.0
# Sequential edges: residues i and i+1, i+2, ... always connected up to
# SEQ_ADJACENCY, reflecting chain connectivity (polypeptide backbone).
SEQ_ADJACENCY = 2

# ---------------------------------------------------------------------------
# Ensembles (data sources)
# ---------------------------------------------------------------------------
# Description of the three PED ensembles used in the study, including how the
# residue numbering in the downloaded files maps onto the canonical Tau-441
# (2N4R isoform, UniProt P10636-8) numbering. For K18 constructs the PDB file
# numbering runs 1..130 = M243..E372, i.e. Tau numbering = PDB numbering + 242.
# For PED00422 the PDB numbering already is the Tau-441 numbering (1..441).
ENSEMBLES = {
    "PED00422": {
        "archive": "PED00422_ensembles.tar.gz",
        "member": "PED00422e002.tar.gz",     # nested tar.gz inside the archive
        "inner": "pdbfile.pdb",              # PDB file inside the nested archive
        "n_expected_models": 1000,
        "n_expected_residues": 441,
        "tau_offset": 0,                     # PDB numbering == Tau-441 numbering
        "construct": "Tau-441 (2N4R, full-length)",
        "method": "IDPConformerGenerator (Teixeira et al., 2022, J. Phys. Chem. A)",
        "notes": "Computational conformer pool; no experimental restraints used.",
    },
    "PED00192": {
        "archive": "PED00192_ensembles.tar.gz",
        "member": "PED00192e001.tar.gz",     # nested tar.gz inside the archive
        "inner": "PED00192e002.pdb",         # PDB file inside the nested archive
        "n_expected_models": 75,
        "n_expected_residues": 130,
        "tau_offset": 242,                   # PDB 1 = Tau M243
        "construct": "K18 (M243-E372)",
        "method": "Bayesian reweighted MD (Fisher, Huang & Stultz, 2010, PLoS Comput. Biol.)",
        "notes": ("Experimentally constrained ensemble (NMR RDC, chemical shifts, "
                  "SAXS). Bayesian weights are not distributed with the PDB file."),
    },
    "PED00443": {
        "archive": "PED00443_ensembles.tar.gz",
        "member": "PED00443e001.tar.gz",     # nested tar.gz inside the archive
        "inner": "pdbfile.pdb",              # PDB file inside the nested archive
        "n_expected_models": 1000,
        "n_expected_residues": 130,
        "tau_offset": 242,                   # PDB 1 = Tau M243
        "construct": "K18 (M243-E372)",
        "method": "idpGAN (Janson, Valdes-Garcia, Heo & Feig, 2023, Nat. Commun. 14:774)",
        "notes": ("Generative ML ensemble derived from PED00192; no experimental "
                  "data folded in during generation."),
    },
}

# Aggregation-prone regions (APRs) of Tau-441 in Tau numbering (1-based,
# inclusive). VQIINK (PHF6*) = 275-280 and VQIVYK (PHF6) = 306-311 (standard
# Tau literature positions; verified against the actual sequences in the PED
# files at runtime).
AGGREGATION_PRONE_MOTIFS = {
    "VQIINK": (275, 280),
    "VQIVYK": (306, 311),
}
