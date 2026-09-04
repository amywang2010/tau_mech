# tau_mech — Multiscale computational framework for Tau mechanobiology

Complete, end-to-end computational study for the manuscript "Do physiologically
relevant mechanical shear forces influence Tau conformational susceptibility
and the accessibility of aggregation-prone regions?"

The framework connects tissue-scale fluid mechanics (smoothed-particle
hydrodynamics, SPH) to protein-scale structural dynamics (residue-level
structural graphs of intrinsically disordered Tau, analyzed with geometric
graph neural networks), to ask whether physiological CSF/interstitial shear
alters the exposure of the aggregation-prone hexapeptide motifs VQIINK and
VQIVYK.

**Headline results** (all machine-extracted from canonical records; full
numbers in `outputs/final_report.md`):

1. **Physiological shear is a quantified null at the condensate scale.**
   The validated SPH solver's droplet deformation response saturates by
   Ca ≈ 0.01 (Taylor saturation); composing the measured mechanical response
   with ensemble conformational sensitivity gives an upper-bound shift in
   APR-1 exposure ≈ 0.45% of native conformational heterogeneity at 1 Pa —
   i.e. mechanical effects at physiological shear are ≥176× smaller than
   thermal conformational fluctuations.
2. **The structural signal in the GNN is demonstrably not sequence leakage**:
   the full GCN (PR-AUC 0.903 ± 0.001, n=5 seeds) and edge-feature GAT
   (0.992) exceed the sequence-only baselines (MLP 0.793; GCN without
   spatial edges 0.791), with seed-robust model ordering
   (GraphSAGE 1.000 ± 0.000 ≥ GAT 0.939 ± 0.012 > GCN 0.902 ± 0.001).
3. **Mutation validation**: the matched-residue comparison deconfounds motif
   composition — dK280 raises surviving VQIINK-residue exposure (+0.033
   rASA) while P301L shows no static-packing effect (its redistribution
   pathway is flagged for MD follow-up).
4. **Solver honesty chain**: zero-shear gate passes all 6 pre-registered
criteria; the wall-coupling layer is shown (pre-registered fixed-point test)
to carry a slow formulation-level mode — reported as a characterized solver
property, not tuned away; the sweep's Ca values use measured local shear
rates, so no result depends on it.

Every numeric claim above traces to a committed JSON record under `outputs/`,
with provenance tags **[machine]/[ref]/[rule]** in `outputs/final_report.md`.

**Quick start** (clone + reproduce):

```bash
git lfs install
git clone https://github.com/amywang2010/tau_mech.git
cd tau_mech
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/          # 62 tests, no data needed
python scripts/build_final_report.py   # rebuild the strict final report from records
python tau_mech/pipeline.py --ensemble PED00422 --data-dir data  # full reprocessing
```

`tau_mech/pipeline.py --ensemble <id>` reprocesses an ensemble end-to-end,
where `<id>` is a PED identifier (see `tau_mech/pipeline.py`); results are
summarized with all provenance in `outputs/final_report.md` (strict mode).
The repo contains the raw PED data archives (Git LFS), all code, all canonical
result records, and a 62-test suite.

---

## 1. Data (verified on disk)

| PED entry | Construct | Conformers | Residues | All-atom incl. H | Generation method |
|---|---|---|---|---|---|
| PED00422e002 | Tau-441 (2N4R, full-length) | 1,000 | 441 | yes (6,428 atoms/model) | IDPConformerGenerator |
| PED00192e002 | K18 (initiator M + Q244–E372) | 75 | 130 | yes (1,209 atoms/model) | Bayesian-reweighted MD (NMR/SAXS) |
| PED00443e001 | K18 | 1,000 | 130 | yes (1,984 atoms/model) | idpGAN (generative ML) |

- Raw archives are **untouched** and integrity-checked (SHA-256 recorded in
  `outputs/provenance.json`; see section 5).
- The raw files are nested gzip tarballs (outer `.tar.gz` → inner `.tar.gz` →
  PDB). The pipeline streams the PDB directly from the compressed archives
  (`tau_mech/io.py::open_ensemble_pdb`), so **nothing is extracted to disk**.
- A residue-position audit was run against the *actual file sequences*:
  the VQIINK motif is detected at Tau residues 275–280 and VQIVYK at 306–311
  in all three ensembles (offset +242 for the K18 files), matching the
  literature positions. `HIP` (protonated His) in PED00422 is normalized to
  His.

### Raw data (bundled via Git LFS)

The three raw PED archives are **included in this repository** under `data/`
(≈140 MB total), stored with **Git LFS** because `PED00422_ensembles.tar.gz`
is 107 MB — over GitHub's 100 MB per-file limit. They are stored
byte-for-byte, so the SHA-256 digests in `outputs/provenance.json` remain
valid for integrity verification.

| Entry | Archive (in `data/`) | Size |
|---|---|---|
| PED00422 | `PED00422_ensembles.tar.gz` | 107 MB |
| PED00192 | `PED00192_ensembles.tar.gz` | 1.5 MB |
| PED00443 | `PED00443_ensembles.tar.gz` | 33 MB |

To clone with the data (requires Git LFS ≥ 3.0):

```bash
git lfs install   # once, if not already installed
git clone https://github.com/amywang2010/tau_mech.git
# if the data appears as small pointer files, run: git lfs pull
```

A normal `git clone` fetches the LFS objects automatically once `git lfs` is
installed. The pipeline reads `data/` by default (`--data-dir data`); the
archives are nested gzip tarballs (outer `.tar.gz` → inner `.tar.gz` → PDB)
streamed directly, never extracted to disk. Original sources for provenance:
https://proteinensemble.org/entries/PED00422/ (and …/PED00192/, …/PED00443/).

The 62-test suite does **not** need the raw data (tests build their own
fixtures), so `pip install -r requirements.txt && pytest tests/` works
immediately after cloning.

## 2. Setup

```
cd tau_mech
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Python 3.14.6; numpy 2.5.1, scipy 1.18.0, pandas 3.0.5, matplotlib 3.11.1,
scikit-learn 1.9.0, torch 2.13.0 (CPU), torch-geometric 2.8.0, pytest
(tested 2026-08-02; GNN stack 2026-08-15). `requirements.txt` pins minimums;
the exact resolved versions are in `outputs/config_used.json` and can be
locked with `pip freeze`.

## 3. Usage

```
# all three ensembles, full run
.venv/Scripts/python scripts/preprocess.py --all

# one ensemble / subset (for quick checks)
.venv/Scripts/python scripts/preprocess.py --ensemble PED00192
.venv/Scripts/python scripts/preprocess.py --ensemble PED00422 --n-models 20

# summary table + comparison with experimental SAXS Rg anchors
.venv/Scripts/python scripts/summarize.py

# tests
.venv/Scripts/python -m pytest tests/ -q
```

Outputs per ensemble (`outputs/<PED_ID>/`):

- `models/model_%05d.npz` — per-conformer residue graph + descriptors
  (resume-capable checkpointing)
- `ensemble_data.npz` — consolidated fixed-shape arrays (all conformers)
- `summary.csv` / `summary.json` — per-conformer descriptors + ensemble statistics
- `config_used.json`, `provenance.json`, `processing_report.json` — reproducibility record

### Graph schema (per conformer, residue-level)

- **Nodes** = residues (441 for Tau-441, 130 for K18)
- **Node features** (23-dim): 21-dim one-hot amino acid code, Kyte–Doolittle
  hydropathy, normalized sequence position
- **Node positions**: C-alpha coordinate and side-chain centroid
- **Edges**: spatial edges where any heavy-atom pair is within `--cutoff` Å
  (default 5.0 Å) **plus** sequential backbone edges (i–i+1, i–i+2)
- **Edge attributes**: [min heavy-atom distance (Å), sequence separation |i−j|]
  (sequential edges carry NaN distance — downstream GNNs must mask/handle this)

### Descriptors per conformer

- radius of gyration (mass-weighted and equal-weight, heavy atoms)
- end-to-end distance (CA₁→CAₙ)
- per-residue SASA and relative SASA (rASA)
- mean rASA of the aggregation-prone regions (APR1 = VQIINK, APR2 = VQIVYK)
- residue-level heavy-atom contact map (5 Å) and per-residue neighbor counts
- graph statistics: edge count, mean degree, graph density

## 4. Methods — exact protocol (recorded in `config_used.json`)

| Parameter | Value | Rationale / reference |
|---|---|---|
| PDB parsing | fixed-width columns, hydrogens dropped | standard PDB 3.30 layout; heavy-atom analyses are the convention |
| SASA algorithm | Shrake–Rupley | Shrake & Rupley 1973, J. Mol. Biol. 79:351–371 |
| SASA probe radius | 1.4 Å (water) | standard |
| vdW radii | C 1.80, N 1.65, O 1.40, S 1.85 (Chothia/NACCESS) | Chothia 1976, J. Mol. Biol. 105:1–14 |
| Probe points/atom | 480 (Fibonacci sphere) | documented compromise; increase for precision |
| rASA reference | Tien et al. 2013, PLoS ONE 8:e80635 (max ASA per residue) | standard for relative accessibility in IDPs |
| Edge cutoff | 5.0 Å heavy-atom (min-pair distance) | van der Waals / H-bond contact regime |
| Numbering | K18: Tau = PDB + 242 (verified by motif detection) | data-driven, audited at runtime |
| APR spans | VQIINK = 275–280, VQIVYK = 306–311 (Tau-441 numbering) | detected in the real file sequences |

## 5. Data provenance & integrity

`outputs/provenance.json` records, for each raw archive: SHA-256 digest,
size, modification date, PED entry URL, and the citation for the generation
method. Download dates are the archive mtimes. The raw files are the canonical
sources; all processed outputs are derived artifacts keyed to them.

Citations recorded: PED (Lazar et al. 2021, NAR), IDPConformerGenerator
(Teixeira et al. 2022, JPCA), Bayesian reweighted MD (Fisher, Huang & Stultz
2010, PLoS Comput. Biol.), idpGAN (Janson et al. 2023, Nat. Commun.),
Shrake–Rupley 1973, Chothia 1976, Tien et al. 2013, Kyte–Doolittle 1982.

## 6. Validation

- **Unit tests** (62, all passing): parser (multi-model, heavy filtering,
  nested-archive streaming, REMARK metadata), numbering/motif mapping, geometry
  (Rg, end-to-end, contacts), SASA (single-sphere analytic limit, occlusion,
  order invariance), graph construction (edge semantics, dedup, no self-loops),
  SPH core (kernel normalization, pair search, lattice density, deformation
  descriptors, transient fit, step stability, wall pinning), CSF operator
  symmetry (3 regression tests), Couette profile measurement layer
  (particle conservation, exact-fit recovery).
- **Experimental anchors** (reported by `scripts/summarize.py`):
  - K18 SAXS Rg ≈ 38 Å (Mukrasch et al. 2005, JBC; He et al. 2024, JCIM) —
    computed ensemble mean ≈ 36 Å (equal-weight; within ~5%)
  - Tau-441 SAXS Rg ≈ 65–69 Å (He et al. 2022, ACS Chem. Neurosci.; SASBDB
    SASDLU4 69 Å) — checked after the full run
- **Runtime audits**: per-model residue-count equality, sequence-content
  consistency (all conformers), motif-position expectations (warns on
  mismatch instead of failing silently).

## 7. Assumptions

1. Heavy-atom representation is sufficient for SASA, contact, and graph
   analyses (hydrogens excluded — standard practice).
2. Residue-level graphs capture aggregation-relevant structure; atom-level
   graphs are a future extension.
3. Ensemble members are treated as equally weighted except where noted
   (PED00192's Bayesian weights are not in the downloaded bundle — see §8).
4. SASA compares conformers within a consistent protocol; absolute values are
   protocol-dependent (Chothia radii, 480 points) and should not be compared
   numerically with tools using other protocols.

## 8. Known limitations & decisions (honest record)

1. **PED00192 Bayesian weights are unavailable in the download.** The 75
   conformers in the file are the weighted *subset*; the per-conformer weights
   that would support weighted ensemble averages must be fetched separately
   from the PED entry. Until then, PED00192 statistics are unweighted over the
   subset (documented in `provenance.json`).
2. **PED00443 is GAN-generated, not experimental.** The comparison it enables
   is "experimentally-constrained (PED00192) vs. fully generative (PED00443)",
   not "experimental vs. generated". PED00422 is likewise a computational
   conformer pool built without experimental restraints.
3. **Sample-size asymmetry**: 75 vs 1,000 vs 1,000 conformers. Distributional
   comparisons between ensembles should account for this (see Phase 2 notes).
4. **Taichi is not installable on the available Python 3.14** (no wheels; the
   project is in maintenance). The SPH stage (Phase 3) will need a secondary
   Python 3.11/3.12, or an alternative engine (NVIDIA Warp, JAX). The engine
   choice should follow a documented validation protocol (see §9), not
   convenience.
5. **Development failures documented**: an early version of the pipeline passed
   per-atom residue arrays into the numbering layer (crashed on real data),
   and the archive-streaming context manager was initially a plain generator.
   Both were caught by the test suite + real-data smoke run and fixed; the
   tests that caught them are retained (tests/test_numbering.py,
   tests/test_io.py).
6. **Phase 3 zero-shear droplet drift (RESOLVED 2026-09-02)**. The no-shear
   control originally showed `D` drifting from ~0.009 to ~0.078 — a spurious
   elongation under *zero* applied shear. Root cause found by operator-level
   audit and **fixed**: the CSF curvature stencil was pairwise-asymmetric
   (net internal force 10.8% of |f_surf|); after the symmetric-stencil fix
   the solver is permutation-invariant to ~1e-15 and the full-duration
   zero-shear gate (6 pre-registered criteria) **passes all criteria**
   (trend 1.35e-5 vs limit 5e-5; max|D−D₀| 0.0098 vs limit 0.02). Primary
   record: `outputs/sph/audits/zero_shear_baseline.json`; full audit trail:
   `docs/PHASE3_CSFFIX_AUDIT.md`; the original open finding is preserved in
   `docs/PHASE3_SPH_DRIFT_FINDING.md` (now carries a resolution banner).

## 9. Phase status (all complete; scientific-rigor notes)

- **Phase 1 — Preprocessing** (COMPLETE): see §1 and §4. Results in
  `docs/PHASES_2_5_REPORT.md`.
- **Phase 2 — EDA** (COMPLETE): distributional statistics with proper tests
  (K–S + Cohen's d), rASA interpretation, PCA/t-SNE clustering. Results in
  `docs/PHASES_2_5_REPORT.md`.
- **Phase 3 — SPH** (COMPLETE — validated engine, zero-shear gate PASSED
  2026-09-02 after the CSF symmetric-stencil fix, canonical physiological
  sweep merged, wall-coupling characterization complete): CPU
  numpy+scipy WCSPH engine validated against analytic Couette (R² = 0.998,
  symmetric profile) and Laplace limits (fresh fixed-solver calibration:
  σ_eff = 1.064 = 106.4% of input, linearity dP vs 1/R = 0.9999, per-radius
  dP·R converging toward σ_input as h/R → 0; both estimators carried in
  `outputs/sph/laplace_calibration.json`). No-shear control included.
  Kernel normalization audited (published Monaghan 1992 form).
  **Correctness audit (2026-08-11)**: the periodic-x neighbour search, the
  seam duplicate in the lattice packing, the x-wrap fold and the
  wall-lattice alignment were all fixed — these were the root cause of a
  spurious, sigma-independent "droplet shape oscillation" that an
  artificial velocity-drag quench had masked (that parameter is now
  removed). **Second audit (2026-08-14)**: a velocity-Verlet factor-of-2
  bug (the CSF surface force was applied only in the second half-step,
  halving its dt weight) made the Laplace jump read 0.5·σ/R instead of σ/R;
  it was mis-attributed to a "band-split" and now has a regression test
  (test_step_applies_csf_in_both_half_steps). **Third audit (2026-09-02)**:
  the CSF curvature stencil was pairwise-asymmetric (net internal force
  10.8%); after the symmetric-stencil fix the solver is
  permutation-invariant to ~1e-15, the full-duration zero-shear gate passes
  all 6 pre-registered criteria, the Laplace calibration was re-measured on
  the fixed solver, and the Couette slope deficit was attributed by
  diagnostic (XSPH + wall momentum-transmission slip; resolution study
  pre-registered) — **the acceptance band was not relaxed post hoc**.
  Final sweep: 6 rates incl. no-shear control, pre-registered acceptance
  v1.2 (below-noise-floor censoring registered at commit `b71aec6`),
  merged canonical record `outputs/sph/sph_shear_sweep.json`. **Framing**: physiological
  shear (~0.1–1 Pa) is far too weak to deform a *single* 10 nm protein
  directly (thermal forces dominate); the mechanistically defensible
  pathway is tissue-scale shear → droplet/condensate deformation →
  altered local concentration/interfacial stress → APR exposure. The
  manuscript states this causal chain explicitly; the coupling record
  `outputs/coupling/coupling_sph_apr.json` quantifies the bounded
  transfer (  affine upper bound: mechanically induced ΔAPR1 rASA ≈ 0.45% of native
  conformational heterogeneity at 1 Pa). Results in
  `docs/PHASES_2_5_REPORT.md` and `outputs/final_report.md`.
- **Wall-coupling resolution study** (COMPLETE 2026-09-04, pre-registered
  `docs/A3_COUETTE_V21_PREREGISTRATION.md`): the fixed-point test with
  analytic steady-state initialization established (i) initial-condition
  independence of the slip metric (rest-IC vs analytic-IC agree to 9e-4 at
  t = 3τ) and (ii) a slow formulation-level mode in the wall-coupling layer
  (slip creeps 0.865→0.910 across 1.5× windows with r² ≥ 0.9996) — reported
  per the pre-registered falsifiability rule as a characterized solver
  property, not tuned away. Records:
  `outputs/sph/audits/couette_resolution_study.json` (+ v1 fixed-h control
  and v2 abort records retained); the sweep's Ca values use *measured*
  local shear rates, so no downstream claim depends on the wall-slip
  attribution.- **Phase 4 — GNN** (COMPLETE): PyTorch Geometric dataset from `models/*.npz`;
  GCN/GAT/GraphSAGE/MLP baselines; structure-aware graph-level splits;
  cross-ensemble transfer matrix; GNNExplainer; ablations (incl. the corrected
  structural no-spatial ablation); seed-robustness extension (GAT/GraphSAGE
  n=3, GCN n=5; `outputs/gnn/seed_polish.json`). Results in
  `docs/PHASES_2_5_REPORT.md`.
- **Phase 5 — ML evaluation** (COMPLETE): ROC/PR/confusion matrices, training
  curves, model comparison, transfer matrix, embeddings viz, GNNExplainer,
  permutation importance, ablations, seed control. Results in
  `docs/PHASES_2_5_REPORT.md`. **Mutation validation** (COMPLETE 2026-09-02):
  conformationally-fixed mutate-and-recompute SASA on the WT ensemble
  (`scripts/mutation_sasa_validation.py` → `outputs/mutation/mutation_sasa.json`):
  P301L shows exactly zero static-packing effect on APR exposure (verified:
  mutation applied, P301 35.8 Å from VQIINK — outside any occlusion sphere;
  redistribution effects flagged for MD follow-up); dK280 raises surviving
  VQIINK-residue exposure by +0.033 rASA (matched-residue comparison
  deconfounds the motif-composition artifact). **Mechanical coupling**
  (COMPLETE): `scripts/coupling_sph_apr.py` →
  `outputs/coupling/coupling_sph_apr.json` — the SPH-measured droplet
  response composed with ensemble sensitivity gives a falsifiable effect
  size (quantified null at physiological Ca). Machine-generated summary of
  all phases: `outputs/final_report.md`
  (`scripts/build_final_report.py`, strict mode).

## 10. Reproducibility checklist

- [x] exact package versions (config_used.json + this README)
- [x] SHA-256 of raw data (provenance.json)
- [x] all numeric protocol parameters recorded (config_used.json)
- [x] unit tests + real-data smoke tests
- [x] lock file (`requirements.lock`, pip freeze of the validated environment, committed 2026-09-02)
- [x] git repository with the raw archives bundled via Git LFS (byte-for-byte;
      hashes in provenance.json)
- [x] resolve the Phase 3 zero-shear drift (RESOLVED 2026-09-02 — see §8.6,
      `docs/PHASE3_CSFFIX_AUDIT.md`, `outputs/sph/audits/zero_shear_baseline.json`)
- [x] wall-coupling characterization + GNN seed robustness (2026-09-04 —
      `outputs/sph/audits/couette_resolution_study.json`,
      `outputs/gnn/seed_polish.json`)
- [x] machine-generated strict final report with provenance tags
      (`outputs/final_report.md`; every number extracted from a canonical
      record; build fails if any record is missing)

## 11. Repository map

```
tau_mech/            core library (io, pipeline, eda, sph, gnn, …)
scripts/             stage drivers + audit/diagnostic scripts
outputs/             canonical records, figures, final report
  sph/               SPH: sweep records, calibration, audits/, traces
  gnn/               GNN: summary, supplement, seed_polish
  coupling/          SPH→APR transfer record + figure
  mutation/          mutation validation record
  PED00*/            per-ensemble npz + per-model graphs
  figures/           EDA figures
data/                raw PED archives (Git LFS, SHA-256 in provenance.json)
docs/                audit trail + pre-registrations + phase reports
tests/               62-test suite (fixtures built in-test; no data needed)
```
