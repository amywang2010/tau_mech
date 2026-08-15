# tau_mech — Multiscale computational framework for Tau mechanobiology

**Phase 1: data preprocessing** for the IEEE submission "Do physiologically
relevant mechanical shear forces influence Tau conformational susceptibility
and the accessibility of aggregation-prone regions?"

This package parses the three Protein Ensemble Database (PED) Tau ensembles,
standardizes residue numbering, computes structural and aggregation descriptors
(radius of gyration, solvent-accessible surface area, aggregation-prone region
exposure, contact maps), and builds residue-level geometric protein graphs for
downstream graph neural network (GNN) and smoothed-particle-hydrodynamics (SPH)
stages.

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

The 55-test suite does **not** need the raw data (tests build their own
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

- **Unit tests** (46, all passing): parser (multi-model, heavy filtering,
  nested-archive streaming, REMARK metadata), numbering/motif mapping, geometry
  (Rg, end-to-end, contacts), SASA (single-sphere analytic limit, occlusion,
  order invariance), graph construction (edge semantics, dedup, no self-loops),
  SPH core (kernel normalization, pair search, lattice density, deformation
  descriptors, transient fit, step stability, wall pinning).
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
6. **Phase 3 zero-shear droplet drift (OPEN)**. The no-shear control of the SPH
   shear sweep shows the droplet deformation `D` drifting monotonically from
   ~0.009 to ~0.078 over the ~406-time-unit measurement window — a spurious
   ~7 % elongation under *zero* applied shear, larger than the low-Ca signal.
   The drift is solver-level (present with all surface forces disabled) and was
   missed because prior validation only ran ~8–48 time units. Full record,
   primary data, and options in `PHASE3_SPH_DRIFT_FINDING.md`. The
   physiological conclusion (Ca ~ 1e-3 → negligible deformation) rests on the
   analytic Taylor limit and does not depend on the affected sweep.

## 9. Roadmap to the remaining phases (scientific-rigor notes)

- **Phase 2 — EDA** (COMPLETE): distributional statistics with proper tests
  (K–S + Cohen's d), rASA interpretation, PCA/t-SNE clustering. Results in
  `PHASES_2_5_REPORT.md`.
- **Phase 3 — SPH** (engine + validation complete; **shear-sweep results are
  BLOCKED by a spurious zero-shear droplet drift — see
  `PHASE3_SPH_DRIFT_FINDING.md`**): CPU
  numpy+scipy WCSPH engine validated against analytic Couette (R²=0.998,
  symmetric profile) and Laplace (σ_eff = 0.998 after the factor-2 fix below)
  limits; no-shear control included; kernel normalization audited (published
  Monaghan 1992 form). **Correctness audit (2026-08-11)**: the periodic-x
  neighbour search, the seam duplicate in the lattice packing, the x-wrap fold
  and the wall-lattice alignment were all fixed — these were the root cause of
  a spurious, sigma-independent "droplet shape oscillation" that an artificial
  velocity-drag quench had masked (that parameter is now removed). **Second
  audit (2026-08-14)**: a velocity-Verlet factor-of-2 bug (the CSF surface
  force was applied only in the second half-step, halving its dt weight) made
  the Laplace jump read 0.5·σ/R instead of σ/R; it was mis-attributed to a
  "band-split" and now has a regression test
  (test_step_applies_csf_in_both_half_steps). See PHASES_2_5_REPORT.md, CSF
  audit trail item 8. **Sweep runtime**: the single-threaded engine runs
  ~0.34 s/step, so the 6-rate sweep (~5.5e5 steps) is ~31 h sequential; the
  rates are independent (each rebuilds its own droplet), so
  `scripts/sph_sweep_parallel.py` runs them concurrently on separate cores
  (~6x wall-clock speedup, no change to physics). **Framing**: physiological
  shear (~0.1–1 Pa) is far too
  weak to deform a *single* 10 nm protein directly (thermal forces dominate);
  the mechanistically defensible pathway is tissue-scale shear →
  droplet/condensate deformation → altered local concentration/interfacial
  stress → APR exposure. The manuscript should state this causal chain
  explicitly. Results in `PHASES_2_5_REPORT.md`.
- **Phase 4 — GNN** (COMPLETE): PyTorch Geometric dataset from `models/*.npz`;
  GCN/GAT/GraphSAGE/MLP baselines; structure-aware graph-level splits;
  cross-ensemble transfer matrix; GNNExplainer; ablations (incl. the corrected
  structural no-spatial ablation). Results in `PHASES_2_5_REPORT.md`.
- **Phase 5 — ML evaluation** (COMPLETE): ROC/PR/confusion matrices, training
  curves, model comparison, transfer matrix, embeddings viz, GNNExplainer,
  permutation importance, ablations, seed control. Results in
  `PHASES_2_5_REPORT.md`. **Validation against pathogenic mutants** (e.g.,
  P301L, ΔK280) is the documented next experiment: does the pipeline predict
  higher APR exposure for aggregation-promoting mutations?

## 10. Reproducibility checklist

- [x] exact package versions (config_used.json + this README)
- [x] SHA-256 of raw data (provenance.json)
- [x] all numeric protocol parameters recorded (config_used.json)
- [x] unit tests + real-data smoke tests
- [ ] lock file (`pip freeze > requirements.lock`) — recommended before submission
- [x] git repository with the raw archives bundled via Git LFS (byte-for-byte;
      hashes in provenance.json)
- [ ] resolve the Phase 3 zero-shear drift (see `PHASE3_SPH_DRIFT_FINDING.md`)
