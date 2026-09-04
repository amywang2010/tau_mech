# Do physiologically relevant mechanical shear forces influence Tau conformational susceptibility and the accessibility of aggregation-prone regions?

**Manuscript skeleton** — tau_mech study, 2026-09-04.
Every numeric claim below is copied from `outputs/final_report.md`
(provenance-tagged, machine-extracted from canonical records). Reference
hooks `[key]` point into `docs/references.bib`. `(Fig. N)` and `(Table N)`
mark placement.

---

## Title

Do physiologically relevant mechanical shear forces influence Tau
conformational susceptibility and the accessibility of aggregation-prone
regions? A quantified multiscale analysis.

*Alternative short title:* Multiscale limits of mechanically driven Tau
condensate deformation.

## Abstract (~200 words)

- **Gap:** Tau undergoes LLPS before aggregating [ambadipudi2017llps,
  wegmann2018llps]; brain extracellular fluids experience pulsatile flow
  [mestre2018csf]; whether physiological shear influences condensate
  morphology or APR exposure is unquantified.
- **Approach:** validated SPH solver → measured droplet deformation D(Ca)
  (six rates incl. no-shear control; Taylor saturation observed) →
  bounded affine/compliant transfer to APR-1 rASA using ensemble
  sensitivity (n = 1,000 conformers) → GNN-based structural analysis
  (2,075 conformers, three ensembles) with baseline separation.
- **Result:** physiological shear (0.1–1 Pa) shifts APR-1 exposure ≤0.57%
  of native conformational heterogeneity — **≥176× smaller** than thermal
  fluctuations (data-anchored upper bound; ≥1,764× under compliant
  transfer).
- **Significance:** a falsifiable, measured-response answer; defines the
  stress regime where mechanics *could* matter; reusable transfer-function
  methodology.

**Keywords:** Tau; liquid–liquid phase separation; biomolecular condensates;
smoothed particle hydrodynamics; geometric graph neural networks; shear
stress; intrinsically disordered proteins.

## 1. Introduction

1.1 Tau, IDPs, and aggregation: microtubule-associated IDP; VQIINK 275–280 /
VQIVYK 306–311 nucleate fibrils; neurofibrillary tangles define AD and
tauopathies.

1.2 LLPS as the physiological on-ramp: repeats and full-length Tau undergo
LLPS [ambadipudi2017llps, wegmann2018llps]; condensates promote
conformational expansion and aggregation [wen2021expansion]; synaptic,
activity-dependent Tau condensates [longfield2023synaptic]; review
[boyko2022review].

1.3 The mechanical environment: pulsatile CSF in perivascular spaces
[mestre2018csf]; ISF bulk flow [hladky2014fluids]; physiological shear
0.1–1 Pa (review: [kelley2023csf]).

1.4 Prior mechanics × condensates work and the gap: generic shear-mediated
liquid-to-solid transition in FUS/P-granule condensates [shen2020shear];
**no tau-specific, exposure-quantified, end-to-end multiscale analysis
exists** → the open question.

1.5 Contribution statement: (i) validated solver + pre-registered
acceptance protocol; (ii) measured D(Ca) with Taylor saturation; (iii)
quantified null with ≥176× margin; (iv) structure-separated GNN framework
for APR exposure in IDP ensembles; (v) mutation validation.

## 2. Results

### 2.1 Tau conformational ensembles and experimental anchoring

Three PED ensembles (Table 1); Rg agreement with SAXS (67.6 ± 14.6 Å vs
≈65–69 Å for Tau-441; 37.5 ± 10.6 Å vs ≈38 Å for K18).
(Fig. 1: ensemble overview — Rg/e2e/SASA distributions;
outputs/figures/fig_rg_distribution.png, fig_e2e_distribution.png,
fig_total_sasa_distribution.png)
idpGAN ensemble is more compact (d = +1.32, p = 1.7e-15) — honest
characterization of generative-vs-reweighted ensembles.

### 2.2 A validated mesoscale solver for condensate mechanics

Validation chain: CSF operator symmetry (permutation invariance ~1e-15);
zero-shear gate all 6 criteria (trend 1.35e-5 vs 5e-5; max|D−D₀| 0.0098
vs 0.02); Laplace calibration (σ_eff = 106.4%, linearity 0.9999,
h/R-convergence); Couette profile (R² = 0.9985) with dissipation-mechanism
diagnostic. (Fig. 2: validation summary schematic or zero-shear trace.)

### 2.3 Measured deformation response: Taylor saturation

D(Ca) across six rates, measured-Ca calibration 0.62–0.76× nominal;
rise → quasi-plateau → turnover at Ca ≈ 1.78 (D_∞ 0.838 at Ca 0.521 vs
0.689 at Ca 1.779) — consistent with classical droplet physics
[taylor1934emulsions, grace1982dispersion]; pre-declared prediction
(docs/A2_TAYLOR_SATURATION_PREDICTION.md, committed before the confirming
data). (Fig. 3: outputs/sph/sph_deformation_vs_Ca.png; Table 2 = sweep
table.)

### 2.4 The multiscale bridge: a quantified null

Ensemble sensitivity d(APR-1 rASA)/d(Rg) = 4.876e-4 Å⁻¹ (95% CI
[1.485e-4, 8.264e-4], n = 1,000). Composition with the measured mechanical
arm at 0.1–1 Pa: ΔAPR-1 rASA ≤ 4.87e-4 = 0.06–0.57% of native SD (0.086)
→ **≥176× (affine) / ≥1,764× (compliant) margin**; analytic Taylor
reference ~223× (superseded by the data-anchored bound).
(Fig. 4: outputs/coupling/coupling_transfer_curve.png.)

### 2.5 Structural analysis of APR exposure in ensembles (GNN)

Baseline separation: MLP 0.793 / GCN-no-spatial 0.791 (sequence-only
level) vs GCN 0.903 / GAT+edge 0.992 / GraphSAGE 1.000 — structural signal
enters through spatial edges, not sequence leakage. Seed robustness:
GCN n=5 0.902 ± 0.001; GAT n=3 0.939 ± 0.012; GraphSAGE n=3 0.9997 ±
0.0000. Explainability: GNNExplainer + permutation importance (one-hot AA
0.66, hydropathy 0.22, seq-position 0.73). Cross-construct transfer
reported without embellishment (0.899 / 0.402 / 0.334).
(Fig. 5: outputs/gnn/pr_roc_curves.png + supplement_ablations.png +
transfer_matrix.png; Table 3 = model comparison with SDs.)

### 2.6 Mutation validation

dK280: +0.033 rASA on surviving VQIINK residues (matched-residue,
deconfounded). P301L: exactly zero static-packing effect (verified
applied; 35.8 Å from VQIINK) → redistribution pathway flagged for MD.

## 3. Discussion

3.1 The null in context: thermal noise dominates physiological shear at
the single-condensate scale; implications for interpreting mechanical
correlations in vivo; amplification candidates (surfaces, confinement,
collective dynamics, repeated transits).

3.2 Taylor saturation connects condensate mechanics to classical droplet
phenomenology; the measured D(Ca) is reusable as a calibration anchor.

3.3 Methodological contribution: pre-registered, machine-reported,
measured-Ca protocol; structure-separated GNN evaluation as a standard
others should meet.

3.4 Limitations (as in the report): upper-bound transfer; discretization
uncertainty (σ_eff 106.4%, h/R-convergent); characterized wall-layer slow
mode with no downstream dependence (measured-Ca protocol); static-packing
mutation scope; exposure correlations not kinetics; single-split seed
robustness; Newtonian-like droplet material model.

3.5 Outlook: MD-coupled redistribution; viscoelastic condensate models;
microfluidic experimental test of the predicted threshold stress.

## 4. Methods

4.1 Data and preprocessing (PED sources [lazar2021ped, sasbdb_sasdlu4,
janson2023idpgan]; numbering; SASA protocol [shrake1973sasa,
chothia1976surfaces]).

4.2 Graph construction and descriptors.

4.3 SPH solver and validation stack (kernel [monaghan1992sph];
pre-registration chain; docs/A3_COUETTE_V21_PREREGISTRATION.md).

4.4 Shear sweep protocol and acceptance rules (pre-registered v1.2→v1.4.1;
measured-Ca diagnostic).

4.5 Coupling model (affine upper bound; compliant variant; assumptions).

4.6 GNN architecture, splits, seeds, explainability.

4.7 Mutation analysis.

4.8 Reproducibility (records, hashes, tests, LFS data).

## Back matter

- **Data availability:** all raw PED archives bundled via Git LFS;
  SHA-256 in outputs/provenance.json; repository URL.
- **Code availability:** full code + 62-test suite + machine-generated
  report builder; requirements.lock.
- **Author contributions, acknowledgments, competing interests:** to fill.

## Figures (final placement)

| # | Content | File |
|---|---|---|
| 1 | Ensemble overview (sizes, Rg, e2e, SASA) | outputs/figures/fig_ensemble_sizes.png, fig_rg_distribution.png, fig_e2e_distribution.png, fig_total_sasa_distribution.png |
| 2 | Residue-level structure (rASA profile, flexibility, contact maps) | outputs/figures/fig_rsa_profile.png, fig_flexibility.png, fig_contact_maps.png |
| 3 | D(Ca) response with Taylor saturation (canonical) | outputs/sph/sph_deformation_vs_Ca.png |
| 4 | SPH→APR transfer curve at physiological stress | outputs/coupling/coupling_transfer_curve.png |
| 5 | GNN performance + ablations + transfer matrix | outputs/gnn/pr_roc_curves.png, supplement_ablations.png, transfer_matrix.png |
| 6 | Embeddings (PCA/t-SNE) | outputs/gnn/embeddings_pca.png, embeddings_tsne.png |
| S1 | Graph properties + Uversky plane | outputs/figures/fig_degree_distribution.png, fig_graph_density.png, fig_charge_hydropathy.png |
| S2 | Residue frequency, training curves, seed robustness | outputs/figures/fig_residue_frequency.png, outputs/gnn/training_curves.png + seed_polish record |

## Tables

| # | Content | Source |
|---|---|---|
| 1 | Ensembles (construct, n, method, role, SAXS anchor) | README §1 |
| 2 | Shear sweep (Ca_meas, D_∞, class, R²) | outputs/sph/sph_shear_sweep.json |
| 3 | Model comparison with seed SDs | outputs/gnn/summary.json + seed_polish.json |
| 4 | Coupling bounds (affine/compliant at 0.1–1 Pa) | outputs/coupling/coupling_sph_apr.json |

## Writing notes

- Voice: measured, quantitative, no adjectives doing scientific work.
- The null is the headline; frame as defining the regime where mechanics
  can matter, not as a failure to find.
- Never cite the pre-fix archived figure or superseded records as results.
- All numbers must match `outputs/final_report.md` verbatim; the report is
  the single source of truth and is rebuilt from records.
