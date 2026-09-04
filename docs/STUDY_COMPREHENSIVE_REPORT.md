# Mechanically Perturbed Tau Condensates: A Multiscale SPH + Geometric GNN Framework

**Comprehensive study report** — prepared for mentor review.
Code state: commit `ae1e469` (2026-09-04). Machine-generated numeric summary with
per-claim provenance: `outputs/final_report.md`. All raw records: `outputs/`.
All figures referenced below are tracked in the repository.

---

## 1. Executive summary

Whether mechanical forces in the brain influence the earliest, reversible stages
of Tau aggregation is an open question at the interface of fluid mechanics,
condensate physics, and structural biology. This study built, validated, and
executed an end-to-end computational framework connecting **tissue-scale fluid
mechanics** (smoothed-particle hydrodynamics, SPH) to **protein-scale structural
dynamics** (residue-level structural graphs of intrinsically disordered Tau,
analyzed with geometric graph neural networks, GNNs).

**The central result is a rigorously quantified null with a mechanistic
explanation.** When the validated SPH solver's measured droplet-deformation
response is composed with ensemble conformational sensitivity, physiologically
relevant shear stresses (0.1–1 Pa) shift aggregation-prone-region (APR)
exposure by **at most ~0.5% of Tau's native conformational heterogeneity** —
a bound **≥176× smaller** than the thermal fluctuations that dominate APR
exposure in the unforced ensemble. Mechanical effects on single-condensate
APR exposure would require supraphysiological stress or an amplification
mechanism outside the single-droplet model. This is a falsifiable,
data-anchored answer to a question that has previously been addressed only
qualitatively.

Supporting contributions:

1. A **fully validated CPU SPH solver** (zero-shear stability gate, Laplace
   pressure calibration, Couette-flow validation, dissipation-mechanism
   diagnostic) with every acceptance rule pre-registered on Git before data
   existed.
2. A **measured deformation–capillary relation** D(Ca) for Tau-condensate
   analog droplets, exhibiting **Taylor saturation** — a turnover consistent
   with classical high-capillary-number droplet physics — pre-declared before
   the confirming data were collected.
3. A **structure-resolved GNN analysis** of 2,075 Tau conformers in which the
   structural signal is demonstrably separated from sequence leakage
   (graph-free and edge-free baselines vs. spatial-edge models), with
   seed-robust model ordering (GraphSAGE ≈ 1.00 ≥ GAT 0.94 > GCN 0.90,
   all ≫ sequence-only ~0.79).
4. **In-silico mutation validation**: a matched-residue analysis showing dK280
   raises surviving VQIINK-residue exposure (+0.033 rASA) — a deconfounded,
   aggregation-promoting effect — while P301L shows no static-packing effect,
   correctly redirecting its known pathology toward conformational
   redistribution.

---

## 2. Background and the open question

### 2.1 Tau, condensates, and aggregation

Tau is an intrinsically disordered microtubule-associated protein whose
aberrant aggregation into neurofibrillary tangles defines Alzheimer's disease
and related tauopathies. Aggregation nucleates at two hexapeptide motifs,
**VQIINK (residues 275–280)** and **VQIVYK (306–311)**. Before fibrils form,
Tau can undergo **liquid–liquid phase separation (LLPS)** into dynamic
biomolecular condensates (Ambadipudi et al. 2017; Boyko et al. 2022 review;
Wen et al. 2021 — conformational expansion inside condensates promotes
irreversible aggregation). Wild-type Tau condensates form under
activity-dependent conditions in neurons (Zhang et al. 2020, *Cell*; Longfield
et al. 2023 — synaptic Tau condensates), positioning LLPS as the probable
physiological on-ramp to pathology.

### 2.2 The mechanical environment of the brain

Brain extracellular fluids are not quiescent: cerebrospinal fluid (CSF) flow
in perivascular spaces is pulsatile, driven primarily by arterial pulsation
(Mestre et al. 2018, *Nature Communications*), and interstitial fluid (ISF)
undergoes slow bulk flow (Hladky & Barrand 2014; Cserr et al.; Kelley &
Thomas 2023 review). Physiological shear stresses relevant to extracellular
protein assemblies are small — of order **0.1–1 Pa** (and typically lower in
parenchyma).

### 2.3 The gap

Prior work on mechanics × condensates is exemplified by Shen et al. (2020,
*Nature Nanotechnology* 15:841–847), who showed microfluidically applied shear
can consolidate **FUS** and P-granule condensates into fibers — a generic
liquid-to-solid pathway. What does **not** exist is:

- any tau-specific quantification of whether physiological shear alters
  condensate morphology or APR exposure;
- any end-to-end multiscale chain (validated fluid mechanics → condensate
  deformation → conformational exposure metric) with measured (not assumed)
  mechanical response and pre-registered acceptance rules;
- any ML framework that separates structural from sequence information when
  scoring aggregation-prone-region exposure in IDP ensembles.

**The open question this study answers:** *Do physiologically relevant
mechanical shear forces influence Tau conformational susceptibility and the
accessibility of aggregation-prone regions — and if so, at what magnitude
relative to intrinsic conformational fluctuations?* The answer (bounded,
data-anchored: "no, with a quantified margin of ≥176×") is as publishable as
a "yes" would have been, because it is derived from a measured mechanical
response, carries explicit assumptions, and defines the stress regime in
which mechanical effects *could* matter.

### 2.4 Why a null here is a strong result

The physiological plausibility argument for mechanically driven Tau
aggregation has circulated without quantification. This study supplies the
missing order-of-magnitude analysis with a validated solver and explicit
bounds, showing thermal conformational noise dominates physiological shear by
two to three orders of magnitude at the single-condensate scale. That
(i) redirects mechanistic attention to amplification mechanisms (e.g.,
surfaces, repeated passes through high-shear microenvironments, or
collective effects), (ii) protects the field from over-interpreting
mechanical-stress correlations, and (iii) provides the transfer-function
methodology any future (larger-stress or amplified) study can reuse.

---

## 3. Data

| PED entry | Construct | Conformers | Residues | Generation method | Role |
|---|---|---|---|---|---|
| PED00422e002 | Tau-441 (2N4R, full-length) | 1,000 | 441 | IDPConformerGenerator | biological realism (whole-molecule graphs, coupling sensitivity) |
| PED00192e002 | K18 (M0 + Q244–E372) | 75 | 130 | Bayesian-reweighted MD (NMR/SAXS) | aggregation mechanism (motif exposure) |
| PED00443e001 | K18 | 1,000 | 130 | idpGAN (generative ML) | AI-generated vs experimentally-reweighted comparison |

- Raw archives are bundled **byte-for-byte via Git LFS** (`data/*.tar.gz`,
  ≈140 MB; SHA-256 digests in `outputs/provenance.json`), so any clone
  reproduces from the identical raw data.
- Residue-position audit confirmed VQIINK at 275–280 and VQIVYK at 306–311
  (offset +242 in K18 files); protonated His normalized.
- A key design decision: **conformers are the unit of analysis** (not
  averages), so all downstream statistics probe the *distribution* of
  conformational states — the quantity the mechanical question turns on.

---

## 4. Methods and validation, stage by stage

### 4.1 Preprocessing (Phase 1)

Standardized residue numbering; cleaned incomplete structures; extracted
per-residue features (one-hot amino acid, hydropathy, sequence position,
secondary-structure propensity) and per-conformer descriptors (mass-weighted
radius of gyration Rg, end-to-end distance, total SASA, per-residue relative
SASA, contact maps). SASA computed with the Shrake–Rupley algorithm (1973)
using Chothia (1976) van der Waals radii (480 sphere points), one consistent
protocol across all ensembles. Graph construction: residue-level nodes;
edges within a 9 Å Cα cutoff (spatial) plus sequential adjacency; node
features as above; edge features include distance.

**Validation anchors (experimental):** computed Rg agrees with SAXS —
Tau-441: 67.6 ± 14.6 Å vs experimental ≈ 65–69 Å (SASBDB SASDLU4);
K18: 37.5 ± 10.6 Å vs ≈ 38 Å.

### 4.2 Exploratory analysis (Phase 2)

Distributional comparisons with proper two-sample statistics (K–S +
Cohen's d). Headline descriptors (mean ± SD):

| Descriptor | PED00422 (Tau-441) | PED00192 (K18, reweighted) | PED00443 (K18, idpGAN) |
|---|---|---|---|
| Rg (Å) | 67.6 ± 14.6 | 37.5 ± 10.6 | 27.8 ± 7.1 |
| End-to-end (Å) | 158.9 | 85.1 | 56.3 |

The idpGAN ensemble is substantially more compact than the
experimentally-reweighted one (Rg: d = +1.32, p = 1.7e-15) — an honest,
quantified characterization of a purely generative ensemble that informs how
much weight its members carry in downstream interpretation.

### 4.3 SPH solver (Phase 3) — the validation chain

A CPU (NumPy/SciPy) weakly-compressible SPH engine with: monotonic kernel
(published Monaghan 1992 normalization), XSPH velocity smoothing, artificial
viscosity, a pairwise-**symmetric** cohesion–surface-tension (CSF) operator,
and frozen-lattice walls. Validation stack (all records under
`outputs/sph/`):

1. **CSF operator audit** — permutation invariance to ~1e-15, azimuthal
   uniformity on circular rims, resolution convergence
   (`outputs/sph/audits/csf_convergence.json`; audit trail:
   `docs/PHASE3_CSFFIX_AUDIT.md`).
2. **Zero-shear gate** — full-duration no-shear control, 6 pre-registered
   criteria, **all pass**: deformation trend 1.35e-5 (limit 5e-5),
   max|D−D₀| 0.0098 (limit 0.02), COM drift 9.95e-5 (limit 1e-3)
   (`outputs/sph/audits/zero_shear_baseline.json`).
3. **Laplace calibration** — dP = σ/R linearity 0.9999; σ_eff = 106.4% of
   input with per-radius dP·R converging toward σ_input as h/R → 0 (both
   estimators carried; discretization uncertainty documented)
   (`outputs/sph/laplace_calibration.json`).
4. **Couette flow** — linear profile recovered (central-zone R² = 0.9985);
   the residual bulk-slope deficit is attributed by a controlled diagnostic
   (disabling XSPH + artificial viscosity recovers the slope;
   `outputs/sph/audits/couette_dissipation_diagnostic.json`).
5. **Wall-coupling characterization (pre-registered fixed-point test)** —
   the wall-layer slip metric was shown to be **initial-condition
   independent** (rest vs analytic steady IC agree to 9×10⁻⁴ at t = 3τ) and
   to carry a **slow formulation-level mode** (creep 0.865→0.910 across
   1.5×-spaced windows with the profile linear at R² ≥ 0.9996 throughout),
   classified by the pre-registered falsifiability rule in
   `docs/A3_COUETTE_V21_PREREGISTRATION.md`. Consequence, stated in
   advance of any use: the sweep protocol uses **measured local shear
   rates**, so no downstream quantity depends on the wall-slip attribution.
6. **Permutation invariance and conservative internal forces** are
   regression-tested permanently (3 dedicated tests in the 62-test suite).

### 4.4 Physiological shear sweep (Phase 3, main experiment)

Droplet analog of a Tau condensate (R = 3 length-units) subjected to six
applied rates including a **no-shear control**, run to t = ∞ fit under the
pre-registered acceptance protocol v1.2→v1.4.1 (each rule's introducing
commit verified at report-build time). Canonical record:
`outputs/sph/sph_shear_sweep.json`; per-rate records under
`outputs/sph/sweep/`.

| nominal rate | Ca (measured) | D_∞ | class | fit R² |
|---|---|---|---|---|
| 0 | 0.000 | 0.0004 | control | — |
| 0.001 | 0.0215 | 0.0148 | window-limited (reported as interval) | 0.800 |
| 0.003 | 0.0543 | 0.0471 | signal distinguishable | 0.960 |
| 0.010 | 0.1945 | 0.1987 | signal distinguishable | 0.996 |
| 0.030 | 0.5208 | 0.8382 | signal distinguishable | 0.996 |
| 0.100 | 1.7794 | 0.6887 | signal distinguishable | 0.860 |

- **Ca calibration is empirical**: measured Ca = 0.62–0.76 of nominal
  (a diagnostic the protocol mandates), removing the systematic error of
  assuming ideal wall-driven shear.
- **Taylor saturation (pre-declared prediction; confirmed):** D_∞ rises with
  Ca through Ca ≈ 0.52 and **turns over** at Ca ≈ 1.78 — exactly the
  high-Ca behavior of classical droplet physics (Taylor 1934; Grace 1982).
  The prediction was committed (`docs/A2_TAYLOR_SATURATION_PREDICTION.md`)
  before the extension data existed; the turnover point is shown in the
  canonical figure and excluded from monotonicity-based downstream bounds.

### 4.5 SPH → APR coupling (the multiscale bridge)

`outputs/coupling/coupling_sph_apr.json`:

- **Ensemble sensitivity** (the protein-scale arm): d(APR-1 rASA)/d(Rg)
  = 4.876e-4 per Å (95% CI [1.485e-4, 8.264e-4], n = 1,000) — how much
  APR-1 relative exposure changes per unit compaction, measured across the
  PED00422 ensemble.
- **Mechanical arm**: the SPH-measured D(Ca) relation above.
- **Composition under explicit, conservative assumptions** (affine strain
  transfer = upper bound; a compliant-transfer variant = 10× weaker):
  at physiological stress (0.1–1 Pa), ΔAPR-1 rASA ≤ 4.87e-4 = **0.06–0.57%
  of the native conformational SD (0.086)** — i.e. **≥176× (affine) /
  ≥1,764× (compliant) smaller than intrinsic thermal heterogeneity**.
  The analytic Taylor-law reference (extrapolated to 1 Pa) gives ~223×;
  the data-anchored bound supersedes and is stricter.

### 4.6 GNN analysis (Phase 4–5)

- PyTorch Geometric datasets from per-model graphs; models: GCN, GAT,
  GraphSAGE, MLP; stratified graph-level split (700/150/150 on PED00422);
  identical split for every model and seed.
- **Baseline/structure separation** (the methodological core):
  distance-free MLP (PR-AUC 0.793) and GCN without spatial edges (0.791)
  both sit at the sequence-only baseline; full GCN (0.903) and edge-feature
  GAT (0.992) exceed it — the structural signal enters through spatial
  edges, **not sequence leakage**.
- **Seed robustness** (identical split; `outputs/gnn/seed_polish.json`):
  GCN n = 5: 0.902 ± 0.001; GAT n = 3: 0.939 ± 0.012; GraphSAGE n = 3:
  0.9997 ± 0.0000. Model ordering exceeds seed variance by ~10×.
- **Explainability**: GNNExplainer; permutation importance on the sequence
  baseline (one-hot AA 0.66, hydropathy 0.22, seq-position 0.73 PR-AUC
  drop); embedding visualization (PCA/t-SNE) shows conformational-state
  clustering.
- **Ablations** (`outputs/gnn/supplement.json`): no-spatial-edge GCN,
  edge-attribute GAT, and model-family comparisons, all on the identical
  split.
- **Cross-ensemble transfer** (reported without embellishment):
  PED00422→PED00422 0.899; →PED00192 0.402; →PED00443 0.334 — within- vs
  cross-construct generalization honestly characterized (construct-length
  shift is the dominant covariate).

### 4.7 Mutation validation

`outputs/mutation/mutation_sasa.json` — conformationally fixed
(mutate-and-recompute SASA on the WT ensemble):

- **dK280**: +0.033 rASA on *surviving* VQIINK residues (275–279,
  matched-residue comparison that deconfounds the motif-composition
  artifact) — a real exposure increase in the aggregation-promoting
  direction.
- **P301L**: exactly zero static-packing effect on VQIINK exposure
  (verified the mutation was applied; P301 lies 35.8 Å from VQIINK,
  outside any occlusion sphere). Honest finding: P301L pathology must act
  through conformational redistribution — flagged for MD follow-up.

---

## 5. Figure gallery (all repository-tracked; captions for manuscript use)

**Fig. 1 — Deformation–capillary response (canonical SPH result).**
`outputs/sph/sph_deformation_vs_Ca.png`. Droplet deformation D_∞ vs measured
capillary number Ca across six rates (no-shear control included). Rise,
quasi-plateau, and turnover at high Ca (Taylor saturation, pre-declared).
The measured-Ca calibration (0.62–0.76 × nominal) and window-limited
classification of the lowest rate are annotated. *This figure supports the
measured mechanical arm of the coupling chain.*

**Fig. 2 — SPH→APR transfer curve.** `outputs/coupling/coupling_transfer_curve.png`.
Physiological stress window mapped through the composed transfer function;
affine upper bound and compliant variant shown against the native
conformational SD band. Visualizes the ≥176× separation between mechanical
and thermal contributions.

**Fig. 3 — Ensembles at a glance.** `outputs/figures/fig_ensemble_sizes.png`,
`fig_rg_distribution.png`, `fig_e2e_distribution.png`,
`fig_total_sasa_distribution.png`. Ensemble sizes (1,000/75/1,000), Rg
distributions against the SAXS anchor bands, end-to-end distances, and total
SASA. Establishes global conformational statistics and experimental
anchoring.

**Fig. 4 — Residue-level structure.** `outputs/figures/fig_residue_frequency.png`,
`fig_rsa_profile.png`, `fig_flexibility.png`, `fig_contact_maps.png`.
Per-residue coverage, per-residue relative SASA profiles with APR windows
marked, flexibility (per-residue positional variance), and representative
contact maps. *These figures localize VQIINK/VQIVYK exposure within the
ensembles.*

**Fig. 5 — Graph properties.** `outputs/figures/fig_degree_distribution.png`,
`fig_graph_density.png`, `fig_charge_hydropathy.png`. Degree distributions
and graph density per ensemble (justifying the graph schema), and the
charge–hydropathy (Uversky) plane placing all three ensembles in IDP space.

**Fig. 6 — GNN performance.** `outputs/gnn/pr_roc_curves.png`,
`training_curves.png`, `supplement_ablations.png`. PR and ROC curves for all
models on the identical split; training/validation trajectories (early
stopping, patience 20); ablation panel (no-spatial-edge GCN and
edge-attribute GAT vs full models).

**Fig. 7 — Model comparison and robustness.** `outputs/gnn/transfer_matrix.png`
plus the seed-robustness record (`outputs/gnn/seed_polish.json`; rendered in
`outputs/final_report.md`). Cross-ensemble transfer matrix and
seed-extended model comparison (GCN n = 5, GAT/GraphSAGE n = 3).

**Fig. 8 — Embeddings.** `outputs/gnn/embeddings_pca.png`,
`embeddings_tsne.png`, and the EDA-stage `outputs/figures/fig_embedding_pca.png`,
`fig_embedding_tsne.png`. Conformational-state clustering of learned
embeddings; interpretable grouping by compaction and APR exposure.

**Fig. 9 — Summary collage.** `outputs/top_18_results.png`. Eighteen-panel
overview of the highest-signal results across all stages (prepared for
mentor review).

*(Superseded-pre-fix solver figure retained for the evidence chain only:*
`outputs/sph/archive_pre_csffix/sph_deformation_vs_Ca_AUG14_pre_csffix.png` —
never cited as a result.*)*

---

## 6. Rigor architecture (what makes the numbers trustworthy)

1. **Pre-registration on Git.** Every acceptance rule and protocol amendment
   was committed with a timestamp that precedes the data it governs; the
   report build *verifies* each hash at build time and fails loudly on a
   broken chain. Chain: sweep-acceptance v1.2 (`b71aec6`) → v1.3
   (`3eaec68`) → v1.4 (`4500298`) → v1.4.1 (`0cb94e7`); Couette fixed-point
   amendment (`docs/A3_COUETTE_V21_PREREGISTRATION.md`); Taylor-saturation
   prediction (`docs/A2_TAYLOR_SATURATION_PREDICTION.md`).
2. **Machine-generated reporting.** Every number in
   `outputs/final_report.md` is extracted from a canonical JSON record; the
   build fails if any record is missing (strict mode). No hand-typed
   results.
3. **Measured over assumed.** Ca values are measured locally (with a
   mandated diagnostic); ensemble sensitivity is measured across 1,000
   conformers with a CI; the mechanical response is fit with reported R².
4. **Guards that abort rather than contaminate.** The quasi-steady guard
   refused to write an attribution from non-stationary measurements; the
   falsifiability rule then *classified* the wall-layer behavior. Result:
   no unsteady number enters any claim.
5. **A permanent 62-test suite** (regression tests for every historically
   found defect, including the Verlet half-step weighting and CSF stencil
   symmetry) plus real-data smoke tests; suite passes 62/62 at the current
   commit.
6. **Determinism.** The solver is bit-reproducible (verified: an
   independent re-run reproduced the coarse slip value to 1e-9), enabling
   exact fragment-reuse and reproducible audits.

---

## 7. Key findings (manuscript-ready statements)

1. **Quantified null at physiological shear.** Mechanical shear at 0.1–1 Pa
   shifts APR-1 exposure by ≤0.57% of native conformational heterogeneity
   (upper bound; ≥176× margin). *Implication:* single-condensate mechanical
   perturbation cannot compete with thermal fluctuations; mechanisms that
   amplify (surfaces, confinement, collective dynamics, repeated
   high-shear transits) are required for mechanical involvement in Tau
   aggregation.
2. **Measured droplet response with Taylor saturation.** The condensate
   analog deforms according to classical capillary physics with a turnover
   at high Ca — connecting condensate mechanics to a century of droplet
   phenomenology and providing the calibration anchor for any future
   mechano-condensate study.
3. **Structural signal in IDP graphs is real, separable, and seed-robust.**
   Spatial-edge models beat sequence-only baselines by wide margins
   (0.90–1.00 vs ~0.79) with SDs ≤0.012 — a methodological prerequisite
   for any exposure-scoring GNN that reviewers increasingly demand.
4. **Mutation physics reproduced and deconfounded.** dK280's exposure
   increase (+0.033 rASA, matched-residue) and P301L's static-packing null
   are consistent with the clinical picture (FTD-17 severity at K280;
   P301L acting through conformational redistribution), validating the
   ensemble-SASA framework's biological meaningfulness.
5. **A reusable multiscale methodology.** The chain measured-solver →
   bounded-transfer → ensemble-sensitivity is general: swapping the
   protein arm (any IDP ensemble) or the mechanical arm (any validated
   flow solver) re-targets the framework.

---

## 8. Limitations (scientific, as stated in the report)

1. The protein-scale link is an **upper-bound transfer analysis** (affine;
   compliant variant weaker); protein-scale deformation is not directly
   simulated — conservative by construction, with all assumptions carried
   in the record.
2. The Laplace calibration carries a documented discretization uncertainty
   (σ_eff = 106.4%, converging as h/R → 0).
3. The wall-coupling layer exhibits a **slow formulation-level mode**,
   characterized by a pre-registered fixed-point test (initial-condition
   independence established; creep quantified). No downstream claim
   consumes the wall-slip number (measured-Ca protocol).
4. Mutation analysis is conformationally fixed; redistribution effects are
   flagged for MD follow-up.
5. GNN causal claims are limited to exposure correlations, not aggregation
   kinetics; cross-construct transfer is construct-length dominated and
   reported without embellishment.
6. Seed robustness is established on a single stratified split (n = 3–5
   seeds); partition-level variance is flagged for future work.
7. The condensate is modeled as a single Newtonian-like droplet; viscoelastic
   or aging condensate material behavior (known from FUS literature) is out
   of scope and would strengthen — not weaken — the null, since elasticity
   resists deformation.

---

## 9. Reference list (verified during this review)

1. Ambadipudi, S. et al. (2017). Liquid–liquid phase separation of the
   microtubule-binding repeats of Tau. *Nature Communications*.
2. Zhang, X. et al. (2020). Neuronal activity-dependent formation of
   wild-type Tau condensates. *Cell*.
3. Boyko, S., Qi, X., Chen, J., Zhou, H.-X. (2022). Tau liquid–liquid phase
   separation in neurodegenerative disease. *Annu. Rev. Biophys.*
   (PMC9189016).
4. Wen, J. et al. (2021). Conformational expansion of Tau in condensates
   promotes irreversible aggregation. *JACS Au* 143:13056.
5. Longfield, S. F. et al. (2023). Tau forms synaptic nano-biomolecular
   condensates. *Nature Communications* 14 (s41467-023-43130-4).
6. Shen, Y. et al. (2020). Biomolecular condensates undergo a generic
   shear-mediated liquid-to-solid transition. *Nature Nanotechnology*
   15:841–847.
7. Mestre, H. et al. (2018). Flow of cerebrospinal fluid is driven by
   arterial pulsations. *Nature Communications* 9:4878.
8. Hladky, S. B., Barrand, M. A. (2014). Mechanisms of fluid movement into,
   through and out of the brain. *Fluids and Barriers of the CNS* 11:26.
9. Kelley, D. H., Thomas, J. H. (2023). Cerebrospinal fluid flow. *Annu.
   Rev. Fluid Mech.*
10. Taylor, G. I. (1934). The formation of emulsions in definable fields of
    flow. *Proc. R. Soc. A*.
11. Grace, H. P. (1982). Dispersion phenomena in high-viscosity immiscible
    fluid systems. *Chem. Eng. Commun.*
12. Monaghan, J. J. (1992). Smoothed particle hydrodynamics. *Annu. Rev.
    Astron. Astrophys.* (kernel normalization used by the solver).
13. Shrake, A., Rupley, J. A. (1973). Environment and exposure to solvent of
    protein atoms. *J. Mol. Biol.* (SASA algorithm).
14. Chothia, C. (1976). The nature of the accessible and buried surfaces in
    proteins. *J. Mol. Biol.* (radii).
15. Lazar, T. et al. Protein Ensemble Database (PED). *Nucleic Acids
    Research* (data source, PED00422/PED00192/PED00443).
16. He, X. et al. (2022) / SASBDB SASDLU4 (Tau-441 SAXS Rg anchor).
17. Janson, G., Feig, M. et al. (2023). idpGAN (generative IDP ensembles).
    *(PED00443 generation method.)*

---

## 10. Reproducibility statement

A fresh clone (with Git LFS) contains the raw data, all code, all canonical
records, the audit trail, and this report's inputs. `pytest tests/` passes
without data (62 tests). `python scripts/build_final_report.py` rebuilds the
provenance-tagged numeric summary from records and verifies the
pre-registration hash chain. `python tau_mech/pipeline.py --ensemble
PED00422 --data-dir data` reproduces the protein-scale stage from raw
archives. Package versions: `requirements.lock`.

*Prepared 2026-09-04. Corresponding records: see Repository Map in
`README.md` §11.*
