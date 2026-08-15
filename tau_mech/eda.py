"""Phase 2 - Exploratory Data Analysis for the Tau mechanobiology study.

Consumes the Phase 1 outputs (outputs/<PED_ID>/ensemble_data.npz + summary.json)
and produces:

  * per-ensemble distribution statistics and figures (Rg, end-to-end, SASA,
    graph degree, graph density, residue frequency, contact maps)
  * per-residue profiles (mean +/- SD rSA, flexibility) with the aggregation-
    prone regions highlighted
  * LLPS-relevant sequence profiles (net charge, hydropathy) for K18
  * pairwise ensemble comparisons (two-sample Kolmogorov-Smirnov,
    Mann-Whitney U, Cohen's d) -- reported with the honest caveat that
    PED00443 vs PED00192 is "fully generative vs experimentally-constrained"
  * conformational embeddings (PCA, t-SNE) of per-conformer descriptors

All figures go to outputs/figures/, the numeric report to outputs/eda_report.json.
Every figure function is isolated so a single failure cannot abort the run.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .constants import AGGREGATION_PRONE_MOTIFS, HYDROPATHY

# Per-conformer descriptor columns used for PCA/t-SNE.
EMBED_FEATURES = [
    "rg_equal_weight",
    "end_to_end",
    "mean_degree",
    "graph_density",
    "n_contacts",
    "apr_mean_rsa_0",
    "apr_mean_rsa_1",
    "mean_res_rsa",
    "mean_res_sasa",
    "mean_neighbors",
]

ENSEMBLE_COLORS = {"PED00422": "#1f77b4", "PED00192": "#ff7f0e", "PED00443": "#2ca02c"}
ENSEMBLE_LABELS = {
    "PED00422": "PED00422 (Tau-441, IDPConfGen)",
    "PED00192": "PED00192 (K18, Bayesian/exp-constrained)",
    "PED00443": "PED00443 (K18, idpGAN)",
}


class EnsembleData:
    """Lazy-ish container for one ensemble's Phase 1 outputs."""

    def __init__(self, out_dir: str, eid: str):
        self.eid = eid
        self.npz = np.load(os.path.join(out_dir, eid, "ensemble_data.npz"))
        with open(os.path.join(out_dir, eid, "summary.json")) as f:
            self.summary = json.load(f)
        self.n = int(self.npz["rg_equal_weight"].shape[0])
        self.n_res = int(self.npz["tau_resseq"].shape[0])
        self.npz_handle = self.npz

    # --- per-conformer scalars -------------------------------------------
    @property
    def rg(self) -> np.ndarray:
        return self.npz["rg_equal_weight"]

    @property
    def e2e(self) -> np.ndarray:
        return self.npz["end_to_end"]

    @property
    def apr(self) -> np.ndarray:
        return self.npz["apr_mean_rsa"]  # (M, 2): VQIINK, VQIVYK

    @property
    def degree(self) -> np.ndarray:
        return self.npz["mean_degree"]

    @property
    def density(self) -> np.ndarray:
        return self.npz["graph_density"]

    @property
    def contacts(self) -> np.ndarray:
        return self.npz["n_contacts"]

    @property
    def n_edges(self) -> np.ndarray:
        return self.npz["n_edges"]

    # --- per-conformer / per-residue arrays -------------------------------
    @property
    def res_rsa(self) -> np.ndarray:
        return self.npz["res_rsa"]  # (M, n_res)

    @property
    def res_sasa(self) -> np.ndarray:
        return self.npz["res_sasa"]

    @property
    def neighbor_counts(self) -> np.ndarray:
        return self.npz["neighbor_counts"]

    @property
    def contact_map(self) -> np.ndarray:
        return self.npz["contact_map"]  # (M, n_res, n_res) uint8

    @property
    def aa_codes(self) -> np.ndarray:
        return self.npz["aa_codes"]

    @property
    def tau_resseq(self) -> np.ndarray:
        return self.npz["tau_resseq"]

    def feature_matrix(self) -> np.ndarray:
        """(M, F) standardized per-conformer descriptor matrix for embedding."""
        cols = [
            self.rg,
            self.e2e,
            self.degree,
            self.density,
            self.contacts.astype(float),
            self.apr[:, 0],
            self.apr[:, 1],
            self.res_rsa.mean(axis=1),
            self.res_sasa.mean(axis=1),
            self.neighbor_counts.mean(axis=1),
        ]
        X = np.column_stack(cols)
        return StandardScaler().fit_transform(X)

    def motif_mask(self, motif: str) -> np.ndarray:
        """Boolean per-residue mask for an APR motif in Tau numbering."""
        s, e = AGGREGATION_PRONE_MOTIFS[motif]
        return (self.tau_resseq >= s) & (self.tau_resseq <= e)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD); sign: positive = a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def compare_pairs(datasets: Sequence[EnsembleData], metric_getter, names: Sequence[str],
                  desc: str) -> List[dict]:
    """Pairwise distributional comparisons (KS, MWU, Cohen's d) for one metric."""
    out = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a, b = metric_getter(datasets[i]), metric_getter(datasets[j])
            ks = stats.ks_2samp(a, b)
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            out.append({
                "metric": desc,
                "group_a": names[i], "group_b": names[j],
                "n_a": int(len(a)), "n_b": int(len(b)),
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                "mw_u": float(mw.statistic), "mw_p": float(mw.pvalue),
                "cohens_d": cohens_d(a, b),
            })
    return out


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apr_span(tau_resseq: np.ndarray) -> List[tuple]:
    """(start, end) APR windows in the plotting coordinate (tau numbering)."""
    return [(s, e) for s, e in AGGREGATION_PRONE_MOTIFS.values()
            if e >= tau_resseq.min() and s <= tau_resseq.max()]


def fig_ensemble_sizes(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([d.eid for d in datasets], [d.n for d in datasets],
           color=[ENSEMBLE_COLORS[d.eid] for d in datasets])
    ax.set_ylabel("conformers"); ax.set_title("Ensemble sizes")
    for d in datasets:
        ax.text(d.eid, d.n + 8, str(d.n), ha="center")
    p = os.path.join(out_dir, "fig_ensemble_sizes.png")
    _save(fig, p); return p


def fig_rg_distribution(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in datasets:
        ax.hist(d.rg, bins=40, density=True, alpha=0.45, histtype="stepfilled",
                color=ENSEMBLE_COLORS[d.eid], label=ENSEMBLE_LABELS[d.eid])
    ax.axvspan(65, 69, color="gray", alpha=0.25, label="Tau-441 SAXS 65-69 A")
    ax.axvline(38, color="k", ls="--", lw=1, label="K18 SAXS ~38 A")
    ax.set_xlabel("radius of gyration, Rg (A)"); ax.set_ylabel("density")
    ax.set_title("Rg distribution"); ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_rg_distribution.png")
    _save(fig, p); return p


def fig_e2e_distribution(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in datasets:
        ax.hist(d.e2e, bins=40, density=True, alpha=0.45, histtype="stepfilled",
                color=ENSEMBLE_COLORS[d.eid], label=ENSEMBLE_LABELS[d.eid])
    ax.set_xlabel("end-to-end distance (A)"); ax.set_ylabel("density")
    ax.set_title("End-to-end distance distribution"); ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_e2e_distribution.png")
    _save(fig, p); return p


def fig_total_sasa_distribution(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in datasets:
        tot = d.res_sasa.sum(axis=1)
        ax.hist(tot, bins=40, density=True, alpha=0.45, histtype="stepfilled",
                color=ENSEMBLE_COLORS[d.eid], label=ENSEMBLE_LABELS[d.eid])
    ax.set_xlabel("total heavy-atom SASA (A^2)"); ax.set_ylabel("density")
    ax.set_title("Total SASA distribution"); ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_total_sasa_distribution.png")
    _save(fig, p); return p


def fig_rsa_profile(datasets, out_dir) -> str:
    """Per-residue mean +/- SD relative SASA vs Tau position (APRs shaded)."""
    n_plots = len(datasets)
    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 3.2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for ax, d in zip(axes, datasets):
        mean = d.res_rsa.mean(axis=0)
        sd = d.res_rsa.std(axis=0)
        x = d.tau_resseq
        ax.plot(x, mean, lw=1.2, color=ENSEMBLE_COLORS[d.eid])
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.2, color=ENSEMBLE_COLORS[d.eid])
        for (s, e) in _apr_span(d.tau_resseq):
            ax.axvspan(s, e, color="crimson", alpha=0.22)
            ax.text((s + e) / 2, ax.get_ylim()[1] * 0.95, "APR", ha="center",
                    fontsize=7, color="crimson")
        ax.set_ylabel("mean rSA (+/- SD)")
        ax.set_title(ENSEMBLE_LABELS[d.eid], fontsize=9)
    axes[-1].set_xlabel("Tau-441 residue position")
    fig.suptitle("Per-residue relative solvent accessibility", y=1.0)
    p = os.path.join(out_dir, "fig_rsa_profile.png")
    _save(fig, p); return p


def fig_flexibility(datasets, out_dir) -> str:
    """Conformer-to-conformer flexibility: SD of per-residue rSA."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for d in datasets:
        flex = d.res_rsa.std(axis=0)
        ax.plot(d.tau_resseq, flex, lw=1.1, color=ENSEMBLE_COLORS[d.eid],
                label=ENSEMBLE_LABELS[d.eid])
    for (s, e) in _apr_span(datasets[0].tau_resseq):
        ax.axvspan(s, e, color="crimson", alpha=0.15)
    ax.set_xlabel("Tau-441 residue position"); ax.set_ylabel("SD of per-residue rSA")
    ax.set_title("Residue flexibility (rSA variability across conformers)")
    ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_flexibility.png")
    _save(fig, p); return p


def fig_contact_maps(datasets, out_dir) -> str:
    """Average heavy-atom contact maps (probability of residue pair contact)."""
    sel = [d for d in datasets if d.eid in ("PED00192", "PED00443")]
    fig, axes = plt.subplots(1, len(sel), figsize=(7 * len(sel), 6))
    if len(sel) == 1:
        axes = [axes]
    for ax, d in zip(axes, sel):
        cm = d.contact_map.mean(axis=0).astype(float)
        im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{d.eid}: P(residue pair contact)", fontsize=9)
        ax.set_xlabel("residue (file numbering)"); ax.set_ylabel("residue")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Average contact maps (5 A heavy-atom cutoff)")
    p = os.path.join(out_dir, "fig_contact_maps.png")
    _save(fig, p); return p


def fig_degree_distribution(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in datasets:
        ax.hist(d.degree, bins=30, density=True, alpha=0.45, histtype="stepfilled",
                color=ENSEMBLE_COLORS[d.eid], label=ENSEMBLE_LABELS[d.eid])
    ax.set_xlabel("mean node degree"); ax.set_ylabel("density")
    ax.set_title("Residue-graph mean-degree distribution"); ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_degree_distribution.png")
    _save(fig, p); return p


def fig_graph_density(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for d in datasets:
        ax.hist(d.density, bins=30, density=True, alpha=0.45, histtype="stepfilled",
                color=ENSEMBLE_COLORS[d.eid], label=ENSEMBLE_LABELS[d.eid])
    ax.set_xlabel("graph density"); ax.set_ylabel("density")
    ax.set_title("Graph density distribution"); ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_graph_density.png")
    _save(fig, p); return p


def fig_residue_frequency(datasets, out_dir) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    for i, d in enumerate(datasets):
        counts = {aa: 0 for aa in aas}
        for code in d.aa_codes:
            if code in counts:
                counts[code] += 1
        freq = np.asarray([counts[a] / d.n_res for a in aas])
        ax.bar(np.arange(20) + (i - 1) * width, freq, width,
               color=ENSEMBLE_COLORS[d.eid], label=d.eid)
    ax.set_xticks(np.arange(20)); ax.set_xticklabels(aas)
    ax.set_ylabel("fraction of residues"); ax.set_title("Residue frequency")
    ax.legend(fontsize=7)
    p = os.path.join(out_dir, "fig_residue_frequency.png")
    _save(fig, p); return p


def fig_charge_hydropathy(datasets, out_dir) -> str:
    """Net charge and hydropathy profiles (LLPS-relevant) for the K18 data."""
    k18 = [d for d in datasets if d.eid in ("PED00192", "PED00443")]
    if not k18:
        return ""
    seq = "".join(k18[0].aa_codes.tolist())
    x = k18[0].tau_resseq
    charge = np.array([(s.count("K") + s.count("R")) - (s.count("D") + s.count("E"))
                       for s in [seq]])
    charge_all = np.array([1 if a in "KR" else (-1 if a in "DE" else 0) for a in seq],
                          dtype=float)
    hyd = np.array([HYDROPATHY.get(a, 0.0) for a in seq])
    w = 10
    kern = np.ones(w) / w
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(x, np.convolve(charge_all, kern, mode="same"), color="#d62728",
             label="net charge (window 10)")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Tau-441 residue position"); ax1.set_ylabel("window net charge")
    ax2 = ax1.twinx()
    ax2.plot(x, np.convolve(hyd, kern, mode="same"), color="#9467bd",
             label="hydropathy (window 10)")
    ax2.set_ylabel("window hydropathy (Kyte-Doolittle)")
    ax1.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax1.set_title(f"K18 sequence profiles ({k18[0].eid}) -- charge balance is LLPS-relevant")
    p = os.path.join(out_dir, "fig_charge_hydropathy.png")
    _save(fig, p); return p


def fig_embedding(datasets, out_dir, method: str = "pca") -> str:
    """2D embedding of per-conformer descriptors, colored by ensemble."""
    X_all = np.vstack([d.feature_matrix() for d in datasets])
    labels = np.concatenate([np.full(d.n, d.eid) for d in datasets])
    if method == "pca":
        emb = PCA(n_components=2, random_state=0).fit_transform(X_all)
    else:
        emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=min(40, len(X_all) - 1),
                   learning_rate="auto").fit_transform(X_all)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for eid in ENSEMBLE_COLORS:
        m = labels == eid
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.55, color=ENSEMBLE_COLORS[eid],
                   label=ENSEMBLE_LABELS[eid])
    ax.set_xlabel(f"{method.upper()} 1"); ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"Conformational embedding ({method.upper()}, standardized descriptors)")
    ax.legend(fontsize=7, markerscale=2)
    p = os.path.join(out_dir, f"fig_embedding_{method}.png")
    _save(fig, p); return p


def _metric_table(datasets) -> dict:
    return {
        d.eid: {
            "n_conformers": d.n,
            "rg_mean": float(d.rg.mean()), "rg_std": float(d.rg.std()),
            "e2e_mean": float(d.e2e.mean()), "e2e_std": float(d.e2e.std()),
            "apr1_vqiink_mean_rsa": float(d.apr[:, 0].mean()),
            "apr2_vqivyk_mean_rsa": float(d.apr[:, 1].mean()),
            "mean_degree": float(d.degree.mean()),
            "graph_density": float(d.density.mean()),
            "mean_n_contacts": float(d.contacts.mean()),
            "mean_residue_rsa": float(d.res_rsa.mean()),
            "mean_residue_sasa": float(d.res_sasa.mean()),
        }
        for d in datasets
    }


def run_eda(out_dir: str = "outputs", fig_dir: Optional[str] = None,
            ensemble_ids: Optional[Sequence[str]] = None) -> dict:
    """Run the full EDA; returns the report dict."""
    ids = list(ensemble_ids or ["PED00422", "PED00192", "PED00443"])
    datasets = [EnsembleData(out_dir, eid) for eid in ids]
    fig_dir = fig_dir or os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    figures = {}
    for fn in [
        fig_ensemble_sizes, fig_rg_distribution, fig_e2e_distribution,
        fig_total_sasa_distribution, fig_rsa_profile, fig_flexibility,
        fig_contact_maps, fig_degree_distribution, fig_graph_density,
        fig_residue_frequency, fig_charge_hydropathy,
    ]:
        try:
            figures[fn.__name__] = fn(datasets, fig_dir)
        except Exception as exc:  # isolated failures must not abort the run
            figures[fn.__name__] = f"FAILED: {exc}"
    for method in ("pca", "tsne"):
        try:
            figures[f"fig_embedding_{method}"] = fig_embedding(datasets, fig_dir, method)
        except Exception as exc:
            figures[f"fig_embedding_{method}"] = f"FAILED: {exc}"

    # pairwise statistical comparisons
    names = [d.eid for d in datasets]
    tests = []
    tests += compare_pairs(datasets, lambda d: d.rg, names, "radius_of_gyration")
    tests += compare_pairs(datasets, lambda d: d.e2e, names, "end_to_end")
    tests += compare_pairs(datasets, lambda d: d.apr[:, 0], names, "apr1_vqiink_mean_rsa")
    tests += compare_pairs(datasets, lambda d: d.apr[:, 1], names, "apr2_vqivyk_mean_rsa")
    tests += compare_pairs(datasets, lambda d: d.degree, names, "mean_degree")
    tests += compare_pairs(datasets, lambda d: d.density, names, "graph_density")

    report = {
        "phase": 2,
        "note": ("PED00443 vs PED00192 is a 'fully generative (idpGAN)' vs "
                 "'experimentally-constrained (Bayesian reweighting of MD with "
                 "NMR/SAXS)' comparison, not experimental vs generated. Sample "
                 "sizes differ (1000 vs 75 vs 1000)."),
        "metrics": _metric_table(datasets),
        "pairwise_tests": tests,
        "figures": figures,
    }
    with open(os.path.join(out_dir, "eda_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def print_report(report: dict) -> None:
    print("=" * 74)
    print("Phase 2 EDA summary")
    print("=" * 74)
    for eid, m in report["metrics"].items():
        print(f"{eid}: n={m['n_conformers']}  Rg={m['rg_mean']:.1f}+/-{m['rg_std']:.1f} A  "
              f"e2e={m['e2e_mean']:.1f} A  APR1 rSA={m['apr1_vqiink_mean_rsa']:.3f}  "
              f"APR2 rSA={m['apr2_vqivyk_mean_rsa']:.3f}  deg={m['mean_degree']:.2f}")
    print("--- pairwise comparisons (KS p / Cohen's d) ---")
    for t in report["pairwise_tests"]:
        print(f"{t['metric']:26s} {t['group_a']} vs {t['group_b']}: "
              f"KS_p={t['ks_p']:.2e} d={t['cohens_d']:+.2f}")
    print(f"figures -> {report['figures'].get('fig_rg_distribution', '')}")
