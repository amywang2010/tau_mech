"""Pipeline configuration.

All numeric parameters that affect results are centralized here (with the
values actually used serialized to outputs/config_used.json at run time), so
the entire study is reproducible from a single recorded configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from .constants import ENSEMBLES
from .features import GraphConfig


@dataclass
class PipelineConfig:
    """Global pipeline configuration."""

    # data
    data_dir: str = ".."                      # directory containing the *.tar.gz archives
    output_dir: str = "outputs"               # where processed outputs are written

    # parsing
    heavy_only: bool = True                   # drop hydrogens
    with_occupancy: bool = False              # parse occupancy/bfactor columns

    # SASA
    probe_radius: float = 1.4                 # Angstrom
    n_probe_points: int = 480

    # contact / neighbor metrics (documented cutoffs, kept in sync with the
    # graph edge cutoff so a reviewer can reproduce every reported number)
    contact_cutoff: float = 5.0               # Angstrom (heavy-atom contact map)
    neighbor_cutoff: float = 8.0              # Angstrom (per-residue neighbor counts)

    # graphs
    graph: GraphConfig = field(default_factory=GraphConfig)

    # ensemble selection
    ensemble_ids: Optional[list] = None       # None -> all in ENSEMBLES
    n_models: Optional[int] = None            # None -> all models
    start_at: int = 0                         # skip N models (debug)

    def resolved_ensembles(self) -> Dict[str, dict]:
        ids = self.ensemble_ids or list(ENSEMBLES.keys())
        return {k: ENSEMBLES[k] for k in ids if k in ENSEMBLES}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["graph"] = asdict(self.graph)
        return d
