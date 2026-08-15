"""tau_mech: multiscale computational framework for Tau mechanobiology.

Phase 1 - data preprocessing: parse PED conformational ensembles, standardize
residue numbering, build residue-level geometric graphs, and compute
structure/aggregation descriptors (Rg, SASA, APR exposure, contacts).

See README.md for methods, decisions, assumptions, and limitations.
"""

__version__ = "0.1.0"

from .config import PipelineConfig          # noqa: F401
from .constants import ENSEMBLES           # noqa: F401
from .pipeline import process_ensemble     # noqa: F401
from .provenance import write_provenance   # noqa: F401
