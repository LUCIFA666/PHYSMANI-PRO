"""Explicit add-only closure; inherited models/proofs/math remain frozen."""
from pathlib import Path
import hashlib
from ..cube_corrected_live_20261104_v1.matched_sources import matched_corrected_source_map
ROOT=Path(__file__).resolve().parents[2]

def bounded_reobserve_source_map():
    sources=matched_corrected_source_map()
    names=('our_mani/cube_bounded_reobserve_live_20261104_v1/__init__.py',
        'our_mani/cube_bounded_reobserve_live_20261104_v1/supervision.py',
        'our_mani/cube_bounded_reobserve_live_20261104_v1/producer.py',
        'our_mani/cube_bounded_reobserve_live_20261104_v1/sources.py',
        'our_mani/cube_stale_reobserve_20261104_v1/__init__.py',
        'our_mani/cube_stale_reobserve_20261104_v1/recovery.py',
        'our_mani/generic_reference_mpc_20261104_v1/__init__.py',
        'our_mani/generic_reference_mpc_20261104_v1/reference_mpc.py',
        'our_mani/generic_reference_mpc_20261104_v1/contracts.py')
    for name in names:
        p=ROOT/name;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return sources
