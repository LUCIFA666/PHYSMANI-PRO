"""Separate add-only closure; previous eight live sources remain frozen."""
from pathlib import Path
import hashlib
from .sources import corrected_live_source_map
from ..cube_matched_candidate_20261104_v1.model import model_source_map

def matched_corrected_source_map():
    result=corrected_live_source_map()
    for p,h in model_source_map().items():result[str(Path(p).resolve())]=h
    for name in ('matched_sources.py','matched_candidate_runtime.py','matched_producer.py'):
        p=Path(__file__).with_name(name);result[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
