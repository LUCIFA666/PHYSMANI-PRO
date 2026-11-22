"""Add-only runtime ABI correction; original trained sources/weights unchanged."""
from pathlib import Path
import hashlib
from ..cube_corrected_live_20261104_v1.matched_sources import matched_corrected_source_map
ROOT=Path(__file__).resolve().parents[2]

def anchor_consistent_corrected_source_map():
    sources=matched_corrected_source_map()
    for name in ('__init__.py','binding.py','producer.py','sources.py','reobserve_producer.py'):
        p=Path(__file__).with_name(name);sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    # New reobserve wrapper inherits its already stopped public supervisor and
    # runner helpers explicitly; it never edits or monkey-patches that family.
    from ..cube_bounded_reobserve_live_20261104_v1.sources import bounded_reobserve_source_map
    sources.update(bounded_reobserve_source_map())
    return sources
