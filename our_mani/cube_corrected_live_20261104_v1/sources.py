"""Explicit source closure for this independent typed composition."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]

def corrected_live_source_map():
    from ..local_mpc_control_20261103.public_candidate_stable_dwell_v72 import stable_dwell_producer_source_map
    from ..cube_motion_residual_20261104_v1.runtime import source_map as motion_sources
    from ..cube_candidate_matched_20261104_v2.data import source_map as feature_sources
    result=stable_dwell_producer_source_map()
    for mapping in (motion_sources(),feature_sources()):
        for p,h in mapping.items():
            path=Path(p);path=path if path.is_absolute() else ROOT/path
            result[str(path.resolve())]=h
    names=('__init__.py','motion_bridge.py','binding.py','measurement_bridge.py','reverification_bridge.py','candidate_runtime.py','producer.py','sources.py')
    for name in names:
        path=Path(__file__).with_name(name)
        result[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    # These classes are genuinely inherited/imported, so their modules belong
    # to this closure even though their former candidate provider is not used.
    inherited=('our_mani/cube_v72_live_20261104_v1/__init__.py',
               'our_mani/cube_v72_live_20261104_v1/producer.py',
               'our_mani/cube_v72_live_20261104_v1/backend.py',
               'our_mani/cube_v72_live_20261104_v1/candidate_runtime.py',
               'our_mani/candidate_transformer_cube_20261104_v1/__init__.py',
               'our_mani/candidate_transformer_cube_20261104_v1/model.py',
               'our_mani/candidate_transformer_cube_20261104_v1/losses.py')
    for name in inherited:
        path=ROOT/name
        result[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result
