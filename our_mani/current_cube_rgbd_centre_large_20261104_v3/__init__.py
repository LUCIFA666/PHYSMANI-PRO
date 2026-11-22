"""Unqualified foreground-surface proposal research; no training/runtime entry."""
VERSION = 'large_CF_surface_proposal_current_RGBD_centre_20261104_v3'


def source_map():
    """Explicit frozen predecessor closure plus the four new prototype files."""
    import hashlib
    from pathlib import Path
    from ..current_cube_rgbd_centre_large_20261104_v2.training import training_source_map
    result = dict(training_source_map())
    root = Path(__file__).resolve().parents[2]
    for name in ('__init__.py', 'losses.py'):
        path = root/'our_mani/near_goal_depth_refiner_20261103_v5'/name
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('__init__.py', 'model.py', 'data.py', 'losses.py'):
        path = Path(__file__).with_name(name).resolve()
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
