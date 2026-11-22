"""Load each dynamics model only when it is requested.

NGFF can run on point clouds without loading the Gaussian dataset, video
pipeline, or the optional torch_geometric baseline.
"""

from importlib import import_module

_EXPORTS = {
    "PointTransformer": (".pointformer", "PointTransformer"),
    "GCN": (".gcn", "GCN"),
    "IN": (".interaction_net", "IN"),
    "MPMSimulator": (".mpm_fake", "MPMSimulator"),
    "NGFFobj": (".ngff", "NGFFobj"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

