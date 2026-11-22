"""TRAIN-mesh collision geometry with explicit rotation uncertainty.

This data-only helper never reads native state. Unknown or axis-only rotations
use a solid cube enclosing the original mesh under every rotation about its
declared body-frame origin. This fills cavities deliberately: it is a safe
uncertainty fallback, not an exact reconstruction of the object.
"""
from dataclasses import dataclass
import hashlib
import itertools

import numpy as np


SCHEMA = 'r10_TRAIN_collision_rotation_uncertainty_v1'


def mesh_sha256(vertices, triangles):
    """Same float32/int32 storage digest as SharedMultiObjectMap."""
    return hashlib.sha256(np.asarray(vertices, dtype=np.float32).tobytes()
                          + np.asarray(triangles, dtype=np.int32).tobytes()).hexdigest()


@dataclass(frozen=True)
class CollisionGeometry:
    vertices: np.ndarray
    triangles: np.ndarray
    receipt: dict


def collision_geometry_for_orientation(vertices, triangles, *, orientation_observability):
    """Retain full-oriented triangles, otherwise bound arbitrary rotation.

    No currently verified finite-quotient collision-pose contract is supplied
    by this interface. In particular an axis/gauge representative is never
    sufficient to select the original asymmetric mesh.
    """
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    if (orientation_observability not in ('full', 'axis', 'none')
            or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
            or vertices.dtype.kind not in 'fiu' or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('finite TRAIN mesh and explicit full/axis/none annotation required')
    original_vertices = np.asarray(vertices, float)
    radius = float(np.linalg.norm(original_vertices, axis=1).max())
    if not np.isfinite(radius) or radius <= 0 or radius >= np.finfo(np.float32).max:
        raise ValueError('TRAIN mesh requires a finite positive float32-representable origin radius')
    original_sha = mesh_sha256(vertices, triangles)
    if orientation_observability == 'full':
        result_vertices, result_triangles = vertices.copy(), triangles.copy()
        kind, cavities, half_extent = 'exact_TRAIN_triangles', True, None
        rule = 'original TRAIN-local triangle surface with fully observed body rotation; no hull or cavity filling'
    else:
        # One float32 ulp upwards also covers a rounded-down cast. The simulator
        # stores mesh vertices as float32; a nominal float64 radius alone could
        # shrink there and break the all-rotation inclusion claim.
        half_extent = float(np.nextafter(np.float32(radius), np.float32(np.inf)))
        if not np.isfinite(half_extent):
            raise ValueError('rotation bound overflows native mesh storage')
        result_vertices = np.asarray(list(itertools.product((-half_extent, half_extent), repeat=3)), dtype=np.float32)
        result_triangles = np.asarray(((0,1,3),(0,3,2),(4,6,7),(4,7,5),
            (0,4,5),(0,5,1),(2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)), dtype=np.int32)
        kind, cavities = 'rotation_uncertainty_cube', False
        rule = ('solid body-origin cube [-r,+r]^3, r=max TRAIN-local vertex norm rounded upward to float32; '
                'contains every rotation of every original triangle; no verified finite-quotient collision pose supplied')
    receipt = dict(schema=SCHEMA, orientation_observability=orientation_observability,
        collision_geometry_kind=kind, source_vertices_max_radius_m=radius,
        registered_cube_half_extent_m=half_extent, cavities_preserved=cavities,
        construction_rule=rule, original_mesh_sha256=original_sha,
        registered_mesh_sha256=mesh_sha256(result_vertices, result_triangles),
        vertex_count=len(result_vertices), triangle_count=len(result_triangles),
        unknown_rotation_identity_is_not_exact_CAD=orientation_observability != 'full',
        verified_finite_quotient_collision_pose_used=False,
        native_pose_or_future_state_used=False, TRAIN_asset_unchanged=True)
    return CollisionGeometry(result_vertices, result_triangles, receipt)
