"""TRAIN-only native mesh export and explicitly shared instance-ID bindings.

Native meshes are a declared simulation-assisted TRAIN/CAD prior. They are
not described as RGB-D reconstructions. Only the exporter reads body poses;
the runtime binding helper reads names and instance handles, never poses.
Neither function supplies native target state, success sensors or schedules
to a policy. Surface samples preserve the source triangular surface; collision
assets retain native physical triangles rather than filling cavities with a
hull. TRAIN flags separate rendered surfaces, invisible collision meshes, and
nonphysical helpers; current flags never extend frozen visual bindings.
"""
from collections.abc import Mapping
import hashlib
import json

import numpy as np

from .native_tasks import resolve_native_roles
from .protocol import MultitaskProtocolError
from .safe_native_mesh import read_native_mesh, VERSION as MESH_READER_VERSION
from .component_flags import (SCHEMA as COMPONENT_ROLE_SCHEMA, partition_components,
                              read_native_component_flags, bind_frozen_visual_components,
                              component_roles_from_native_values)

ASSET_VERSION = 'r10_shared_native_TRAIN_mesh_v2'


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _check_asset(asset):
    if (not isinstance(asset, Mapping) or asset.get('version') != ASSET_VERSION
            or asset.get('split') != 'train' or asset.get('shared_with_baseline') is not True
            or not asset.get('bodies') or 'asset_sha256' not in asset
            or asset.get('component_role_schema') != COMPONENT_ROLE_SCHEMA):
        raise MultitaskProtocolError('declare an immutable shared TRAIN native mesh asset')
    payload = {key: value for key, value in asset.items() if key != 'asset_sha256'}
    if _sha(payload) != asset['asset_sha256']:
        raise MultitaskProtocolError('TRAIN mesh asset fingerprint differs')
    for name, body in asset['bodies'].items():
        try:
            classification = partition_components(body['owned_component_annotations'])
            for row in body['owned_component_annotations']:
                raw = component_roles_from_native_values(row['native_getter_values'])
                if raw['flags'] != row['flags'] or raw['native_true_values'] != row['native_true_values']:
                    raise ValueError('normalized flag annotations disagree with native raw getter values')
        except (KeyError, ValueError) as error:
            raise MultitaskProtocolError('TRAIN component flags missing/invalid: ' + name) from error
        if (body.get('visual_component_names') != classification['visual_component_names']
                or body.get('collision_component_names') != classification['collision_component_names']
                or [row['name'] for row in body['components']] != classification['visual_component_names']
                or body.get('empty_visual_role') != classification['empty_visual_role']
                or body.get('empty_collision_role') != classification['empty_collision_role']
                or body.get('visual_geometry_status') != ('native_nonrenderable' if
                    classification['empty_visual_role'] else 'renderable_components')
                or (classification['empty_collision_role'] and
                    (body['collision_vertices_body'] or body['collision_triangles']))):
            raise MultitaskProtocolError('TRAIN visual/physical component partition differs: ' + name)


def _owned_shapes(env, role_ids, *, shape_type=None):
    if shape_type is None:
        from pyrep.const import ObjectType
        shape_type = ObjectType.SHAPE
    roles = resolve_native_roles(env.task, env.spec, scene_object_resolver=env.resolve_scene_object)
    physical = {role.name for role in env.spec.roles if role.kind == 'body'} - set(env.spec.audit_only_goal_roles)
    selected = tuple(sorted(physical) if role_ids is None else role_ids)
    if not selected or len(set(selected)) != len(selected) or set(selected) - physical:
        raise MultitaskProtocolError('geometry assets must use unique non-audit physical TaskSpec roles')
    # Partition by all declared body roots even when exporting only a subset:
    # plate/spokes/cups remain independent articulated or attachable bodies.
    trees = {}
    for name in physical:
        root = roles[name]
        tree = {int(shape.get_handle()): shape
                for shape in root.get_objects_in_tree(object_type=shape_type)}
        if hasattr(root, 'get_mesh_data'):
            tree[int(root.get_handle())] = root
        trees[name] = tree
    owned = {}
    for name in selected:
        root_handle = int(roles[name].get_handle())
        tree = dict(trees[name])
        for other, subtree in trees.items():
            if other != name and int(roles[other].get_handle()) != root_handle:
                # A named child body's whole subtree belongs to that child.
                # A sibling does not intersect; an ancestor contains our root
                # and must not subtract our own tree.
                if root_handle not in subtree:
                    for handle in subtree:
                        tree.pop(handle, None)
        if not tree:
            raise MultitaskProtocolError('declared physical role has no owned shape mesh: ' + name)
        owned[name] = tuple(sorted(tree.values(), key=lambda shape: shape.get_name()))
    handles = [int(shape.get_handle()) for shapes in owned.values() for shape in shapes]
    if len(handles) != len(set(handles)):
        raise MultitaskProtocolError('native role geometry ownership is ambiguous')
    return roles, owned


def _surface_points(vertices, triangles, count):
    faces = np.asarray(vertices)[np.asarray(triangles)]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise MultitaskProtocolError('native TRAIN mesh has no finite nondegenerate surface')
    cumulative = np.cumsum(areas) / areas.sum()
    index = np.searchsorted(cumulative, (np.arange(count) + .5) / count)
    chosen = faces[np.minimum(index, len(faces) - 1)]
    # Deterministic area-weighted triangular sampling; no RNG/native stepping.
    u = np.sqrt(np.mod((np.arange(count) + .5) * .6180339887498949, 1.))
    v = np.mod((np.arange(count) + .5) * .4142135623730951, 1.)
    return ((1 - u)[:, None] * chosen[:, 0] + (u * (1 - v))[:, None] * chosen[:, 1]
            + (u * v)[:, None] * chosen[:, 2])


def export_train_geometry(env, source, *, role_ids=None, surface_samples=384, shape_type=None):
    """Read TRAIN native geometry after a source-seeded development reset.

    Returns JSON-serializable data with no world poses or motion schedules.
    This privileged TRAIN call is forbidden on VAL/TEST runtimes. The collector
    must write/freeze this artifact before the final policy uses its geometry.
    """
    if (getattr(env, 'development_split', None) != 'train' or source.get('split', 'train') != 'train'
            or source.get('task') != env.spec.task_name
            or source.get('raw_episode_id') != env.development_source.get('raw_episode_id')
            or not getattr(env, '_aligned', False) or not hasattr(env, 'reset_receipt')):
        raise MultitaskProtocolError('native mesh/role supervision is TRAIN-only after a completed reset')
    if type(surface_samples) is not int or surface_samples < 6:
        raise MultitaskProtocolError('declare at least six surface samples per component')
    before = env.physics_time(), float(env.task.t)
    roles, owned = _owned_shapes(env, role_ids, shape_type=shape_type)
    bodies = {}
    for name, shapes in owned.items():
        body_world = np.asarray(roles[name].get_matrix())
        inverse = np.linalg.inv(body_world)
        components, all_vertices, all_triangles, offset, annotations = [], [], [], 0, []
        for shape in shapes:
            try:
                flags = read_native_component_flags(shape)
            except (AttributeError, ValueError) as error:
                raise MultitaskProtocolError('native TRAIN component flag evidence unavailable: ' + shape.get_name()) from error
            annotations.append(dict(name=shape.get_name(), flags=flags['flags'],
                native_getter_values=flags['native_getter_values'], native_true_values=flags['native_true_values']))
            if not flags['visual'] and not flags['physical_collision']:
                continue
            vertices, triangles, _ = read_native_mesh(shape)
            vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
            triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
            child = inverse @ np.asarray(shape.get_matrix())
            local = vertices @ child[:3, :3].T + child[:3, 3]
            if flags['visual']:
                components.append(dict(name=shape.get_name(),
                    TRAIN_instance_ids=[int(shape.get_handle())],
                    points_body=_surface_points(local, triangles, surface_samples).tolist()))
            if flags['physical_collision']:
                all_vertices.append(local);all_triangles.append(triangles + offset);offset += len(local)
        classification = partition_components(annotations)
        bodies[name] = dict(components=components,
            owned_component_annotations=annotations,
            visual_component_names=classification['visual_component_names'],
            collision_component_names=classification['collision_component_names'],
            empty_visual_role=classification['empty_visual_role'],
            empty_collision_role=classification['empty_collision_role'],
            visual_geometry_status='native_nonrenderable' if classification['empty_visual_role'] else 'renderable_components',
            collision_vertices_body=np.concatenate(all_vertices).tolist() if all_vertices else [],
            collision_triangles=np.concatenate(all_triangles).tolist() if all_triangles else [],
            frame_definition='native TRAIN role object frame; transformed component vertices only',
            orientation_observability_requires_explicit_annotation=True)
    payload = dict(version=ASSET_VERSION, split='train',
        task=env.spec.task_name, source_group=f"train:{source['task']}:{source['raw_episode_id']}",
        source_hdf5_sha256=env.reset_receipt['source_hdf5_sha256'],
        native_task_source_sha256=env.spec.native_source_sha256,
        reset_kind=env.reset_receipt['reset_kind'], source_original_motion_replayed=False,
        geometry_source='native TRAIN-generated scene meshes; declared shared CAD prior',
        native_mesh_reader=MESH_READER_VERSION,
        component_role_schema=COMPONENT_ROLE_SCHEMA,
        component_flag_source='raw native TRAIN is_renderable/is_collidable/is_respondable getters, before any filtering',
        visual_geometry_rule='renderable', collision_geometry_rule='collidable OR respondable',
        component_flags_are_TRAIN_annotations_not_runtime_visibility=True,
        shared_with_baseline=True, world_pose_or_motion_schedule_in_asset=False,
        mask_encoding='instance_handle = round(R*255) + 256*round(G*255) + 65536*round(B*255)',
        component_ownership='nearest declared physical body root; articulated/attachable named children excluded',
        bodies=bodies)
    if env.physics_time() != before[0] or float(env.task.t) != before[1]:
        raise MultitaskProtocolError('TRAIN geometry read advanced physics/task clock')
    payload['asset_sha256'] = _sha(payload)
    return payload


def declared_instance_bindings(env, asset, *, shared_permission=False, shape_type=None):
    """Bind shared semantic component names to current mask IDs; no pose reads.

    Native IDs need not remain numerically identical between fresh processes.
    Both methods must receive and disclose the same semantic-ID permission.
    This helper never discovers unknown geometry or calibrates from VAL/TEST.
    """
    _check_asset(asset)
    if shared_permission is not True or asset['task'] != env.spec.task_name:
        raise MultitaskProtocolError('current semantic/instance bindings require declared common permission')
    before = env.physics_time(), float(env.task.t)
    _, owned = _owned_shapes(env, tuple(asset['bodies']), shape_type=shape_type)
    result = {}
    for name, body in asset['bodies'].items():
        current = {shape.get_name(): int(shape.get_handle()) for shape in owned[name]}
        try:
            result[name] = bind_frozen_visual_components(body['owned_component_annotations'],
                {component: (handle,) for component, handle in current.items()})
        except ValueError as error:
            raise MultitaskProtocolError('current declared component topology differs from frozen TRAIN mesh') from error
    if env.physics_time() != before[0] or float(env.task.t) != before[1]:
        raise MultitaskProtocolError('instance-ID binding advanced physics/task clock')
    return result


def body_templates_from_asset(asset, instance_bindings, *, orientation_observability,
                              sphere_radii=None, symmetry_axes=None, axis_polarity=None,
                              finite_rotation_symmetries=None,cup_feature_models=None):
    """Build data-only registration templates with explicit symmetry annotation."""
    from .perception import BodyTemplate, GeometryComponent, PriorProvenance
    _check_asset(asset)
    if set(instance_bindings) != set(asset['bodies']) or set(orientation_observability) != set(asset['bodies']):
        raise MultitaskProtocolError('bind every TRAIN role and explicitly annotate its rotation observability')
    axis_roles = {name for name, observability in orientation_observability.items() if observability == 'axis'}
    finite_rotation_symmetries = {} if finite_rotation_symmetries is None else finite_rotation_symmetries
    cup_feature_models = {} if cup_feature_models is None else cup_feature_models
    if (not isinstance(cup_feature_models,Mapping) or set(cup_feature_models)-{'cup0','cup1','cup2'}
            or set(cup_feature_models)-set(asset['bodies'])):
        raise MultitaskProtocolError('cup feature priors must name actual frozen TRAIN cup roles')
    if (not isinstance(finite_rotation_symmetries, Mapping) or set(finite_rotation_symmetries)-{'ring'}
            or set(finite_rotation_symmetries)-set(asset['bodies'])):
        raise MultitaskProtocolError('finite rotation evidence is restricted to the certified TRAIN ring role')
    symmetry_axes = {} if symmetry_axes is None else symmetry_axes
    if not isinstance(symmetry_axes, Mapping) or set(symmetry_axes) != axis_roles:
        raise MultitaskProtocolError('provide an explicit body-local symmetry axis for every and only axis-observed role')
    if axis_polarity is not None and (not isinstance(axis_polarity,Mapping) or set(axis_polarity)!=axis_roles
            or any(value not in ('directed','undirected') for value in axis_polarity.values())):
        raise MultitaskProtocolError('axis polarity must explicitly match every and only axis-observed role')
    for name, value in symmetry_axes.items():
        vector = np.asarray(value, dtype=float)
        if (vector.shape != (3,) or not np.isfinite(vector).all()
                or abs(float(np.linalg.norm(vector))-1.) > 1e-6):
            raise MultitaskProtocolError('symmetry axes must be finite body-local unit vectors: ' + name)
    templates = []
    sphere_radii = {} if sphere_radii is None else sphere_radii
    for name, body in asset['bodies'].items():
        provenance = PriorProvenance('declared_cad', 'frozen TRAIN native mesh asset:' + asset['asset_sha256'],
            body['frame_definition'], True, 'train', (asset['source_group'],), asset['asset_sha256'])
        components = tuple(GeometryComponent(component['name'],
            tuple(instance_bindings[name][component['name']]), component['points_body'])
            for component in body['components'])
        kwargs = dict(orientation_observability=orientation_observability[name], sphere_radius=sphere_radii.get(name))
        if body['empty_visual_role']:
            kwargs['visual_geometry_status'] = 'native_nonrenderable'
        if name in symmetry_axes:
            kwargs['symmetry_axis'] = symmetry_axes[name]
        if axis_polarity is not None and name in axis_polarity:
            kwargs['axis_polarity'] = axis_polarity[name]
        if name in finite_rotation_symmetries:
            kwargs['finite_rotation_symmetry'] = finite_rotation_symmetries[name]
        if name in cup_feature_models:
            kwargs['cup_feature_model'] = cup_feature_models[name]
        templates.append(BodyTemplate(name, components, provenance, **kwargs))
    return tuple(templates)


def register_collision_asset(map_service, asset, *, orientation_observability,axis_swept_assets=None,
                             finite_rotation_symmetries=None):
    """Register TRAIN geometry with explicit per-role rotation uncertainty.

    Full orientation preserves the original cavities. Axis/none default to a
    rotation-invariant cube. Only an explicit bin/direct-X frozen-asset permit
    enables the inflated yaw sweep; every query still needs current axis
    readiness, enforced by the runtime collision port. The source is unchanged.
    """
    from .collision_geometry import collision_geometry_for_orientation, SCHEMA
    _check_asset(asset)
    if (not isinstance(orientation_observability, Mapping)
            or set(orientation_observability) != set(asset['bodies'])
            or any(value not in ('full','axis','none') for value in orientation_observability.values())):
        raise MultitaskProtocolError('explicit collision orientation annotation required for every TRAIN body')
    axis_swept_assets={} if axis_swept_assets is None else axis_swept_assets
    finite_rotation_symmetries={} if finite_rotation_symmetries is None else finite_rotation_symmetries
    if (not isinstance(finite_rotation_symmetries, Mapping) or set(finite_rotation_symmetries)-{'ring'}
            or set(finite_rotation_symmetries)-set(asset['bodies'])):
        raise MultitaskProtocolError('finite collision quotient is restricted to the certified TRAIN ring role')
    if (not isinstance(axis_swept_assets,Mapping) or set(axis_swept_assets)-{'bin'}
            or set(axis_swept_assets)-set(asset['bodies'])
            or any(orientation_observability[name]!='axis' for name in axis_swept_assets)):
        raise MultitaskProtocolError('optional directed-axis sweep permission is restricted to an explicitly axis-observed bin')
    receipts = {}
    for name, body in asset['bodies'].items():
        if body['empty_collision_role']:
            if name in axis_swept_assets:
                raise MultitaskProtocolError('axis sweep cannot invent a mesh for a declared noncolliding role')
            declaration = getattr(map_service, 'declare_noncolliding_geometry', None)
            if declaration is None:
                raise MultitaskProtocolError('map must explicitly accept TRAIN noncolliding role: ' + name)
            receipt = dict(schema=SCHEMA, orientation_observability=orientation_observability[name],
                collision_geometry_kind='declared_noncolliding', construction_rule='no owned collidable OR respondable TRAIN component',
                TRAIN_asset_sha256=asset['asset_sha256'], original_mesh_sha256=None,
                registered_mesh_sha256=None, cavities_preserved=None, vertex_count=0,triangle_count=0,
                native_pose_or_future_state_used=False, TRAIN_asset_unchanged=True)
            declaration(name, provenance='TRAIN_geometry_model', source_sha256=asset['asset_sha256'],
                        collision_geometry_receipt=receipt)
            receipts[name] = receipt
            continue
        if name in finite_rotation_symmetries:
            from .rotation_equivalence import VerifiedFiniteSymmetry
            symmetry=finite_rotation_symmetries[name]
            if (not isinstance(symmetry,VerifiedFiniteSymmetry) or orientation_observability[name]!='none'
                    or symmetry.certificate.get('CAD_asset_sha256')!=asset['asset_sha256']
                    or symmetry.certificate.get('body_id')!=name):
                raise MultitaskProtocolError('finite collision quotient certificate differs from the frozen TRAIN ring CAD')
            geometry=collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],
                orientation_observability='full')
            geometry.receipt.update(orientation_observability='none',verified_finite_quotient_collision_pose_used=True,
                finite_rotation_certificate_sha256=symmetry.certificate['certificate_sha256'],
                finite_surface_equivalence_tolerance_m=symmetry.certificate['surface_tolerance_m'],
                full_orientation_promoted=False,
                construction_rule='original TRAIN triangles; use only under the current typed same-CAD finite rotation quotient')
        elif name in axis_swept_assets:
            from .bin_swept_assets import load_axis_swept_collision_geometry
            try:
                geometry=load_axis_swept_collision_geometry(axis_swept_assets[name],asset,role_id=name)
            except (ValueError,KeyError,TypeError,OSError) as error:
                raise MultitaskProtocolError('explicit TRAIN bin sweep asset failed verification') from error
        else:
            geometry = collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],
                orientation_observability=orientation_observability[name])
        receipt = dict(geometry.receipt, TRAIN_asset_sha256=asset['asset_sha256'])
        fallback = None
        if geometry.receipt['collision_geometry_kind'] != 'rotation_uncertainty_cube':
            cube = collision_geometry_for_orientation(body['collision_vertices_body'],body['collision_triangles'],
                orientation_observability='none')
            fallback = dict(vertices=cube.vertices,triangles=cube.triangles,
                collision_geometry_receipt=dict(cube.receipt,TRAIN_asset_sha256=asset['asset_sha256']))
        map_service.register_estimated_geometry(name, geometry.vertices, geometry.triangles,
            provenance='TRAIN_geometry_model', source_sha256=asset['asset_sha256'],
            collision_geometry_receipt=receipt, **({'fallback_mesh': fallback} if fallback is not None else {}))
        receipts[name] = receipt
    return receipts
