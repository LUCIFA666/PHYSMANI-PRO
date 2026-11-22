"""Parent-side assembly; native semantic bindings never become policy poses."""
import json
from pathlib import Path

from ..contracts import Pose
from .evaluation import digest, file_sha
from .perception import MultiBodySE3Tracker, PriorProvenance, RigidChildConstraint, RegistrationConfig
from .protocol import MultitaskProtocolError


def artifact_path(root, pointer):
    root = Path(root).resolve()
    if not isinstance(pointer, dict) or not {'path', 'sha256'} <= set(pointer):
        raise MultitaskProtocolError('learned artifact needs its exact path and SHA256')
    path = (root / pointer['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha(path) != pointer['sha256']:
        raise MultitaskProtocolError('learned artifact missing, changed or outside desktop project')
    return path


def load_rigid_layout(path, task):
    path = Path(path)
    value = json.loads(path.read_text())
    if (value.get('version') != 'r10_TRAIN_observed_rigid_layout_v1'
            or value.get('split') != 'train' or value.get('task') != task
            or value.get('shared_with_baseline') is not True
            or value.get('evidence_sha256') != digest({k: v for k, v in value.items() if k != 'evidence_sha256'})):
        raise MultitaskProtocolError('only validated TRAIN rigid-layout priors may be loaded')
    provenance = PriorProvenance('train_calibrated_rigid_layout', str(path.resolve()),
        'child frame expressed in parent native TRAIN CAD frame', True, 'train',
        tuple(value.get('source_groups', [value['source_group']])), value['evidence_sha256'])
    return tuple(RigidChildConstraint(row['parent'], row['child'],
        Pose(row['child_to_parent_position'], row['child_to_parent_rotation']), provenance)
        for row in value['accepted'])


def load_finite_rotation_symmetries(root, protocol, task, asset):
    """Reproduce explicit TRAIN surface certificates; never promote full R."""
    from .rotation_equivalence import VerifiedFiniteSymmetry
    pointers = protocol.get('finite_rotation_symmetries', {}).get(task, {})
    if not isinstance(pointers, dict) or set(pointers)-{'ring'} or set(pointers)-set(asset['bodies']):
        raise MultitaskProtocolError('finite rotation certificates must name the same TRAIN ring CAD')
    result = {}
    for role, pointer in pointers.items():
        artifact_path(root, pointer)
        try:
            symmetry = VerifiedFiniteSymmetry.load(root, pointer)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise MultitaskProtocolError('frozen TRAIN finite rotation certificate failed reproduction') from error
        if symmetry.certificate['task'] != task or symmetry.certificate['CAD_asset_sha256'] != asset['asset_sha256']:
            raise MultitaskProtocolError('finite rotation certificate task or TRAIN CAD differs')
        result[role] = symmetry
    return result


def load_cup_feature_models(root, protocol, task, asset):
    """Load the explicit TRAIN CAD recipe; current frames still decide pose validity."""
    permission = protocol.get('cup_feature_priors', {}).get(task)
    if permission is None:
        return {}
    from .cup_features import load_frozen_cup_features
    try:
        return load_frozen_cup_features(permission,asset,root=root)
    except (ValueError,KeyError,TypeError,OSError) as error:
        raise MultitaskProtocolError('frozen TRAIN cup feature prior failed verification') from error


def continuation_identity(root, protocol, task, predictor, library, candidate_config, mpc_config):
    """Bind full observation/control continuation, excluding the yet-unfitted critic."""
    from .interaction import continuation_fingerprint, GripperTimingConfig
    root = Path(root)
    artifacts = protocol['learned_artifacts'][task]
    relevant = {key: artifacts[key] for key in ('geometry', 'interaction', 'rigid_layout', 'colour')
                if artifacts.get(key) is not None}
    if predictor == 'context':
        relevant['context'] = protocol['context_checkpoint']
    modules = ('pipeline.py', 'forecasting.py', 'mode_observer.py', 'perception.py',
               'public_roles.py', 'maps.py', 'specs.py', 'factory.py', 'simulation.py',
               'protocol.py', 'native_tasks.py', 'training_geometry.py', 'actuation.py',
               'evaluation.py', 'reset_parity.py', 'safe_native_mesh.py', 'component_flags.py',
               'collision_geometry.py', 'axis_evidence.py', 'depth_pose_consistency.py',
               'bin_swept_geometry.py', 'bin_swept_assets.py', 'rotation_equivalence.py', 'ring_affordance.py',
               'cup_features.py')
    sources = {f'our_mani/multitask/{name}': file_sha(root / 'our_mani/multitask' / name) for name in modules}
    for name in ('contracts.py', 'reachability.py', 'intercept_planner.py'):
        sources[f'our_mani/{name}'] = file_sha(root / 'our_mani' / name)
    for name in ('scripts/evaluate_multitask_case_r10.py', 'our_mani/aligned/protocol.py',
                 'our_mani/sensor_alignment/author_rgbd.py', 'scripts/run_local_sim.sh'):
        sources[name] = file_sha(root / name)
    if predictor == 'context':
        name = 'our_mani/multitask/context_se3.py'
        sources[name] = file_sha(root / name)
    physical = {key: protocol[key] for key in ('dt', 'simulation_seconds', 'gripper_velocity',
        'global_joint_acceleration', 'attachment_association_distance_m',
        'shared_semantic_instance_binding', 'semantic_binding_available_to_baseline',
        'shared_TRAIN_CAD_geometry', 'information_setting', 'body_orientation_observability',
        'gripper_timing_config', 'joint_limit_recovery_margin_rad')}
    value = dict(version='r10_observed_full_continuation_identity_v1', task=task, predictor=predictor,
        core_sha256=continuation_fingerprint(library, candidate_config, mpc_config,
            GripperTimingConfig(**protocol['gripper_timing_config']),
            protocol['joint_limit_recovery_margin_rad']),
        observation_control_sources=sources, artifact_pointers=relevant, shared_physical_protocol=physical,
        forecast_config=protocol.get('forecast_config', {}),
        body_sphere_radii=protocol.get('body_sphere_radii', {}).get(task, {}),
        body_symmetry_axes=protocol.get('body_symmetry_axes', {}).get(task, {}),
        body_axis_polarity=protocol.get('body_axis_polarity', {}).get(task, {}),
        axis_swept_assets=protocol.get('axis_swept_assets', {}).get(task, {}),
        finite_rotation_symmetries=protocol.get('finite_rotation_symmetries', {}).get(task, {}),
        ring_affordance=protocol.get('ring_affordances', {}).get(task),
        cup_feature_priors=protocol.get('cup_feature_priors', {}).get(task),
        registration_config=protocol.get('registration_config', {}),
        continuation_selection_policy='single_value_selection_then_frozen_geometry',
        downstream_selection_policy='geometry', attached_dynamics='measured_RGBD_proprioceptive_attachment_FK')
    return dict(value, sha256=digest(value))


def create_actor(root, env, protocol, method, descriptions, *, purpose='evaluation', exploration_seed=None):
    """Call only in the parent evaluator, after reset and before actor.act()."""
    from .forecasting import ForecastConfig, MultiBodyForecastBridge
    from .interaction import (CandidateConfig, InteractionLibrary, MPCConfig, GripperTimingConfig,
                              TRAINUniformFirstCandidateSampler)
    from .candidate_success import CandidateSuccessModel
    from .pipeline import PipelineActor
    from .public_roles import PublicRoleResolver
    from .training_geometry import (body_templates_from_asset, declared_instance_bindings,
                                   register_collision_asset)
    if method not in ('cv_geometry', 'cv_value', 'context_geometry', 'context_value'):
        raise MultitaskProtocolError('unknown ours comparison method')
    if (protocol.get('shared_semantic_instance_binding') is not True
            or protocol.get('semantic_binding_available_to_baseline') is not True
            or protocol.get('shared_TRAIN_CAD_geometry') is not True):
        raise MultitaskProtocolError('declare extra shared CAD/semantic information explicitly')
    if exploration_seed is not None and (purpose != 'train_collection' or not method.endswith('_geometry')):
        raise MultitaskProtocolError('only TRAIN geometry collection can explore the first candidate')
    task = env.spec.task_name
    pointers = protocol['learned_artifacts'][task]
    asset = json.loads(artifact_path(root, pointers['geometry']).read_text())
    finite_symmetries = load_finite_rotation_symmetries(root, protocol, task, asset)
    cup_models = load_cup_feature_models(root,protocol,task,asset)
    bindings = declared_instance_bindings(env, asset, shared_permission=True)
    templates = body_templates_from_asset(asset, bindings,
        orientation_observability=protocol['body_orientation_observability'][task],
        sphere_radii=protocol.get('body_sphere_radii', {}).get(task),
        symmetry_axes=protocol.get('body_symmetry_axes', {}).get(task),
        axis_polarity=protocol.get('body_axis_polarity', {}).get(task),
        finite_rotation_symmetries=finite_symmetries,cup_feature_models=cup_models)
    rigid = (load_rigid_layout(artifact_path(root, pointers['rigid_layout']), task)
             if pointers.get('rigid_layout') else ())
    tracker = MultiBodySE3Tracker(templates, task_spec=env.spec, rigid_children=rigid,
        config=RegistrationConfig(**protocol.get('registration_config', {})))
    register_collision_asset(env.map_service, asset,
        orientation_observability=protocol['body_orientation_observability'][task],
        axis_swept_assets=protocol.get('axis_swept_assets', {}).get(task),
        finite_rotation_symmetries=finite_symmetries)
    predictor, ranking = method.split('_', 1)
    context = None
    if predictor == 'context':
        from .context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(artifact_path(root, protocol['context_checkpoint']))
    bridge = MultiBodyForecastBridge(tuple(template.role_id for template in templates),
        method=predictor, context=context, rigid_children=rigid, task_spec=env.spec,
        config=ForecastConfig(**protocol.get('forecast_config', {})))
    library = InteractionLibrary.load(artifact_path(root, pointers['interaction']))
    if library.task_spec.task_name != task:
        raise MultitaskProtocolError('skill library task differs from native case')
    fit = library.metadata.get('fit_configuration', {})
    if (library.metadata.get('development_support_override') is not False
            or fit.get('development_only') is not False
            or fit.get('minimum_completed_parent_sources', 0) < 8):
        raise MultitaskProtocolError('production actor requires eight completed TRAIN parents per skill; development library is not released')
    candidate = CandidateConfig(**protocol['candidate_config'])
    mpc = MPCConfig(**protocol['mpc_config'])
    gripper_timing = GripperTimingConfig(**protocol['gripper_timing_config'])
    continuation = continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    expected = protocol.get('continuation_identities', {}).get(task, {}).get(predictor)
    if expected is not None and expected != continuation['sha256']:
        raise MultitaskProtocolError('frozen full continuation identity changed')
    resolver = None
    if env.spec.family in ('insert_onto_rotating_peg', 'push_moving_button'):
        resolver = PublicRoleResolver(env.spec, json.loads(artifact_path(root, pointers['colour']).read_text()),
                                      shared_semantic_permission=True)
    success_model = (CandidateSuccessModel(artifact_path(root, pointers[predictor + '_value']),
        continuation_sha256=continuation['sha256'], library_sha256=library.sha256) if ranking == 'value' else None)
    sampler = (TRAINUniformFirstCandidateSampler(seed=exploration_seed, purpose='train')
               if exploration_seed is not None else None)
    ring_affordance = None
    if protocol.get('ring_affordances', {}).get(task) is not None:
        from .ring_affordance import VerifiedRingAffordance
        pointer = protocol['ring_affordances'][task]
        artifact_path(root, pointer)
        ring_affordance = VerifiedRingAffordance.load(root, pointer)
    actor = PipelineActor(env.spec, templates, tracker, bridge, library, env.robot_model, env.map_service,
        candidate, mpc, predictor=predictor, ranking=ranking, descriptions=descriptions,
        role_resolver=resolver, success_model=success_model,
        attachment_association_distance_m=protocol['attachment_association_distance_m'],
        continuation_sha256=continuation['sha256'], purpose=purpose, first_candidate_sampler=sampler,
        gripper_timing_config=gripper_timing,
        joint_limit_recovery_margin_rad=protocol['joint_limit_recovery_margin_rad'],
        ring_affordance=ring_affordance)
    return actor, continuation
