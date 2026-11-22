"""Geometry actor with source-bound feedback and empirical servo compensation.

Observation, learned models, native physics and the original sampled motion
gates are shared with the preceding feedback backend. Installation order is
measured-feedback controller first, causal servo-response wrapper second.
"""
from copy import deepcopy
from pathlib import Path

from .evaluation import digest, file_sha
from .protocol import MultitaskProtocolError
from . import factory
from scripts.evaluate_multitask_case_r10 import NativeBackend

OVERLAY_SCHEMA = 'r10_geometry_feedback_servo_response_policy_overlay_v1'
IDENTITY_SCHEMA = 'r10_geometry_feedback_servo_response_continuation_v1'
METHODS = ('cv_geometry', 'context_geometry')


def require(value, message):
    if not value:
        raise MultitaskProtocolError(message)


def _source(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'servo-response source leaves the project')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def servo_policy_overlay(root, feedback_configuration, servo_configuration):
    """Bind both normalized configs and the three executable overlay sources."""
    from .feedback_interaction import feedback_binding
    from .online_servo_response import servo_response_binding
    root = Path(root).resolve()
    feedback = feedback_binding(feedback_configuration)
    servo = servo_response_binding(servo_configuration)
    return dict(schema=OVERLAY_SCHEMA,
        configurations=dict(feedback=deepcopy(feedback['normalized_config']),
            servo_response=deepcopy(servo['normalized_config'])),
        sources=dict(actor_backend=_source(root, __file__),
            feedback_controller=_source(root, root / 'our_mani/multitask/feedback_interaction.py'),
            servo_response=_source(root, root / 'our_mani/multitask/online_servo_response.py')))


def verify_servo_policy_overlay(root, protocol):
    overlay = protocol.get('policy_overlay')
    require(isinstance(overlay, dict)
        and set(overlay) == {'schema', 'configurations', 'sources'}
        and overlay.get('schema') == OVERLAY_SCHEMA,
        'explicit geometry feedback/servo-response overlay is required')
    configurations = overlay['configurations']
    require(isinstance(configurations, dict) and set(configurations) == {'feedback', 'servo_response'},
        'both normalized controller configurations are required')
    expected = servo_policy_overlay(root, configurations['feedback'], configurations['servo_response'])
    require(overlay == expected, 'servo-response source or normalized configurations changed')
    require(protocol.get('dt') == .05 and protocol.get('simulation_seconds') == 10.,
        'servo-response overlay preserves the original 50ms/10-second task budget')
    return expected


def servo_continuation_identity(root, protocol, task, predictor, library, candidate, mpc):
    require(predictor in ('cv', 'context'), 'servo-response predictor must be CV or Context')
    overlay = verify_servo_policy_overlay(root, protocol)
    original = factory.continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    value = dict(version=IDENTITY_SCHEMA, task=task, predictor=predictor,
        original_continuation=original, policy_overlay=overlay,
        continuation_selection_policy='geometry', downstream_selection_policy='geometry',
        original_physics_and_observation_sources_unchanged=True,
        installation_order=['observed_feedback_controller', 'online_servo_response'],
        sampled_response_interval_approximation=True, actual_tracking_guaranteed=False)
    return dict(value, sha256=digest(value))


def servo_actor_model_preflight(root, prepared):
    """Check CPU model loaders and frozen executable bindings before native use."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig
    from scripts.evaluate_multitask_case_r10 import locked_digest
    root = Path(root).resolve()
    args, config, files = prepared['args'], prepared['config'], prepared['files']
    require(args.method in METHODS, 'servo-response backend accepts geometry methods only')
    overlay = verify_servo_policy_overlay(root, config)
    for value in overlay['sources'].values():
        require(locked_digest(root / value['path'], files) == value['sha256'],
            'servo-response executable source omitted from actual frozen lock')
    artifacts = config['learned_artifacts'][args.task]
    library = InteractionLibrary.load(factory.artifact_path(root, artifacts['interaction']))
    require(library.task_spec.task_name == args.task, 'servo-response fitted library task differs')
    fit = library.metadata.get('fit_configuration', {})
    require(library.metadata.get('development_support_override') is False
        and fit.get('development_only') is False and fit.get('minimum_completed_parent_sources', 0) >= 8,
        'servo-response actor retains the original completed-TRAIN support gate')
    predictor = args.method.split('_', 1)[0]
    identity = servo_continuation_identity(root, config, args.task, predictor, library,
        CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))
    require(config.get('continuation_identities', {}).get(args.task, {}).get(predictor) == identity['sha256'],
        'freeze the actual feedback/servo-response continuation identity')
    context = None
    if predictor == 'context':
        from .context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(factory.artifact_path(root, config['context_checkpoint']))
        source = context.metadata['training_source_manifest']
        locked_digest(context.path.parent / source['path_relative_to_checkpoint'], files)
    return dict(interaction_library_sha256=library.sha256, continuation_identity=identity,
        policy_overlay=overlay, context_checkpoint_sha256=None if context is None else context.checkpoint_sha256,
        all_loaders_CPU_only=True, native_started=False, geometry_only=True)


def create_servo_response_actor(root, env, protocol, method, descriptions, *,
        purpose='evaluation', exploration_seed=None):
    """Keep original observation/physics and install both wrappers before act()."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig, InteractionController
    from .feedback_interaction import install_feedback_controller
    from .online_servo_response import install_servo_response_compensator
    require(method in METHODS, 'servo-response backend accepts geometry methods only')
    predictor, ranking = method.split('_', 1)
    require(exploration_seed is None or purpose == 'train_collection',
        'first-candidate exploration remains TRAIN geometry only')
    task = env.spec.task_name
    library = InteractionLibrary.load(factory.artifact_path(root, protocol['learned_artifacts'][task]['interaction']))
    candidate, mpc = CandidateConfig(**protocol['candidate_config']), MPCConfig(**protocol['mpc_config'])
    extended = servo_continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    require(protocol.get('continuation_identities', {}).get(task, {}).get(predictor) == extended['sha256'],
        'actual servo-response actor continuation differs from its frozen protocol')
    base_protocol = deepcopy(protocol)
    base_protocol.setdefault('continuation_identities', {}).setdefault(task, {})[predictor] = extended['original_continuation']['sha256']
    actor, original = factory.create_actor(root, env, base_protocol, method, descriptions,
        purpose=purpose, exploration_seed=exploration_seed)
    require(original == extended['original_continuation'], 'original actor construction changed')
    old = actor.controller
    actor.controller = InteractionController(old.library, old.generator, old.mpc,
        predictor=predictor, ranking=ranking, continuation_sha256=extended['sha256'], success_model=None,
        first_candidate_sampler=old.first_candidate_sampler,
        collection_purpose='train' if purpose == 'train_collection' else None,
        gripper_timing_config=old.gripper_timing_config)
    actor.ranking = ranking
    configurations = extended['policy_overlay']['configurations']
    install_feedback_controller(actor, configurations['feedback'])
    install_servo_response_compensator(actor, configurations['servo_response'])
    return actor, extended


class ServoResponseNativeBackend(NativeBackend):
    def actor(self, prepared, env, descriptions):
        args = prepared['args']
        root = Path(__file__).resolve().parents[2]
        return create_servo_response_actor(root, env, prepared['config'], args.method, descriptions,
            purpose='train_collection' if args.split == 'train' else
                'validation_collection' if args.split == 'validation' else 'evaluation',
            exploration_seed=args.exploration_seed)
