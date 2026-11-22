"""Explicit, source-bound feedback actor; original perception and physics stay in use."""
from copy import deepcopy
from pathlib import Path

from .evaluation import digest, file_sha
from .protocol import MultitaskProtocolError
from . import factory
from scripts.evaluate_multitask_case_r10 import NativeBackend

OVERLAY_SCHEMA = 'r10_observed_feedback_policy_overlay_v1'
IDENTITY_SCHEMA = 'r10_full_observed_feedback_continuation_v1'


def require(value, message):
    if not value:
        raise MultitaskProtocolError(message)


def _source(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'feedback source leaves the project')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def policy_overlay(root, configuration):
    """Prepare explicit actual source/config bindings, without creating an actor."""
    from .feedback_interaction import feedback_binding
    root = Path(root).resolve()
    binding = feedback_binding(configuration)
    return dict(schema=OVERLAY_SCHEMA, configuration=deepcopy(binding['normalized_config']),
        sources=dict(actor_backend=_source(root, __file__),
            controller=_source(root, root / 'our_mani/multitask/feedback_interaction.py')))


def verify_policy_overlay(root, protocol):
    root = Path(root).resolve()
    overlay = protocol.get('policy_overlay')
    require(isinstance(overlay, dict) and set(overlay) == {'schema', 'configuration', 'sources'}
        and overlay.get('schema') == OVERLAY_SCHEMA, 'explicit observed-feedback overlay is required')
    expected = policy_overlay(root, overlay['configuration'])
    require(overlay == expected, 'feedback source or normalized configuration changed')
    require(protocol.get('dt') == .05 and protocol.get('simulation_seconds') == 10.,
        'feedback overlay preserves the original 50ms/10-second task budget')
    return expected


def feedback_continuation_identity(root, protocol, task, predictor, library, candidate, mpc):
    overlay = verify_policy_overlay(root, protocol)
    original = factory.continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    value = dict(version=IDENTITY_SCHEMA, task=task, predictor=predictor,
        original_continuation=original, policy_overlay=overlay,
        continuation_selection_policy='single_value_selection_then_frozen_geometry',
        downstream_selection_policy='geometry', original_physics_and_observation_sources_unchanged=True)
    return dict(value, sha256=digest(value))


def feedback_actor_model_preflight(root, prepared):
    """Load actual models on CPU, rejecting old-policy critics before native launch."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig
    from scripts.evaluate_multitask_case_r10 import locked_digest
    root = Path(root).resolve()
    args, config, files = prepared['args'], prepared['config'], prepared['files']
    require(args.method in ('cv_geometry', 'context_geometry'), 'all Value work is stopped; geometry methods only')
    overlay = verify_policy_overlay(root, config)
    for value in overlay['sources'].values():
        require(locked_digest(root / value['path'], files) == value['sha256'],
            'feedback source omitted from actual frozen lock')
    artifacts = config['learned_artifacts'][args.task]
    library = InteractionLibrary.load(factory.artifact_path(root, artifacts['interaction']))
    require(library.task_spec.task_name == args.task, 'feedback fitted library task differs')
    fit = library.metadata.get('fit_configuration', {})
    require(library.metadata.get('development_support_override') is False
        and fit.get('development_only') is False and fit.get('minimum_completed_parent_sources', 0) >= 8,
        'feedback actor retains the original completed-TRAIN support gate')
    identity = feedback_continuation_identity(root, config, args.task, args.method.split('_')[0], library,
        CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))
    predictor, ranking = args.method.split('_', 1)
    require(config.get('continuation_identities', {}).get(args.task, {}).get(predictor) == identity['sha256'],
        'freeze the actual extended feedback continuation identity')
    context = critic = None
    if predictor == 'context':
        from .context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(factory.artifact_path(root, config['context_checkpoint']))
        source = context.metadata['training_source_manifest']
        locked_digest(context.path.parent / source['path_relative_to_checkpoint'], files)
    return dict(interaction_library_sha256=library.sha256, continuation_identity=identity,
        policy_overlay=overlay, context_checkpoint_sha256=None if context is None else context.checkpoint_sha256,
        fitted_matching_critic_sha256=None if critic is None else critic.artifact_sha256,
        all_loaders_CPU_only=True, native_started=False)


def create_feedback_actor(root, env, protocol, method, descriptions, *, purpose='evaluation', exploration_seed=None):
    """Construct unchanged observation modules, then install the explicit controller before act()."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig, InteractionController
    from .feedback_interaction import install_feedback_controller
    predictor, ranking = method.split('_', 1)
    require(predictor in ('cv', 'context') and ranking == 'geometry', 'all Value work is stopped; geometry methods only')
    require(exploration_seed is None or (ranking == 'geometry' and purpose == 'train_collection'),
        'first-candidate exploration remains TRAIN geometry only')
    task = env.spec.task_name
    library = InteractionLibrary.load(factory.artifact_path(root, protocol['learned_artifacts'][task]['interaction']))
    candidate, mpc = CandidateConfig(**protocol['candidate_config']), MPCConfig(**protocol['mpc_config'])
    extended = feedback_continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    require(protocol.get('continuation_identities', {}).get(task, {}).get(predictor) == extended['sha256'],
        'actual feedback actor continuation differs from its frozen protocol')
    base_protocol = deepcopy(protocol)
    base_protocol.setdefault('continuation_identities', {}).setdefault(task, {})[predictor] = extended['original_continuation']['sha256']
    actor, original = factory.create_actor(root, env, base_protocol, predictor + '_geometry', descriptions,
        purpose=purpose, exploration_seed=exploration_seed)
    require(original == extended['original_continuation'], 'original actor construction changed')
    old = actor.controller
    critic = None
    actor.controller = InteractionController(old.library, old.generator, old.mpc,
        predictor=predictor, ranking=ranking, continuation_sha256=extended['sha256'], success_model=critic,
        first_candidate_sampler=old.first_candidate_sampler,
        collection_purpose='train' if purpose == 'train_collection' else None,
        gripper_timing_config=old.gripper_timing_config)
    actor.ranking = ranking
    install_feedback_controller(actor, protocol['policy_overlay']['configuration'])
    return actor, extended


class FeedbackNativeBackend(NativeBackend):
    def actor(self, prepared, env, descriptions):
        args = prepared['args']
        root = Path(__file__).resolve().parents[2]
        return create_feedback_actor(root, env, prepared['config'], args.method, descriptions,
            purpose='train_collection' if args.split == 'train' else
                'validation_collection' if args.split == 'validation' else 'evaluation',
            exploration_seed=args.exploration_seed)
