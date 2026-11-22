"""Geometry actor with persistent verified Hermite references and direct setpoints.

Observation, learned models, native physics and the original sampled motion
gates are shared with the preceding feedback backend. Installation order is
measured-feedback controller, causal retarget, then persistent reference stream.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType

from .evaluation import digest, file_sha
from .protocol import MultitaskProtocolError
from . import factory
from scripts.evaluate_multitask_case_r10 import NativeBackend

OVERLAY_SCHEMA = 'r10_geometry_continuous_reference_direct_policy_overlay_v16'
IDENTITY_SCHEMA = 'r10_geometry_continuous_reference_direct_continuation_v16'
METHODS = ('cv_geometry', 'context_geometry')


def require(value, message):
    if not value:
        raise MultitaskProtocolError(message)


def _source(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), 'persistent-reference source leaves the project')
    return dict(path=str(path.relative_to(root)), sha256=file_sha(path))


def persistent_reference_policy_overlay(root, feedback_configuration, reference_configuration, retarget_configuration):
    """Bind the normalized control settings and all overlay sources."""
    from .feedback_interaction import feedback_binding
    from .bin_reference_v16 import continuous_reference_binding
    from .causal_retarget_interaction import retarget_binding
    root = Path(root).resolve()
    feedback = feedback_binding(feedback_configuration)
    reference = continuous_reference_binding(reference_configuration)
    retarget = retarget_binding(retarget_configuration)
    return dict(schema=OVERLAY_SCHEMA,
        configurations=dict(feedback=deepcopy(feedback['normalized_config']),
            persistent_reference=deepcopy(reference['normalized_config']),
            causal_retarget=deepcopy(retarget['normalized_config'])),
        sources=dict(actor_backend=_source(root, __file__),
            feedback_controller=_source(root, root / 'our_mani/multitask/feedback_interaction.py'),
            persistent_reference=_source(root, root / 'our_mani/multitask/bin_reference_v16.py'),
            direct_dispatch=_source(root, root / 'our_mani/multitask/bin_dispatch_v16.py'),
            causal_retarget=_source(root, root / 'our_mani/multitask/causal_retarget_interaction.py'),
            bounded_approach=_source(root, root / 'our_mani/multitask/bounded_approach_v16.py')))


def verify_persistent_reference_policy_overlay(root, protocol):
    overlay = protocol.get('policy_overlay')
    require(isinstance(overlay, dict)
        and set(overlay) == {'schema', 'configurations', 'sources'}
        and overlay.get('schema') == OVERLAY_SCHEMA,
        'explicit geometry feedback/persistent-reference overlay is required')
    configurations = overlay['configurations']
    require(isinstance(configurations, dict) and set(configurations) == {'feedback', 'persistent_reference', 'causal_retarget'},
        'all normalized controller configurations are required')
    expected = persistent_reference_policy_overlay(root, configurations['feedback'], configurations['persistent_reference'], configurations['causal_retarget'])
    require(overlay == expected, 'persistent-reference source or normalized configurations changed')
    require(protocol.get('dt') == .05 and protocol.get('simulation_seconds') == 10.,
        'persistent-reference overlay preserves the original 50ms/10-second task budget')
    return expected


def persistent_reference_continuation_identity(root, protocol, task, predictor, library, candidate, mpc):
    require(predictor in ('cv', 'context'), 'persistent-reference predictor must be CV or Context')
    overlay = verify_persistent_reference_policy_overlay(root, protocol)
    original = factory.continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    value = dict(version=IDENTITY_SCHEMA, task=task, predictor=predictor,
        original_continuation=original, policy_overlay=overlay,
        continuation_selection_policy='geometry', downstream_selection_policy='geometry',
        original_physics_and_observation_sources_unchanged=True,
        installation_order=['observed_feedback_controller', 'continuous_reference_with_retarget_before_commit'],
        fresh_public_sampled_collision=True, actual_tracking_guaranteed=False)
    return dict(value, sha256=digest(value))


def persistent_reference_actor_model_preflight(root, prepared):
    """Check CPU model loaders and frozen executable bindings before native use."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig
    from scripts.evaluate_multitask_case_r10 import locked_digest
    root = Path(root).resolve()
    args, config, files = prepared['args'], prepared['config'], prepared['files']
    require(args.method in METHODS, 'persistent-reference backend accepts geometry methods only')
    require(args.task == 'put_rubbish_in_moving_bin',
        'this executor qualification is limited to normal-speed rubbish bin')
    overlay = verify_persistent_reference_policy_overlay(root, config)
    for value in overlay['sources'].values():
        require(locked_digest(root / value['path'], files) == value['sha256'],
            'persistent-reference executable source omitted from actual frozen lock')
    from .causal_retarget_interaction import retarget_binding
    from .bin_reference_v16 import continuous_reference_binding
    for dependencies in (retarget_binding(overlay['configurations']['causal_retarget']),
            continuous_reference_binding(overlay['configurations']['persistent_reference'])):
        for name, checksum in dependencies['original_sources_sha256'].items():
            require(locked_digest(root / name, files) == checksum,
                'controller original dependency differs from frozen inputs')
    artifacts = config['learned_artifacts'][args.task]
    library = InteractionLibrary.load(factory.artifact_path(root, artifacts['interaction']))
    require(library.task_spec.task_name == args.task, 'persistent-reference fitted library task differs')
    fit = library.metadata.get('fit_configuration', {})
    require(library.metadata.get('development_support_override') is False
        and fit.get('development_only') is False and fit.get('minimum_completed_parent_sources', 0) >= 8,
        'persistent-reference actor retains the original completed-TRAIN support gate')
    predictor = args.method.split('_', 1)[0]
    identity = persistent_reference_continuation_identity(root, config, args.task, predictor, library,
        CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))
    require(config.get('continuation_identities', {}).get(args.task, {}).get(predictor) == identity['sha256'],
        'freeze the actual feedback/persistent-reference continuation identity')
    context = None
    if predictor == 'context':
        from .context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(factory.artifact_path(root, config['context_checkpoint']))
        source = context.metadata['training_source_manifest']
        locked_digest(context.path.parent / source['path_relative_to_checkpoint'], files)
    return dict(interaction_library_sha256=library.sha256, continuation_identity=identity,
        policy_overlay=overlay, context_checkpoint_sha256=None if context is None else context.checkpoint_sha256,
        all_loaders_CPU_only=True, native_started=False, geometry_only=True)


def create_persistent_reference_actor(root, env, protocol, method, descriptions, *,
        purpose='evaluation', exploration_seed=None):
    """Keep original observation/physics and install both wrappers before act()."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig, InteractionController
    from .feedback_interaction import install_feedback_controller
    from .bin_reference_v16 import install_continuous_reference
    require(method in METHODS, 'persistent-reference backend accepts geometry methods only')
    predictor, ranking = method.split('_', 1)
    require(exploration_seed is None or purpose == 'train_collection',
        'first-candidate exploration remains TRAIN geometry only')
    task = env.spec.task_name
    library = InteractionLibrary.load(factory.artifact_path(root, protocol['learned_artifacts'][task]['interaction']))
    candidate, mpc = CandidateConfig(**protocol['candidate_config']), MPCConfig(**protocol['mpc_config'])
    extended = persistent_reference_continuation_identity(root, protocol, task, predictor, library, candidate, mpc)
    require(protocol.get('continuation_identities', {}).get(task, {}).get(predictor) == extended['sha256'],
        'actual persistent-reference actor continuation differs from its frozen protocol')
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
    install_continuous_reference(actor, configurations['persistent_reference'],
        retarget_config=configurations['causal_retarget'])
    # Record only already supplied state; no extra camera, FK or physics query.
    observed_trace = []
    previous_step = actor.controller.step
    def traced_step(controller, state, forecast, remaining_seconds):
        command = previous_step(state, forecast, remaining_seconds)
        ball = state.bodies.get('rubbish')
        goal = controller.goal
        observed_trace.append(dict(
            time=float(state.robot.timestamp),
            tcp_xyzw=state.robot.pose.xyzw().tolist(),
            joints=state.robot.joints.tolist(),
            joint_velocities=state.robot.joint_velocities.tolist(),
            rubbish_xyzw=None if ball is None else ball.pose.xyzw().tolist(),
            rubbish_visible=None if ball is None else bool(ball.visible),
            rubbish_observed_at=None if ball is None else float(ball.timestamp),
            phase_index=int(controller.phase_index),
            active_candidate_id=None if goal is None else goal.candidate_id,
            committed_reference_xyzw=None if goal is None else goal.reference_pose.xyzw().tolist(),
            phase_epoch=None if goal is None else float(goal.selected_at),
            phase_deadline=None if goal is None else float(goal.selected_at + goal.phase_budget),
            observed_contact=bool(state.robot.contact),
            observed_gripper_open=bool(state.robot.gripper_open),
            requested_opening_before_outer_dispatch=actor.requested_opening,
            public_motion_modes={key:info.get('motion_mode') for key,info in state.perception_metadata.get('bodies',{}).items()},
            controller_diagnostics=deepcopy(controller.last_diagnostics),
            inner_proposed_command=command.get('command'),
            inner_proposal_is_not_outer_actual_issued_receipt=True))
        return command
    actor.controller.step = MethodType(traced_step, actor.controller)
    previous_metrics = actor.metrics_report
    def traced_metrics(instance):
        report = dict(previous_metrics())
        report['executor_observed_state_trace'] = deepcopy(observed_trace)
        return report
    actor.metrics_report = MethodType(traced_metrics, actor)
    return actor, extended


class ContinuousReferenceNativeBackend(NativeBackend):
    def __init__(self):
        from .bin_dispatch_v16 import CountedDirectDispatcher
        self.direct_dispatcher = CountedDirectDispatcher()

    def dispatch(self, env, command):
        return self.direct_dispatcher.dispatch(env, command)

    def actor(self, prepared, env, descriptions):
        args = prepared['args']
        root = Path(__file__).resolve().parents[2]
        actor, identity = create_persistent_reference_actor(root, env, prepared['config'], args.method, descriptions,
            purpose='train_collection' if args.split == 'train' else
                'validation_collection' if args.split == 'validation' else 'evaluation',
            exploration_seed=args.exploration_seed)
        self.direct_dispatcher.bind_manager(actor.continuous_reference_manager)
        previous_metrics = actor.metrics_report
        dispatcher = self.direct_dispatcher
        def metrics(instance):
            return dict(previous_metrics(), bin_dispatch_v16=dispatcher.receipt())
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
