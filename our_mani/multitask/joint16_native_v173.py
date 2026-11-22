"""Shared16 development preflight retains all original v69 gates except obsolete old14 rejection.

The complete original preflight is copied explicitly. Its unconditional buzz
Context prohibition is replaced by actual shared16 completion and exact TRAIN
roster/conditional rotation verification. The native actor remains v69.
"""
from pathlib import Path
from types import MethodType
from .task_actor_backend_v69 import (METHODS, require, factory,
    verify_persistent_reference_policy_overlay, persistent_reference_continuation_identity,
    ContinuousReferenceNativeBackend)
from .specs import TASK_SPECS
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

def model_preflight(root, prepared):
    """Check CPU model loaders and frozen executable bindings before native use."""
    from .interaction import InteractionLibrary, CandidateConfig, MPCConfig
    from scripts.evaluate_multitask_case_r10 import locked_digest
    root = Path(root).resolve()
    checked_subset(root, prepared['files'], {str(Path(__file__)): file_sha(__file__)})
    args, config, files = prepared['args'], prepared['config'], prepared['files']
    require(args.method in METHODS, 'persistent-reference backend accepts geometry methods only')
    require(args.task in TASK_SPECS, 'only the fourteen registered multitask actors are accepted')
    overlay = verify_persistent_reference_policy_overlay(root, config)
    for value in overlay['sources'].values():
        require(locked_digest(root / value['path'], files) == value['sha256'],
            'persistent-reference executable source omitted from actual frozen lock')
    from .causal_retarget_interaction import retarget_binding
    from .task_reference_v37 import continuous_reference_binding
    for dependencies in (retarget_binding(overlay['configurations']['causal_retarget']),
            continuous_reference_binding(overlay['configurations']['persistent_reference'])):
        for name, checksum in dependencies['original_sources_sha256'].items():
            require(locked_digest(root / name, files) == checksum,
                'controller original dependency differs from frozen inputs')
    artifacts = config['learned_artifacts'][args.task]
    if TASK_SPECS[args.task].family in ('insert_onto_rotating_peg','push_moving_button'):
        for path,checksum in config['colour_runtime_v18']['inputs_sha256'].items():
            require(locked_digest(Path(path),files)==checksum,'colour runtime evidence omitted from actual lock')
    library = InteractionLibrary.load(factory.artifact_path(root, artifacts['interaction']))
    require(library.task_spec.task_name == args.task, 'persistent-reference fitted library task differs')
    fit = library.metadata.get('fit_configuration', {})
    require(library.metadata.get('development_support_override') is False
        and fit.get('development_only') is False and fit.get('minimum_completed_parent_sources', 0) >= 8,
        'persistent-reference actor retains the original completed-TRAIN support gate')
    predictor = args.method.split('_', 1)[0]
    from .joint16_checkpoint_v172 import verified_checkpoint
    shared = verified_checkpoint(root, prepared) if predictor == 'context' else None
    identity = persistent_reference_continuation_identity(root, config, args.task, predictor, library,
        CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))
    require(config.get('continuation_identities', {}).get(args.task, {}).get(predictor) == identity['sha256'],
        'freeze the actual feedback/persistent-reference continuation identity')
    if args.task=='insert_onto_rotating_peg':
        for name,checksum in config['ring_runtime_v25'][args.task]['inputs_sha256'].items():
            require(locked_digest(root/name,files)==checksum,'ring complete proof input missing from actual lock')
    context = None
    if predictor == 'context':
        from .context_se3 import MultiBodyContextDynamics
        context = MultiBodyContextDynamics(factory.artifact_path(root, config['context_checkpoint']))
        source = context.metadata['training_source_manifest']
        locked_digest(context.path.parent / source['path_relative_to_checkpoint'], files)
    return dict(interaction_library_sha256=library.sha256, continuation_identity=identity,
        policy_overlay=overlay, context_checkpoint_sha256=None if context is None else context.checkpoint_sha256,
        all_loaders_CPU_only=True, native_started=False, geometry_only=True, joint16_development_checkpoint=shared)


class SharedJoint16NativeBackend(ContinuousReferenceNativeBackend):
    """Original policy and actual-window dispatch; omit optional force/contact debug."""
    def actor(self, prepared, env, descriptions):
        actor, identity = super().actor(prepared, env, descriptions)
        previous = actor.metrics_report
        def metrics(instance):
            return dict(previous(), shared_joint16_native_v173=dict(
                checkpoint=prepared['config']['context_checkpoint'],
                original_actor_and_dispatch_unchanged=True,
                optional_native_force_contact_debug_enabled=False,
                original_actual_window_motion_metrics_retained=True,
                formal_VAL_TEST=False))
        actor.metrics_report = MethodType(metrics, actor)
        return actor, identity
