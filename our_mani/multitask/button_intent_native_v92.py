"""TRAIN-only native entry for separately supervised button command targets."""
from pathlib import Path
from types import MethodType
from .task_native_backend_v69 import NativeActuatorDiagnosticBackend,diagnostic_binding
from .button_intent_actor_v91 import binding,create_actor,SOURCES
from .button_intent_library_v71 import ButtonIntentLibrary,TASKS
from .evaluation import file_sha


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    root=Path(root).resolve();args=prepared['args'];c=prepared['config'];files=prepared['files']
    if args.split!='train' or args.task not in TASKS or args.method not in ('cv_geometry','context_geometry'):raise ValueError('button command candidate is original TRAIN development only')
    model=c['learned_artifacts'][args.task]['interaction'];locked_digest(root/model['path'],files)
    library=ButtonIntentLibrary(root,root/model['path'],purpose='train_development');predictor=args.method.split('_')[0]
    identity=binding(root,c,args.task,predictor)
    if c['continuation_identities'][args.task][predictor]!=identity['sha256']:raise ValueError('actual button continuation source changed')
    for name,checksum in identity['sources'].items():
        if locked_digest(root/name,files)!=checksum:raise ValueError('button source omitted from frozen input lock')
    for pointer in identity['shared_executor_overlay']['sources'].values():
        if locked_digest(root/pointer['path'],files)!=pointer['sha256']:raise ValueError('shared executor dependency changed')
    locked_digest(Path(__file__),files)
    if predictor=='context':
        from .context_se3 import MultiBodyContextDynamics
        cp=c['context_checkpoint'];locked_digest(root/cp['path'],files);context=MultiBodyContextDynamics(root/cp['path'])
        source=context.metadata['training_source_manifest'];locked_digest(context.path.parent/source['path_relative_to_checkpoint'],files)
    return dict(interaction_library_sha256=library.sha256,continuation_identity=identity,all_loaders_CPU_only=True,
        supervision='original_author_command_goals',completed_skill_model_admitted=False,physical_press_completion_capability_granted=False,
        formal_VAL_TEST_ready=False,native_started=False)


class ButtonIntentNativeBackend(NativeActuatorDiagnosticBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=create_actor(Path(__file__).resolve().parents[2],env,prepared['config'],prepared['args'].method,descriptions,purpose='train_collection')
        self.direct_dispatcher.bind_manager(actor.continuous_reference_manager)
        before=actor.metrics_report;dispatcher=self.direct_dispatcher
        def metrics(instance):return dict(before(),task_dispatch_v43=dispatcher.receipt())
        actor.metrics_report=MethodType(metrics,actor)
        from .all_phase_collision_diagnostic_v17 import CollisionObserver
        observer=CollisionObserver(actor,env);old=actor.metrics_report
        def observed(instance):return dict(old(),public_collision_query_diagnostics_v6=observer.receipt())
        actor.metrics_report=MethodType(observed,actor);self.instrument_actor(actor,env)
        return actor,identity
