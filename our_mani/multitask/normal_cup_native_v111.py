"""TRAIN-only explicit consumer of the initial-sequence cup model domain."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
from . import interaction
from .normal_cup_primary_library_v109 import PrimarySequenceLibrary
from .normal_cup_primary_fitting_v107 import TASK
from .task_native_backend_v69 import NativeActuatorDiagnosticBackend
from .task_actor_backend_v69 import persistent_reference_actor_model_preflight
from .evaluation import file_sha

ORIGINAL_LIBRARY=interaction.InteractionLibrary
SOURCES=('our_mani/multitask/normal_cup_native_v111.py',
    'our_mani/multitask/normal_cup_primary_library_v109.py',
    'our_mani/multitask/normal_cup_primary_fitting_v107.py')


def binding(root,protocol):
    root=Path(root).resolve();pointer=protocol['normal_cup_primary_report']
    path=root/pointer['path']
    if file_sha(path)!=pointer['sha256']:raise ValueError('initial-sequence cup report changed')
    report=json.loads(path.read_text())
    if report['model']!=protocol['learned_artifacts'][TASK]['interaction']:
        raise ValueError('actual cup model differs from declared scope report')
    return dict(schema='r10_normal_cup_explicit_primary_TRAIN_consumer_v111',report=pointer,
        model=report['model'],scope=report['policy_scope'],
        sources_sha256={name:file_sha(root/name) for name in SOURCES},
        original_full_scope_gate_passed=False,original_support_and_error_caps_unchanged=True,
        native_success_source_unchanged=True,TRAIN_only=True,formal_VAL_TEST=False)


@contextmanager
def library_scope(root,protocol,task):
    if task!=TASK or protocol.get('normal_cup_primary_v111')!=binding(root,protocol):
        raise ValueError('freeze exact explicit initial-sequence consumer/domain binding')
    if interaction.InteractionLibrary is not ORIGINAL_LIBRARY:
        raise ValueError('nested or mixed library scopes forbidden')
    interaction.InteractionLibrary=PrimarySequenceLibrary
    try:yield
    finally:interaction.InteractionLibrary=ORIGINAL_LIBRARY


def model_preflight(root,prepared):
    from scripts.evaluate_multitask_case_r10 import locked_digest
    args=prepared['args'];protocol=prepared['config'];root=Path(root).resolve()
    if args.split!='train' or args.task!=TASK or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('new primary cup policy is fixed-source TRAIN development only')
    proof=binding(root,protocol)
    for name,checksum in proof['sources_sha256'].items():
        if locked_digest(root/name,prepared['files'])!=checksum:raise ValueError('new library consumer source omitted from lock')
    report=json.loads((root/proof['report']['path']).read_text())
    locked_digest(root/proof['report']['path'],prepared['files'])
    for name,checksum in report['verified_files_sha256'].items():
        if locked_digest(Path(name),prepared['files'])!=checksum:raise ValueError('original100 fit proof input differs')
    with library_scope(root,protocol,args.task):
        result=persistent_reference_actor_model_preflight(root,prepared)
    return dict(result,explicit_initial_sequence_policy_domain=proof,
        old_full_scope_library_release_claimed=False,formal_VAL_TEST=False)


class NormalCupNativeBackend(NativeActuatorDiagnosticBackend):
    def actor(self,prepared,env,descriptions):
        root=Path(__file__).resolve().parents[2]
        with library_scope(root,prepared['config'],prepared['args'].task):
            actor,identity=super().actor(prepared,env,descriptions)
        previous=actor.metrics_report;scope=deepcopy(binding(root,prepared['config']))
        def metrics(instance):return dict(previous(),normal_cup_primary_domain_v111=deepcopy(scope))
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
