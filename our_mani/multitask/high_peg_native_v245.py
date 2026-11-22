"""Explicit highpeg initial-sequence consumer with all100 TRAIN D4 evidence."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
import json
from . import interaction
from .high_peg_primary_fitting_v231 import PrimarySequenceLibrary, TASK
from .high_peg_ring_runtime_v243 import binding as ring_binding, factory_ring_scope
from .high_peg_operational_gauge_v244 import binding as gauge_binding, install as install_gauge
from .joint16_native_v173 import model_preflight as original_preflight
from .joint16_native_v173 import SharedJoint16NativeBackend as OriginalBackend
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset

ORIGINAL_LIBRARY=interaction.InteractionLibrary
SOURCES=('high_peg_native_v245.py','high_peg_primary_fitting_v231.py','minimax_terminal_v220.py',
    'phase_prefix_v230.py','high_peg_affordance_v241.py','high_peg_ring_runtime_v243.py',
    'high_peg_operational_gauge_v244.py')


def binding(root,protocol):
    root=Path(root).resolve();pointer=protocol['high_peg_primary_report_v232'];path=root/pointer['path']
    if file_sha(path)!=pointer['sha256']:raise ValueError('initial highpeg fit report differs')
    report=json.loads(path.read_text())
    if (report['stage']!='all_initial_sequence_original_gates_passed_new_estimator'
            or report['model']!=protocol['learned_artifacts'][TASK]['interaction']
            or report['original_teacher_parents']!=100 or report['required_phase_failures']):
        raise ValueError('qualified original100 initial highpeg fit required')
    library=PrimarySequenceLibrary.load(root/report['model']['path'])
    ring=ring_binding(root,protocol['ring_affordances'][TASK],
        protocol['high_peg_ring_runtime_v243'][TASK]['replay_report'])
    gauge=gauge_binding(root,protocol,TASK)
    if protocol['high_peg_ring_runtime_v243'][TASK]!=ring or protocol['high_peg_operational_gauge_v244'][TASK]!=gauge:
        raise ValueError('actual highpeg ring proof or operational gauge differs')
    return dict(schema='r10_highpeg_explicit_primary_and_full100_D4_TRAIN_consumer_v245',
        report=pointer,model=report['model'],scope=report['policy_scope'],
        phase_order=list(library.phase_order),original_native_optional_tail=library.metadata['omitted_optional_phases'],
        ring=ring,gauge=gauge,
        sources_sha256={f'our_mani/multitask/{name}':file_sha(root/'our_mani/multitask'/name) for name in SOURCES},
        initial_sequence_only=True,released_recovery_unconditionally_unsupported=True,
        old_full_scope_library_release_claimed=False,original8_point03_point15_fit_gates_unchanged=True,
        original_native_physics_success_collision_motion_and_global10s_unchanged=True,
        TRAIN_only=True,formal_VAL_TEST=False)


@contextmanager
def library_scope(root,protocol,task):
    if task!=TASK or protocol.get('high_peg_primary_v245')!=binding(root,protocol):
        raise ValueError('freeze the explicit highpeg initial-sequence consumer and all100 proof')
    if interaction.InteractionLibrary is not ORIGINAL_LIBRARY:
        raise ValueError('nested or mixed library scopes forbidden')
    interaction.InteractionLibrary=PrimarySequenceLibrary
    try:
        with factory_ring_scope(root,protocol,task):yield
    finally:interaction.InteractionLibrary=ORIGINAL_LIBRARY


def model_preflight(root,prepared):
    args,protocol=prepared['args'],prepared['config'];root=Path(root).resolve()
    if args.task!=TASK or args.split!='train' or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('fixed original highpeg TRAIN development only')
    proof=binding(root,protocol)
    expected={str(root/name):sha for name,sha in proof['sources_sha256'].items()}
    expected[str(root/proof['report']['path'])]=proof['report']['sha256']
    checked_subset(root,prepared['files'],expected)
    report=json.loads((root/proof['report']['path']).read_text())
    checked_subset(root,prepared['files'],report['verified_files_sha256'])
    checked_subset(root,prepared['files'],{str(root/name):sha for name,sha in proof['ring']['inputs_sha256'].items()})
    with library_scope(root,protocol,args.task):result=original_preflight(root,prepared)
    return dict(result,high_peg_primary_v245=proof,old_full_scope_library_release_claimed=False,
        original_native_optional_tail_not_fabricated=True,formal_VAL_TEST=False)


class SharedJoint16NativeBackend(OriginalBackend):
    def actor(self,prepared,env,descriptions):
        root=Path(__file__).resolve().parents[2];protocol=prepared['config']
        with library_scope(root,protocol,prepared['args'].task):
            actor,identity=super().actor(prepared,env,descriptions)
        install_gauge(actor,protocol,root,prepared['args'].task)
        previous=actor.metrics_report;proof=deepcopy(binding(root,protocol))
        def metrics(instance):return dict(previous(),high_peg_primary_v245=deepcopy(proof))
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
