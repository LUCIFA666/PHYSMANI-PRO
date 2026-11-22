"""Runtime admission of an exactly replayed, source-bound full-TRAIN proof.

Full measured action orbits remain an offline proof artifact. Native cases
verify its inputs and use a sealed small capability, avoiding repeated JSON
hashing of all TRAIN trajectories on every collision sample.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
from pathlib import Path
import json

from . import ring_affordance as original
from .high_peg_affordance_v241 import VerifiedRingAffordance as FullTRAINAffordance,PHASES,SCHEMA
from .rotation_equivalence import VerifiedFiniteSymmetry
from .evaluation import digest,file_sha
from .protocol import MultitaskProtocolError

_TOKEN=object()
_active=False
RUNTIME_SCHEMA='r10_source_bound_highpeg_initial_full100_ring_runtime_v243'


def require(value,message):
    if not value:raise MultitaskProtocolError(message)


def checked(root,pointer):
    path=(root/pointer['path']).resolve()
    require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'ring runtime proof bytes changed')
    return path


def evidence(root,permission_pointer,replay_report):
    root=Path(root).resolve();path=checked(root,permission_pointer);report=json.loads(checked(root,replay_report).read_text())
    p=json.loads(path.read_text());rows=[r for r in report['entries'] if r['task']==p['task']]
    require(len(rows)==1 and rows[0]['permission']==permission_pointer and report.get('complete') is True
        and report.get('TRAIN_only') is True and report.get('VAL_read') is False and report.get('TEST_read') is False,
        'completed original full-TRAIN replay report required')
    row=rows[0]
    require(row.get('all_permission_fields_replayed_exactly') is True
        and row.get('reproducible_loaded_permission') is True and row.get('policy_skill_authorized') is True
        and p.get('policy_skill_authorized') is True and p.get('development_only') is False
        and p.get('schema')==SCHEMA and p.get('permission_sha256')==digest({k:v for k,v in p.items() if k!='permission_sha256'})
        and report['source_manifest']==p['TRAIN_manifest'],'ring replay is pending, mismatched or changed')
    checked(root,report['source_script'])
    from .high_peg_affordance_v241 import _native_rule
    require(_native_rule(p['native_gripper_source'],p['native_task_source'])==p['native_grasp_semantics'],
        'actual native grasp/task semantics changed after the complete replay')
    inputs=dict(p['inputs_sha256']);inputs[permission_pointer['path']]=permission_pointer['sha256']
    inputs[replay_report['path']]=replay_report['sha256'];inputs[report['source_script']['path']]=report['source_script']['sha256']
    for name,sha in inputs.items():checked(root,dict(path=name,sha256=sha))
    for phase in PHASES:
        support=p['phase_support'][phase]
        require(support['passed'] is True and support['minimum_completed_parents']==8
            and support['completed_parent_count']==len(set(support['completed_parent_groups']))>=8,
            'original eight-parent phase support gate failed')
    return p,inputs


def binding(root,permission_pointer,replay_report):
    p,inputs=evidence(root,permission_pointer,replay_report)
    root=Path(root).resolve();inputs[str(Path(__file__).relative_to(root))]=file_sha(__file__)
    return dict(schema=RUNTIME_SCHEMA,task=p['task'],permission=permission_pointer,replay_report=replay_report,
        library=p['library'],inputs_sha256=inputs,original_complete_replay_passed=True,
        runtime_payload_contains_TRAIN_action_orbits=False,full_orientation_promoted=False,
        collision_and_original_grasp_gates_changed=False)


@dataclass(frozen=True)
class RuntimeRingAffordance(FullTRAINAffordance,original.VerifiedRingAffordance):
    full_permission_pointer:dict=field(repr=False,compare=False)
    exact_replay_report:dict=field(repr=False,compare=False)
    _projection_sha256:str=field(repr=False,compare=False)

    def __post_init__(self):
        require(self._token is _TOKEN,'ring capability requires verified full-TRAIN admission')
        self._check()

    def _check(self):
        p=self.permission
        require(p.get('schema')==SCHEMA and digest(p)==self._projection_sha256
            and p.get('certificate_sha256')==self.symmetry.certificate['certificate_sha256'],
            'sealed runtime ring capability was mutated')

    @classmethod
    def load(cls,root,pointer,*,replay_report,allow_development=False):
        require(allow_development is False,'production ring admission cannot use one-parent development support')
        p,_=evidence(root,pointer,replay_report)
        keys=('schema','permission_sha256','certificate_sha256','finite_symmetry','task','body_id',
            'allowed_reference_phases','development_only','policy_skill_authorized','status','TRAIN_only',
            'VAL_used','TEST_used','library','templates','phase_support','full_orientation_observable')
        compact={k:p[k] for k in keys}
        return cls(compact,VerifiedFiniteSymmetry.load(root,p['finite_symmetry']),_TOKEN,
            pointer,replay_report,digest(compact))


@contextmanager
def factory_ring_scope(root,protocol,task):
    global _active
    settings=protocol['high_peg_ring_runtime_v243'][task]
    expected=binding(root,protocol['ring_affordances'][task],settings['replay_report'])
    require(settings==expected and task==expected['task'],'explicit actual ring runtime binding required')
    require(expected['library']['file']==protocol['learned_artifacts'][task]['interaction'],
        'ring capability belongs to another fitted skill library')
    require(not _active,'nested ring factory construction rejected')
    original_class=original.VerifiedRingAffordance
    class BoundRuntimeRing(RuntimeRingAffordance):
        @classmethod
        def load(cls,case_root,pointer,*,allow_development=False):
            return super().load(case_root,pointer,replay_report=settings['replay_report'],allow_development=allow_development)
    original.VerifiedRingAffordance=BoundRuntimeRing;_active=True
    try:yield expected
    finally:original.VerifiedRingAffordance=original_class;_active=False
