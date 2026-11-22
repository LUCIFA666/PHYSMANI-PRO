"""Source-bound integration of the fitted q90 colour model for TRAIN smoke.

The original classifier and candidate bytes remain immutable. The scope is a
fresh case's actor construction; only public current RGBD and language reach
the resolver. Formal admission requires a subsequent native qualification.
"""
from contextlib import contextmanager
from pathlib import Path
import json

from . import public_roles
from .colour_distribution import DevelopmentQuantileRoleResolver
from .evaluation import file_sha
from .protocol import MultitaskProtocolError

SCHEMA='r10_shared_TRAIN_q90_colour_runtime_integration_v18'
TASKS=('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed',
       'push_moving_button','push_moving_button_high_speed')
_active=False


def require(value,message):
    if not value:raise MultitaskProtocolError(message)


def binding(root,candidate_pointer):
    root=Path(root).resolve();path=(root/candidate_pointer['path']).resolve()
    require(path.is_relative_to(root) and file_sha(path)==candidate_pointer['sha256'],
        'TRAIN colour candidate pointer changed')
    artifact=json.loads(path.read_text());m=artifact['metadata']
    references=[candidate_pointer,m['TRAIN_manifest'],m['parent_crossfit'],
        *m['source_bindings'].values(),*m['evidence'].values()]
    sources={str(Path(__file__).resolve()):file_sha(__file__)}
    for pointer in references:
        p=(root/pointer['path']).resolve()
        require(p.is_relative_to(root) and file_sha(p)==pointer['sha256'],'colour evidence source changed')
        sources[str(p)]=pointer['sha256']
    return dict(schema=SCHEMA,candidate=candidate_pointer,tasks=list(TASKS),
        inputs_sha256=sources,allowed_purpose='train_collection',
        native_target_or_variation_input=False,original_margin_support_and_freshness_gates=True,
        classifier_refitted=False,formal_VAL_TEST_admission=False)


@contextmanager
def factory_colour_scope(root,protocol,task,*,purpose):
    """Install the explicitly bound TRAIN resolver during one factory call."""
    global _active
    require(task in TASKS and purpose=='train_collection',
        'q90 integration currently accepts explicit task TRAIN smoke only')
    pointer=protocol['learned_artifacts'][task]['colour']
    expected=binding(root,pointer)
    require(protocol.get('colour_runtime_v18')==expected,'explicit actual colour runtime binding required')
    require(not _active,'nested colour factory replacement is not allowed')
    original=public_roles.PublicRoleResolver
    def resolver(task_spec,artifact,*,shared_semantic_permission,**kwargs):
        require(task_spec.task_name==task and not kwargs,'foreign task or undeclared colour factory arguments')
        require(artifact==json.loads((Path(root)/pointer['path']).read_text()),'factory colour model differs from frozen candidate')
        return DevelopmentQuantileRoleResolver(task_spec,artifact,root=root,
            shared_semantic_permission=shared_semantic_permission,allow_development_candidate=True)
    _active=True;public_roles.PublicRoleResolver=resolver
    try:
        yield expected
    finally:
        public_roles.PublicRoleResolver=original;_active=False
