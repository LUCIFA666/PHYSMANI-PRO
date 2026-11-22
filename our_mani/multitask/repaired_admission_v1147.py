"""Admit a frozen repaired policy before reading its existing VAL manifest.

TRAIN192 evidence is produced after the immutable source/control graph.  This
module never permits TEST and never uses a TRAIN success rate as a gate.  The
resource capability belongs to this process for the whole admission lifetime.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import repaired_policy_graph_v1143 as graphs
from .evaluation_roster16_v283 import TASKS, METHODS, validate_roster
from .common_rml_protocol_v530 import fingerprint

SOURCE = 'our_mani/multitask/repaired_admission_v1147.py'
SCHEMA = 'r10_repaired_validation_admission_v1147'
SUITE_SCHEMA = 'r10_repaired_validation_suite_v1147'
READINESS_SCHEMA = 'r10_fresh_TRAIN192_readiness_v1144'
SPLIT_LOCK_SCHEMA = 'r10_full16_post_admission_split_source_lock_v1'
VAL_COUNT = 315
# Expected pointer comes from the already frozen752 suite metadata.  This
# constant does not read the manifest while compiling a new policy graph.
VALIDATION_MANIFEST = dict(path='artifacts/framework/protocol/validation_manifest.json',
    sha256='70498a21eabd28e55235672371a224047ec25f10857b90bdddfde7d24aa9f6bf')
_active = ContextVar('repaired_validation_admission_v1147', default=None)


def require(value, message):
    if not value:
        raise ValueError('repaired admission1147: '+message)


def plain(pointer):
    require(type(pointer) is dict and set(pointer) <= {'root','path','sha256'}
        and pointer.get('root','project') == 'project'
        and set(('path','sha256')) <= set(pointer), 'project evidence pointer required')
    return dict(path=pointer['path'],sha256=pointer['sha256'])


def point(root, path):
    root=Path(root).resolve();path=Path(path)
    require(path.is_file() and path.resolve()==path and path.is_relative_to(root),
        'canonical project evidence file required')
    return dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _read(root, pointer):
    value, normalized=graphs.read_project_json(root, plain(pointer))
    return value, plain(normalized)


def _permit(root, permit):
    from .post_completed_TEST_resource_guard_v1106 import require_permit
    return require_permit(permit, Path(root).resolve())


def _readiness(root, graph_ref, proof_ref):
    from .repaired_readiness_v1144 import require_readiness_proof
    return require_readiness_proof(root, graph_ref, proof_ref)


def _graph(root, graph_ref):
    graph=graphs.read_policy_graph(root, plain(graph_ref))
    contracts=graph['control']['contracts']
    require(contracts == dict(readiness_schema=READINESS_SCHEMA,
        admission_schema=SCHEMA,split_source_lock_schema=SPLIT_LOCK_SCHEMA),
        'frozen repaired readiness/admission/source contracts required')
    current=point(root,Path(root)/SOURCE)
    for component in ('admission_guard','formal_source_reader'):
        require(plain(graph['control']['components'][component])==current,
            'actual admission reader differs from frozen control graph')
    return graph


def _proof(root, graph_ref, graph, proof_ref):
    proof=_readiness(root,plain(graph_ref),plain(proof_ref))
    require(proof.get('schema')==READINESS_SCHEMA and proof.get('complete') is True
        and proof.get('completed')==192 and proof.get('method_case_count')==192
        and proof.get('source_count')==64 and proof.get('task_ready_count')==16
        and proof.get('all_failures_retained') is True
        and proof.get('native_success_is_not_an_admission_threshold') is True
        and proof.get('automatic_retry') is False and proof.get('source_outcome_filtering') is False
        and proof.get('VAL_read') is False and proof.get('TEST_read') is False,
        'complete actual fresh TRAIN192 evidence required before VAL')
    require(plain(proof.get('graph'))==plain(graph_ref)
        and all(proof.get(key)==graph[key] for key in
            ('policy_sha256','physical_policy_sha256','control_graph_sha256')),
        'TRAIN192 proof belongs to a different frozen graph')
    return proof


def _manifest(root, reference):
    require(plain(reference)==VALIDATION_MANIFEST,
        'only the frozen original315 VAL manifest pointer is accepted before any source read')
    manifest, ref=_read(root,reference)
    rows=validate_roster(manifest,'validation')
    require(len(rows)==VAL_COUNT and len({(r['task'],r['raw_episode_id']) for r in rows})==VAL_COUNT,
        'the complete original315-source VAL roster is required')
    return manifest,rows,ref


def routes_from_graph(root, graph):
    """Old742-compatible substrate routes; actual policy approval remains new."""
    profile,_=_read(root,graph['physical']['common_RML']['profile'])
    profile_sha=fingerprint(profile)
    rows=[]
    for route in graph['routes']:
        execution=route['evaluation']
        rows.append(dict(task=route['task'],method=route['method'],GPU_required=route['GPU_required'],
            configuration=plain(route['configuration']),frozen_inputs=plain(route['frozen_inputs']),
            worker=plain(execution['worker']),consumer=plain(execution['consumer']),
            source_preflight=plain(execution['preparer']),common_RML_profile_sha256=profile_sha,
            repaired_policy_sha256=graph['policy_sha256'],
            physical_policy_sha256=graph['physical_policy_sha256'],
            control_graph_sha256=graph['control_graph_sha256'],
            physical_factory=route['physical_factory']))
    return dict(routes=rows,common_author_RML_profile=profile,
        shared_context_checkpoint_sha256=graph['physical']['shared_context_checkpoint']['sha256'],
        original_745_execution_acceptance_granted=False)


def _fresh(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def create_admission(root, graph_ref, readiness_ref, validation_manifest_ref, *, namespace, permit):
    """First-read VAL only after capability, source graph and all192 cases pass."""
    root=Path(root).resolve();_permit(root,permit)
    graph=_graph(root,graph_ref);_proof(root,graph_ref,graph,readiness_ref)
    _permit(root,permit)
    _,_,manifest_ref=_manifest(root,validation_manifest_ref)
    name=graphs._relative(namespace)
    require(name.startswith('artifacts/framework/multitask_r10/') and len(Path(name).parts)>3,
        'explicit new admission artifact namespace required')
    directory=root/name
    require(directory.parent.is_dir() and directory.parent.resolve()==directory.parent,
        'canonical existing namespace parent required')
    directory.mkdir(exist_ok=False)
    route_path=directory/'routes.json';_fresh(route_path,routes_from_graph(root,graph))
    admission=dict(schema=SCHEMA,graph=plain(graph_ref),readiness=plain(readiness_ref),
        **{key:graph[key] for key in ('policy_sha256','physical_policy_sha256','control_graph_sha256')},
        routes=point(root,route_path),source_manifests={'validation':manifest_ref},
        source_count=VAL_COUNT,method_case_count=945,allowed_splits=['validation'],
        tasks=list(TASKS),methods=list(METHODS),complete=True,
        source_read_after_fresh_TRAIN192_admission=True,
        existing_VAL_manifest_bytes_new_policy=True,new_independent_validation_dataset=False,
        formal_TEST_permitted=False,TEST_used_for_fitting=False,
        original_745_execution_acceptance_granted=False)
    path=directory/'admission.json';_fresh(path,admission)
    _permit(root,permit)
    return point(root,path)


def _check_admission(root, graph_ref, admission_ref, permit):
    _permit(root,permit);graph=_graph(root,graph_ref)
    admission,ref=_read(root,admission_ref)
    require(admission.get('schema')==SCHEMA and admission.get('complete') is True
        and plain(admission.get('graph'))==plain(graph_ref)
        and admission.get('allowed_splits')==['validation']
        and admission.get('formal_TEST_permitted') is False
        and admission.get('TEST_used_for_fitting') is False
        and admission.get('source_manifests')=={'validation':VALIDATION_MANIFEST}
        and admission.get('source_count')==VAL_COUNT and admission.get('method_case_count')==945
        and admission.get('tasks')==list(TASKS) and admission.get('methods')==list(METHODS)
        and all(admission.get(key)==graph[key] for key in
            ('policy_sha256','physical_policy_sha256','control_graph_sha256')),
        'new immutable same-policy VAL-only admission required')
    _proof(root,graph_ref,graph,admission['readiness'])
    routes,_=_read(root,admission['routes'])
    require(routes==routes_from_graph(root,graph),'admitted substrate routes differ from frozen graph')
    return dict(root=root,graph_ref=plain(graph_ref),graph=graph,admission_ref=ref,
        admission=admission,permit=permit,manifest_reads=0)


@contextmanager
def admission_scope(root, graph_ref, admission_ref, permit):
    """Enter before742 preparation, including its declared input verification."""
    require(_active.get() is None,'nested or leaked admission scope forbidden')
    current=_check_admission(Path(root).resolve(),graph_ref,admission_ref,permit)
    token=_active.set(current)
    try:
        yield current
    finally:
        _active.reset(token)


def read_formal_source(root, suite, files, split, *, validation_release=None):
    """Signature for the scoped640→532 reader; TEST remains unavailable."""
    require(split=='validation' and validation_release is None,'only new VAL is authorized')
    current=_active.get();root=Path(root).resolve()
    require(current is not None and current['root']==root,'active same-process admission capability required')
    _permit(root,current['permit'])
    admission=current['admission']
    require(type(suite) is dict and suite.get('schema')==SUITE_SCHEMA
        and plain(suite.get('graph'))==current['graph_ref']
        and plain(suite.get('admission'))==current['admission_ref']
        and suite.get('source_manifests')==admission['source_manifests']
        and suite.get('routes')==admission['routes'], 'actual suite differs from admitted graph and sources')
    require(type(files) is dict,'explicit source/evidence lock required')
    for ref in (current['graph_ref'],current['admission_ref'],admission['routes'],
                admission['source_manifests']['validation']):
        path=str(root/ref['path'])
        require(files.get(path)==ref['sha256'],'actual suite evidence lock omits an admitted input')
    latest=_graph(root,current['graph_ref'])
    require(latest==current['graph'],'source/control graph changed within admission scope')
    latest_admission,_=_read(root,current['admission_ref'])
    require(latest_admission==admission,'admission changed within its active scope')
    manifest,rows,_=_manifest(root,admission['source_manifests']['validation'])
    current['manifest_reads']+=1
    proof=deepcopy(admission)
    proof.update(actual_split='validation',component_purpose='evaluation',
        new_admission_reader=SCHEMA,formal_TEST_permitted=False)
    return manifest,rows,proof


def admission_metrics():
    current=_active.get()
    return dict(schema=SCHEMA,active=current is not None,
        admitted_manifest_reads=0 if current is None else current['manifest_reads'],
        TEST_permitted=False)
