"""Exact delayed-import edges proven unreachable for PhysMani VAL.

No module-level import may be removed. Origin sources and the review evidence
stay frozen; accidentally importing an excluded module still fails the actual
runtime dependency observer.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from .evaluation import file_sha

SCHEMA='r10_PhysMani_VAL_unreachable_delayed_import_review_v1'
PRUNED_MODULES=('our_mani.multitask.factory','our_mani.multitask.ring_affordance',
                'scripts.collect_multitask_training_r10',
                'our_mani.multitask.interaction','our_mani.multitask.artifact_fitting',
                'our_mani.multitask.training_view','our_mani.multitask.pipeline','our_mani.multitask.reference_time',
                'our_mani.multitask.acquisition_manifest','our_mani.multitask.release_preflight')
SCOPES={
 'scripts/evaluate_multitask_case_r10.py':{
  'NativeBackend.actor':('our_mani.multitask.factory',),
  'test_release':('our_mani.multitask.release_preflight',)},
 'scripts/run_multitask_fresh_queue_r10.py':{
  'run':('scripts.collect_multitask_training_r10',)}}


def require(value,message):
    if not value:raise ValueError('baseline import scope: '+message)


def import_nodes(root,path):
    path=Path(path).resolve();relative=path.relative_to(Path(root).resolve()).as_posix()
    package=list(Path(relative).with_suffix('').parts[:-1]);rows=[]
    class Visitor(ast.NodeVisitor):
        def __init__(self):self.scope=[]
        def nested(self,node):
            self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
        visit_ClassDef=nested;visit_FunctionDef=nested;visit_AsyncFunctionDef=nested
        def visit_Import(self,node):self.add(node,[alias.name for alias in node.names])
        def visit_ImportFrom(self,node):
            base=package[:len(package)-node.level+1] if node.level else []
            name='.'.join(base+([node.module] if node.module else []))
            self.add(node,[name]+[name+'.'+alias.name if name else alias.name for alias in node.names if alias.name!='*'])
        def add(self,node,modules):
            rows.append(dict(source_path=relative,scope='.'.join(self.scope) or '<module>',line=node.lineno,
                statement=ast.unparse(node),modules=modules))
    Visitor().visit(ast.parse(path.read_text(),filename=str(path)))
    return rows


def expected_nodes(root):
    rows=[]
    for path,scopes in SCOPES.items():
        found=set()
        for node in import_nodes(root,Path(root)/path):
            targets=set(scopes.get(node['scope'],())) & set(node['modules'])
            if targets:
                require(node['scope']!='<module>' and all(module in PRUNED_MODULES
                    or any(module.startswith(target+'.') for target in PRUNED_MODULES) for module in node['modules']),
                    'an import statement also names a real shared module')
                node['excluded_targets']=sorted(targets);rows.append(node)
                found.update((node['scope'],target) for target in targets)
        require(found=={(scope,target) for scope,targets in scopes.items() for target in targets},
                'reviewed exact delayed import edge vanished or moved scope: '+path)
    return sorted(rows,key=lambda node:(node['source_path'],node['line']))


def draft_review(root,evidence):
    return dict(schema=SCHEMA,reviewed_before_execution=False,split='validation',method='physmani',
        excluded_modules=list(PRUNED_MODULES),excluded_import_nodes=expected_nodes(root),
        source_bindings_sha256={str((Path(root)/path).resolve()):file_sha(Path(root)/path)
            for path in (*SCOPES,'scripts/evaluate_multitask_case_r10.py',str(Path(__file__).resolve()))},evidence=deepcopy(evidence),
        module_level_imports_pruned=False,unexpected_actual_imports_rejected=True)


def load_review(root,pointer,*,require_reviewed=True):
    root=Path(root).resolve();require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'exact review pointer required')
    path=(root/pointer['path']).resolve()
    require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'review changed or escaped workspace')
    value=json.loads(path.read_text())
    expected=draft_review(root,value.get('evidence'));expected['reviewed_before_execution']=require_reviewed
    require(value==expected,'delayed-edge review widened, source bytes changed, or approval missing')
    evidence=value['evidence']
    require(isinstance(evidence,dict) and set(evidence)=={'path','sha256'},'exact parent branch evidence pointer required')
    evidence_path=(root/evidence['path']).resolve()
    require(evidence_path.is_relative_to(root) and file_sha(evidence_path)==evidence['sha256'],'branch evidence changed')
    parent=json.loads(evidence_path.read_text())
    require(parent.get('schema') in ('r10_ROOT_PhysMani_VAL_ours_only_import_edge_review_v1',
        'r10_ROOT_PhysMani_VAL_ours_only_import_edge_review_v2','r10_ROOT_PhysMani_VAL_ours_only_import_edge_review_v3')
        and parent.get('reviewed_for')==dict(split='validation',method='physmani',replica=0,record_demo=False,demo_only=False)
        and parent.get('required_bound_source_sha256')=={name:file_sha(root/name)
            for name in (*SCOPES,'scripts/evaluate_multitask_case_r10.py')},
        'ROOT parent evidence is stale or belongs to another split/method/owner source')
    return value
