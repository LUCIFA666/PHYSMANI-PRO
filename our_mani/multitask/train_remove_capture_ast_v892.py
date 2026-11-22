"""UNACTIVATED, unqualified TRAIN-only capture-expression observer.

Source-SHA-pinned method bodies are compiled with value-preserving taps. A tap
records only a boolean, public scalar/duration, already computed public Pose,
or the already computed (distance, angle) result of errors(). It never reads
self/env/native objects, predicts, queries held IDs, or calls a predicate again.
Unvisited expressions stay unknown. No tracing/profiling or source-file edit.
Compound conditional expressions (and/or/not/chained comparison) are kept in
their original conditional context and are intentionally not materialized for
logging. Only safe leaf results/operands and already materialized values are
tapped. Uninstrumented compounds remain a distinct unknown category.

This uses scoped class-descriptor replacement in ONE isolated TRAIN process,
because super() bypasses instance-only wrappers. Original globals and __class__
closure cells are retained. Non-target instances call the original methods.
All descriptors/helper globals are restored. CPU parity is NOT native/physical
equivalence, qualification, a fix, or evidence of improved performance.
There is no runner, scheduler, policy change or automatic activation here.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import threading
import types

SCHEMA='r10_unqualified_TRAIN_remove_value_taps_v892'
TASKS=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed')
ROOT=Path(__file__).resolve().parents[2]
_HELPER='_V892_RECORD_ALREADY_EVALUATED'
_MISSING=object()
_RESERVED=set()
_LOCK=threading.Lock()
PINS={
 'physmani_skill_policy_v548.py':'cbc413b3d1c601344f7871b315cbe10abe1ad87533d6ef7210eff55952a982d2',
 'physmani_remove_stages_v611.py':'1d71b23d37d7c847c9614b1672598c907f20d8b7fce4e7f16b8af52b6e8658ba',
 'physmani_terminal_opening_v620.py':'2f27f0a4a1e3a5972ef5c026a05bd02ec489a223f1ec8a67920ac6880811292d',
 'physmani_remove_wp0_v637.py':'c131bb72c35fcca8e5f1981cee938ab0a3146efa1ee1b3652243e4d1d0077c62',
 'physmani_remove_source_roles_v653.py':'a5275003e73d654c20536f335a9c537874bc2e1d1b2220bffced469087d8f199',
 'physmani_remove_intercept_v667.py':'40693b3792250aea65b6b129984bd1a63e31d5997327a4850e2e2e9f7fe337bc',
 'physmani_geometry_policy_v518.py':'ea23722e08bd955675a33411bfc657258f1273534652e7a4a33dd12741c865e0',
}
METHODS={
 'SkillGeometryIntents':('_completion_target','_public_capture','_done','_advance','select','qualify_opening','confirm_opening'),
 'RemoveAuthorIntents':('_done','qualify_opening'),
 'OriginalTerminalOpening':('qualify_opening',),
 'AuthorPreApproachRemoveIntents':('_completion_target','_advance','candidates','select'),
 'AuthorLiveRoleRemoveIntents':('select',),
 'InterceptRemoveIntents':('candidates',),
 'PublicModelFrontend':('ready',),
}


def _sha(raw):return hashlib.sha256(raw).hexdigest()


class _Taps(ast.NodeTransformer):
    def __init__(self):self.labels={}
    def label(self,node,kind,instrumented):
        stem=f'{node.lineno}:{node.col_offset}:{kind}'
        label=stem;duplicate=1
        while label in self.labels:duplicate+=1;label=f'{stem}:{duplicate}'
        self.labels[label]=dict(line=node.lineno,column=node.col_offset,kind=kind,instrumented=instrumented,
            expression=ast.unparse(_Strip().visit(deepcopy(node))))
        return label
    def tap(self,node,kind):
        label=self.label(node,kind,True)
        return ast.copy_location(ast.Call(func=ast.Name(id=_HELPER,ctx=ast.Load()),
            args=[ast.Constant(label),ast.Constant(kind),node],keywords=[]),node)
    def untapped(self,node,kind):
        self.label(node,kind,False);return node
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        return self.untapped(node,'compound_comparison') if len(node.ops)>1 else self.tap(node,'comparison')
    def visit_UnaryOp(self,node):
        value=self.generic_visit(node)
        return self.untapped(value,'compound_not') if isinstance(node.op,ast.Not) else value
    def visit_BoolOp(self,node):
        return self.untapped(self.generic_visit(node),'compound_boolean')
    def visit_If(self,node):
        node=self.generic_visit(node);self.untapped(node.test,'conditional_context');return node
    def visit_While(self,node):
        node=self.generic_visit(node);self.untapped(node.test,'conditional_context');return node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        public_scalar=all(isinstance(t,ast.Name) and t.id in ('desired','command','distance','angle') for t in node.targets)
        node.value=self.tap(node.value,'assigned_public_scalar' if public_scalar else 'assigned_value');return node
    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.ctx,ast.Load) and node.attr in ('timestamp','duration_seconds','phase_epoch'):return self.tap(node,'duration_operand')
        return self.tap(node,'public_boolean_operand') if isinstance(node.ctx,ast.Load) and node.attr in ('contact','gripper_open') else node
    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        clock_field=isinstance(node.slice,ast.Constant) and node.slice.value in ('confirmed_after','issued_at')
        return self.tap(node,'duration_operand') if isinstance(node.ctx,ast.Load) and clock_field else node
    def visit_BinOp(self,node):
        source=ast.unparse(node);node=self.generic_visit(node)
        return self.tap(node,'duration_arithmetic') if any(k in source for k in ('timestamp','duration_seconds','phase_epoch')) else node
    def visit_Call(self,node):
        is_errors=isinstance(node.func,ast.Name) and node.func.id=='errors'
        node=self.generic_visit(node)
        return self.tap(node,'computed_distance_angle') if is_errors else node


class _Strip(ast.NodeTransformer):
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id==_HELPER:return self.visit(node.args[2])
        return self.generic_visit(node)


def compile_tapped(owner,name,*,source_path,expected_sha):
    """Compile one pinned body; stripping taps must reproduce its exact AST."""
    original=owner.__dict__[name]
    if not isinstance(original,types.FunctionType):raise ValueError('plain original instance method required')
    path=Path(source_path);raw=path.read_bytes()
    if _sha(raw)!=expected_sha:raise ValueError('original predicate source SHA changed')
    if Path(original.__code__.co_filename).resolve()!=path.resolve():raise ValueError('method/source identity mismatch')
    tree=ast.parse(raw,filename=str(path))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner.__name__)
    source=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if source.decorator_list:raise ValueError('decorated predicate is outside this minimal observer')
    transformed=deepcopy(source);transformer=_Taps()
    transformed.body=[transformer.visit(n) for n in transformed.body]
    if ast.dump(_Strip().visit(deepcopy(transformed)),include_attributes=False)!=ast.dump(source,include_attributes=False):
        raise ValueError('instrumentation AST parity failed')
    # Do not evaluate the original defaults/annotations a second time. Function
    # construction below restores their original objects without evaluating them.
    transformed.args.defaults=[]
    transformed.args.kw_defaults=[None for _ in transformed.args.kwonlyargs]
    for arg in (*transformed.args.posonlyargs,*transformed.args.args,*transformed.args.kwonlyargs):arg.annotation=None
    if transformed.args.vararg:transformed.args.vararg.annotation=None
    if transformed.args.kwarg:transformed.args.kwarg.annotation=None
    transformed.returns=None
    #A class envelope makes the compiler generate the original super() freevar.
    envelope=ast.ClassDef(name='_V892Envelope',bases=[],keywords=[],body=[transformed],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[envelope],type_ignores=[]))
    namespace={};exec(compile(module,str(path),'exec'),original.__globals__,namespace)
    compiled=namespace['_V892Envelope'].__dict__[name]
    cells=dict(zip(original.__code__.co_freevars,original.__closure__ or ()))
    if set(compiled.__code__.co_freevars)!=set(cells):raise ValueError('original closure/freevar parity required')
    closure=tuple(cells[k] for k in compiled.__code__.co_freevars) or None
    function=types.FunctionType(compiled.__code__,original.__globals__,name,original.__defaults__,closure)
    function.__kwdefaults__=original.__kwdefaults__
    function.__annotations__=original.__annotations__
    return function,transformer.labels,dict(path=str(path),sha256=expected_sha,class_name=owner.__name__,method=name,
        original_MRO_super_closure_preserved=True,stripped_AST_equal=True)


def default_plan(policy):
    plan=[];seen=set()
    for target in (policy,policy.frontend):
        for owner in type(target).__mro__:
            try:sourcefile=inspect.getsourcefile(owner)
            except TypeError:continue
            basename=Path(sourcefile or '').name
            if basename not in PINS:continue
            for name in METHODS.get(owner.__name__,()):
                key=(owner,name)
                if key not in seen and name in owner.__dict__:
                    seen.add(key);plan.append((target,owner,name,inspect.getsourcefile(owner),PINS[basename]))
    required={('SkillGeometryIntents','_done'),('SkillGeometryIntents','_public_capture'),
        ('OriginalTerminalOpening','qualify_opening'),('AuthorPreApproachRemoveIntents','_advance'),('InterceptRemoveIntents','candidates'),('PublicModelFrontend','ready')}
    if not required<={(owner.__name__,name) for _,owner,name,_,_ in plan}:raise ValueError('exact remove policy/frontend MRO required')
    return plan


class TrainRemoveCaptureValueObserver:
    """Disabled by default; authentic TRAIN capability and isolated scope needed."""
    def __init__(self,policy=None,*,domain=None,enabled=False,development_only=False,max_calls=10000):
        self.policy=policy;self.domain=domain;self.enabled=enabled;self.development_only=development_only
        self.max_calls=max_calls;self.calls=[];self.errors=[];self.dropped_calls=0
        self._patches=[];self._globals=[];self._stack=[];self._context=None;self._entered=False;self._reserved=set();self._sequence=0
    def _authorize(self):
        from .rml_component_domain_v640 import ComponentDomain
        if (self.development_only is not True or not isinstance(self.domain,ComponentDomain)
            or self.domain.require(self.policy.task,self.domain.method)!='train_qualification'
            or self.domain.split!='train' or self.policy.task not in TASKS
            or self.domain.method not in ('cv_geometry','context_geometry')):raise ValueError('authentic development-only TRAIN remove capability required')
    def __enter__(self):
        if self._entered:raise RuntimeError('observer cannot be reentered')
        self._entered=True
        if not self.enabled:return self
        try:
            self._authorize()
            if type(self.max_calls) is not int or self.max_calls<1:raise ValueError('positive max_calls required')
            self._thread=threading.get_ident();self._install(default_plan(self.policy));return self
        except BaseException:
            self._restore();raise
    def _install(self,plan):
        reserved={(id(owner),name) for _,owner,name,_,_ in plan}
        with _LOCK:
            if _RESERVED & reserved:raise RuntimeError('overlapping predicate-class scopes prohibited')
            _RESERVED.update(reserved);self._reserved=reserved
        compiled=[]
        for target,owner,name,path,checksum in plan:
            function,labels,receipt=compile_tapped(owner,name,source_path=path,expected_sha=checksum)
            compiled.append((target,owner,name,function,labels,receipt))
        for _,owner,name,function,labels,receipt in compiled:
            namespace=owner.__dict__[name].__globals__
            if not any(namespace is p[0] for p in self._globals):
                if _HELPER in namespace:raise RuntimeError('diagnostic global name already bound')
                namespace[_HELPER]=self._tap;self._globals.append((namespace,_HELPER))
        for target,owner,name,function,labels,receipt in compiled:
            original=owner.__dict__[name]
            wrapper=self._wrapper(target,original,function,labels,receipt)
            self._patches.append((owner,name,original));setattr(owner,name,wrapper)
    def _encode(self,value,kind):
        if value is None or type(value) is bool:return dict(status='recorded',value=value)
        if kind in ('duration_operand','duration_arithmetic','assigned_public_scalar') and type(value) in (int,float):
            return dict(status='recorded',value=value if math.isfinite(value) else None)
        if kind=='computed_distance_angle' and type(value) is tuple and len(value)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in value):
            return dict(status='recorded',distance_m=value[0],angle_rad=value[1])
        if kind=='return_opening_receipt' and type(value) is tuple and len(value)==3 and type(value[0]) in (int,float) and math.isfinite(value[0]):
            return dict(status='recorded_public_opening_only',opening=value[0],proof_and_proposed_identity='not_serialized')
        #Exact public Pose only. Never duck-type/read a native object/getter.
        from ..contracts import Pose
        if type(value) is Pose:
            state=vars(value)
            return dict(status='recorded_public_pose',position=state['position'].tolist(),rotation=state['rotation'].tolist())
        return dict(status='unknown_value_not_on_public_allowlist')
    def _tap(self,label,kind,value):
        if self._stack and threading.get_ident()==self._thread:
            try:self._stack[-1]['evaluations'].setdefault(label,[]).append(self._encode(value,kind))
            except Exception as problem:
                self._stack[-1]['evaluations'].setdefault(label,[]).append(dict(status='unknown_recording_failed',exception_type=type(problem).__name__))
                self.errors.append(dict(operation='encode',type=type(problem).__name__))
        return value
    def _wrapper(self,target,original,function,labels,receipt):
        def observed(owned,*args,**kwargs):
            if owned is not target or threading.get_ident()!=self._thread:return original(owned,*args,**kwargs)
            self._sequence+=1
            event=dict(call_id=self._sequence,parent_call_id=self._stack[-1]['call_id'] if self._stack else None,
                source=receipt,caller_context=dict(self._context) if self._context is not None else None,
                evaluations={},expressions=labels,original_body_executions=1,
                original_function_object_invocations=0,source_body_instrumented=True)
            self._stack.append(event)
            try:
                result=function(owned,*args,**kwargs)
                try:event['return_value']=self._encode(result,'return_opening_receipt' if receipt['method']=='qualify_opening' else 'return')
                except Exception as problem:
                    event['return_value']=dict(status='unknown_recording_failed',exception_type=type(problem).__name__)
                    self.errors.append(dict(operation='return_encode',type=type(problem).__name__))
                event['returned']=True;return result
            except BaseException as problem:
                event['exception']=dict(type=type(problem).__name__);raise
            finally:
                self._stack.pop()
                event['not_evaluated']={key:dict(status='unknown_not_evaluated_on_original_path') for key in labels
                    if labels[key]['instrumented'] and key not in event['evaluations']}
                event['unobserved_compound_conditions']={key:dict(status='unknown_not_instrumented_to_preserve_conditional_truth_semantics')
                    for key in labels if not labels[key]['instrumented']}
                if len(self.calls)<self.max_calls:self.calls.append(event)
                else:self.dropped_calls+=1
        return observed
    def bind_context(self,*,tick,action_id=None,phase=None):
        if not self.enabled:return False
        if not self._entered or threading.get_ident()!=self._thread or type(tick) is not int or tick<0:raise ValueError('active same-thread counted TRAIN context required')
        if action_id is not None and (type(action_id) is not int or action_id<0):raise ValueError('public action index must be a nonnegative int or unknown')
        if phase is not None and (type(phase) is not str or phase not in ('approach','acquire','detach','transport','place','verify')):raise ValueError('known remove policy phase or unknown required')
        self._context=dict(tick=tick,action_id=action_id,phase=phase);return True
    def _restore(self):
        while self._patches:
            owner,name,original=self._patches.pop();setattr(owner,name,original)
        while self._globals:
            namespace,key=self._globals.pop();namespace.pop(key,None)
        with _LOCK:_RESERVED.difference_update(self._reserved)
        self._reserved.clear();self._entered=False
    def __exit__(self,*exc):self._restore();return False
    def receipt(self):
        return dict(schema=SCHEMA,enabled=self.enabled,active=self._entered,restored=not self._patches and not self._globals,
            unqualified=True,native_validated=False,fix_claimed=False,performance_improvement_claimed=False,
            participates_in_formal_qualification=False,participates_in_VAL_TEST=False,
            native_env_teacher_or_held_ID_queries=False,extra_predicate_prediction_or_geometry_calls=0,
            class_descriptor_scope_isolated_process_only=True,full_physical_equivalence_claimed=False,
            unknown_expressions_not_filled=True,calls=deepcopy(self.calls),errors=deepcopy(self.errors),dropped_calls=self.dropped_calls)
