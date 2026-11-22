"""Three fixed TRAIN diagnostic cases; one profiler and no policy change.

The separately locked1154/core1140/block1153 controller, original Native loop and outcome
sampling are inherited. A new diagnostic identity/worker/source lock is explicit;
these cases never impersonate1146 formal invocation or grant formal score admission.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import train_block_preclose_full16_overlay_v1154 as parent
from .rml_case_preflight_v742 import SOURCES as SUBSTRATE_SOURCES
from .passive_execution_diagnostic_v1162 import _PROJECT_SITES,_EXECUTION_SITES

SOURCE = 'our_mani/multitask/passive_execution_case_v1163.py'
WORKER = 'scripts/run_TRAIN_passive_execution_case_v1163_r10.py'
TEST_SOURCE = 'tests/test_passive_execution_case_v1163.py'
SCHEMA = 'r10_fixed_three_TRAIN_passive_execution_case_v1163'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_CPU_integration_evidence'
NAME = 'TRAIN_passive_execution_v1163'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
RUNTIME_PREFIX = 'runtime/framework/' + NAME + '/cases/'
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_block_preclose_full16_v1154_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9'
PARENT_PINS = {
    parent.SOURCE:'8a4b8a2019a979b4483efce9b39f5ebd7671450ed94771dd897fab04d29ca9c0',
    parent.WORKER:'7e64cb2b7fa66017aeeeca3a7f6d4a684910ee6c1d4093844f46aedd7d2b4d2f',
    parent.TEST_SOURCE:'1981d432b3ac106adbd38f4682aa742ef2e3f9374324c11fd307c50d114a76a0',
}
PASSIVE_PINS = {
    'our_mani/multitask/train_public_execution_trace_v1126.py': 'ffa610f07d45fafc1ef43785e66b86f111af20408ab650b9a042e4da122921bc',
    'tests/test_train_public_execution_trace_v1126.py': '7d64bc3d5eb519a56f0113c934f745a23086d9c12a9d9bb7f92371de0145d743',
    'our_mani/multitask/train_public_readiness_probe_v1134.py': '38b85888f726efc887a74f8f0ec47e12824cf613b9e28ea9fd79f4166949d3ec',
    'tests/test_train_public_readiness_probe_v1134.py': '393f56c155ed3e3147a2fb08e79dbbd8cc67c27b46392c401449ddc163f95be0',
    'our_mani/multitask/train_ring_success_tap_v1125.py': '73f12acae977616c4f5f754d3a827eb9d28ca0356dae606878c681551912e003',
    'tests/test_train_ring_success_tap_v1125.py': 'b89f2def4ad5faae895e2ccbc5d1e7b804b3ace7fe3556044fd7370b0ada3c06',
    'our_mani/multitask/train_buzz_failure_tap_v1105.py': '798308727ef9d4f32e3f3247f867d5dc17a07e81bf09e77b2e040223d564bdc4',
    'tests/test_train_buzz_failure_tap_v1105.py': '7e294521ce2e12a172c91a4a2e92f399be07303b82bc107f71a51c6ba184afdd',
}
PASSIVE_PINS.update({
    'our_mani/multitask/passive_execution_diagnostic_v1162.py':'ad4f4e41909b482b725295305b62c5ff30a0d2805e7b94426c605f0afd1918eb',
    'tests/test_passive_execution_diagnostic_v1162.py':'44ed72e6f48d7abed770734619a029e760f5c2dbed4801e80868a2b8e8e3eba0',
})
SELECTIONS = (
    ('insert_onto_rotating_peg','episode112','context_geometry'),
    ('insert_onto_rotating_peg_high_speed','episode102','cv_geometry'),
    ('insert_onto_rotating_peg_high_speed','episode102','context_geometry'),
)
PROFILE_PROJECT_SOURCES=tuple(dict.fromkeys(
    ['our_mani/multitask/'+row[1] for row in _PROJECT_SITES]+[row[1] for row in _EXECUTION_SITES]))
OVERLAY_SOURCES = tuple(dict.fromkeys((SOURCE,WORKER,TEST_SOURCE,*parent.OVERLAY_SOURCES,*SUBSTRATE_SOURCES,*PROFILE_PROJECT_SOURCES,
    parent.SOURCE,parent.WORKER,parent.TEST_SOURCE,PARENT_LOCK,*PASSIVE_PINS,
    'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py',
    'our_mani/multitask/repaired_readiness_v1144.py')))
PHYSICAL_REFERENCE = dict(path='artifacts/framework/multitask_r10/repaired_policy_conservative_v1159_setup/graph/policy_graph.json',
    sha256='607ce1154ebcc8a8fe221afa37ac95cb902d56a3a6069c3ee48068a5c1466b91')
TRAIN_OWNER='runtime/framework/repaired_policy_owner_v1149/train/conservative_v1159_train192'
TRAIN_PLAN_SHA='a547505b01632a2eb5cc55ab5d7c4d7202bf2b918615b63321835c2a1711e046'
NATIVE_SOURCE_NAMES=(
    'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',
    'third_party/RLBench/rlbench/backend/task.py','third_party/RLBench/rlbench/backend/conditions.py',
    'third_party/RLBench/rlbench/tasks/insert_onto_rotating_peg.py',
    'third_party/RLBench/rlbench/tasks/insert_onto_rotating_peg_high_speed.py')
def require(value, message):
    if not value:
        raise ValueError('TRAIN1163: ' + message)


def _parent_lock(root):
    for name, expected in PARENT_PINS.items():
        parent.raw(root, name, expected)
    parent.raw(root, PARENT_LOCK, PARENT_LOCK_SHA)
    value, reference = parent.read_overlay_lock(root, PARENT_LOCK)
    parent.require_native_ready(dict(source_overlay=value))
    parent.raw(root,PHYSICAL_REFERENCE['path'],PHYSICAL_REFERENCE['sha256'])
    return value, reference


def _cpu_evidence(root, reference, files):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}
        and reference['path'].startswith(SETUP_PREFIX), 'explicit new diagnostic CPU evidence pointer required')
    value = json.loads(parent.raw(root, reference['path'], reference['sha256']))
    require(value.get('schema') == CPU_EVIDENCE_SCHEMA and value.get('complete') is True
        and type(value.get('tests_exit_code')) is int and value['tests_exit_code'] == 0
        and value.get('native_started') is False and value.get('GPU_started') is False
        and value.get('source_hashes') == files
        and value.get('parent_source_overlay_lock') == dict(path=PARENT_LOCK, sha256=PARENT_LOCK_SHA)
        and value.get('native_source_hashes') == _native_pins(root),
        'complete CPU evidence must bind every diagnostic source and exact parent lock')


def write_diagnostic_lock(root, name, *, integration_complete=False, integration_cpu_evidence=None):
    """Fresh lock only; pending sources cannot acquire Native permission."""
    root = Path(root).resolve()
    path = parent._path(root, name, prefix=SETUP_PREFIX, must_exist=False)
    require(type(integration_complete) is bool, 'explicit integration state required')
    _, reference = _parent_lock(root)
    for source, expected in PASSIVE_PINS.items(): parent.raw(root, source, expected)
    files = {}; unpinned = []
    for source in OVERLAY_SOURCES:
        if (root/source).exists(): files[source] = parent.pointer(root, source)['sha256']
        else: unpinned.append(source)
    require(not integration_complete or not unpinned, 'all passive diagnostic sources must exist')
    if integration_complete:
        _cpu_evidence(root, integration_cpu_evidence, files)
    else:
        require(integration_cpu_evidence is None, 'pending integration cannot claim CPU completion')
    value = dict(schema=LOCK_SCHEMA, required_sources=list(OVERLAY_SOURCES), files=files,
        unpinned_sources=unpinned, integration_complete=integration_complete,
        integration_cpu_evidence=integration_cpu_evidence, parent_source_overlay_lock=reference,
        native_source_hashes=_native_pins(root),physical_reference_graph=PHYSICAL_REFERENCE,
        fixed_TRAIN_sources=[list(row) for row in SELECTIONS], diagnostic_only=True,
        automatic_retry=False, formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        TEST_used_for_fitting=False, additional_native_getters_or_condition_calls=0)
    parent.write_json(path, value)
    return parent.pointer(root, name)


def read_diagnostic_lock(root, name):
    parent._path(root, name, prefix=SETUP_PREFIX)
    value = json.loads(parent.raw(root, name)); files = value.get('files')
    require(value.get('schema') == LOCK_SCHEMA and value.get('required_sources') == list(OVERLAY_SOURCES)
        and type(files) is dict and set(files) <= set(OVERLAY_SOURCES)
        and value.get('unpinned_sources') == [s for s in OVERLAY_SOURCES if s not in files],
        'honest diagnostic source roster required')
    _, reference = _parent_lock(root)
    for source, expected in PASSIVE_PINS.items(): parent.raw(root, source, expected)
    require(value.get('native_source_hashes') == _native_pins(root)
        and value.get('physical_reference_graph') == PHYSICAL_REFERENCE
        and value.get('parent_source_overlay_lock') == reference
        and value.get('fixed_TRAIN_sources') == [list(row) for row in SELECTIONS]
        and value.get('diagnostic_only') is True and value.get('automatic_retry') is False
        and value.get('formal_score_contribution') is False
        and value.get('formal_VAL_TEST_qualification') is False
        and value.get('TEST_used_for_fitting') is False
        and value.get('additional_native_getters_or_condition_calls') == 0,
        'fixed source-only passive diagnostic contract required')
    for source, expected in files.items(): parent.raw(root, source, expected)
    require(type(value.get('integration_complete')) is bool, 'explicit integration state required')
    if value['integration_complete']:
        require(not value['unpinned_sources'], 'Native cannot use missing diagnostic sources')
        _cpu_evidence(root, value.get('integration_cpu_evidence'), files)
    else:
        require(value.get('integration_cpu_evidence') is None, 'pending integration cannot claim CPU completion')
    return value, parent.pointer(root, name)


def require_native_ready(prepared):
    require(prepared['diagnostic_overlay']['integration_complete'] is True,
            'diagnostic integration pending; Native disabled')
    require(type(prepared.get('training_completion')) is dict,
            'Root actual completed TRAIN192 evidence required before Native')
    parent.require_native_ready(prepared)


def selection(root, args):
    validate_request(args)
    key = (args.task, args.raw_episode_id, args.method)
    require(key in SELECTIONS, 'only three fixed original TRAIN diagnostic sources; no replacement')
    return parent.selection(root, args)


def prepare(root, args):
    root = Path(root).resolve(); validate_request(args)
    training_completion=(require_training_complete(root,getattr(args,'training_exit',None))
                         if args.execute else None)
    chosen = selection(root, args)
    lock, reference = read_diagnostic_lock(root, args.diagnostic_lock)
    require(all(s in lock['files'] for s in (SOURCE, WORKER, TEST_SOURCE)),
            'actual diagnostic worker/preparer/tests must be pinned even for CPU check')
    out = parent._path(root, args.out, prefix=ARTIFACT_PREFIX, must_exist=False)
    require(not out.exists(), 'fresh diagnostic output required; no overwrite/retry')
    forwarded = SimpleNamespace(**vars(args))
    forwarded.overlay_lock = PARENT_LOCK
    # Only the namespace validation constant is scoped; no source/control change.
    with patch.object(parent, 'ARTIFACT_PREFIX', ARTIFACT_PREFIX):
        prepared = parent.prepare(root, forwarded)
    inherited_identity = deepcopy(prepared['identity'])
    inherited_protocol = deepcopy(prepared['execution_protocol'])
    protocol = dict(schema=SCHEMA, diagnostic_only=True,
        actual_worker=parent.pointer(root, WORKER), actual_preparer=parent.pointer(root, SOURCE),
        diagnostic_source_overlay_lock=reference,
        parent_source_overlay_lock=prepared['source_overlay_pointer'],
        inherited_1154_execution_protocol=inherited_protocol,
        inherited_1154_identity_sha256=inherited_identity['sha256'],
        actual_worker_is_parent1154=False,
        scoped_parent_prepare_output_namespace=ARTIFACT_PREFIX,
        fixed_TRAIN_sources=[list(row) for row in SELECTIONS],
        passive_readiness_probe='v1134_original_ready_readiness_candidates_observed_reference_calls',
        passive_public_execution_trace='v1126_existing_inputs_and_path_results_only',
        passive_native_conditions='v1125_original_ring_success_and_four_condition_returns',
        sysprofile_installed_during_native_only=True, simultaneous_profile_scopes=1,
        physical_reference_graph=PHYSICAL_REFERENCE,formal_graph_worker_approval=False,
        execution_observer='v1162_one_profile_owner_with_v1125_callback_only',
        additional_native_getters_or_success_or_condition_calls=0,
        Native_loop_policy_models_outcome_sampling_unchanged=True,
        formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        original_745_execution_acceptance_granted=False, automatic_retry=False)
    protocol['selection_identity']=selection_contract(root,args,chosen,reference)
    identity = dict(inherited_identity); identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA, execution_protocol_sha256=parent.digest(protocol),
        source_overlay_lock_sha256=reference['sha256'],
        inherited_1154_identity_sha256=inherited_identity['sha256'])
    identity['sha256'] = parent.digest(identity)
    prepared.update(identity=identity, execution_protocol=protocol,
        diagnostic_overlay=lock, diagnostic_overlay_pointer=reference,
        diagnostic_lock_name=args.diagnostic_lock, diagnostic_selection=chosen,
        inherited_1154_identity=inherited_identity)
    prepared.update(training_completion=training_completion,
                    training_exit_name=getattr(args,'training_exit',None),
                    selection_identity=protocol['selection_identity'])
    return prepared


def verify(root, prepared):
    parent.verify(root, prepared)
    value, reference = read_diagnostic_lock(root, prepared['diagnostic_lock_name'])
    require(value == prepared['diagnostic_overlay'] and reference == prepared['diagnostic_overlay_pointer'],
            'diagnostic source overlay changed during case')
    require(prepared['execution_protocol']['diagnostic_source_overlay_lock'] == reference
        and prepared['execution_protocol']['parent_source_overlay_lock'] == prepared['source_overlay_pointer'],
        'honest nested diagnostic/parent lock identities required')


def annotated_output(prepared, path, value):
    """Accept the original parent annotation once, then tag before first write."""
    value = deepcopy(value)
    require(type(value) is dict, 'dictionary diagnostic output required')
    value.update(execution_protocol=prepared['execution_protocol'],
        source_overlay_lock=prepared['diagnostic_overlay_pointer'],
        parent_source_overlay_lock=prepared['source_overlay_pointer'], diagnostic_only=True,
        formal_score_contribution=False, new_protocol_formal_VAL_TEST_qualification=False)
    if Path(path).name == 'case.json':
        value['inherited_1154_diagnostic_parent_case_version'] = value.pop('version', None)
        value['version'] = SCHEMA + '_case'
    return value




def validate_request(args):
    # This check precedes any manifest, source-lock or source-byte read.
    require(getattr(args,'split','train')=='train','TRAIN only; VAL/TEST refused before source read')
    require((getattr(args,'task',None),getattr(args,'raw_episode_id',None),getattr(args,'method',None)) in SELECTIONS,
            'only normal112Context/high102CV/high102Context; PhysMani and other sources refused')
    require(bool(getattr(args,'check',False)) != bool(getattr(args,'execute',False)),
            'one explicit --check or --execute required')
    require(all(getattr(args,k,None) is None for k in ('suite','suite_lock','source_inputs','admission','validation_release')),
            'no formal inputs accepted')


def selection_contract(root,args,chosen,diagnostic_ref):
    manifest=json.loads(parent.raw(root,parent.MANIFEST,parent.MANIFEST_SHA))
    row=next(r for r in manifest['episodes'] if (r['task'],r['raw_episode_id'])==(args.task,args.raw_episode_id))
    value=dict(schema=SCHEMA+'_selection',task=args.task,raw_episode_id=args.raw_episode_id,method=args.method,
        source_row_sha256=parent.digest(row),original_route=chosen['original_route'],
        original_route_sha256=chosen['original_route_sha256'],diagnostic_source_overlay_lock=diagnostic_ref,
        parent_source_overlay_lock=dict(path=PARENT_LOCK,sha256=PARENT_LOCK_SHA))
    return dict(value,sha256=parent.digest(value))


def prepare_check(root,args):
    """Hash/metadata only. Do not call inherited742 Context model constructor."""
    validate_request(args);require(args.check and not args.execute,'static CPU-check mode only')
    chosen=selection(root,args);lock,reference=read_diagnostic_lock(root,args.diagnostic_lock)
    require(all(s in lock['files'] for s in (SOURCE,WORKER,TEST_SOURCE)),'actual1163 sources must be pinned')
    out=parent._path(root,args.out,prefix=ARTIFACT_PREFIX,must_exist=False)
    require(not out.exists(),'fresh static-check output namespace required')
    route=chosen['original_route'];config=json.loads(parent.raw(root,route['configuration']['path'],route['configuration']['sha256']))
    files=json.loads(parent.raw(root,parent.TRAIN_INPUTS,parent.TRAIN_INPUTS_SHA))['files']
    pointers=[config['context_checkpoint'],*config['learned_artifacts'][args.task].values(),
              config.get('ring_affordances',{}).get(args.task)]
    checked=[]
    for pointer in pointers:
        if pointer is None:continue
        require(type(pointer) is dict and set(pointer)=={'path','sha256'},'explicit configured project model pointer')
        file=parent._path(root,pointer['path'])
        require(file.suffix not in ('.h5','.hdf5','.npz','.npy','.png','.jpg','.mp4')
            and files.get(str(file))==pointer['sha256'],'configured model bytes must be in unchanged745 substrate')
        parent.raw(root,pointer['path'],pointer['sha256']);checked.append(deepcopy(pointer))
    from .rml_component_domain_v640 import for_train
    domain=for_train(root,files,args.task,args.raw_episode_id,args.method)
    selected=selection_contract(root,args,chosen,reference)
    protocol=dict(schema=SCHEMA+'_static_CPU_check',actual_worker=parent.pointer(root,WORKER),
        actual_preparer=parent.pointer(root,SOURCE),diagnostic_only=True,selection_identity=selected,
        physical_reference_graph=PHYSICAL_REFERENCE,formal_graph_worker_approval=False,
        actual_model_loader_validation=False,model_loader_preflight_deferred_to_execute=True,
        policy_actor_executor_or_Neural_model_constructed=False)
    value=dict(schema=SCHEMA+'_static_CPU_identity',selection_identity=selected,
               execution_protocol_sha256=parent.digest(protocol))
    identity=dict(value,sha256=parent.digest(value))
    return dict(identity=identity,selection_identity=selected,inherited_1154_identity=None,
        component_domain=domain,model_preflight=dict(configured_model_byte_pointers=checked,
            actual_model_loader_validation=False,model_loader_preflight_deferred_to_execute=True,
            model_constructed=False,native_started=False,GPU_started=False),
        execution_protocol=protocol,diagnostic_overlay=lock,diagnostic_overlay_pointer=reference,
        diagnostic_check_args=args)


def verify_check(root,prepared):
    args=prepared['diagnostic_check_args'];chosen=selection(root,args)
    lock,reference=read_diagnostic_lock(root,args.diagnostic_lock)
    require(lock==prepared['diagnostic_overlay'] and reference==prepared['diagnostic_overlay_pointer']
        and selection_contract(root,args,chosen,reference)==prepared['selection_identity'],
        'static CPU selection/source identity changed')
    identity=prepared['identity'];unsigned=dict(identity);unsigned.pop('sha256',None)
    protocol=prepared['execution_protocol']
    require(identity.get('sha256')==parent.digest(unsigned)
        and identity.get('execution_protocol_sha256')==parent.digest(protocol)
        and protocol.get('selection_identity')==prepared['selection_identity']
        and protocol.get('model_loader_preflight_deferred_to_execute') is True
        and protocol.get('policy_actor_executor_or_Neural_model_constructed') is False,
        'static CPU-only proof cannot claim inherited model construction or changed identity')
    for pointer in prepared['model_preflight']['configured_model_byte_pointers']:
        parent.raw(root,pointer['path'],pointer['sha256'])


def _native_pins(root):
    # Only config/source/immutable-map metadata is parsed, never a dataset row file.
    routes=json.loads(parent.raw(root,parent.ROUTES,parent.ROUTES_SHA))['routes']
    refs=[r['configuration'] for r in routes if (r['task'],r['method']) in {(t,m) for t,_,m in SELECTIONS}]
    roots={json.loads(parent.raw(root,p['path'],p['sha256']))['baseline_repo'] for p in refs}
    require(len(roots)==1,'one actual frozen Native source root required')
    native=Path(next(iter(roots)));require(native.resolve()==native,'canonical Native source root')
    old=json.loads(parent.raw(root,parent.TRAIN_INPUTS,parent.TRAIN_INPUTS_SHA))['files']
    result={}
    for name in NATIVE_SOURCE_NAMES:
        file=native/name;require(file.resolve()==file and file.is_file(),'canonical Native source file')
        checksum=hashlib.sha256(file.read_bytes()).hexdigest()
        require(old.get(str(file))==checksum,'Native observer code differs from original immutable substrate')
        result[str(file)]=checksum
    return result


def require_training_complete(root, reference):
    # No permit, Native import or source selection occurs until Root observed exit.
    require(type(reference) is str and reference==TRAIN_OWNER+'/exec_completion.json','actual current TRAIN192 exit pointer required')
    plan=json.loads(parent.raw(root,TRAIN_OWNER+'/plan.json',TRAIN_PLAN_SHA))
    launch=json.loads(parent.raw(root,TRAIN_OWNER+'/launch.json'))
    terminal=json.loads(parent.raw(root,TRAIN_OWNER+'/terminal.json'))
    actual=json.loads(parent.raw(root,reference));status=json.loads(parent.raw(root,TRAIN_OWNER+'/status.json'))
    from .repaired_readiness_v1144 import _not_live
    require(actual.get('process')==launch.get('process')==terminal.get('process')
        and actual.get('actual_tool_exit_observed') is True and actual.get('natural_completion') is True
        and type(actual.get('returncode')) is int and actual['returncode']==0
        and actual.get('plan') in (dict(path=TRAIN_OWNER+'/plan.json',sha256=TRAIN_PLAN_SHA),
                                  dict(root='project',path=TRAIN_OWNER+'/plan.json',sha256=TRAIN_PLAN_SHA))
        and actual.get('terminal',{}).get('sha256')==parent.pointer(root,TRAIN_OWNER+'/terminal.json')['sha256']
        and actual.get('terminal',{}).get('path')==TRAIN_OWNER+'/terminal.json'
        and _not_live(launch['process']),'Root actual tool exit0/dead owner required before Native')
    require(type(plan.get('expected_cases')) is int and plan['expected_cases']==192
        and plan.get('graph',{}).get('sha256')==PHYSICAL_REFERENCE['sha256']
        and terminal.get('complete') is True
        and type(terminal.get('completed')) is int and terminal['completed']==192
        and type(terminal.get('attempted')) is int and terminal['attempted']==192
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed']==0
        and terminal.get('active') is None and terminal.get('automatic_retry') is False
        and status.get('stage')=='complete' and status.get('failed')==[] and status.get('active') is None
        and type(status.get('attempted')) is int and status['attempted']==192
        and [(v.get('task'),v.get('raw_episode_id'),v.get('method')) for v in status.get('completed',[])]==
            [(v.get('task'),v.get('raw_episode_id'),v.get('method')) for v in plan.get('jobs',[])]
        and len(status.get('completed',[]))==len(plan.get('jobs',[]))==192,
        'all192 actual unfiltered completions required')
    return parent.pointer(root,reference)


from .passive_execution_diagnostic_v1162 import PassiveExecutionDiagnostic, frozen_source_sites
from .train_ring_success_tap_v1125 import PassiveRingSuccessTap


class OneProfile(PassiveExecutionDiagnostic):
    """1162 owns the only lifecycle;1125 is an explicitly unentered callback."""
    def __init__(self,sites,task,method,*,trace=None,**limits):
        super().__init__(sites,**limits)
        self.native=PassiveRingSuccessTap(task,method,max_events=5000)
        self.trace=trace;self.ranges=[];self._ranges={};self.callback_errors=0
        self.native_pending=None;self.native_callback_cleared=False

    def observe_profile_event(self,frame,event,arg):
        try:
            site=self._site(frame)
            if site is not None and site.kind=='terminal' and event=='call':
                self._ranges[id(frame)]=dict(plan_start=vars(self.trace).get('plan_calls') if self.trace else None,
                    native_start=self.native.success_calls)
            if site is not None and site.kind=='terminal' and event=='return':
                token=self._active.get(id(frame));row=self._ranges.pop(id(frame),{})
                if len(self.ranges)<24:
                    self.ranges.append(dict(row,call_id=None if token is None else token['call_id'],
                        plan_end=vars(self.trace).get('plan_calls') if self.trace else None,
                        native_end=self.native.success_calls))
                else:self.callback_errors+=1
        except BaseException:self.callback_errors+=1
        super().observe_profile_event(frame,event,arg)
        self.native._profile(frame,event,arg)

    def __exit__(self,exc_type,exc,tb):
        try:return super().__exit__(exc_type,exc,tb)
        finally:
            self.native_pending=dict(success=len(self.native._success_frames),condition=len(self.native._condition_frames))
            self.native._success_frames.clear();self.native._condition_frames.clear();self._ranges.clear()
            self.native_callback_cleared=True

    def combined_receipt(self):
        return dict(schema=SCHEMA+'_one_profile',owner=super().receipt(),
            native_callback=self.native.receipt(),native_callback_only=True,
            native_callback_context_entered=self.native._entered,
            native_callback_lifecycle_fields_authoritative=False,
            native_callback_cleared=self.native_callback_cleared,native_pending_at_exit=self.native_pending,
            callback_errors=self.callback_errors,execution_plan_ranges=list(self.ranges),
            only_profile_owner='1163.OneProfile_inherits1162',simultaneous_profile_scopes=1)


def require_passive_quality(events,case):
    from collections import Counter
    require(events.get('schema')==SCHEMA+'_observers','actual1163 observer envelope required')
    require(events.get('restored') is True and events.get('observer_finalization_errors')==[], 'all observer scopes must restore')
    for key in ('factory_restored','writer_restored','parent_annotation_restored'):
        require(events.get(key) is True,'owned scope must restore: '+key)
    trace=events['public_execution_trace']['trace_receipt'];probe=events['readiness_probe']['readiness_probe']
    require(events['public_execution_trace'].get('restored') is True
        and events['readiness_probe'].get('restored') is True,'both public wrapper scopes must restore')
    require(trace.get('schema')=='r10_evaluation_existing_public_execution_trace_v1126'
        and probe.get('schema')=='r10_evaluation_existing_public_readiness_probe_v1134','actual public observer schemas')
    for stream,keys in ((trace,('recording_errors','frame_dropped','plan_dropped','plan_value_truncations',
            'extra_env_observe_calls','extra_native_getter_or_condition_calls','extra_prediction_calls','extra_path_step_or_FK_calls')),
            (probe,('recording_errors','dropped_events','extra_readiness_reference_forecast_or_native_calls'))):
        require(stream.get('restored') is True and stream.get('errors')==[] and stream.get('scope_exception_type') is None,'public stream restoration')
        require(all(type(stream.get(k)) is int and stream[k]==0 for k in keys),'zero observer drops/errors/extra calls')
    ticks=case['counted_clock']['ticks'];terminal=case['official_task_success'] or case['native_terminal_failure']
    require(trace['geometry_calls']==probe['geometry_calls']==probe['geometry_returns']==1,'one original geometry factory')
    require(trace['frame_calls']==trace['frame_returns']+trace['frame_exceptions']
        and trace['frame_returns']==ticks+(0 if terminal and ticks else 1)
        and len(trace['frames'])==trace['frame_returns']>0,'retain every original public frame')
    require(trace['plan_calls']==trace['plan_returns']+trace['plan_exceptions']==len(trace['plans']),'original planner results/exceptions retained')
    pc=probe['method_call_counts'];rows=probe['events']
    require(Counter(r['method'] for r in rows)==Counter({k:v['calls'] for k,v in pc.items()}),'all original probe calls retained')
    require(all(v['calls']==v['returns']+v['exceptions'] for v in pc.values()),'probe return/exception closure')
    one=events['one_profile'];ex=one['owner'];native=one['native_callback']
    require(one.get('schema')==SCHEMA+'_one_profile'
        and ex.get('schema')=='r10_passive_existing_execution_diagnostic_v1162'
        and native.get('schema')=='r10_fixed_TRAIN_ring_passive_original_success_tree_v1125',
        'actual execution/native callback schemas required')
    require(native.get('task')==case.get('task') and native.get('method')==case.get('method'),
            'Native callback task/method must equal this original case')
    require(one['native_callback_only'] is True and one['native_callback_context_entered'] is False
        and one['native_callback_lifecycle_fields_authoritative'] is False
        and one['native_callback_cleared'] is True and one['native_pending_at_exit']==dict(success=0,condition=0)
        and one['callback_errors']==0 and one['simultaneous_profile_scopes']==1,'single profile callback ownership')
    require(ex['restored'] is True and ex['restoration_error'] is None and ex['scope_exception_type'] is None
        and ex['diagnostic_available'] is True,'actual1162 profile lifecycle')
    require(all(type(ex.get(k)) is int and ex[k]==0 for k in ('dropped_events','value_truncations','recording_errors','active_frame_overflows')),
        'execution diagnostic truncation/error is incomplete')
    counts=ex['method_call_counts'];require(all(v['calls']==v['return_events'] for v in counts.values()),'all original execution call/return events retained')
    labels={s[0] for s in (*_PROJECT_SITES,*_EXECUTION_SITES)}|{'path.step','path.motion'}
    require(set(counts)==labels,'all declared original profile sites must have explicit coverage counters')
    require(all(type(v[k]) is int and 0<=v[k]<=50000 for v in counts.values() for k in ('calls','return_events')),
            'strict bounded execution counters')
    required={'frontend.observe':trace['frame_calls'],'frontend.ready':pc['frontend.ready']['calls'],
        'collision.readiness':pc['collision_port.readiness']['calls'],'reference.observed':pc['generator.observed_reference']['calls']}
    for label,number in required.items():require(counts[label]['calls']==number,'cross-stream original site count: '+label)
    done_refs=[r for r in rows if r['method']=='generator.observed_reference'
        and r.get('context',{}).get('kind')=='done'
        and r['context'].get('source_file')=='physmani_ring_release_v590.py']
    if done_refs:
        for label in ('release.done','skill.completion_target','completion.current529'):
            require(counts[label]['calls']>=len(done_refs),'original done-context helper must be observed: '+label)
    actions=case['native_outcome_sampling']['completed_actions'];pose=[a for a in actions if a['label']=='geometry_execute']
    require(counts['terminal.execute']['calls']==len(pose),'one original terminal executor per pose action')
    dispatch=sum(a['ticks'] for a in pose)
    require(all(counts[k]['calls']==dispatch for k in ('terminal.command','actuator.command','actuator.apply')),'each actual pose tick has one original dispatch')
    require(counts['path.filter']['calls']==counts['path.extract']['calls']==trace['plan_returns'],'each original returned plan modified/extracted once')
    require(counts['path.motion']['calls']<=counts['path.step']['calls'],'actual path motion cannot exceed step calls')
    records=ex['events'];step_parents={row['parent_call_id'] for row in records if row['site']=='path.step' and row['event']=='original_python_call'}
    for label,totals in counts.items():
        require(sum(r['site']==label and r['event']=='original_python_call' for r in records)==totals['calls']
            and sum(r['site']==label and r['event']=='original_python_return_event' for r in records)==totals['return_events'],
            'execution counters must equal retained actual events: '+label)
    calls={r['call_id']:r for r in records if r['event']=='original_python_call'}
    def under(row,ancestor):
        parent_id=row.get('parent_call_id')
        for _ in range(40):
            if parent_id==ancestor:return True
            parent_row=calls.get(parent_id)
            if parent_row is None:return False
            parent_id=parent_row.get('parent_call_id')
        return False
    release_calls=[r for r in calls.values() if r['site']=='release.done'
        and r.get('snapshot',{}).get('goal',{}).get('phase')=='release']
    for call in release_calls:
        for label in ('skill.completion_target','completion.current529'):
            require(sum(r['site']==label and under(r,call['call_id']) for r in calls.values())==1,
                    'actual release done must retain its original helper chain: '+label)
    for row in records:
        if row['site']!='release.done' or row['event']!='original_python_return_event':continue
        if row.get('snapshot',{}).get('goal',{}).get('phase')!='release' or type(row.get('original_return')) is not bool:continue
        leaves=row.get('release_leaf_evidence',{})
        require(type(leaves.get('target_present')) is bool,'actual release target leaf must be observed')
        if leaves['target_present']:
            require(type(leaves.get('gripper_open')) is bool,'evaluated release opening leaf must be observed')
            if leaves['gripper_open']:
                require(type(leaves.get('not_contact')) is bool,'evaluated release contact leaf must be observed')
                if leaves['not_contact']:
                    pair=row.get('direct_done_errors_return')
                    require(type(pair) is list and len(pair)==2 and all(type(v) is float for v in pair),
                            'only an originally evaluated distance requires its actual errors return')
    motion_parents={r['parent_call_id'] for r in records if r['site']=='path.motion' and r['event']=='original_python_call'}
    for row in records:
        if row['site']=='path.step' and row['event']=='original_python_return_event' and row['call_id'] not in motion_parents:
            require(row['return_status']=='none_return_or_python_exception_unwind_unknown',
                'only an original pre-motion exception can omit motion evidence')
    terminal_returns=[row for row in records if row['site']=='terminal.execute' and row['event']=='original_python_return_event']
    for row in terminal_returns:
        returned=row.get('original_return',{})
        if type(returned) is dict and returned.get('path_found') is True:
            require(row['call_id'] in step_parents,'eligible original found path must have step evidence')
    require(len(one['execution_plan_ranges'])==len(pose),'one raw-plan counter range per original action')
    for row in one['execution_plan_ranges']:
        require(type(row['plan_start']) is int and type(row['plan_end']) is int and row['plan_end']-row['plan_start'] in (0,1),'original per-action raw-plan correlation')
    for key in ('dropped_events','observer_errors','registration_truncations','inner_sequence_drops','exception_or_nonstandard_returns',
                'extra_success_calls','extra_condition_calls','extra_physical_getter_calls'):
        require(type(native.get(key)) is int and native[key]==0,'native callback collection incomplete: '+key)
    ns=native['original_success_returns_observed'];nc=native['original_condition_returns_observed']
    require(ns>0 and ns==native['original_success_calls_observed'] and nc==native['original_condition_calls_observed']==5*ns,'original task/ConditionSet/four leaves closure')
    success=[x for x in native['events'] if x['event']=='original_task_success_return']
    require(sum(x['event']=='original_task_success_call' for x in native['events'])==ns
        and sum(x['event']=='original_condition_met_call' for x in native['events'])==nc
        and sum(x['event']=='original_condition_met_return' for x in native['events'])==nc,
        'every original Native call/return event must be retained')
    require(len(success)==ns and all(x['original_four_detector_registration'] is True and len(x['leaf_returns'])==4
        and {v['ordinal'] for v in x['leaf_returns']}=={0,1,2,3}
        and all(type(v['result']) is list and len(v['result'])==2 and all(type(z) is bool for z in v['result'])
                for v in x['leaf_returns']) for x in success),'each original registered four-leaf sample observed')
    coverage={k:dict(calls=v['calls'],status='observed' if v['calls'] else 'not_reached_by_original_branch_unknown') for k,v in counts.items()}
    return dict(complete=True,diagnostic_only=True,profile_owners=1,public_frames=trace['frame_returns'],
        original_pose_actions=len(pose),original_dispatch_ticks=dispatch,site_coverage=coverage,
        tracking_sites_reached=bool(pose),Native_success_is_not_quality=True,NativeFalse_fatal_preserved=True,
        original_planner_and_helper_failures_are_data=True,extra_native_calls=0)


@contextmanager
def runtime_scope(root,prepared,runtime,events):
    require_native_ready(prepared);verify(root,prepared)
    require(require_training_complete(root,prepared['training_exit_name'])==prepared['training_completion'],
            'completed TRAIN192 evidence changed before Native observer scope')
    original_geometry,original_writer=runtime._geometry,runtime.atomic_json
    original_annotation=parent.annotated_output
    trace_events={};probe_events={};original_error=None;finalization=[];one=None
    try:
        with parent.runtime_scope(root,prepared,runtime):
            def annotate(current,path,value):
                require(current is prepared,'owned diagnostic writer identity')
                return annotated_output(current,path,original_annotation(current,path,value))
            with patch.object(parent,'annotated_output',annotate):
                from .train_public_execution_trace_v1126 import public_execution_trace_scope
                from .train_public_readiness_probe_v1134 import readiness_probe_scope
                with public_execution_trace_scope(runtime,trace_events,max_frames=201,max_plans=24) as trace:
                    with readiness_probe_scope(runtime,probe_events,max_events=1000):
                        pins=prepared['diagnostic_overlay']['native_source_hashes']
                        pyrep=next(p for p in pins if p.endswith('/'+NATIVE_SOURCE_NAMES[0]))
                        sites=frozen_source_sites(root,pyrep_path_file=pyrep)
                        one=OneProfile(sites,prepared['args'].task,prepared['args'].method,trace=trace,
                            max_events=50000,max_serialized_bytes=64_000_000)
                        with one:
                            try:yield
                            except BaseException as error:
                                original_error=(error,error.__traceback__);raise
        if original_error is not None:raise original_error[0].with_traceback(original_error[1])
    except BaseException as error:
        if original_error is not None and error is not original_error[0]:
            finalization.append(dict(stage='observer_exit',type=type(error).__name__))
            raise original_error[0].with_traceback(original_error[1]) from error
        raise
    finally:
        profile=None
        if one is not None:
            try:profile=one.combined_receipt()
            except BaseException as error:finalization.append(dict(stage='profile_receipt',type=type(error).__name__))
        events.update(schema=SCHEMA+'_observers',public_execution_trace=trace_events,readiness_probe=probe_events,
            one_profile=profile,observer_finalization_errors=finalization,
            factory_restored=runtime._geometry is original_geometry,writer_restored=runtime.atomic_json is original_writer,
            parent_annotation_restored=parent.annotated_output is original_annotation,
            restored=(runtime._geometry is original_geometry and runtime.atomic_json is original_writer
                and parent.annotated_output is original_annotation and trace_events.get('restored') is True
                and probe_events.get('restored') is True and profile is not None and profile['owner']['restored'] is True))
    verify(root,prepared)
