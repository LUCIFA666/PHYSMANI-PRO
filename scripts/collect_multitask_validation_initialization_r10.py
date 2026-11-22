"""First-execution VAL275 hold collection with explicit development hooks.

This is not recovery: no previous infra receipt, recovery_input, teacher or VAL
geometry export is accepted. Frozen collectors and the TRAIN queue are untouched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import collect_multitask_recovery_r10 as owned

collection = owned.collection
lineage = owned.lineage
require = owned.require
read = owned.read
pointer = owned.pointer
checked = owned.checked
VERSION = 'r10_first_execution_VAL_development_initialization_runner_v1'
REGISTRATION_SCHEMA = 'r10_first_execution_development_initialization_registration_v1'
HOOK_KEYS = ('development_reset_profile', 'RML_factory_hook')
OWNED_SOURCES = (*owned.OWNED_SOURCES, 'scripts/collect_multitask_validation_initialization_r10.py')
DEMONSTRATIONS = ROOT/'artifacts/framework/multitask_r10/demonstrations/manifest.json'
PARENT_MANIFESTS = {split: ROOT/'artifacts/framework/protocol'/name for split,name in
    (('train','train_manifest.json'),('validation','validation_manifest.json'))}


def validate_VAL_roster(plan):
    require(plan.get('version') == collection.BEHAVIOR_PLAN_VERSION and plan.get('split') == 'validation'
        and plan.get('validation_mode') == 'hold' and plan.get('frozen_policy') is None
        and plan.get('context_primary_behavior') == collection.VAL_HOLD_BEHAVIOR,
        'first initialization runner is ordinary VAL hold only')
    require('recovery_input' not in plan['protocol'], 'first VAL execution cannot invent a prior recovery input')
    require(plan['inputs'].get(str(DEMONSTRATIONS)) == collection.sha(DEMONSTRATIONS),
            'original 1400/275 demonstration metadata must be a plan input')
    _, rows = collection.demonstration_registry(DEMONSTRATIONS)
    parents = {collection.source_identity(row)[1:]: row for row in rows if row['split'] == 'validation'}
    require(len(parents) == 275 and len(plan['cases']) == 275, 'complete fixed275 VAL roster required')
    seen, ids = set(), set()
    for case in plan['cases']:
        key = (case['task'], case['parent_raw_episode_id'])
        require(key in parents and key not in seen and case['case_id'] not in ids,
                'missing, duplicate or non-original VAL source')
        seen.add(key); ids.add(case['case_id'])
        raw = case['parent_raw_episode_id']
        require(case['split'] == 'validation' and case['behavior'] == collection.VAL_HOLD_BEHAVIOR
            and case['replica'] == case['behavior_replica'] == 0
            and case['raw_episode_id'] == raw+'__'+collection.VAL_HOLD_BEHAVIOR+'_r0'
            and case['parent_source_group'] == 'validation:'+case['task']+':'+raw
            and case['maximum_collection_ticks'] == 60 and case['teacher_used'] is False
            and case['context_primary_view'] is True and case['interaction_eligible'] is False
            and case['source']['task'] == case['task'] and case['source']['raw_episode_id'] == raw
            and Path(case['source']['path']).resolve() == Path(parents[key]['source_path']).resolve()
            and type(case['source'].get('variation')) is int,
            'VAL hold source/behavior/seed replica/tick contract changed')
        asset = case.get('template_asset')
        require(asset and str(Path(asset).resolve()) in plan['inputs']
            and collection.sha(asset) == plan['inputs'][str(Path(asset).resolve())],
            'VAL must use an immutable pre-existing TRAIN geometry asset')
    require(seen == set(parents), 'all original275 validation parents are required')
    budget = plan['augmentation_budget']
    require(budget.get('selected_parent_sources') == 275 and budget.get('planned_native_cases') == 275
        and budget.get('version') == 'r10_fixed_behavior_budget_v1'
        and set(budget.get('behaviors', {})) == {collection.VAL_HOLD_BEHAVIOR}, 'fixed VAL behavior budget differs')
    profile = budget['behaviors'][collection.VAL_HOLD_BEHAVIOR]
    require(profile == dict(replicas=1, replica_indices=[0], maximum_ticks=60, teacher_used=False,
                            context_primary_view=True, interaction_eligible=False), 'fixed VAL hold profile differs')
    require(plan['original_demonstration_budget'].get('train_sources') == 1400
        and plan['original_demonstration_budget'].get('validation_sources') == 275, 'original source budget differs')


def validate_transition(original, plan):
    """Only creation metadata, input closure and two explicit hooks may differ."""
    validate_VAL_roster(original)
    validate_VAL_roster(plan)
    require(not any(key in original['protocol'] for key in (*HOOK_KEYS, 'recovery_input')),
            'base VAL plan must be the original unexecuted profile')
    expected = deepcopy(original)
    for key in HOOK_KEYS:
        expected['protocol'][key] = deepcopy(plan['protocol'][key])
    ignored = {'created_utc', 'plan_sha256', 'inputs', 'protocol_path'}
    require({key:value for key,value in expected.items() if key not in ignored}
        == {key:value for key,value in plan.items() if key not in ignored},
        'first VAL plan changes original cases, seed, geometry, labels or physical protocol')
    for name, checksum in original['inputs'].items():
        require(plan['inputs'].get(name) == checksum, 'original VAL input removed or rebound: '+name)


def validate_permissions(plan):
    owned.load_rml_permission(plan['protocol']['RML_factory_hook'], plan)
    from our_mani.multitask.development_reset_recovery import load_permission
    value = load_permission(plan['protocol']['development_reset_profile'], root=ROOT,
                            native_repo=plan['protocol']['baseline_repo'])
    for key in HOOK_KEYS:
        path = checked(plan['protocol'][key])
        require(plan['inputs'].get(str(path)) == collection.sha(path), 'initialization permission is unbound')
    for relative, checksum in value['native_inputs_sha256'].items():
        require(plan['inputs'].get(str((Path(plan['protocol']['baseline_repo'])/relative).resolve())) == checksum,
                'native development initialization source is unbound')


def prepare_plan(base_plan_path, source_closure_plan_path, *, physical_permission,
                 rml_permission, protocol_output, plan_output, extra_inputs=()):
    original = collection.verify_plan(base_plan_path)
    closure = collection.verify_plan(source_closure_plan_path)
    validate_VAL_roster(original)
    plan = deepcopy(original)
    plan['created_utc'] = collection.utc()
    plan['protocol'].update(development_reset_profile=deepcopy(physical_permission),
                            RML_factory_hook=deepcopy(rml_permission))
    inputs = dict(original['inputs'])
    for name, checksum in closure['inputs'].items():
        if Path(name).suffix in ('.py', '.sh'):
            require(name not in inputs or inputs[name] == checksum, 'frozen source closure conflicts with original VAL')
            inputs[name] = checksum
    for relative in OWNED_SOURCES:
        name = str((ROOT/relative).resolve())
        checksum = collection.sha(name)
        require(name not in inputs or inputs[name] == checksum, 'owned source conflicts with frozen source closure')
        inputs[name] = checksum
    for key in HOOK_KEYS:
        inputs.update(owned.permission_dependencies(plan['protocol'][key]))
    from our_mani.multitask.development_reset_recovery import load_permission
    physical = load_permission(physical_permission, root=ROOT, native_repo=plan['protocol']['baseline_repo'])
    for relative, checksum in physical['native_inputs_sha256'].items():
        name = str((Path(plan['protocol']['baseline_repo'])/relative).resolve())
        require(collection.sha(name) == checksum and (name not in inputs or inputs[name] == checksum),
                'native physical source conflicts with frozen original/closure')
        inputs[name] = checksum
    for value in (base_plan_path, source_closure_plan_path, *PARENT_MANIFESTS.values(), *extra_inputs):
        path = Path(value).resolve()
        require(str(path) not in inputs or inputs[str(path)] == collection.sha(path), 'extra input conflicts with original')
        inputs[str(path)] = collection.sha(path)
    protocol_output = collection.destination(protocol_output)
    plan_output = collection.destination(plan_output)
    require(protocol_output != Path(original['protocol_path']).resolve()
        and plan_output not in (Path(base_plan_path).resolve(), Path(source_closure_plan_path).resolve()),
        'new VAL plan and protocol paths required')
    plan['inputs'] = inputs
    plan['protocol_path'] = str(protocol_output)
    validate_transition(original, plan)
    validate_permissions(plan)
    collection.write_json(protocol_output, plan['protocol'])
    plan['inputs'][str(protocol_output)] = collection.sha(protocol_output)
    plan.pop('plan_sha256', None)
    plan['plan_sha256'] = collection.canonical_sha(plan)
    collection.write_json(plan_output, plan)
    collection.verify_plan(plan_output)
    return plan


def verify_execution(plan_path, registration_path, *, output=None, case_ids=None):
    registry = read(registration_path)
    require(set(registry) == {'schema', 'original_plan', 'original_collection', 'source_closure_plan', 'execution',
        'original_TRAIN_parents', 'original_VAL_parents'}
        and registry['schema'] == REGISTRATION_SCHEMA, 'exact first VAL execution registration required')
    files = {}
    original_path = checked(registry['original_plan'], files)
    closure_path = checked(registry['source_closure_plan'], files)
    original = collection.verify_plan(original_path)
    closure = collection.verify_plan(closure_path)
    actual = lineage.load_archived_plan(ROOT, registry['execution'], files)
    require(actual['path'] == Path(plan_path).resolve(), 'VAL CLI plan differs from actual source archive')
    plan = collection.verify_plan(plan_path)
    require(plan == actual['plan'], 'actual and archived VAL plans differ')
    validate_transition(original, plan)
    source_hashes = read(DEMONSTRATIONS)['inputs']['manifests_sha256']
    for key, split in (('original_TRAIN_parents','train'),('original_VAL_parents','validation')):
        parent_path = checked(registry[key], files)
        require(registry[key]['sha256'] == source_hashes[split]
            and plan['inputs'].get(str(parent_path)) == source_hashes[split],
            'original parent metadata must match the audited demonstration source SHA and actual plan')
    lineage.validate_fixed_roster(plan, read(checked(registry['original_TRAIN_parents'], files)),
                                  read(checked(registry['original_VAL_parents'], files)))
    require(all(closure['protocol'].get(key) == plan['protocol'][key] for key in HOOK_KEYS),
            'VAL permissions differ from actual frozen TRAIN shared profile')
    for value in (registry['original_plan'], registry['source_closure_plan']):
        path = checked(value, files)
        require(plan['inputs'].get(str(path)) == value['sha256'], 'VAL original/closure evidence is unbound')
    for name, checksum in closure['inputs'].items():
        if Path(name).suffix in ('.py', '.sh'):
            require(plan['inputs'].get(name) == checksum, 'VAL plan omits frozen execution source closure')
    for relative in OWNED_SOURCES:
        owned.source_bound(plan, relative)
    validate_permissions(plan)
    collection.destination(actual['collection'])
    original_collection = collection.destination(ROOT/registry['original_collection'])
    require(original_collection != actual['collection']
        and not original_collection.is_relative_to(actual['collection'])
        and not actual['collection'].is_relative_to(original_collection), 'first VAL requires its own new namespace')
    require(not any(original_collection.glob('cases/*/*/*/case.json')),
            'original VAL already has completions; first-execution runner cannot replace them')
    require(not (original_collection/'infrastructure_attempts').exists(),
            'original VAL has previous native attempts; explicit recovery lineage is required')
    if output is not None:
        require(Path(output).resolve().is_relative_to(actual['collection']), 'VAL output escaped actual registered namespace')
    if case_ids is not None:
        known = {case['case_id'] for case in plan['cases']}
        require(case_ids and len(case_ids) == len(set(case_ids)) and set(case_ids) <= known,
                'VAL queue subset must be unique cases of unchanged full275 plan')
    return dict(plan=plan, actual=actual, registry=registry, original=original,
                registration_pointer=pointer(registration_path), files=files)


def validate_case(path, context, case):
    record = collection.validate_case_result(path, context['plan'], case)
    own = record.get('development_initialization_execution', {})
    require(set(own) == {'version', 'hook_receipt', 'base_collector_receipt', 'execution_registration'}
        and own['version'] == VERSION and own['execution_registration'] == context['registration_pointer']
        and 'recovery_execution' not in record, 'actual first VAL initialization witness required')
    hooks = read(checked(own['hook_receipt']))
    base = read(checked(own['base_collector_receipt']))
    require(hooks.get('version') == VERSION and hooks.get('complete') is True and hooks.get('error') is None
        and hooks.get('case_id') == case['case_id'] and hooks.get('plan') == context['actual']['registration']['plan']
        and hooks.get('plan_sha256') == context['plan']['plan_sha256']
        and hooks.get('execution_registration') == context['registration_pointer']
        and hooks.get('original_plan') == context['registry']['original_plan']
        and hooks.get('source_closure_plan') == context['registry']['source_closure_plan']
        and hooks.get('actual_runner') == pointer(Path(__file__))
        and hooks.get('permissions') == {key:context['plan']['protocol'][key] for key in HOOK_KEYS}
        and hooks.get('first_execution') is True and hooks.get('previous_infrastructure_failure_required') is False
        and hooks.get('policy_failure_label_assigned') is False and hooks.get('historical_outputs_modified') is False
        and hooks.get('teacher_actions_used') is False and hooks.get('teacher_paths_requested') is False
        and hooks.get('VAL_geometry_exported') is False and hooks.get('RML_path_recovery_receipts') == []
        and 'failed_parent_receipts' not in hooks, 'first VAL hook identity/scope differs')
    resets = hooks.get('reset_recovery_receipts', [])
    expected = lineage.expected_reset(case, context['plan']['protocol'])
    require(len(resets) == 1 and resets[0].get('complete') is True
        and record.get('native_reset_receipt', {}).get('reset_recovery_receipt') == resets[0]
        and all(resets[0].get(key) == value for key,value in expected.items())
        and resets[0].get('permission') == context['plan']['protocol']['development_reset_profile']
        and resets[0].get('task') == case['task'] and resets[0].get('split') == 'validation'
        and resets[0].get('source') == case['source'] and resets[0].get('replica') == case['replica']
        and resets[0].get('source_resampled') is False and resets[0].get('official_TEST_allowed') is False,
        'VAL requires its measured complete same-source reset receipt, identical to embedded native receipt')
    require(record.get('teacher_actions_used') is False and record.get('teacher_paths_requested') is False
        and record.get('teacher_action_rows') == 0
        and record.get('policy_history_contains_native_GT') is False
        and record['entry'].get('source_kind') == 'VAL_native_generated_non_teacher', 'VAL may not use teacher actions/paths/GT history')
    require(not any(Path(name).name in ('TRAIN_color_annotations.json', 'TRAIN_rigid_layout.json',
                'TRAIN_body_relative_interaction_library.json') for name in record['files_sha256']),
            'VAL may not calibrate new TRAIN annotations/layout/library')
    geometry = [name for name in record['files_sha256'] if Path(name).name == 'TRAIN_geometry.json']
    require(len(geometry) == 1 and read(ROOT/geometry[0]) == read(case['template_asset']),
            'VAL must reuse exact frozen TRAIN CAD, not export new geometry')
    stripped = {key:value for key,value in record.items() if key not in ('development_initialization_execution', 'parent_process_receipt')}
    stripped['files_sha256'] = {key:value for key,value in stripped['files_sha256'].items()
        if key not in (own['hook_receipt']['path'], own['base_collector_receipt']['path'])}
    require(stripped == base, 'first VAL wrapper changed original collector labels/outputs')
    for key in ('hook_receipt', 'base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path']) == own[key]['sha256'], 'VAL witness missing from output manifest')
    if 'parent_process_receipt' in record:
        parent = record['parent_process_receipt']
        require(read(ROOT/parent['child_case_path']) == {key:value for key,value in record.items() if key!='parent_process_receipt'},
                'VAL canonical changes preserved raw child')
        attempt_output = Path(lineage.command_argument(parent['command'], '--output')).resolve()
        receipt = read(attempt_output.parent/'parent_receipt.json')
        require(receipt.get('status') == 'completed' and receipt.get('returncode') == 0
            and receipt.get('process_id') == parent['process_id'] and receipt.get('command') == parent['command']
            and receipt.get('execution_registration') == context['registration_pointer'], 'VAL completed parent receipt differs')
    return record


def collect_case(plan_path, case_id, output, guard_report, registration_path):
    context = verify_execution(plan_path, registration_path, output=output, case_ids=[case_id])
    plan = context['plan']; case = next(row for row in plan['cases'] if row['case_id']==case_id)
    require(Path(output).resolve() == context['actual']['collection']/'infrastructure_attempts'/case_id/'0/output',
            'VAL child must use its explicit fresh attempt0 output')
    out = collection.case_directory(output, case)
    require(not out.exists(), 'existing VAL child directory; no hidden retry/resume')
    captures = dict(simulations=[], paths=[]); started = collection.utc(); error = None; record = None
    try:
        with owned.scoped_owned_hooks(plan, captures):
            record = collection.collect_native_case(plan_path, case_id, output, guard_report)
    except BaseException as failure:
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
        raise
    finally:
        resets = [deepcopy(getattr(env, 'reset_recovery_receipt', None)) for env in captures['simulations']]
        paths = [deepcopy(path.recovery_receipt) for path in captures['paths']]
        hooks = dict(version=VERSION, case_id=case_id, plan=pointer(plan_path), plan_sha256=plan['plan_sha256'],
            execution_registration=context['registration_pointer'], original_plan=context['registry']['original_plan'],
            source_closure_plan=context['registry']['source_closure_plan'], actual_runner=pointer(Path(__file__)),
            started_utc=started, finished_utc=collection.utc(), permissions={key:deepcopy(plan['protocol'][key]) for key in HOOK_KEYS},
            reset_recovery_receipts=resets, RML_path_recovery_receipts=paths, error=error,
            complete=bool(record is not None and error is None and len(resets)==1
                and isinstance(resets[0],dict) and resets[0].get('complete') is True and not paths),
            first_execution=True, previous_infrastructure_failure_required=False, policy_failure_label_assigned=False,
            historical_outputs_modified=False, teacher_actions_used=False, teacher_paths_requested=False, VAL_geometry_exported=False)
        collection.write_json(out/'development_initialization_hook_receipt.json', hooks)
    require(hooks['complete'] is True, 'completed VAL requires one measured reset and zero teacher/RML paths')
    base = out/'base_collector_receipt.json'; (out/'case.json').rename(base)
    record = deepcopy(record)
    record['development_initialization_execution'] = dict(version=VERSION,
        hook_receipt=pointer(out/'development_initialization_hook_receipt.json'),
        base_collector_receipt=pointer(base), execution_registration=context['registration_pointer'])
    for path in (base, out/'development_initialization_hook_receipt.json'):
        record['files_sha256'][str(path.relative_to(ROOT))] = collection.sha(path)
    collection.write_json(out/'case.json', record)
    validate_case(out/'case.json', context, case)
    return record


def worker_command(plan_path, case_id, output, guard_report, registration_path):
    return ['bash', str(ROOT/'scripts/run_local_sim.sh'), str(Path(__file__).resolve()), '--collect-case',
        '--plan', str(Path(plan_path).resolve()), '--case-id', case_id, '--output', str(Path(output).resolve()),
        '--guard-report', str(Path(guard_report).resolve()), '--execution-registration', str(Path(registration_path).resolve())]


def run_queue(plan_path, output, guard_report, registration_path, *, case_ids=None, workers=2, case_timeout=900):
    context = verify_execution(plan_path, registration_path, output=output, case_ids=case_ids)
    require(Path(output).resolve() == context['actual']['collection'], 'VAL parent output must equal actual registration namespace')
    require(type(workers) is int and 1<=workers<=8 and type(case_timeout) is int and case_timeout>0,
            'VAL CPU workers1..8 and positive wall timeout required')
    require(context['plan']['protocol'].get('draft') is not True
        and context['plan']['protocol'].get('native_execution_authorized') is not False, 'draft VAL cannot launch')
    selected = [case for case in context['plan']['cases'] if case_ids is None or case['case_id'] in set(case_ids)]
    gate = collection.require_r09_release(guard_report); output = collection.destination(output)
    for case in selected:
        canonical = collection.case_directory(output, case)/'case.json'
        if canonical.exists():
            validate_case(canonical, context, case)
        else:
            require(not (output/'infrastructure_attempts'/case['case_id']).exists(),
                    'no automatic VAL retry; unresolved attempt requires explicit new recovery lineage')
    def one(case):
        canonical = collection.case_directory(output, case)/'case.json'
        if canonical.exists():
            return dict(case_id=case['case_id'],status='resumed_verified',process_id=validate_case(canonical,context,case)['process_id'])
        if collection.COLLECTION_STOP.exists():
            return dict(case_id=case['case_id'],status='cancelled_stop_marker',policy_failure_label_assigned=False)
        attempt = collection.destination(output/'infrastructure_attempts'/case['case_id']/'0')
        attempt.mkdir(parents=True,exist_ok=False)
        command = worker_command(plan_path,case['case_id'],attempt/'output',guard_report,registration_path)
        started,status,detail,code,pid = collection.utc(),'infrastructure_error','',None,None
        try:
            with (attempt/'native.log').open('wb') as log:
                process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                             MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                pid=process.pid
                try:
                    code=process.wait(timeout=case_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(pid,signal.SIGKILL);process.wait()
                    detail='VAL wall timeout; preserved evidence, no policy outcome assigned'
            if code==0:
                fresh=verify_execution(plan_path,registration_path,output=output,case_ids=[case['case_id']])
                child=collection.case_directory(attempt/'output',case)/'case.json'
                record=deepcopy(validate_case(child,fresh,case))
                require(type(record.get('process_id')) is int and record['process_id']>0,
                        'VAL Python child must preserve its own positive PID independently of xvfb wrapper')
                launcher=ROOT/'scripts/run_local_sim.sh'
                record['parent_process_receipt']=dict(launcher=str(launcher.relative_to(ROOT)),launcher_sha256=collection.sha(launcher),
                    process_id=pid,command=command,child_case_path=str(child.relative_to(ROOT)),child_case_sha256=collection.sha(child))
                collection.write_json(canonical,record);status='completed'
        except Exception as error:
            detail=''.join(traceback.format_exception(error))
        receipt=dict(version=VERSION,case_id=case['case_id'],status=status,detail=detail,returncode=code,process_id=pid,
            started_utc=started,finished_utc=collection.utc(),command=command,attempt=0,policy_failure_label_assigned=False,gate=gate,
            execution_registration=context['registration_pointer'],actual_plan=pointer(plan_path),first_execution=True,automatic_retries=False)
        collection.write_json(attempt/'parent_receipt.json',receipt)
        return receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts=[future.result() for future in as_completed([pool.submit(one,case) for case in selected])]
    merged=collection.merge_completed(plan_path,output)
    merged['shared_profile_execution_registration']=context['registration_pointer']
    merged['development_initialization_execution_version']=VERSION
    collection.write_json(output/'manifest.json',merged,overwrite=True)
    collection.write_json(output/'queue_report.json',dict(version=VERSION,created_utc=collection.utc(),receipts=receipts,
        expected=275,selected_case_ids=[case['case_id'] for case in selected],workers=workers,complete=merged['complete'],
        selected_complete=all(row['status'] in ('completed','resumed_verified') for row in receipts),
        automatic_retries=False,GPU_used=False,first_execution=True,teacher_actions_used=False,
        execution_registration=context['registration_pointer']),overwrite=True)
    return merged


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-plan','check','collect-case','run-queue'):
        action.add_argument('--'+name,action='store_true')
    for name in ('plan','base-plan','source-closure-plan','protocol-output','output','physical-permission','rml-permission','execution-registration'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--extra-input',type=Path,action='append',default=[])
    parser.add_argument('--case-id');parser.add_argument('--queue-case-id',action='append')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--case-timeout',type=int,default=900)
    parser.add_argument('--guard-report',type=Path,default=ROOT/'artifacts/framework/aligned_r09/final_report.json')
    args=parser.parse_args()
    def need(*names):
        for name in names:
            if getattr(args,name) is None:parser.error('this action requires --'+name.replace('_','-'))
    need('plan')
    if args.prepare_plan:
        need('base_plan','source_closure_plan','protocol_output','physical_permission','rml_permission')
        plan=prepare_plan(args.base_plan,args.source_closure_plan,physical_permission=pointer(args.physical_permission),
            rml_permission=pointer(args.rml_permission),protocol_output=args.protocol_output,plan_output=args.plan,extra_inputs=args.extra_input)
        result=dict(plan=pointer(args.plan),cases=len(plan['cases']),native_started=False)
    else:
        need('execution_registration')
        if args.check:
            context=verify_execution(args.plan,args.execution_registration,output=args.output,case_ids=args.queue_case_id)
            result=dict(plan_sha256=context['plan']['plan_sha256'],cases=275,first_execution=True,native_started=False)
        elif args.collect_case:
            need('case_id','output')
            record=collect_case(args.plan,args.case_id,args.output,args.guard_report,args.execution_registration)
            result=dict(case_id=record['case_id'],status=record['status'])
        else:
            need('output')
            merged=run_queue(args.plan,args.output,args.guard_report,args.execution_registration,
                case_ids=args.queue_case_id,workers=args.workers,case_timeout=args.case_timeout)
            result=dict(completed=merged['completed'],expected=merged['expected'])
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
