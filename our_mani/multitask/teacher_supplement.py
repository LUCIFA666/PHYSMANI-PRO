"""Strict independent fixed-200 TRAIN teacher acquisition and typed bundle audit.

The original 2800-case primary acquisition and Context K1 contract are not
extended. This module owns a separately declared replica1 teacher budget.
Historical source checks use actual archived bytes, never invented plan SHAs.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import acquisition_lineage as lineage

PLAN_SCOPE = 'r10_fixed_all_parent_TRAIN_teacher_supplement_v1'
REGISTRATION_SCHEMA = 'r10_fixed_teacher_supplement_execution_registration_v1'
RUNNER_VERSION = 'r10_fixed_teacher_supplement_first_execution_runner_v1'
AUX_MANIFEST_VERSION = 'r10_fixed_auxiliary_TRAIN_teacher_tracks_v1'
BUNDLE_VERSION = 'r10_TRAIN_primary2800_plus_fixed_aux200_v1'
TASKS = ('moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed')
BEHAVIOR = 'TRAIN_teacher_committed_waypoint'
HOOKS = ('development_reset_profile', 'RML_factory_hook')
BUDGET = dict(version='r10_separate_fixed_TRAIN_teacher_supplement_budget_v1', parents=200,
    parents_per_task=100, extra_teacher_replicas_per_parent=1, replica_indices=[1], expected_cases=200,
    maximum_ticks_per_case=200, maximum_total_ticks=40000, extra_hold_cases=0,
    context_primary_view=False, original_primary_budget_unchanged=True)
OWNED_SOURCES = ('scripts/collect_teacher_supplement_r10.py', 'our_mani/multitask/teacher_supplement.py',
    'scripts/collect_multitask_training_r10.py', 'scripts/collect_multitask_recovery_r10.py',
    'scripts/run_local_sim.sh', 'our_mani/multitask/acquisition_lineage.py',
    'our_mani/multitask/development_reset_recovery.py', 'our_mani/multitask/simulation.py',
    'our_mani/multitask/rml_path_repair.py', 'our_mani/multitask/rml_path_recovery.py',
    'our_mani/multitask/train_teacher.py','our_mani/multitask/training_view.py',
    'our_mani/multitask/acquisition_manifest.py','our_mani/multitask/evaluation.py',
    'our_mani/multitask/motion_labels.py')


def require(value, message):
    lineage.require(value, message)


def expected_cases(primary):
    require(primary.get('split') == 'train' and len(primary['cases']) == 2800,
        'supplement requires the complete original primary2800 roster')
    parents = [row for row in primary['cases'] if row['task'] in TASKS and row.get('behavior') == BEHAVIOR]
    require(len(parents) == 200 and all(sum(c['task'] == task for c in parents) == 100 for task in TASKS)
        and len({c['parent_source_group'] for c in parents}) == 200, 'both exact original100-parent teacher tasks required')
    result = []
    for before in parents:
        require(before['replica'] == before['behavior_replica'] == 0 and before['teacher_used'] is True
            and before['maximum_collection_ticks'] == 200 and before['context_primary_view'] is False
            and before['interaction_eligible'] is True, 'original teacher behavior contract changed')
        row = deepcopy(before)
        row.update(replica=1, behavior_replica=1,
            raw_episode_id=before['parent_raw_episode_id'] + '__' + BEHAVIOR + '_r1')
        row['case_id'] = lineage.canonical_sha(dict(split='train', task=before['task'],
            raw=before['parent_raw_episode_id'], replica=1, behavior=BEHAVIOR))[:16]
        result.append(row)
    require(len({r['case_id'] for r in result}) == 200
        and not ({r['case_id'] for r in result} & {r['case_id'] for r in primary['cases']}), 'auxiliary case identity overlaps primary K1')
    return result


def verify_registration(root, registry_pointer, *, live=False, require_ready=False):
    """Validate a separate aux roster/namespace; never call recovery transition."""
    root = Path(root).resolve(); files = {}
    if not isinstance(registry_pointer, dict):
        registry_pointer = lineage.pointer(root, registry_pointer)
    registry_path = lineage.checked_pointer(root, registry_pointer, files)
    registry = lineage.read(registry_path)
    require(set(registry) == {'schema', 'draft_preparation_report', 'primary_plan', 'shared_profile_plan',
        'execution', 'execution_ready', 'fixed_budget'} and registry['schema'] == REGISTRATION_SCHEMA,
        'exact typed auxiliary execution registration required')
    require(type(registry['execution_ready']) is bool and (not require_ready or registry['execution_ready']),
        'auxiliary candidate registration requires ROOT first-job review before native execution')
    require(registry['fixed_budget'] == BUDGET, 'auxiliary budget differs from fixed200 teacher-only replica1')
    draft_path = lineage.checked_pointer(root, registry['draft_preparation_report'], files)
    draft_report = lineage.read(draft_path)
    require(draft_report.get('schema') == 'r10_fixed_all_parent_TRAIN_teacher_supplement_draft_v1'
        and draft_report.get('approved_for_execution') is False and draft_report.get('fixed_cases') == 200
        and draft_report.get('automatic_retry') is False and draft_report.get('outcome_selection') is False,
        'the original unapproved fixed200 draft must remain intact')
    draft = lineage.read(lineage.checked_pointer(root, draft_report['plan'], files))
    lineage.checked_pointer(root, draft_report['source_archive'], files)
    proposal=lineage.read(lineage.checked_pointer(root,draft_report['proposal'],files))
    require(draft.get('supplement_proposal')==draft_report['proposal']
        and registry['shared_profile_plan']==proposal['shared_profile_reference_plan'],
        'auxiliary shared profile differs from the actual fixed TRAIN proposal')
    require(draft['protocol'].get('draft') is True and draft['protocol'].get('native_execution_authorized') is False
        and draft.get('native_execution_ready') is False, 'original draft permission was rewritten')
    primary_path = lineage.checked_pointer(root, registry['primary_plan'], files)
    primary = lineage.read(primary_path)
    reviewed_path = lineage.checked_pointer(root, registry['shared_profile_plan'], files)
    reviewed = lineage.read(reviewed_path)
    require(registry['primary_plan'] == draft_report['primary_plan'] == draft['primary_K1_plan'],
        'auxiliary primary-plan identity differs from original K1')
    require(reviewed['cases'] == primary['cases'], 'accepted shared profile changed original case identities')
    current = lineage.load_archived_plan(root, registry['execution'], files)
    plan = current['plan']; expected = expected_cases(primary)
    demonstrations_path=root/'artifacts/framework/multitask_r10/demonstrations/manifest.json'
    require(plan['inputs'].get(str(demonstrations_path))==lineage.file_sha(demonstrations_path),
        'original demonstration metadata is not bound by auxiliary actual plan')
    demonstrations=lineage.read(lineage.checked_pointer(root,lineage.pointer(root,demonstrations_path),files))
    parents={}
    for split,name in (('train','train_manifest.json'),('validation','validation_manifest.json')):
        metadata_path=root/'artifacts/framework/protocol'/name
        checksum=demonstrations['inputs']['manifests_sha256'][split]
        require(plan['inputs'].get(str(metadata_path))==checksum,'auxiliary original parent metadata differs from audited demonstrations')
        parents[split]=lineage.read(lineage.checked_pointer(root,dict(path=str(metadata_path),sha256=checksum),files))
    lineage.validate_fixed_roster(primary,parents['train'],parents['validation'])
    require(plan['cases'] == draft['cases'] == expected and plan['augmentation_budget'] == BUDGET
        and plan.get('auxiliary_teacher_scope') == PLAN_SCOPE and plan.get('primary_K1_plan') == registry['primary_plan']
        and plan.get('primary_K1_roster_unmodified') is True and plan.get('primary_Context_hold_view_unmodified') is True
        and plan.get('native_execution_ready') is True, 'actual auxiliary plan scope/identity/roster differs')
    require(plan.get('auxiliary_collection')==registry['execution']['collection'],
        'auxiliary actual output namespace is not bound by its immutable plan')
    from scripts import collect_multitask_training_r10 as collection
    require(plan['version'] == collection.BEHAVIOR_PLAN_VERSION, 'reuse the unchanged teacher collector plan format')
    expected_protocol = deepcopy(reviewed['protocol']); expected_protocol.pop('recovery_input', None)
    expected_protocol.update(draft=False, native_execution_authorized=True)
    require(plan['protocol'] == expected_protocol and 'recovery_input' not in plan['protocol'],
        'auxiliary collection changed physics/protocol or impersonated infrastructure recovery')
    # Every non-metadata plan property remains the original draft's value.
    variable = {'created_utc', 'protocol', 'protocol_path', 'inputs', 'plan_sha256',
        'auxiliary_teacher_scope', 'native_execution_ready','auxiliary_collection'}
    require({k:v for k,v in plan.items() if k not in variable} == {k:v for k,v in draft.items() if k not in variable},
        'undeclared auxiliary plan semantic change')
    require(current['collection'].is_relative_to(root/'artifacts/framework/multitask_r10')
        and current['collection'] != (root/draft_report['collection']).resolve()
        and not current['collection'].is_relative_to(primary_path.parent/'TRAIN_full_collection_v11'),
        'auxiliary output needs a fresh independent production-candidate namespace')
    for relative in OWNED_SOURCES:
        source = str((root/relative).resolve())
        require(source in current['sources'], 'auxiliary executable is missing from actual archived source closure: ' + relative)
    for key in HOOKS:
        require(plan['protocol'][key] == reviewed['protocol'][key], 'shared physical/RML hook changed for auxiliary collection')
        permission_path = lineage.checked_pointer(root, plan['protocol'][key], files)
        require(plan['inputs'].get(str(permission_path)) == plan['protocol'][key]['sha256'], 'shared permission is not an actual auxiliary plan input')
    if live:
        require(collection.verify_plan(current['path']) == plan, 'live auxiliary plan/input bytes differ from archive')
        for relative in OWNED_SOURCES:
            source = (root/relative).resolve()
            require(plan['inputs'].get(str(source)) == lineage.file_sha(source), 'auxiliary imported live executable is not bound')
        from scripts import collect_multitask_recovery_r10 as recovery
        recovery.load_rml_permission(plan['protocol']['RML_factory_hook'], plan)
        from .development_reset_recovery import load_permission
        permission = load_permission(plan['protocol']['development_reset_profile'], root=root,
            native_repo=plan['protocol']['baseline_repo'])
        for name, checksum in permission['native_inputs_sha256'].items():
            require(plan['inputs'].get(str((Path(plan['protocol']['baseline_repo'])/name).resolve())) == checksum,
                'auxiliary physical permission native executable is unbound')
    return dict(root=root, registry=registry, registry_pointer=registry_pointer, registry_path=registry_path,
        current=current, plan=plan, primary=primary, primary_path=primary_path, files=files,
        expected_cases={row['case_id']:row for row in expected})


def validate_result(root, path, context, case, *, files=None):
    """Strict raw/base/hook witnesses; historical runner code comes from archive."""
    root = Path(root).resolve(); files = {} if files is None else files
    path = lineage.local(root, path); record = lineage.read(path)
    require(record.get('status') == 'completed' and record.get('case_id') == case['case_id']
        and record.get('plan_sha256') == context['plan']['plan_sha256'] and record.get('task') == case['task']
        and record.get('split') == 'train' and type(record.get('process_id')) is int and record['process_id'] > 0
        and 'recovery_execution' not in record and 'development_initialization_execution' not in record,
        'raw auxiliary case identity/status differs or impersonates recovery/VAL')
    own = record.get('teacher_supplement_execution')
    require(isinstance(own,dict) and set(own) == {'version','hook_receipt','base_collector_receipt','execution_registration'}
        and own['version'] == RUNNER_VERSION and own['execution_registration'] == context['registry_pointer'],
        'raw auxiliary case lacks its independent typed execution witness')
    hook_path=lineage.checked_pointer(root, own['hook_receipt'], files)
    base_path=lineage.checked_pointer(root, own['base_collector_receipt'], files)
    hooks = lineage.read(hook_path)
    base = lineage.read(base_path)
    require(hook_path.parent==base_path.parent,'auxiliary raw base/hook output directories differ')
    require(hooks.get('version') == RUNNER_VERSION and hooks.get('execution_kind') == 'fixed_auxiliary_TRAIN_teacher_replica1_first_execution'
        and hooks.get('case_id') == case['case_id'] and hooks.get('plan') == context['current']['registration']['plan']
        and hooks.get('plan_sha256') == context['plan']['plan_sha256']
        and hooks.get('execution_registration') == context['registry_pointer'] and hooks.get('primary_plan') == context['registry']['primary_plan']
        and hooks.get('permissions') == {key:context['plan']['protocol'][key] for key in HOOKS}
        and hooks.get('complete') is True and hooks.get('error') is None and hooks.get('first_execution') is True
        and hooks.get('is_infrastructure_recovery') is False and hooks.get('previous_infrastructure_failure_required') is False
        and hooks.get('outcome_selected') is False and hooks.get('primary_outputs_modified') is False
        and hooks.get('policy_failure_label_assigned') is False and hooks.get('automatic_retry') is False
        and 'failed_parent_receipts' not in hooks, 'auxiliary hook scope/identity/permission/first-execution witness differs')
    runner = str((root/hooks['actual_runner']['path']).resolve())
    require(runner == str((root/'scripts/collect_teacher_supplement_r10.py').resolve())
        and runner in context['current']['sources']
        and hooks['actual_runner']['sha256'] == context['current']['sources'][runner]['sha256'],
        'actual auxiliary runner differs from registered archived executable')
    reset = hooks.get('reset_recovery_receipts', [])
    require(len(reset) == 1 and isinstance(reset[0],dict) and reset[0].get('complete') is True
        and reset[0].get('permission') == context['plan']['protocol']['development_reset_profile'],
        'auxiliary physical reset witness is missing or not the shared profile')
    expected_reset = lineage.expected_reset(case, context['plan']['protocol'])
    physical = reset[0]
    physical_permission=lineage.read(lineage.local(root,context['plan']['protocol']['development_reset_profile']['path']))
    require(physical.get('version')==physical_permission['version']
        and all(physical.get(name)==value for name,value in expected_reset.items()) and physical.get('source_resampled') is False
        and physical.get('official_TEST_allowed') is False, 'auxiliary physical seed/variation/resampling witness differs')
    paths = hooks.get('RML_path_recovery_receipts')
    require(isinstance(paths,list) and all(isinstance(row,dict) and row.get('close_complete') is True
        and row.get('version')=='r10_explicit_owned_RML_recovery_adapter_v1'
        and row.get('real_native_step_motion_replaced') is False and row.get('physics_step_called_by_adapter') is False
        and row.get('historical_fault_reproduced_in_native_diagnostic') is False for row in paths),
        'auxiliary RML hook was not closed or changed native motion/physics')
    stripped = deepcopy({key:value for key,value in record.items() if key not in ('teacher_supplement_execution','parent_process_receipt')})
    stripped['files_sha256'] = {name:checksum for name,checksum in stripped['files_sha256'].items()
        if name not in (own['hook_receipt']['path'],own['base_collector_receipt']['path'])}
    require(stripped == base, 'auxiliary wrapper changed base collector bytes/labels/seed/clock')
    for key in ('hook_receipt','base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path']) == own[key]['sha256'], 'auxiliary witness omitted from raw output hashes')
    require(lineage.instant(hooks['started_utc']) <= lineage.instant(base['started_utc'])
        <= lineage.instant(base['finished_utc']) <= lineage.instant(hooks['finished_utc']), 'auxiliary hook/base chronological witness reversed')
    require(record.get('case_isolation')=='fresh_simulator_process_per_episode'
        and record['teacher_actions_used'] is True and record['teacher_paths_requested'] is True
        and record.get('policy_history_contains_native_GT') is False, 'auxiliary is teacher-only TRAIN with causal policy observations')
    entry = record['entry']
    for name in ('task','raw_episode_id','parent_raw_episode_id','parent_source_group','behavior','behavior_replica',
            'maximum_collection_ticks','teacher_used','context_primary_view','interaction_eligible'):
        require(entry.get(name) == case[name], 'auxiliary entry mutated fixed original-derived identity: ' + name)
    require(entry.get('dynamics_target_label_schema')==context['plan']['protocol'].get('dynamics_target_label_schema')
        and entry.get('original_HDF5_motion_replayed') is False
        and entry.get('observation_source')=='causal_RGBD_and_proprioception'
        and entry.get('target_source')=='TRAIN_VAL_native_supervision'
        and entry.get('augmentation_replica') == 1 and entry.get('source_kind') == 'TRAIN_native_generated_augmentation'
        and entry.get('source_demonstration') == case['offline_demonstration']
        and entry.get('context_training_eligible') is False and entry.get('context_primary_view') is False,
        'auxiliary cannot enter Context primary hold or use another source cache')
    native = record['native_reset_receipt']
    require(all(native.get(name) == value for name,value in expected_reset.items())
        and native.get('replica') == 1 and native.get('task') == case['task']
        and native.get('raw_episode_id') == case['parent_raw_episode_id'] and native.get('source_split') == 'train'
        and Path(native['source_path']).resolve() == Path(case['source']['path']).resolve()
        and native.get('official_TEST') is False and native.get('original_source_replayed') is False,
        'auxiliary native reset/source identity differs')
    require(native.get('reset_recovery_receipt')==physical
        and native.get('development_reset_profile')==context['plan']['protocol']['development_reset_profile']
        and physical.get('task')==case['task'] and physical.get('split')=='train'
        and physical.get('replica')==1 and physical.get('source')==case['source'],
        'auxiliary hook reset witness is not the actual same-source native embedded receipt')
    for name,checksum in record['files_sha256'].items():
        output_path=lineage.checked_pointer(root,dict(path=name,sha256=checksum),files)
        require(output_path.is_relative_to(base_path.parent),'auxiliary output file reused from another case/attempt')
    require(record['files_sha256'].get(entry['cache']) == entry['cache_sha256'], 'auxiliary arrays are not hash-bound')
    geometry=[name for name in record['files_sha256'] if Path(name).name=='TRAIN_geometry.json']
    require(len(geometry)==1 and case['template_asset'] in context['plan']['inputs']
        and lineage.read(lineage.local(root,geometry[0]))==lineage.read(lineage.local(root,case['template_asset'])),
        'auxiliary actual geometry differs from same original frozen TRAIN CAD')
    from .training_view import validate_arrays
    import numpy as np
    with np.load(lineage.local(root,entry['cache']),allow_pickle=False) as stored:
        arrays = {key:stored[key].copy() for key in stored.files}
    validate_arrays(arrays,entry,record)
    require(not np.count_nonzero(arrays['teacher_action_valid']) or bool(paths),
        'executed auxiliary teacher actions require an actual owned RML path witness')
    return record


def audit_collection(root, registry_pointer, *, require_complete=False):
    """Read committed cases only; no output rewriting, outcome selection or retry."""
    root = Path(root).resolve(); context = verify_registration(root,registry_pointer,require_ready=require_complete)
    files = dict(context['files']); current=context['current']; expected=context['expected_cases']
    directory=current['collection']; cases=[]; entries=[]; proofs=[]; pending=[]; infra=[]
    discovered=list((directory/'cases/train').glob('*/*/case.json'))
    require(len({path.parent.name for path in discovered}) == len(discovered)
        and all(path.parent.name in expected and path.parent.parent.name == expected[path.parent.name]['task'] for path in discovered),
        'auxiliary unknown/duplicate canonical case or task')
    attempt_roots=list((directory/'infrastructure_attempts').glob('*'))
    require(all(path.name in expected and path.is_dir() for path in attempt_roots), 'unknown auxiliary infrastructure attempt')
    for attempt_root in attempt_roots:
        require({p.name for p in attempt_root.iterdir()} <= {'0'}, 'auxiliary automatic retry/second attempt is forbidden')
        parent_path=attempt_root/'0/parent_receipt.json'
        if parent_path.exists():
            parent=lineage.read(lineage.checked_pointer(root,lineage.pointer(root,parent_path),files))
            require(parent.get('version') == RUNNER_VERSION and parent.get('execution_registration') == context['registry_pointer']
                and parent.get('automatic_retries') is False and parent.get('policy_failure_label_assigned') is False
                and parent.get('actual_plan') == current['registration']['plan'] and parent.get('attempt') == 0,
                'auxiliary raw parent witness differs')
            require(parent['status'] in ('completed','infrastructure_error'), 'unknown auxiliary parent outcome')
            if parent.get('process_started') is False:
                require(parent['status']=='infrastructure_error' and parent.get('process_id') is None
                    and parent.get('returncode') is None and parent.get('case_id')==attempt_root.name,
                    'only a genuine prelaunch infrastructure failure may lack a process PID')
                command=parent['command']
                require(Path(lineage.command_argument(command,'--plan')).resolve()==current['path']
                    and lineage.command_argument(command,'--case-id')==attempt_root.name
                    and Path(lineage.command_argument(command,'--output')).resolve()==parent_path.parent/'output'
                    and all(str(Path(name).resolve()) in current['sources'] for name in command[1:3]),
                    'prelaunch auxiliary command/source identity differs')
            else:
                require(parent.get('process_started') is True,'auxiliary parent must explicitly distinguish actual process from prelaunch failure')
                lineage.verify_attempt_parent(root,lineage.pointer(root,parent_path),current,attempt_root.name,files,status=parent['status'])
            if parent['status'] == 'infrastructure_error':
                require(not (directory/'cases/train'/expected[attempt_root.name]['task']/attempt_root.name/'case.json').exists(),
                    'failed auxiliary parent was promoted to a completed policy case')
                partial={}
                for item in parent_path.parent.rglob('*'):
                    if item.is_file():
                        value=lineage.pointer(root,item);lineage.checked_pointer(root,value,files);partial[value['path']]=value['sha256']
                require((parent_path.parent/'native.log').is_file(),'auxiliary infra native log must be preserved')
                infra.append(dict(case_id=attempt_root.name,parent_receipt=lineage.pointer(root,parent_path),partial_outputs_sha256=partial))
        else:
            pending.append(dict(case_id=attempt_root.name,reason='raw_parent_commit_pending'))
    for cid,case in expected.items():
        path=directory/'cases/train'/case['task']/cid/'case.json'
        if not path.exists():
            continue
        parent_path=directory/'infrastructure_attempts'/cid/'0/parent_receipt.json'
        if not parent_path.exists():
            continue
        record=validate_result(root,path,context,case,files=files)
        proof=lineage.verify_completed(root,dict(case_id=cid,canonical_receipt=lineage.pointer(root,path),
            parent_receipt=lineage.pointer(root,parent_path)),current,case,files)
        require(lineage.read(parent_path).get('execution_registration') == context['registry_pointer'], 'auxiliary parent used another execution registration')
        hooks=lineage.read(lineage.local(root,record['teacher_supplement_execution']['hook_receipt']['path']))
        parent=lineage.read(parent_path)
        require(lineage.instant(parent['started_utc'])<=lineage.instant(hooks['started_utc'])
            <=lineage.instant(hooks['finished_utc'])<=lineage.instant(parent['finished_utc']),
            'auxiliary hook clock lies outside its actual raw parent process')
        entries.append(record['entry']);proofs.append(proof)
        cases.append(dict(case_id=cid,**lineage.pointer(root,path),actual_plan_sha256=context['plan']['plan_sha256']))
    complete=len(cases)==200 and not pending and not infra
    require(not require_complete or complete, 'complete fixed auxiliary200 committed cases required; failed/censored/unfinished sources cannot be dropped')
    mapping={row['case_id']:context['plan']['plan_sha256'] for row in cases}
    projection=dict(version=AUX_MANIFEST_VERSION,split='train',TEST_used=False,complete=complete,expected=200,
        completed=len(cases),entries=entries,case_receipts=cases,missing_case_ids=sorted(set(expected)-set(mapping)),
        plan_sha256=context['plan']['plan_sha256'],actual_plan=context['registry']['execution']['plan'],
        execution_registration=context['registry_pointer'],augmentation_budget=BUDGET,
        fixed_teacher_supplement=True,primary_K1_roster_unmodified=True,Context_primary_view_unmodified=True,
        auxiliary_parent_sources=200,parent_grouping='same original TRAIN parent; replica1 is not an independent CV parent',
        first_execution=True,is_infrastructure_recovery=False)
    for name,checksum in files.items():
        require(lineage.file_sha(name)==checksum,'auxiliary audit input changed during committed-case scan')
    return dict(schema='r10_fixed_auxiliary_teacher_acquisition_audit_v1',audit_passed=True,complete=complete,
        execution_kind='fixed_auxiliary_TRAIN_teacher_replica1_first_execution',acquisition_projection=projection,
        per_case_actual_plan_sha256=mapping,actual_plan_sha256_values=[context['plan']['plan_sha256']],
        base_plan_sha256=context['primary']['plan_sha256'],verified_files_sha256=files,case_proofs=proofs,
        pending=pending,infrastructure_errors=infra,execution_registration=context['registry_pointer'])


def audit_manifest(root, path, *, require_complete=True):
    root=Path(root).resolve();path=lineage.local(root,path);value=lineage.read(path)
    require(value.get('version') == AUX_MANIFEST_VERSION, 'typed auxiliary teacher manifest required')
    proof=audit_collection(root,value['execution_registration'],require_complete=require_complete)
    require(value == proof['acquisition_projection'], 'auxiliary manifest differs from exact committed raw200 projection')
    proof['verified_files_sha256'][str(path)] = lineage.file_sha(path)
    return proof


def audit_bundle(root,path,*,require_complete=True):
    """Primary2800 AND auxiliary200; same1400 parents,1600 teacher rollouts."""
    root=Path(root).resolve();path=lineage.local(root,path);bundle=lineage.read(path);files={}
    require(set(bundle) == {'version','primary_manifest','auxiliary_teacher_manifest','primary_Context_view_unchanged'}
        and bundle['version'] == BUNDLE_VERSION and bundle['primary_Context_view_unchanged'] is True,
        'strict typed primary-plus-auxiliary bundle required')
    from .acquisition_manifest import load_acquisition_manifest,COMPOSITE_VERSION
    primary_path=lineage.checked_pointer(root,bundle['primary_manifest'],files)
    primary_header=lineage.read(primary_path)
    require(primary_header.get('schema') == COMPOSITE_VERSION or primary_header.get('version') == COMPOSITE_VERSION,
        'bundle requires actual full historical primary composite')
    primary,primary_proof=load_acquisition_manifest(root,primary_path)
    require(primary_proof is not None and primary_proof.get('complete') is True and primary['complete'] is True
        and primary['expected']==primary['completed']==len(primary['entries'])==2800,
        'primary complete2800 gate cannot be substituted by auxiliary data')
    original=primary['entries'];parents={r['parent_source_group'] for r in original}
    require(len(parents)==1400 and sum(r['teacher_used'] is True for r in original)==1400
        and sum(r['context_primary_view'] is True for r in original)==1400
        and all(r['augmentation_replica']==r['behavior_replica']==0 for r in original), 'original primary K1 teacher/hold coverage changed')
    aux_path=lineage.checked_pointer(root,bundle['auxiliary_teacher_manifest'],files)
    auxiliary=audit_manifest(root,aux_path,require_complete=require_complete)
    aux=auxiliary['acquisition_projection']
    require(auxiliary['complete'] is True and aux['expected']==aux['completed']==200,
        'bundle requires complete fixed auxiliary200, not an outcome-selected subset')
    require(auxiliary['base_plan_sha256']==primary['base_plan_sha256'], 'auxiliary belongs to another original primary plan')
    aux_parents={r['parent_source_group'] for r in aux['entries']}
    expected_aux={r['parent_source_group'] for r in original if r['task'] in TASKS and r['teacher_used']}
    require(aux_parents == expected_aux and aux_parents<=parents, 'auxiliary all200 original-parent coverage differs')
    mapping=dict(primary_proof['per_case_actual_plan_sha256'])
    require(not set(mapping)&set(auxiliary['per_case_actual_plan_sha256']), 'auxiliary duplicates/replaces an original primary case')
    mapping.update(auxiliary['per_case_actual_plan_sha256'])
    entries=primary['entries']+aux['entries'];receipts=primary['case_receipts']+aux['case_receipts']
    require(len(entries)==len(receipts)==len(mapping)==3000 and sum(r['teacher_used'] for r in entries)==1600,
        'typed bundle requires original2800 plus fixed200; parent/teacher budget cannot be conflated')
    for proof in (primary_proof,auxiliary):
        for name,checksum in proof['verified_files_sha256'].items():
            require(name not in files or files[name]==checksum, 'historical source/archive SHA conflict in typed bundle')
            files[name]=checksum
    files[str(path)]=lineage.file_sha(path)
    projection=dict(version=BUNDLE_VERSION,split='train',TEST_used=False,complete=True,expected=3000,completed=3000,
        entries=entries,case_receipts=receipts,missing_case_ids=[],plan_sha256=None,single_plan_SHA_claimed=False,
        projection_rewrites_original_receipts=False,per_case_actual_plan_sha256=mapping,
        actual_plan_sha256_values=sorted(set(mapping.values())),base_plan_sha256=primary['base_plan_sha256'],
        primary_manifest=bundle['primary_manifest'],auxiliary_teacher_manifest=bundle['auxiliary_teacher_manifest'],
        primary_complete2800=True,auxiliary_complete200=True,TRAIN_parent_sources=1400,teacher_rollouts=1600,
        original_Context_K1_hold_rollouts=1400,primary_Context_view_unchanged=True,
        original_primary_augmentation_budget=primary['augmentation_budget'],auxiliary_augmentation_budget=BUDGET)
    return dict(schema=BUNDLE_VERSION,audit_passed=True,complete=True,execution_kind='primary2800_plus_separate_fixed_aux_teacher200',
        acquisition_projection=projection,per_case_actual_plan_sha256=mapping,
        actual_plan_sha256_values=projection['actual_plan_sha256_values'],verified_files_sha256=files,
        primary_proof=primary_proof,auxiliary_proof=auxiliary)
