"""First-execution VAL audit with real immutable files and all275 parents."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.multitask import validation_lineage as audit
from our_mani.multitask.acquisition_lineage import TASKS,SOURCE_SCHEMA,canonical_sha,file_sha,pointer,expected_reset,AcquisitionLineageError

ROOT=Path(__file__).resolve().parents[1]


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');return path


def freeze_plan(path,value):
    value.pop('plan_sha256',None);value['plan_sha256']=canonical_sha(value);put(path,value);return pointer(path.parents[0],path)


def fixture(root,completed=1):
    root=Path(root);train=dict(split='train',episodes=[dict(task=task,raw_episode_id='train'+str(i),path='/readonly/'+task+'/train'+str(i)+'.h5')
        for task in TASKS for i in range(100)])
    val=dict(split='validation',episodes=[dict(task=TASKS[i%14],raw_episode_id='val'+str(i),path='/readonly/val'+str(i)+'.h5') for i in range(275)])
    train_path=put(root/'train_parents.json',train);val_path=put(root/'VAL_parents.json',val)
    demonstration_path=put(root/'artifacts/framework/multitask_r10/demonstrations/manifest.json',
        dict(inputs=dict(manifests_sha256=dict(train=file_sha(train_path),validation=file_sha(val_path)))))
    sources={}
    for name in ('collector.py','launcher.sh','our_mani/multitask/simulation.py','our_mani/multitask/development_reset_recovery.py',
        'our_mani/multitask/rml_path_recovery.py','scripts/collect_multitask_validation_initialization_r10.py','native/task.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen executable '+name+'\n');sources[name]=path
    geometry={task:put(root/'CAD'/(task+'.json'),dict(split='train',task=task,source_group='train:'+task+':train0',
        world_pose_or_motion_schedule_in_asset=False,shared_with_baseline=True,mesh='same frozen TRAIN geometry')) for task in TASKS}
    offline=root/'offline.npz';np.savez(offline,times=np.array([0.,.05]))
    profile=dict(replicas=1,replica_indices=[0],maximum_ticks=60,teacher_used=False,context_primary_view=True,interaction_eligible=False)
    cases=[]
    for i,parent in enumerate(val['episodes']):
        raw=parent['raw_episode_id'];task=parent['task'];cases.append(dict(case_id='case'+str(i),split='validation',task=task,
            raw_episode_id=raw+'__'+audit.VAL_HOLD+'_r0',parent_raw_episode_id=raw,parent_source_group='validation:'+task+':'+raw,
            replica=0,behavior=audit.VAL_HOLD,behavior_replica=0,maximum_collection_ticks=60,teacher_used=False,
            context_primary_view=True,interaction_eligible=False,source=dict(parent,variation=0),template_asset=str(geometry[task]),
            offline_demonstration=dict(path=str(offline),sha256=file_sha(offline))))
    protocol=dict(seed=0,dt=.05,baseline_repo=str(root/'native'),dynamics_target_label_schema='fixedlabels')
    original_config=put(root/'original_protocol.json',protocol)
    original_inputs={str(path):file_sha(path) for path in (sources['collector.py'],sources['launcher.sh'],sources['our_mani/multitask/simulation.py'],
        offline,original_config,demonstration_path,*geometry.values())}
    original=dict(version='original_VAL_plan',created_utc='2026-10-18T00:00:00+00:00',split='validation',cases=cases,
        validation_mode='hold',frozen_policy=None,context_primary_behavior=audit.VAL_HOLD,protocol=protocol,
        protocol_path=str(original_config),inputs=original_inputs,original_demonstration_budget=dict(train_sources=1400,validation_sources=275),
        augmentation_budget=dict(version='r10_fixed_behavior_budget_v1',selected_parent_sources=275,planned_native_cases=275,behaviors={audit.VAL_HOLD:profile}))
    original_path=root/'original_plan.json';freeze_plan(original_path,original)
    diagnostic=put(root/'diagnostic.json',dict(limited_evidence=True))
    physical=put(root/'physical_permission.json',dict(allowed_splits=['train','validation'],official_TEST_allowed=False,
        resample_source_seed_variation_or_scene=False,strict_physical_guards=True,diagnostic_report=pointer(root,diagnostic),
        native_inputs_sha256={'task.py':file_sha(sources['native/task.py'])},
        base_simulation_sha256=file_sha(sources['our_mani/multitask/simulation.py'])))
    rml=put(root/'RML_permission.json',dict(physics_step_called_by_factory=False,real_native_step_motion_replaced=False,
        historical_fault_reproduced_in_native_diagnostic=False,diagnostic_report=pointer(root,diagnostic),
        inputs_sha256={str(sources['our_mani/multitask/rml_path_recovery.py']):file_sha(sources['our_mani/multitask/rml_path_recovery.py'])}))
    permissions=dict(development_reset_profile=pointer(root,physical),RML_factory_hook=pointer(root,rml))
    closure=dict(version='shared_TRAIN_source_closure',split='train',cases=[],protocol=dict(protocol,**permissions),
        inputs={str(path):file_sha(path) for path in sources.values()})
    closure_path=root/'shared_TRAIN_plan.json';freeze_plan(closure_path,closure)
    plan=deepcopy(original);plan['created_utc']='2026-10-18T00:00:01+00:00';plan['protocol'].update(permissions)
    actual_config=put(root/'actual_protocol.json',plan['protocol']);plan['protocol_path']=str(actual_config)
    plan['inputs'].update({str(path):file_sha(path) for path in (*sources.values(),physical,rml,original_path,closure_path,actual_config,train_path,val_path)})
    actual_path=root/'actual_VAL_plan.json';freeze_plan(actual_path,plan)
    archive=[]
    for name,path in sources.items():
        copy=root/'executed_sources'/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
        archive.append(dict(original_path=str(path),**pointer(root,copy)))
    snapshot=put(root/'executed_sources/manifest.json',dict(version=SOURCE_SCHEMA,plan=dict(**pointer(root,actual_path),internal_sha256=plan['plan_sha256']),
        source_count=len(archive),entries=archive,all_plan_inputs_verified_before_and_after=True,source_files_modified=False,case_records_or_labels_modified=False))
    registry=dict(schema=audit.REGISTRATION_SCHEMA,original_plan=pointer(root,original_path),original_collection='original_VAL_not_executed',
        source_closure_plan=pointer(root,closure_path),execution=dict(plan=pointer(root,actual_path),source_snapshot=pointer(root,snapshot),collection='actual_VAL'),
        original_TRAIN_parents=pointer(root,train_path),original_VAL_parents=pointer(root,val_path))
    registry_path=put(root/'execution_registration.json',registry);registry_pointer=pointer(root,registry_path)
    entries=[];receipts=[];paths={}
    for case in cases[:completed]:
        cid=case['case_id'];attempt=root/'actual_VAL/infrastructure_attempts'/cid/'0';output=attempt/'output'
        caseout=output/'cases/validation'/case['task']/cid;caseout.mkdir(parents=True)
        cache=caseout/'body_tracks.npz';np.savez(cache,times=np.array([0.,.05]),teacher_action_valid=np.array([False,False]))
        copied=put(caseout/'TRAIN_geometry.json',json.loads(geometry[case['task']].read_text()))
        entry={key:case[key] for key in ('split','task','raw_episode_id','parent_raw_episode_id','parent_source_group','behavior','behavior_replica',
            'maximum_collection_ticks','teacher_used','context_primary_view','interaction_eligible')}
        entry.update(augmentation_replica=0,source_demonstration=case['offline_demonstration'],dynamics_target_label_schema='fixedlabels',
            original_HDF5_motion_replayed=False,observation_source='causal_RGBD_and_proprioception',target_source='TRAIN_VAL_native_supervision',
            source_kind='VAL_native_generated_non_teacher',cache=str(cache.relative_to(root)),cache_sha256=file_sha(cache),frames=2,
            teacher_command_receipt=None,phase_completion_events_receipt=None)
        reset=dict(expected_reset(case,plan['protocol']),complete=True,permission=permissions['development_reset_profile'],task=case['task'],
            split='validation',source=case['source'],replica=0,source_resampled=False,official_TEST_allowed=False)
        native=dict(expected_reset(case,plan['protocol']),task=case['task'],raw_episode_id=case['parent_raw_episode_id'],source_split='validation',
            source_path=case['source']['path'],replica=0,official_TEST=False,original_source_replayed=False,reset_recovery_receipt=reset)
        raw=dict(version='r10_single_case_collection_receipt_v1',status='completed',case_id=cid,plan_sha256=plan['plan_sha256'],split='validation',
            case_isolation='fresh_simulator_process_per_episode',teacher_actions_used=False,teacher_paths_requested=False,teacher_action_rows=0,
            policy_history_contains_native_GT=False,process_id=1000+int(cid[4:]),entry=entry,native_reset_receipt=native,
            started_utc='2026-10-18T00:00:04+00:00',finished_utc='2026-10-18T00:00:05+00:00',
            files_sha256={str(path.relative_to(root)):file_sha(path) for path in (cache,copied)},
            alignment=dict(clock=dict(ticks=1,dt=.05,budget_seconds=10.,simulation_seconds=.05),actual_physics_seconds=.05,actual_task_seconds=.05))
        base=put(caseout/'base_collector_receipt.json',raw)
        hooks=dict(version=audit.EXECUTION_VERSION,case_id=cid,plan=pointer(root,actual_path),plan_sha256=plan['plan_sha256'],
            execution_registration=registry_pointer,original_plan=registry['original_plan'],source_closure_plan=registry['source_closure_plan'],
            actual_runner=pointer(root,sources['scripts/collect_multitask_validation_initialization_r10.py']),permissions=permissions,
            started_utc='2026-10-18T00:00:03.500000+00:00',finished_utc='2026-10-18T00:00:05.500000+00:00',
            reset_recovery_receipts=[reset],RML_path_recovery_receipts=[],error=None,complete=True,first_execution=True,
            previous_infrastructure_failure_required=False,policy_failure_label_assigned=False,historical_outputs_modified=False,
            teacher_actions_used=False,teacher_paths_requested=False,VAL_geometry_exported=False)
        hook_path=put(caseout/'development_initialization_hook_receipt.json',hooks)
        raw=deepcopy(raw);raw['development_initialization_execution']=dict(version=audit.EXECUTION_VERSION,
            hook_receipt=pointer(root,hook_path),base_collector_receipt=pointer(root,base),execution_registration=registry_pointer)
        raw['files_sha256'].update({str(path.relative_to(root)):file_sha(path) for path in (base,hook_path)})
        raw_path=put(caseout/'case.json',raw)
        command=['bash',str(sources['launcher.sh']),str(sources['scripts/collect_multitask_validation_initialization_r10.py']),
            '--collect-case','--plan',str(actual_path),'--case-id',cid,'--output',str(output),'--execution-registration',str(registry_path)]
        parent=put(attempt/'parent_receipt.json',dict(version=audit.EXECUTION_VERSION,case_id=cid,status='completed',attempt=0,returncode=0,
            process_id=2000+int(cid[4:]),command=command,started_utc='2026-10-18T00:00:03+00:00',finished_utc='2026-10-18T00:00:06+00:00',
            policy_failure_label_assigned=False,execution_registration=registry_pointer,actual_plan=pointer(root,actual_path),first_execution=True,automatic_retries=False))
        (attempt/'native.log').write_text('measured native stdout preserved\n')
        canonical=dict(raw,parent_process_receipt=dict(launcher=str(sources['launcher.sh'].relative_to(root)),launcher_sha256=file_sha(sources['launcher.sh']),
            process_id=2000+int(cid[4:]),command=command,child_case_path=str(raw_path.relative_to(root)),child_case_sha256=file_sha(raw_path)))
        canonical_path=put(root/'actual_VAL/cases/validation'/case['task']/cid/'case.json',canonical)
        entries.append(entry);receipts.append(dict(case_id=cid,**pointer(root,canonical_path)))
        paths[cid]=dict(raw=raw_path,canonical=canonical_path,parent=parent,base=base,hooks=hook_path,cache=cache,geometry=copied)
    manifest=dict(version=audit.TRACK_VERSION,plan_sha256=plan['plan_sha256'],entries=entries,case_receipts=receipts,
        complete=completed==275,expected=275,completed=completed,missing_case_ids=[case['case_id'] for case in cases[completed:]],
        cache_episode_count=completed,source_episode_count=completed,original_demonstration_budget=plan['original_demonstration_budget'],
        augmentation_budget=dict(plan['augmentation_budget'],completed_native_cases=completed,native_started=bool(completed)),
        dynamics_target_label_schema='fixedlabels',TEST_used=False,validation_teacher_actions_used=False,
        shared_profile_execution_registration=registry_pointer)
    # Do not mistake later live executable changes for historical corruption.
    sources['scripts/collect_multitask_validation_initialization_r10.py'].write_text('# newer unexecuted live wrapper\n')
    return manifest,dict(plan=plan,registry=registry,registry_path=registry_path,actual_path=actual_path,paths=paths,sources=sources)


def reseal_case(root,manifest,paths,mutate_record=None,mutate_hook=None,mutate_base=None):
    record=json.loads(paths['canonical'].read_text())
    if mutate_record:mutate_record(record)
    for key,mutate in (('hook_receipt',mutate_hook),('base_collector_receipt',mutate_base)):
        if mutate:
            field='hooks' if key=='hook_receipt' else 'base';path=paths[field];value=json.loads(path.read_text());mutate(value);put(path,value)
            ptr=pointer(root,path);record['development_initialization_execution'][key]=ptr;record['files_sha256'][ptr['path']]=ptr['sha256']
    raw={key:value for key,value in record.items() if key!='parent_process_receipt'};put(paths['raw'],raw)
    record['parent_process_receipt']['child_case_sha256']=file_sha(paths['raw']);put(paths['canonical'],record)
    manifest['entries'][0]=record['entry'];manifest['case_receipts'][0]=dict(case_id=record['case_id'],**pointer(root,paths['canonical']))


class FirstVALLineageTests(unittest.TestCase):
    def test_complete275_hold_sources_archive_raw_parent_base_hook_and_single_actual_plan(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,state=fixture(root,275);proof=audit.audit_validation_acquisition_lineage(root,manifest)
            self.assertTrue(proof['complete']);self.assertEqual(proof['verified_completed_acquisitions'],275)
            self.assertEqual(proof['acquisition_projection'],manifest);self.assertEqual(len(proof['per_case_actual_plan_sha256']),275)
            self.assertEqual(proof['actual_plan_sha256_values'],[state['plan']['plan_sha256']])
            self.assertFalse(proof['prior_infrastructure_failure_required']);self.assertFalse(proof['VAL_RML_paths_created'])
            self.assertNotIn(str(state['sources']['scripts/collect_multitask_validation_initialization_r10.py']),proof['verified_files_sha256'])
            for name,checksum in proof['verified_files_sha256'].items():self.assertEqual(file_sha(name),checksum)

    def test_partial_is_honest_and_complete_gate_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,_=fixture(root)
            proof=audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)
            self.assertFalse(proof['complete']);self.assertEqual(len(proof['missing_case_ids']),274)
            with self.assertRaisesRegex(AcquisitionLineageError,'incomplete'):audit.audit_validation_acquisition_lineage(root,manifest)

    def test_missing_duplicate_or_false275_coverage_reject(self):
        for kind in ('missing','duplicate','source_count','complete'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,_=fixture(root)
                if kind=='missing':manifest['entries']=[];manifest['case_receipts']=[];manifest['completed']=0
                elif kind=='duplicate':manifest['entries']*=2;manifest['case_receipts']*=2;manifest['completed']=2
                elif kind=='source_count':manifest['source_episode_count']=275
                else:manifest['complete']=True;manifest['missing_case_ids']=[]
                with self.assertRaises(AcquisitionLineageError):audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_old_native_attempts_cannot_be_relabelled_as_first_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,_=fixture(root);(root/'original_VAL_not_executed/infrastructure_attempts').mkdir(parents=True)
            with self.assertRaisesRegex(AcquisitionLineageError,'already has native attempts'):
                audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_hook_no_teacher_no_RML_no_export_and_same_reset_scope_are_strict(self):
        for kind in ('teacher','RML','export','reset','failed_receipts','source','profile'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,state=fixture(root)
                def mutate(hooks):
                    if kind=='teacher':hooks['teacher_actions_used']=True
                    elif kind=='RML':hooks['RML_path_recovery_receipts']=[dict(close_complete=True)]
                    elif kind=='export':hooks['VAL_geometry_exported']=True
                    elif kind=='reset':hooks['reset_recovery_receipts'][0]['complete']=False
                    elif kind=='failed_receipts':hooks['failed_parent_receipts']=[]
                    elif kind=='source':hooks['reset_recovery_receipts'][0]['source']['variation']=1
                    else:hooks['permissions']['development_reset_profile']={}
                reseal_case(root,manifest,state['paths']['case0'],mutate_hook=mutate)
                with self.assertRaises(AcquisitionLineageError):audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_base_label_change_missing_actual_hook_and_native_teacher_rows_reject(self):
        for kind in ('base','hook','teacher'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,state=fixture(root)
                if kind=='base':reseal_case(root,manifest,state['paths']['case0'],mutate_base=lambda base:base['entry'].update(source_kind='teacher'))
                elif kind=='hook':reseal_case(root,manifest,state['paths']['case0'],mutate_record=lambda record:record.pop('development_initialization_execution'))
                else:reseal_case(root,manifest,state['paths']['case0'],mutate_record=lambda record:record.update(teacher_action_rows=1))
                with self.assertRaises(AcquisitionLineageError):audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_parent_failure_hidden_retry_and_wrong_registration_command_reject(self):
        for kind in ('returncode','retry','command'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,state=fixture(root);path=state['paths']['case0']['parent'];parent=json.loads(path.read_text())
                if kind=='returncode':parent['returncode']=1;put(path,parent)
                elif kind=='retry':put(path.parent.parent/'1/parent_receipt.json',parent)
                else:
                    parent['command'][parent['command'].index('--execution-registration')+1]=str(root/'wrong.json');put(path,parent)
                with self.assertRaises(AcquisitionLineageError):audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_original_cases_physical_rules_profile_and_parent_roster_cannot_change(self):
        for kind in ('variation','seed','ticks','profile','parents'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,state=fixture(root);plan=state['plan'];registry=state['registry']
                if kind=='parents':
                    path=root/registry['original_VAL_parents']['path'];value=json.loads(path.read_text());value['episodes'].pop();put(path,value)
                    registry['original_VAL_parents']=pointer(root,path)
                else:
                    if kind=='variation':plan['cases'][0]['source']['variation']=1
                    elif kind=='seed':plan['protocol']['seed']=1
                    elif kind=='ticks':plan['cases'][0]['maximum_collection_ticks']=61
                    else:plan['protocol']['development_reset_profile']={}
                    freeze_plan(state['actual_path'],plan);registry['execution']['plan']=pointer(root,state['actual_path'])
                    source_path=root/registry['execution']['source_snapshot']['path'];snapshot=json.loads(source_path.read_text())
                    snapshot['plan']=dict(**registry['execution']['plan'],internal_sha256=plan['plan_sha256']);put(source_path,snapshot)
                    registry['execution']['source_snapshot']=pointer(root,source_path)
                put(state['registry_path'],registry)
                with self.assertRaises(AcquisitionLineageError):audit.verify_validation_execution_registration(root,pointer(root,state['registry_path']))

    def test_rehashed_teacher_cache_changed_clock_or_new_geometry_reject(self):
        for kind in ('teacher','clock','geometry'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,state=fixture(root);paths=state['paths']['case0']
                if kind=='geometry':
                    geometry=json.loads(paths['geometry'].read_text());geometry['mesh']='new VAL exported mesh';put(paths['geometry'],geometry)
                else:
                    np.savez(paths['cache'],times=np.array([0.,.1 if kind=='clock' else .05]),
                        teacher_action_valid=np.array([False,kind=='teacher']))
                def mutate(record):
                    for name in list(record['files_sha256']):record['files_sha256'][name]=file_sha(root/name)
                    record['entry']['cache_sha256']=file_sha(paths['cache'])
                reseal_case(root,manifest,paths,mutate_record=mutate)
                with self.assertRaises(AcquisitionLineageError):audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)

    def test_archived_executable_corruption_is_not_hidden_by_live_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,_=fixture(root)
            (root/'executed_sources/scripts/collect_multitask_validation_initialization_r10.py').write_text('# corrupted historical bytes\n')
            with self.assertRaisesRegex(AcquisitionLineageError,'lineage bytes changed'):
                audit.audit_validation_acquisition_lineage(root,manifest,require_complete=False)


if __name__=='__main__':unittest.main()
