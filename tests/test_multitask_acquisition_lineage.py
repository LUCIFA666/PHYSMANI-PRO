"""Fixed full-roster coverage and explicitly declared historical transitions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.multitask.acquisition_lineage import (TASKS,TRAIN_TEACHER,TRAIN_HOLD,canonical_sha,
    pointer,validate_fixed_roster,verify_recovery_transition,expected_reset,AcquisitionLineageError,
    audit_acquisition_lineage,SCHEMA,SOURCE_SCHEMA,file_sha,load_archived_plan,verify_recovery_execution)

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n');return path


def refresh_manifest(value):
    value.pop('lineage_sha256',None);value['lineage_sha256']=canonical_sha(value);return value


def historical_fixture(root,*,recovered=False):
    """A complete fixed roster, one measured completion, real archive/receipt bytes."""
    root=Path(root);base,train,val=full_fixture()
    geometry=write_json(root/'CAD.json',dict(collision='fixed TRAIN CAD'))
    demo=root/'offline.npz';np.savez(demo,times=np.array([0.,.05]))
    for case in base['cases']:
        case['template_asset']=str(geometry);case['offline_demonstration']=dict(cache=str(demo),cache_sha256=file_sha(demo))
    collector=root/'collector.py';collector.write_text('# original collector\n')
    launcher=root/'launcher.sh';launcher.write_text('# original launcher\n')
    def register(plan,index):
        config=write_json(root/('protocol'+str(index)+'.json'),plan['protocol']);plan['protocol_path']=str(config)
        plan['inputs']={str(path):file_sha(path) for path in (collector,launcher,geometry,demo,config)}
        plan.pop('plan_sha256',None);plan['plan_sha256']=canonical_sha(plan)
        plan_path=write_json(root/('plan'+str(index)+'.json'),plan)
        entries=[]
        for source in (collector,launcher):
            archived=root/('sources'+str(index))/source.name;archived.parent.mkdir(parents=True,exist_ok=True);archived.write_bytes(source.read_bytes())
            entries.append(dict(original_path=str(source),**pointer(root,archived)))
        snapshot=write_json(root/('sources'+str(index))/'manifest.json',dict(version=SOURCE_SCHEMA,
            plan=dict(**pointer(root,plan_path),internal_sha256=plan['plan_sha256']),all_plan_inputs_verified_before_and_after=True,
            source_count=2,entries=entries,source_files_modified=False,case_records_or_labels_modified=False))
        return dict(plan=pointer(root,plan_path),source_snapshot=pointer(root,snapshot),collection='collection'+str(index))
    old=register(base,0);case=base['cases'][0];cid=case['case_id'];old_key=base['plan_sha256'];registrations=[old];chain=[]
    def command(reg,attempt):
        return ['bash',str(launcher),str(collector),'--collect-case','--plan',str(root/reg['plan']['path']),
            '--case-id',cid,'--output',str(attempt/'output')]
    if recovered:
        failed_dir=root/old['collection']/'infrastructure_attempts'/cid/'0';failed_dir.mkdir(parents=True)
        failed=write_json(failed_dir/'parent_receipt.json',dict(case_id=cid,status='infrastructure_error',policy_failure_label_assigned=False,
            process_id=11,command=command(old,failed_dir),started_utc='2026-10-18T00:00:00+00:00',finished_utc='2026-10-18T00:00:01+00:00'))
        (failed_dir/'native.log').write_text('infrastructure failed before measured collection\n')
        chain=[dict(actual_plan_sha256=old_key,parent_receipt=pointer(root,failed),native_log=pointer(root,failed_dir/'native.log'),
            started_utc='2026-10-18T00:00:00+00:00',finished_utc='2026-10-18T00:00:01+00:00')]
        collector.write_text('# reviewed infrastructure recovery collector\n')
        new=deepcopy(base);new['version']='fixturev2';new['protocol']['version']='v2';new['created_utc']='2026-10-18T00:00:02+00:00'
        current=register(new,1);current['predecessor_plan_sha256']=old_key
        change=write_json(root/'change.json',dict(schema='r10_declared_infrastructure_only_acquisition_transition_v1',
            created_utc='2026-10-18T00:00:02+00:00',from_plan_sha256=old_key,to_plan_sha256=new['plan_sha256'],reason='reviewed infrastructure fix',
            preserves_case_seed_variation_geometry_labels_budget=True,allowed_recovery_case_ids=[cid],protocol_changes=[],
            source_changes=[dict(original_path=str(collector),before_sha256=base['inputs'][str(collector)],after_sha256=new['inputs'][str(collector)])]))
        current['infrastructure_change_declaration']=pointer(root,change);registrations.append(current);actual=new
    else:current=old;actual=base
    attempt=root/current['collection']/'infrastructure_attempts'/cid/'0';output=attempt/'output';output.mkdir(parents=True)
    cache=output/'cache.npz';np.savez(cache,times=np.array([0.,.05]))
    actual_geometry=write_json(output/'TRAIN_geometry.json',dict(collision='fixed TRAIN CAD'))
    entry={name:case[name] for name in ('task','raw_episode_id','parent_raw_episode_id','parent_source_group','behavior','behavior_replica',
        'maximum_collection_ticks','teacher_used','context_primary_view','interaction_eligible')}
    entry.update(dynamics_target_label_schema='fixedlabels',augmentation_replica=0,source_demonstration=case['offline_demonstration'],
        original_HDF5_motion_replayed=False,observation_source='causal_RGBD_and_proprioception',target_source='TRAIN_VAL_native_supervision',
        source_kind='TRAIN_native_generated_augmentation',cache=str(cache),cache_sha256=file_sha(cache),frames=2)
    reset=dict(expected_reset(case,actual['protocol']),task=case['task'],raw_episode_id=case['parent_raw_episode_id'],source_split='train',replica=0,
        source_path=case['source']['path'],official_TEST=False,original_source_replayed=False)
    raw=dict(status='completed',case_id=cid,plan_sha256=actual['plan_sha256'],split='train',case_isolation='fresh_simulator_process_per_episode',
        teacher_actions_used=True,policy_history_contains_native_GT=False,entry=entry,native_reset_receipt=reset,
        started_utc='2026-10-18T00:00:04+00:00',finished_utc='2026-10-18T00:00:05+00:00',
        files_sha256={str(cache):file_sha(cache),str(actual_geometry):file_sha(actual_geometry)},
        alignment=dict(clock=dict(ticks=1,dt=.05,budget_seconds=10.,simulation_seconds=.05),actual_physics_seconds=.05,actual_task_seconds=.05))
    raw_path=write_json(output/'cases/train'/case['task']/cid/'case.json',raw)
    parent=write_json(attempt/'parent_receipt.json',dict(case_id=cid,status='completed',returncode=0,policy_failure_label_assigned=False,process_id=22,
        command=command(current,attempt),started_utc='2026-10-18T00:00:03+00:00',finished_utc='2026-10-18T00:00:06+00:00'))
    canonical=dict(raw,parent_process_receipt=dict(launcher=str(launcher.relative_to(root)),launcher_sha256=file_sha(launcher),process_id=22,
        command=command(current,attempt),child_case_path=str(raw_path.relative_to(root)),child_case_sha256=file_sha(raw_path)))
    canonical_path=write_json(root/current['collection']/'cases/train'/case['task']/cid/'case.json',canonical)
    manifest=dict(schema=SCHEMA,base_plan_sha256=old_key,actual_plans=registrations,
        original_TRAIN_parents=pointer(root,write_json(root/'train.json',train)),original_VAL_parents=pointer(root,write_json(root/'val.json',val)),
        completed_acquisitions=[dict(case_id=cid,actual_plan_sha256=actual['plan_sha256'],canonical_receipt=pointer(root,canonical_path),
            parent_receipt=pointer(root,parent),preceding_infrastructure_attempts=chain)],
        coverage_contract=dict(original_TRAIN_parents=1400,original_VAL_parents=275,minimum_completed_phase_parents=8),case_records_or_labels_rewritten=False)
    # Historical execution is intentionally independent of changed live code.
    collector.write_text('# present source is not the historical source\n')
    return refresh_manifest(manifest),dict(cid=cid,canonical=canonical_path,raw=raw_path,parent=parent,collector=collector,cache=cache)


def hook_witness_fixture(root):
    """Isolate the hook-witness JSON validator after the common lineage audit."""
    manifest,paths=historical_fixture(root,recovered=True);proof=audit_acquisition_lineage(root,manifest)
    files={};previous=load_archived_plan(root,manifest['actual_plans'][0],files)
    current=load_archived_plan(root,manifest['actual_plans'][1],files)
    keys=('development_reset_profile','RML_factory_hook','recovery_input')
    for key in keys:current['plan']['protocol'][key]=pointer(root,write_json(root/(key+'.json'),dict(fixed_permission=key)))
    runner=root/'collect_multitask_recovery_r10.py';runner.write_text('# actual owned recovery runner\n')
    archive=root/'sources1/owned_recovery_runner.py';archive.write_bytes(runner.read_bytes())
    current['sources'][str(runner)]=dict(path=str(archive),sha256=file_sha(archive));files[str(archive)]=file_sha(archive)
    registry=write_json(root/'execution_registration.json',dict(schema='r10_explicit_recovery_execution_registration_v1',
        original=previous['registration'],recovery=current['registration']))
    completion=proof['completed_acquisitions'][0];record=json.loads(paths['canonical'].read_text())
    reset=dict(completion['reset_identity'],complete=True,permission=current['plan']['protocol']['development_reset_profile'],
        task=record['entry']['task'],split='train',source_resampled=False,official_TEST_allowed=False)
    record['native_reset_receipt']['reset_recovery_receipt']=reset
    base={key:value for key,value in record.items() if key!='parent_process_receipt'}
    output=paths['parent'].parent/'output';base_path=write_json(output/'base_collector_receipt.json',base)
    version='r10_explicit_same_case_infrastructure_recovery_runner_v1'
    hooks=dict(version=version,complete=True,error=None,case_id=paths['cid'],plan=current['registration']['plan'],
        plan_sha256=current['plan']['plan_sha256'],original_plan=previous['registration']['plan'],
        execution_registration=pointer(root,registry),permissions={key:current['plan']['protocol'][key] for key in keys},
        failed_parent_receipts=[row['parent_receipt'] for row in completion['preceding_infrastructure_attempts']],actual_runner=pointer(root,runner),
        reset_recovery_receipts=[reset],RML_path_recovery_receipts=[dict(close_complete=True,physics_step_called_by_adapter=False,
            real_native_step_motion_replaced=False,historical_fault_reproduced_in_native_diagnostic=False)],
        started_utc='2026-10-18T00:00:03.500000+00:00',finished_utc='2026-10-18T00:00:05.500000+00:00',
        policy_failure_label_assigned=False,historical_v11_outputs_modified=False,partial_historical_rows_reconstructed=False,new_attempt_is_original_v11=False)
    hook_path=write_json(output/'recovery_hook_receipt.json',hooks)
    record['recovery_execution']=dict(version=version,hook_receipt=pointer(root,hook_path),base_collector_receipt=pointer(root,base_path),
        execution_registration=pointer(root,registry))
    def commit():
        for key,path in (('hook_receipt',hook_path),('base_collector_receipt',base_path)):
            old=record['recovery_execution'][key];record['files_sha256'].pop(old['path'],None)
            ptr=pointer(root,path);record['recovery_execution'][key]=ptr;record['files_sha256'][ptr['path']]=ptr['sha256']
        write_json(paths['canonical'],record);completion['canonical_receipt']=pointer(root,paths['canonical'])
    commit()
    # Only historical archived runner bytes matter to this validator.
    runner.write_text('# newer live runner version\n')
    return dict(completion=completion,current=current,previous=previous,files={},record=record,hooks=hooks,hook_path=hook_path,
        base_path=base_path,registry=registry,commit=commit)


def full_fixture():
    train=dict(split='train',episodes=[dict(task=task,raw_episode_id='episode'+str(i),path='/readonly/'+task+'/episode'+str(i)+'.h5')
        for task in TASKS for i in range(100)])
    val=dict(split='validation',episodes=[dict(task=TASKS[i%14],raw_episode_id='val'+str(i),path='/readonly/val'+str(i)+'.h5') for i in range(275)])
    cases=[];profiles={TRAIN_TEACHER:dict(replicas=1,replica_indices=[0],maximum_ticks=200,teacher_used=True,context_primary_view=False,interaction_eligible=True),
        TRAIN_HOLD:dict(replicas=1,replica_indices=[0],maximum_ticks=60,teacher_used=False,context_primary_view=True,interaction_eligible=False)}
    for parent in train['episodes']:
        for behavior,profile in profiles.items():
            raw=parent['raw_episode_id'];task=parent['task'];cases.append(dict(case_id=task+'_'+raw+'_'+behavior,split='train',task=task,
                raw_episode_id=raw+'__'+behavior+'_r0',parent_raw_episode_id=raw,parent_source_group='train:'+task+':'+raw,
                behavior=behavior,behavior_replica=0,replica=0,source=dict(parent,variation=0),
                maximum_collection_ticks=profile['maximum_ticks'],teacher_used=profile['teacher_used'],
                context_primary_view=profile['context_primary_view'],interaction_eligible=profile['interaction_eligible'],template_asset='/frozen/CAD.json'))
    plan=dict(version='fixturev1',created_utc='2026-10-18T00:00:00+00:00',split='train',cases=cases,
        original_demonstration_budget=dict(train_sources=1400,validation_sources=275),
        augmentation_budget=dict(version='r10_fixed_behavior_budget_v1',selected_parent_sources=1400,planned_native_cases=2800,behaviors=profiles),
        protocol=dict(version='v1',seed=0,dt=.05,geometry='fixed',dynamics_target_label_schema='fixedlabels',gripper_velocity=.2),
        protocol_path='/metadata/v1.json',inputs={'/source/collector.py':'a'*64,'/metadata/v1.json':'c'*64})
    return plan,train,val


class FixedRosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan,cls.train,cls.val=full_fixture()

    def test_exact1400_teacher_hold_pairs_and275_VAL_requirement(self):
        result=validate_fixed_roster(self.plan,self.train,self.val)
        self.assertEqual(result['planned_acquisitions'],2800);self.assertEqual(result['minimum_completed_phase_parents'],8)

    def test_missing_duplicate_and_hold_substitute_are_rejected(self):
        for kind in ('missing','duplicate','hold_substitute'):
            plan=deepcopy(self.plan)
            if kind=='missing':plan['cases'].pop()
            elif kind=='duplicate':plan['cases'][-1]=deepcopy(plan['cases'][0])
            else:plan['cases'][0]['behavior']=TRAIN_HOLD
            with self.subTest(kind=kind),self.assertRaises(AcquisitionLineageError):validate_fixed_roster(plan,self.train,self.val)

    def test_budget_labels_replica_and_VAL_coverage_cannot_be_relaxed(self):
        for kind in ('ticks','teacher','replica','VAL'):
            plan=deepcopy(self.plan);val=deepcopy(self.val)
            if kind=='ticks':plan['cases'][0]['maximum_collection_ticks']=199
            elif kind=='teacher':plan['cases'][0]['teacher_used']=False
            elif kind=='replica':plan['cases'][0]['replica']=1
            else:val['episodes'].pop()
            with self.subTest(kind=kind),self.assertRaises(AcquisitionLineageError):validate_fixed_roster(plan,self.train,val)

    def test_teacher_and_hold_share_derived_parent_seed_but_preserve_variation(self):
        self.assertEqual(expected_reset(self.plan['cases'][0],self.plan['protocol']),expected_reset(self.plan['cases'][1],self.plan['protocol']))
        changed=deepcopy(self.plan['cases'][0]);changed['source']['variation']=1
        self.assertNotEqual(expected_reset(changed,self.plan['protocol']),expected_reset(self.plan['cases'][0],self.plan['protocol']))


class DeclaredTransitionTests(unittest.TestCase):
    def transition_fixture(self,root,mutate=None,declare_protocol=False,declare_source=True):
        before,_,_=full_fixture();before['plan_sha256']=canonical_sha(before)
        after=deepcopy(before);after.pop('plan_sha256');after['version']='fixturev2';after['created_utc']='2026-10-18T01:00:00+00:00'
        after['protocol']['version']='v2';after['protocol_path']='/metadata/v2.json'
        after['inputs']={'/source/collector.py':'b'*64,'/metadata/v2.json':'d'*64}
        evidence=root/'review.json';evidence.write_text('{}\n')
        protocol_changes=[]
        if mutate:mutate(after)
        if declare_protocol:
            protocol_changes=[dict(json_pointer='/protocol/development_reset_profile',old=dict(present=False),
                new=dict(present=True,value=after['protocol']['development_reset_profile']),reason='explicit shared development reset permission',evidence=pointer(root,evidence))]
        after['plan_sha256']=canonical_sha(after)
        change=dict(schema='r10_declared_infrastructure_only_acquisition_transition_v1',created_utc='2026-10-18T00:30:00+00:00',
            from_plan_sha256=before['plan_sha256'],to_plan_sha256=after['plan_sha256'],reason='infrastructure fix',
            preserves_case_seed_variation_geometry_labels_budget=True,allowed_recovery_case_ids=[before['cases'][0]['case_id']],
            source_changes=[dict(original_path='/source/collector.py',before_sha256='a'*64,after_sha256='b'*64)] if declare_source else [],
            protocol_changes=protocol_changes)
        file=root/'change.json';file.write_text(json.dumps(change))
        previous=dict(plan=before,sources={'/source/collector.py':dict(sha256='a'*64)},registration=dict(plan=dict(path='old.json',sha256='1'*64)))
        current=dict(plan=after,sources={'/source/collector.py':dict(sha256='b'*64)},
            registration=dict(predecessor_plan_sha256=before['plan_sha256'],infrastructure_change_declaration=pointer(root,file)))
        return previous,current

    def test_only_version_metadata_plus_exactly_declared_source_diff_passes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);before,after=self.transition_fixture(root)
            self.assertEqual(verify_recovery_transition(root,before,after,{})['reason'],'infrastructure fix')

    def test_undeclared_source_or_reset_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory)
            for source,reset in ((False,False),(True,True)):
                before,after=self.transition_fixture(root,mutate=(lambda plan:plan['protocol'].update(development_reset_profile={'enabled':True})) if reset else None,
                    declare_source=source)
                with self.subTest(source=source,reset=reset),self.assertRaises(AcquisitionLineageError):verify_recovery_transition(root,before,after,{})

    def test_exact_declared_development_reset_transition_is_explicitly_accepted(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);before,after=self.transition_fixture(root,
                mutate=lambda plan:plan['protocol'].update(development_reset_profile={'enabled':True,'schema':'fixture'}),declare_protocol=True)
            result=verify_recovery_transition(root,before,after,{})
            self.assertEqual(result['protocol_changes'][0]['json_pointer'],'/protocol/development_reset_profile')

    def test_seed_dt_geometry_labels_action_budget_and_case_changes_are_forbidden(self):
        mutations=(lambda p:p['protocol'].update(seed=1),lambda p:p['protocol'].update(dt=.1),
            lambda p:p['protocol'].update(geometry='new'),lambda p:p['protocol'].update(dynamics_target_label_schema='new'),
            lambda p:p['protocol'].update(gripper_velocity=.3),lambda p:p['cases'][0].update(maximum_collection_ticks=199),
            lambda p:p['cases'][0]['source'].update(variation=1))
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory)
            for index,mutation in enumerate(mutations):
                before,after=self.transition_fixture(root,mutate=mutation)
                with self.subTest(index=index),self.assertRaises(AcquisitionLineageError):verify_recovery_transition(root,before,after,{})

    def test_recovery_metadata_failure_list_must_be_unique_and_cover_every_declared_case(self):
        for kind in ('valid','duplicate','missing'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);before,after=self.transition_fixture(root)
                original=write_json(root/'original.json',before['plan']);before['path']=original
                before['registration']['plan']=pointer(root,original)
                cid=before['plan']['cases'][0]['case_id']
                failed=write_json(root/'failed.json',dict(status='infrastructure_error',case_id=cid,
                    command=['--plan',str(original)]))
                failures=[pointer(root,failed)]*(2 if kind=='duplicate' else 1)
                payload=write_json(root/'recovery_input.json',dict(original_plan=before['registration']['plan'],
                    failed_parent_receipts=failures,development_reset_profile=None,RML_factory_hook=None))
                after['plan']['protocol']['recovery_input']=pointer(root,payload)
                after['plan'].pop('plan_sha256');after['plan']['plan_sha256']=canonical_sha(after['plan'])
                change_path=root/'change.json';change=json.loads(change_path.read_text())
                change['to_plan_sha256']=after['plan']['plan_sha256']
                if kind=='missing':change['allowed_recovery_case_ids'].append(before['plan']['cases'][1]['case_id'])
                change['protocol_changes']=[dict(json_pointer='/protocol/recovery_input',old=dict(present=False),
                    new=dict(present=True,value=pointer(root,payload)),reason='same-case failure lineage binding',evidence=pointer(root,root/'review.json'))]
                write_json(change_path,change);after['registration']['infrastructure_change_declaration']=pointer(root,change_path)
                if kind=='valid':verify_recovery_transition(root,before,after,{})
                else:
                    with self.assertRaises(AcquisitionLineageError):verify_recovery_transition(root,before,after,{})

    def test_declaring_a_seed_change_or_confusing_absent_with_null_is_rejected(self):
        for kind in ('seed','absent_null'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);before,after=self.transition_fixture(root,
                    mutate=lambda plan:plan['protocol'].update(development_reset_profile={'enabled':True}),declare_protocol=True)
                path=root/'change.json';change=json.loads(path.read_text())
                if kind=='seed':change['protocol_changes'][0]['json_pointer']='/protocol/seed'
                else:change['protocol_changes'][0]['old']=dict(present=True,value=None)
                write_json(path,change);after['registration']['infrastructure_change_declaration']=pointer(root,path)
                with self.assertRaises(AcquisitionLineageError):verify_recovery_transition(root,before,after,{})


class HistoricalAcquisitionTests(unittest.TestCase):
    def test_archived_sources_survive_live_changes_and_projection_preserves_actual_receipts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,paths=historical_fixture(root)
            proof=audit_acquisition_lineage(root,manifest)
            self.assertFalse(proof['complete']);self.assertEqual(proof['verified_completed_acquisitions'],1)
            self.assertEqual(len(proof['missing_case_ids']),2799)
            projected=proof['acquisition_projection'];self.assertIsNone(projected['plan_sha256'])
            self.assertEqual(projected['entries'][0],json.loads(paths['raw'].read_text())['entry'])
            self.assertEqual(projected['case_receipts'][0]['sha256'],file_sha(paths['canonical']))
            self.assertEqual(proof['per_case_actual_plan_sha256'][paths['cid']],manifest['base_plan_sha256'])
            self.assertNotIn(str(paths['collector']),proof['verified_files_sha256'])
            self.assertIn(str(root/'sources0/collector.py'),proof['verified_files_sha256'])
            with self.assertRaisesRegex(AcquisitionLineageError,'missing teacher/hold'):
                audit_acquisition_lineage(root,manifest,require_complete=True)

    def test_recovery_uses_own_plan_source_archive_and_preserves_preceding_infra_chain(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,paths=historical_fixture(root,recovered=True)
            proof=audit_acquisition_lineage(root,manifest);actual=manifest['completed_acquisitions'][0]['actual_plan_sha256']
            self.assertNotEqual(actual,manifest['base_plan_sha256'])
            self.assertEqual(proof['per_case_actual_plan_sha256'],{paths['cid']:actual})
            self.assertEqual(len(proof['completed_acquisitions'][0]['preceding_infrastructure_attempts']),1)
            archives=proof['verified_files_sha256']
            self.assertNotEqual(archives[str(root/'sources0/collector.py')],archives[str(root/'sources1/collector.py')])

    def test_duplicate_or_hidden_completed_and_prior_chain_omission_reject(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,_=historical_fixture(root,recovered=True)
            for kind in ('duplicate','omission','chain'):
                changed=deepcopy(manifest)
                if kind=='duplicate':changed['completed_acquisitions'].append(deepcopy(changed['completed_acquisitions'][0]))
                elif kind=='omission':changed['completed_acquisitions']=[]
                else:changed['completed_acquisitions'][0]['preceding_infrastructure_attempts']=[]
                with self.subTest(kind=kind),self.assertRaises(AcquisitionLineageError):
                    audit_acquisition_lineage(root,refresh_manifest(changed))

    def test_recovery_without_predecessor_infra_and_completed_replacement_reject(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,paths=historical_fixture(root,recovered=True)
            old_collection=root/manifest['actual_plans'][0]['collection']
            old_parent=old_collection/'infrastructure_attempts'/paths['cid']/'0/parent_receipt.json'
            preserved=old_parent.read_bytes();old_parent.unlink()
            with self.assertRaisesRegex(AcquisitionLineageError,'no preceding same-case'):
                audit_acquisition_lineage(root,manifest)
            old_parent.write_bytes(preserved)
            canonical=json.loads(paths['canonical'].read_text());canonical['plan_sha256']=manifest['base_plan_sha256']
            write_json(old_collection/'cases/train'/canonical['entry']['task']/paths['cid']/'case.json',canonical)
            with self.assertRaisesRegex(AcquisitionLineageError,'replacement'):
                audit_acquisition_lineage(root,manifest)

    def test_rehashed_canonical_cannot_rewrite_raw_child_or_seed_and_outputs_remain_hash_bound(self):
        for kind in ('canonical','seed','output'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,paths=historical_fixture(root)
                if kind=='output':paths['cache'].write_bytes(b'tampered cache')
                else:
                    record=json.loads(paths['canonical'].read_text())
                    if kind=='canonical':record['entry']['frames']=3
                    else:record['native_reset_receipt']['seed']+=1
                    write_json(paths['canonical'],record)
                    manifest['completed_acquisitions'][0]['canonical_receipt']=pointer(root,paths['canonical'])
                    refresh_manifest(manifest)
                with self.assertRaises(AcquisitionLineageError):audit_acquisition_lineage(root,manifest)

    def test_actual_command_cannot_point_outside_archived_launcher_or_attempt_output(self):
        for kind in ('launcher','output'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);manifest,paths=historical_fixture(root)
                parent=json.loads(paths['parent'].read_text());command=parent['command']
                if kind=='launcher':command[1]=str(root/'undeclared.sh')
                else:command[command.index('--output')+1]=str(root/'other/output')
                write_json(paths['parent'],parent)
                manifest['completed_acquisitions'][0]['parent_receipt']=pointer(root,paths['parent']);refresh_manifest(manifest)
                with self.assertRaisesRegex(AcquisitionLineageError,'archived launcher/collector'):
                    audit_acquisition_lineage(root,manifest)

    def test_completed_parent_with_nonzero_exit_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);manifest,paths=historical_fixture(root)
            parent=json.loads(paths['parent'].read_text());parent['returncode']=1;write_json(paths['parent'],parent)
            manifest['completed_acquisitions'][0]['parent_receipt']=pointer(root,paths['parent']);refresh_manifest(manifest)
            with self.assertRaisesRegex(AcquisitionLineageError,'successful native return code'):
                audit_acquisition_lineage(root,manifest)


class RecoveryExecutionWitnessTests(unittest.TestCase):
    def verify(self,root,value):
        return verify_recovery_execution(root,value['completion'],value['current'],value['previous'],
            value['completion']['preceding_infrastructure_attempts'],value['files'])

    def test_same_case_hook_base_registry_and_archived_runner_witness_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);value=hook_witness_fixture(root);self.verify(root,value)
            self.assertNotIn(str(root/'collect_multitask_recovery_r10.py'),value['files'])

    def test_missing_hook_cannot_be_treated_as_execution_of_declared_permissions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
            root=Path(directory);value=hook_witness_fixture(root);value['record'].pop('recovery_execution')
            canonical=root/value['completion']['canonical_receipt']['path'];write_json(canonical,value['record'])
            value['completion']['canonical_receipt']=pointer(root,canonical)
            with self.assertRaisesRegex(AcquisitionLineageError,'actual recovery execution witness'):self.verify(root,value)

    def test_rehashed_incomplete_or_wrong_hook_witnesses_reject(self):
        for kind in ('reset','RML','permission','runner','failure_chain','resample','historical_rewrite'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);value=hook_witness_fixture(root);hooks=value['hooks']
                if kind=='reset':hooks['reset_recovery_receipts'][0]['complete']=False
                elif kind=='RML':hooks['RML_path_recovery_receipts'][0]['close_complete']=False
                elif kind=='permission':hooks['permissions']['RML_factory_hook']={}
                elif kind=='runner':hooks['actual_runner']['sha256']='0'*64
                elif kind=='failure_chain':hooks['failed_parent_receipts']=[]
                elif kind=='resample':hooks['reset_recovery_receipts'][0]['source_resampled']=True
                else:hooks['partial_historical_rows_reconstructed']=True
                write_json(value['hook_path'],hooks);value['commit']()
                with self.assertRaises(AcquisitionLineageError):self.verify(root,value)

    def test_rehashed_base_label_change_and_other_execution_registry_reject(self):
        for kind in ('base','registry'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=ROOT/'runtime') as directory:
                root=Path(directory);value=hook_witness_fixture(root)
                if kind=='base':
                    base=json.loads(value['base_path'].read_text());base['entry']['teacher_used']=False;write_json(value['base_path'],base)
                    value['commit']()
                else:
                    registry=json.loads(value['registry'].read_text());registry['recovery']['collection']='other_namespace'
                    write_json(value['registry'],registry);ptr=pointer(root,value['registry'])
                    value['record']['recovery_execution']['execution_registration']=ptr;value['hooks']['execution_registration']=ptr
                    write_json(value['hook_path'],value['hooks']);value['commit']()
                with self.assertRaises(AcquisitionLineageError):self.verify(root,value)


if __name__=='__main__':unittest.main()
