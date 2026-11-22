"""Fixed acquisition accounting and exact parent/window coverage, CPU only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from our_mani.multitask.context_se3 import SE3ContextNetwork
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.track_dataset import BodyTrackEpisode,examples,read_tracks
from our_mani.multitask.training_view import merge_training_view,window_statistics,require_production_behavior_budget


def arrays(n):
    t=np.arange(n)*.05;b=3;positions=np.tile(np.c_[t,t*0,t*0+.8][:,None],(1,b,1))
    result=dict(times=t,body_ids=np.asarray(['button','plate','wrap']),positions=positions,
        rotations=np.tile(np.eye(3),(n,b,1,1)),linear_velocities=np.tile([1.,0.,0.],(n,b,1)),
        angular_velocities=np.zeros((n,b,3)),confidence=np.ones((n,b)),valid=np.ones((n,b),bool),
        modes=np.zeros((n,b),np.int8),rotation_observable=np.ones((n,b),bool))
    for name in ('positions','rotations','linear_velocities','angular_velocities'):
        result['target_'+name]=result[name].copy()
    result['target_valid']=result['valid'].copy();result['target_rotation_observable']=result['valid'].copy()
    return result


class TrainingViewTests(unittest.TestCase):
    def test_formal_behavior_budget_cannot_silently_expand_registered_K(self):
        profile=dict(version='r10_fixed_behavior_budget_v1',behaviors={
            'TRAIN_hold_motion_identification':dict(replicas=1,replica_indices=[0])})
        require_production_behavior_budget([dict(fixed_augmentation_budget=profile)])
        profile['behaviors']['TRAIN_hold_motion_identification'].update(replicas=2,replica_indices=[0,1])
        with self.assertRaisesRegex(ValueError,'preregistered K1'):
            require_production_behavior_budget([dict(fixed_augmentation_budget=profile)])
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ('track_dataset.py','context_se3.py'):
            path=self.root/'our_mani/multitask'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# CPU fixture source receipt only\n')
        self.launcher=self.root/'launch.sh';self.launcher.write_text('# mock; never launched\n')
    def tearDown(self): self.temp.cleanup()
    def source_manifest(self,split,parents):
        path=self.root/(split+'_sources.json')
        path.write_text(json.dumps(dict(split=split,episodes=[dict(task=task,raw_episode_id=raw,path='/read-only/source')
            for task,raw in parents])))
        return path
    def acquisition(self,split,parent,replica,n,task='push_moving_button',behavior=None,v6=False):
        teacher=(split=='train') if behavior is None else behavior=='TRAIN_teacher_committed_waypoint'
        raw=parent+(f'__native_aug_r{replica}' if behavior is None else f'__{behavior}_r{replica}')
        folder=self.root/split/task/raw;folder.mkdir(parents=True)
        data=arrays(n)
        if v6:
            from our_mani.multitask.motion_labels import backward_pose_supervision
            for name in ('native_pose_valid','mode_valid','dynamics_segment_valid'):data[name]=data['valid'].copy()
            data,_=backward_pose_supervision(data)
        cache=folder/'body_tracks.npz';np.savez_compressed(cache,**data)
        entry=dict(split=split,task=task,raw_episode_id=raw,parent_raw_episode_id=parent,
            augmentation_replica=replica,
            parent_source_group=f'{split}:{task}:{parent}',cache=str(cache.relative_to(self.root)),
            cache_sha256=file_sha(cache),frames=n,original_HDF5_motion_replayed=False,
            observation_source='causal_RGBD_and_proprioception',target_source='TRAIN_VAL_native_supervision',
            source_kind=('TRAIN_native_generated_augmentation' if teacher else 'TRAIN_native_generated_hold')
                if split=='train' else 'VAL_native_generated_non_teacher')
        if behavior is not None:
            entry.update(behavior=behavior,behavior_replica=replica,teacher_used=teacher,
                maximum_collection_ticks=200 if teacher else 60,
                context_primary_view=not teacher,interaction_eligible=teacher)
        if v6:
            from our_mani.multitask.motion_labels import SCHEMA
            entry['dynamics_target_label_schema']=SCHEMA
        ticks=n-1;record=dict(entry=entry,status='completed',split=split,case_id=raw,
            plan_sha256='a'*64,case_isolation='fresh_simulator_process_per_episode',process_id=42+replica,
            teacher_actions_used=teacher,policy_history_contains_native_GT=False,
            files_sha256={entry['cache']:entry['cache_sha256']},alignment=dict(
                clock=dict(ticks=ticks,dt=.05,budget_seconds=10.,simulation_seconds=ticks*.05),
                actual_physics_seconds=ticks*.05,actual_task_seconds=ticks*.05,
                native_success_latched=n<5,native_terminal_failure_latched=False,native_terminal_latched=n<5))
        child=folder/'raw_child.json';child.write_text(json.dumps(record))
        record['parent_process_receipt']=dict(child_case_path=str(child.relative_to(self.root)),
            child_case_sha256=file_sha(child),launcher=str(self.launcher.relative_to(self.root)),
            launcher_sha256=file_sha(self.launcher),process_id=1000+replica)
        case=folder/'case.json';case.write_text(json.dumps(record))
        return entry,dict(path=str(case.relative_to(self.root)),sha256=file_sha(case),case_id=raw),cache
    def manifest(self,name,acquisitions):
        path=self.root/(name+'.json');entries=[a[0] for a in acquisitions]
        parents={(entry['task'],entry['parent_raw_episode_id']) for entry in entries}
        budget=dict(replicas=len(entries)//len(parents),selected_parent_sources=len(parents),
            planned_native_cases=len(entries),completed_native_cases=len(entries))
        if any('behavior' in entry for entry in entries):
            names=('TRAIN_teacher_committed_waypoint','TRAIN_hold_motion_identification') if entries[0]['split']=='train' else (
                'VAL_hold_motion_identification',)
            profiles={behavior:dict(replicas=1,replica_indices=[0],
                maximum_ticks=200 if 'teacher' in behavior else 60,teacher_used='teacher' in behavior,
                context_primary_view='teacher' not in behavior,interaction_eligible='teacher' in behavior) for behavior in names}
            budget.update(version='r10_fixed_behavior_budget_v1',replicas=None,behaviors=profiles,
                planned_cases_by_behavior={behavior:sum(e['behavior']==behavior for e in entries) for behavior in profiles})
        metadata={}
        if any('dynamics_target_label_schema' in entry for entry in entries):
            metadata['dynamics_target_label_schema']=next(entry['dynamics_target_label_schema'] for entry in entries
                if 'dynamics_target_label_schema' in entry)
        path.write_text(json.dumps(dict(version='r10_causal_body_tracks_v1',complete=True,TEST_used=False,**metadata,
            expected=len(entries),completed=len(entries),entries=entries,case_receipts=[a[1] for a in acquisitions],
            missing_case_ids=[],plan_sha256='a'*64,augmentation_budget=budget)))
        return path
    def merge(self,train,val,train_parents,val_parents,name='view'):
        tp=self.source_manifest('train',train_parents);vp=self.source_manifest('validation',val_parents)
        return merge_training_view(self.root,[train],[val],tp,vp,name,expected_train=len(train_parents),
            expected_validation=len(val_parents),development_only=True)

    def test_short_terminal_is_retained_zero_weight_without_losing_parent(self):
        task='push_moving_button'
        acquisitions=[self.acquisition('train','source0',0,3),self.acquisition('train','source0',1,30)]
        train=self.manifest('train',acquisitions)
        val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        report=self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertTrue(report['training_view_ready']);self.assertFalse(report['formal_comparison_ready'])
        ledger=json.loads((self.root/'view/acquisition_ledger.json').read_text())
        self.assertEqual(len(ledger['entries']),3)
        short=ledger['entries'][0];self.assertEqual(short['training_weight'],0)
        self.assertEqual(short['cache'],acquisitions[0][0]['cache'])
        self.assertTrue(short['native_acquisition_alignment']['native_success_latched'])
        self.assertFalse(short['native_outcome_used_for_selection'])
        manifest,episodes=read_tracks(self.root,self.root/'view/manifest.json')
        self.assertEqual(len(episodes),2)
        self.assertEqual({row['cache'] for row in manifest['entries']},
            {acquisitions[1][0]['cache'],json.loads(val.read_text())['entries'][0]['cache']})

    def test_a_parent_with_only_short_acquisition_fails_instead_of_disappearing(self):
        task='push_moving_button';train=self.manifest('train',[self.acquisition('train','source0',0,3)])
        val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        report=self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse(report['training_view_ready'])
        self.assertEqual(report['parent_coverage']['train']['missing_usable'],[(task,'source0')])
        self.assertFalse((self.root/'view/manifest.json').exists())
        self.assertTrue((self.root/'view/acquisition_ledger.json').exists())

    def test_statistics_match_examples_with_permanent_target_and_mode_boundaries(self):
        a=arrays(40);a['target_valid'][8,0]=False;a['modes'][12:14,1]=1
        a['valid'][20,2]=False
        entry=dict(split='train',task='push_moving_button',raw_episode_id='source0')
        actual,_=examples([BodyTrackEpisode(entry,a)],SE3ContextNetwork(),steps=22,stride=2)
        stats=window_statistics(a)
        self.assertEqual(stats['usable_causal_examples'],len(actual))
        self.assertEqual(stats['full_1p1s_examples'],sum(row['mask'][-1] for row in actual))
        first=next(row for row in actual if row['body_id']=='button' and row['anchor']==4)
        self.assertFalse(first['mask'][4:].any())

    def test_source_sha_or_missing_fixed_attempt_prevents_a_view(self):
        task='push_moving_button';t=self.acquisition('train','source0',0,30)
        train=self.manifest('train',[t]);val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        t[2].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())
        self.assertTrue((self.root/'view/infrastructure_error.json').exists())

    def test_usable_parents_still_need_a_full_horizon_for_each_task(self):
        normal='push_moving_button';high='push_moving_button_high_speed'
        train=self.manifest('train',[self.acquisition('train','source0',0,30),
            self.acquisition('train','source1',0,12,task=high)])
        val=self.manifest('val',[self.acquisition('validation','source8',0,30),
            self.acquisition('validation','source9',0,30,task=high)])
        report=self.merge(train,val,[(normal,'source0'),(high,'source1')],[(normal,'source8'),(high,'source9')])
        self.assertFalse(report['training_view_ready'])
        self.assertEqual(report['parent_coverage']['train']['usable_parent_sources'],2)
        self.assertEqual(report['full_horizon_coverage']['train']['missing_full_horizon_tasks'],[high])

    def test_raw_parent_overlap_between_train_and_val_is_rejected(self):
        task='push_moving_button';train=self.manifest('train',[self.acquisition('train','same',0,30)])
        val=self.manifest('val',[self.acquisition('validation','same',0,30)])
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.merge(train,val,[(task,'same')],[(task,'same')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_incomplete_declared_budget_cannot_be_relabelled_short_terminal(self):
        task='push_moving_button';train=self.manifest('train',[self.acquisition('train','source0',0,30)])
        val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        value=json.loads(train.read_text());value.update(complete=False,expected=2,missing_case_ids=['unfinished'])
        train.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'every fixed-budget'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_fixed_replica_indices_cannot_be_inferred_from_arbitrary_attempts(self):
        task='push_moving_button';train=self.manifest('train',[self.acquisition('train','source0',0,30)])
        val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        value=json.loads(train.read_text());value['entries'][0]['augmentation_replica']=10
        train.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'undeclared replica'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_val_teacher_labels_are_forbidden_even_with_consistent_file_hashes(self):
        task='push_moving_button';train=self.manifest('train',[self.acquisition('train','source0',0,30)])
        acquisition=self.acquisition('validation','source9',0,30)
        path=self.root/acquisition[1]['path'];record=json.loads(path.read_text())
        record['teacher_actions_used']=True
        child=self.root/record['parent_process_receipt']['child_case_path']
        child.write_text(json.dumps({k:v for k,v in record.items() if k!='parent_process_receipt'}))
        record['parent_process_receipt']['child_case_sha256']=file_sha(child)
        path.write_text(json.dumps(record));acquisition[1]['sha256']=file_sha(path)
        val=self.manifest('val',[acquisition])
        with self.assertRaisesRegex(ValueError,'wrongly supervised'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_v5_teacher_context_examples_cannot_replace_a_short_primary_hold(self):
        task='push_moving_button'
        teacher=self.acquisition('train','source0',0,30,behavior='TRAIN_teacher_committed_waypoint')
        hold=self.acquisition('train','source0',0,3,behavior='TRAIN_hold_motion_identification')
        train=self.manifest('train',[teacher,hold]);val=self.manifest('val',[
            self.acquisition('validation','source9',0,30,behavior='VAL_hold_motion_identification')])
        report=self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse(report['training_view_ready'])
        self.assertTrue(report['primary_hold_motion_context_view'])
        self.assertEqual(report['parent_coverage']['train']['usable_parent_sources'],0)
        self.assertGreater(report['parent_coverage']['train']['diagnostic_examples_in_all_acquisitions'],0)
        self.assertEqual(report['full_horizon_coverage']['train']['full_horizon_tasks'],[])
        self.assertFalse((self.root/'view/manifest.json').exists())
        ledger=json.loads((self.root/'view/acquisition_ledger.json').read_text())['entries']
        self.assertEqual(len(ledger),3)
        self.assertEqual(ledger[0]['exclusion_reason'],'not_primary_hold_motion_context_view')
        self.assertEqual(ledger[0]['training_weight'],0)
        self.assertEqual(ledger[0]['cache'],teacher[0]['cache'])

    def test_v5_context_view_contains_only_actual_primary_hold_caches(self):
        task='push_moving_button'
        teacher=self.acquisition('train','source0',0,30,behavior='TRAIN_teacher_committed_waypoint')
        hold=self.acquisition('train','source0',0,30,behavior='TRAIN_hold_motion_identification')
        val_hold=self.acquisition('validation','source9',0,30,behavior='VAL_hold_motion_identification')
        train=self.manifest('train',[teacher,hold]);val=self.manifest('val',[val_hold])
        report=self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertTrue(report['training_view_ready'])
        self.assertFalse(report['teacher_acquisitions_can_replace_missing_hold_parents'])
        manifest,episodes=read_tracks(self.root,self.root/'view/manifest.json')
        self.assertEqual({row['cache'] for row in manifest['entries']},{hold[0]['cache'],val_hold[0]['cache']})
        self.assertEqual(len(episodes),2)

    def test_v5_missing_teacher_acquisition_still_fails_declared_budget(self):
        task='push_moving_button'
        train=self.manifest('train',[
            self.acquisition('train','source0',0,30,behavior='TRAIN_hold_motion_identification')])
        val=self.manifest('val',[
            self.acquisition('validation','source9',0,30,behavior='VAL_hold_motion_identification')])
        with self.assertRaisesRegex(ValueError,'every parent must retain'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_legacy_teacher_view_cannot_be_mixed_into_v5_hold_coverage(self):
        task='push_moving_button'
        train=self.manifest('train',[
            self.acquisition('train','source0',0,30,behavior='TRAIN_teacher_committed_waypoint'),
            self.acquisition('train','source0',0,30,behavior='TRAIN_hold_motion_identification')])
        val=self.manifest('val',[self.acquisition('validation','source9',0,30)])
        with self.assertRaisesRegex(ValueError,'legacy teacher views cannot replace'):
            self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertFalse((self.root/'view/manifest.json').exists())

    def test_v6_schema_propagates_and_raw_v5_cannot_supply_other_split(self):
        from our_mani.multitask.motion_labels import SCHEMA
        task='push_moving_button'
        train=self.manifest('train',[
            self.acquisition('train','source0',0,30,behavior='TRAIN_teacher_committed_waypoint',v6=True),
            self.acquisition('train','source0',0,30,behavior='TRAIN_hold_motion_identification',v6=True)])
        val=self.manifest('val',[
            self.acquisition('validation','source9',0,30,behavior='VAL_hold_motion_identification',v6=True)])
        report=self.merge(train,val,[(task,'source0')],[(task,'source9')])
        self.assertTrue(report['training_view_ready'])
        self.assertEqual(report['dynamics_target_label_schema'],SCHEMA)
        manifest=json.loads((self.root/'view/manifest.json').read_text())
        self.assertEqual(manifest['dynamics_target_label_schema'],SCHEMA)
        self.assertTrue(all(entry['dynamics_target_label_schema']==SCHEMA for entry in manifest['entries']))
        raw_val=self.manifest('raw_val',[
            self.acquisition('validation','source8',0,30,behavior='VAL_hold_motion_identification')])
        with self.assertRaisesRegex(ValueError,'v5 raw and v6 derived-twist'):
            self.merge(train,raw_val,[(task,'source0')],[(task,'source8')],name='mixed')


if __name__=='__main__': unittest.main()
