"""CPU proofs for projection/closure/raw identity; never execute native policy."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from types import MappingProxyType
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from our_mani.multitask import baseline_component as b
from our_mani.multitask.evaluation import native_completion_record


def put(root, name, value):
    path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')
    return dict(path=name, sha256=b.file_sha(path))


def config():
    return dict(version='early', created_utc='2026-01-01T00:00:00+00:00', formal_comparison_ready=False,
        baseline_repo='/fixed/repo', baseline_checkpoint='/fixed/epoch_99999.pth', baseline_decision_ticks=10,
        baseline_executor='released_waypoints_counted_actuator_author_rgbd_terminal_v7', seed=0, dt=.05,
        simulation_seconds=10., gripper_velocity=.2, current_instance_masks=True, information_setting='shared',
        development_reset_profile={'path':'permission.json','sha256':'a'*64}, context_checkpoint=None,
        continuation_identities={}, initial_condition_tolerance_profile={'path':'frozen-profile.json','sha256':'c'*64},
        learned_artifacts={task:dict(geometry={'path':'CAD.json','sha256':'b'*64},
            interaction={'path':'library.json','sha256':'d'*64}, cv_value=None, context_value=None)
            for task in b.TASK_SPECS})


def camera_file(root):
    path = root/'source/data.h5'; path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for camera in ('front', '8', '16', '36'):
            for key in ('extrinsics', 'intrinsics', 'near', 'far'):
                f['cam_params/'+camera+'/'+key] = np.eye(4) if key=='extrinsics' else np.eye(3) if key=='intrinsics' else np.array(.1)
        f['unused/future_motion'] = np.arange(3)
    return dict(task=next(iter(b.TASK_SPECS)), raw_episode_id='episode0', path=str(path))


class ProjectionTests(unittest.TestCase):
    def setUp(self): self.early = config()

    def test_exact_four_ours_artifact_categories_can_change_but_raw_configs_differ(self):
        final = deepcopy(self.early); final.update(context_checkpoint={'path':'new.pt','sha256':'f'*64},
            continuation_identities={'new':'ours-only'}, version='final', formal_comparison_ready=True)
        for row in final['learned_artifacts'].values():
            row.update(cv_value={'path':'cv.pt','sha256':'e'*64}, context_value={'path':'ctx.pt','sha256':'d'*64})
        proof = b.compare_config_dependencies(self.early, final)
        self.assertNotEqual(proof['early_full_config_sha256'], proof['final_full_config_sha256'])
        self.assertFalse(proof['full_config_identities_claimed_equal'])
        self.assertNotIn('context_checkpoint', proof['projected_config'])
        self.assertTrue(all('cv_value' not in row and 'context_value' not in row
            for row in proof['projected_config']['learned_artifacts'].values()))

    def test_unknown_metadata_name_heuristics_and_physical_changes_are_rejected(self):
        changes = ({'new_training_metadata': True}, {'dt':.1}, {'seed':1}, {'baseline_executor':'different'},
            {'baseline_checkpoint':'other.pth'}, {'development_reset_profile':None},
            {'initial_condition_tolerance_profile':{'path':'relaxed.json','sha256':'e'*64}})
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,'execution config changed'):
                b.compare_config_dependencies(self.early, dict(self.early, **change))
        for field in ('geometry','unknown_unreviewed_artifact'):
            final=deepcopy(self.early); final['learned_artifacts'][next(iter(b.TASK_SPECS))][field]={'path':'other','sha256':'e'*64}
            with self.subTest(field=field), self.assertRaises(ValueError): b.compare_config_dependencies(self.early,final)

    def test_reviewed_ours_skill_colour_rigid_leaves_are_absent_not_merely_unread(self):
        early=deepcopy(self.early); final=deepcopy(self.early)
        for task in b.TASK_SPECS:
            early['learned_artifacts'][task].update(interaction={'poison':'must never reach baseline'},
                colour={'poison':'must never reach baseline'},rigid_layout={'poison':'must never reach baseline'})
            final['learned_artifacts'][task].update(interaction={'path':'later-skill.json'},
                colour={'path':'later-colour.json'},rigid_layout={'path':'later-layout.json'})
        proof=b.compare_config_dependencies(early,final)
        for row in proof['projected_config']['learned_artifacts'].values():
            self.assertTrue(set(b.OURS_LEAVES).isdisjoint(row));self.assertIn('geometry',row)

    def test_roster_exact275_and_task_denominators_no_subset_or_substitution(self):
        rows=[]
        for task in b.TASK_SPECS:
            n=18 if task in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed') else 19 if task=='insert_onto_rotating_peg_high_speed' else 20
            rows += [dict(task=task, raw_episode_id='episode'+str(i), path='/readonly/'+task+str(i)) for i in range(n)]
        manifest=dict(split='validation',episodes=rows)
        self.assertEqual(len(b.canonical_VAL_roster(manifest)),275)
        for changed in (dict(manifest,split='train'),dict(manifest,episodes=rows[:-1]),dict(manifest,episodes=rows[:-1]+[rows[0]])):
            with self.assertRaises(ValueError): b.canonical_VAL_roster(changed)


class BackendAndContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=b.ROOT/'runtime'); self.root=Path(self.temp.name)
        self.row=camera_file(self.root); self.config=config()
        ledger=put(self.root,'camera_ledger.json',dict(schema='r10_canonical_VAL_camera_dependency_ledger_v1',
            entries=[b.calibration_dependency(self.row)]))
        self.contract=put(self.root,'contract.json',dict(calibration_ledger=ledger,schema=b.CONTRACT,version=b.VERSION,
            approved_before_VAL_execution=True,projected_backend_required=True,
            projected_config_sha256=b.digest(b.project_baseline_config(self.config))))
    def tearDown(self): self.temp.cleanup()

    def prepared(self,split='validation'):
        return dict(args=SimpleNamespace(method='physmani',split=split,task=self.row['task'],replica=0),
            config=self.config,row=self.row)

    def test_camera_digest_binds_all_only_used_datasets_and_original_source_identity(self):
        binding=b.calibration_dependency(self.row)
        self.assertEqual(len(binding['camera_dataset_dependencies']),16)
        self.assertFalse(binding['future_motion_images_or_expert_arrays_read'])
        with h5py.File(self.row['path'],'r+') as source: source['cam_params/16/near'][...]=.2
        self.assertNotEqual(binding,b.calibration_dependency(self.row))

    def test_backend_passes_projected_config_to_simulator_and_policy_and_keeps_full_prepared(self):
        for row in self.config['learned_artifacts'].values():
            row.update(colour={'poison':True},rigid_layout={'poison':True})
        calls=[]; receipt=dict(complete=True,permission=self.config['development_reset_profile'],
            source_resampled=False,official_TEST_allowed=False)
        class Simulation:
            def __init__(_self,*args,**kwargs):
                calls.append(('environment',kwargs)); _self.reset_recovery_receipt=receipt
            def reset_development(_self,**kwargs): calls.append(('reset',kwargs)); return ['native'],None
            def close(_self): calls.append(('close',None))
        adapter=SimpleNamespace(predict=lambda *a: np.zeros(8),execute=lambda *a,**kw:'unchanged-execute')
        inner=SimpleNamespace(baseline=lambda p,*a:(calls.append(('baseline',p)) or adapter,'model'),
            initial_conditions=lambda *a:{'metric_only':True})
        backend=b.ProjectedBaselineBackend(execution_contract=self.contract,root=self.root,inner=inner,simulation_class=Simulation)
        p=self.prepared(); original=deepcopy(p['config']); env,desc=backend.environment(p)
        initial=backend.initial_conditions(env,None,None)
        with patch.object(backend,'_imports'):
            proxy,model=backend.baseline(p,env,None,None)
            self.assertEqual(proxy.execute(None),'unchanged-execute')
        self.assertEqual(p['config'],original); self.assertEqual(desc,['native'])
        self.assertNotIn('context_checkpoint',calls[0][1]['protocol'])
        for supplied in (calls[0][1]['protocol'],calls[2][1]['config']):
            self.assertTrue(all(set(b.OURS_LEAVES).isdisjoint(row) for row in supplied['learned_artifacts'].values()))
        self.assertNotIn('continuation_identities',calls[2][1]['config'])
        self.assertIs(initial['baseline_execution_dependency_witness'],backend.witness)
        self.assertEqual(backend.witness['shared_reset_receipt'],receipt)
        self.assertTrue(backend.witness['complete']); self.assertEqual(model,'model')

    def test_reset_failure_closes_once_and_official_TEST_delegates_original_archive(self):
        calls=[]
        class Bad:
            def __init__(self,*a,**kw): pass
            def reset_development(self,**kw): raise RuntimeError('guard rejected')
            def close(self): calls.append('close')
        inner=SimpleNamespace(environment=lambda p:('original-official','unchanged'))
        backend=b.ProjectedBaselineBackend(execution_contract=self.contract,root=self.root,inner=inner,simulation_class=Bad)
        with self.assertRaisesRegex(RuntimeError,'guard rejected'): backend.environment(self.prepared())
        self.assertEqual(calls,['close'])
        self.assertEqual(backend.environment(self.prepared('test')),('original-official','unchanged'))

    def test_unbound_late_import_is_rejected_instead_of_assumed_unused(self):
        put(self.root,'contract.json',dict(dependency_groups={'local_sources':{str(self.root/'allowed.py'):'a'*64}}))
        self.contract['sha256']=b.file_sha(self.root/'contract.json')
        unknown=self.root/'late.py'; unknown.write_text('# unexpected import\n')
        backend=b.ProjectedBaselineBackend(execution_contract=self.contract,root=self.root,inner=SimpleNamespace())
        backend.witness={'imported_dependency_files_sha256':{}}
        with patch.object(b.sys,'modules',{'late':SimpleNamespace(__file__=str(unknown))}), self.assertRaisesRegex(ValueError,'outside frozen closure'):
            backend._imports()

    def test_missing_closure_or_old_smoke_style_contract_is_never_promoted(self):
        lock={'files':{str(self.root/'contract.json'):self.contract['sha256']}}
        with self.assertRaisesRegex(ValueError,'pre-execution approved'):
            b.verify_execution_contract(self.root,self.contract,self.config,lock,[self.row])

    def test_real_contract_checks_package_inventory_archives_assets_and_lock_closure(self):
        def asset(name, data=b'frozen'):
            path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
            return str(path),b.file_sha(path)
        cfg=deepcopy(self.config); cfg['baseline_repo']=str(self.root/'repo')
        permission=put(self.root,'physical_permission.json',{'explicit':'CPU fixture'})
        cfg['development_reset_profile']=permission
        local=dict(asset(name,b'# frozen config reader\n') for name in b.REVIEW_SOURCE_FILES)
        entrypoints={}
        for field,name in (('parent_source','queue.py'),('child_source','projected_child.py'),('launcher_source','scripts/run_aligned_gpu.sh')):
            name_,checksum=asset(name,b'# reviewed execution entrypoint\n');local[name_]=checksum
            entrypoints[field]=dict(path=name,sha256=checksum)
        policies={}; inventories=[]
        for package in ('physmani_policy','physmani_wm','physmani_bench'):
            pair=asset('repo/packages/'+package+'/'+package+'/__init__.py',b'# native package\n')
            policies[pair[0]]=pair[1]
            inventories.append(dict(root=str(Path(pair[0]).parent),files_sha256=dict([pair])))
        checkpoint=asset('epoch_99999.pth');cfg['baseline_checkpoint']=checkpoint[0]
        models=dict([checkpoint,asset('runtime/framework/clip/RN50.pt'),
            asset('repo/tasks/18_peract_tasks_location_bounds.json'),
            asset('repo/instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl')])
        native=dict([asset('repo/third_party/RLBench/rlbench/task_design.ttt')]+
            [asset('repo/third_party/RLBench/rlbench/task_ttms/'+task+'.ttm') for task in b.TASK_SPECS])
        runtime_bin=dict([asset('native_runtime.so')])
        ledger=put(self.root,'full_camera_ledger.json',dict(schema='r10_canonical_VAL_camera_dependency_ledger_v1',
            entries=[b.calibration_dependency(self.row)]))
        groups=dict(local_sources=local,policy_and_WM_sources=policies,model_assets=models,
            native_sources_and_scenes=native,runtime_and_extension_assets=runtime_bin,
            calibration_inputs={str(self.root/ledger['path']):ledger['sha256'],str(self.root/permission['path']):permission['sha256']})
        runtime=put(self.root,'runtime_inventory.json',dict(schema='r10_PhysMani_actual_runtime_dependency_inventory_v1',
            complete=True,CUDA_and_native_extensions_bound=True,files_sha256=runtime_bin))
        review=put(self.root,'config_access_review.json',dict(schema='r10_PhysMani_config_access_code_review_v1',approved=True,
            branch='physmani_TRAIN_VAL_only',baseline_read_keys=list(b.BASELINE_READ_KEYS),excluded_roots=list(b.OURS_ROOTS),
            excluded_artifact_leaves=list(b.OURS_LEAVES),excluded_metadata=list(b.METADATA),
            sources_sha256={str(self.root/name):local[str(self.root/name)] for name in b.REVIEW_SOURCE_FILES},execution_entrypoints=entrypoints))
        archived=[]
        for i,(name,checksum) in enumerate({**local,**policies}.items()):
            target=self.root/('archive/source'+str(i)+'.py');target.parent.mkdir(exist_ok=True);target.write_bytes(Path(name).read_bytes())
            archived.append(dict(original_path=name,path=str(target.relative_to(self.root)),sha256=checksum))
        archive=put(self.root,'source_archive.json',dict(schema=b.SOURCE_ARCHIVE,all_inputs_verified_before_and_after=True,entries=archived))
        contract=put(self.root,'full_contract.json',dict(schema=b.CONTRACT,version=b.VERSION,
            approved_before_VAL_execution=True,projected_backend_required=True,actual_dependency_closure_reviewed=True,
            projected_config_sha256=b.digest(b.project_baseline_config(cfg)),canonical_VAL_roster_sha256=b.digest([self.row]),
            shared_development_reset_profile=permission,dependency_groups=groups,calibration_ledger=ledger,
            config_access_review=review,execution_entrypoints=entrypoints,complete_source_root_inventories=inventories,runtime_inventory=runtime,source_archive=archive))
        files={k:v for values in groups.values() for k,v in values.items()}
        files.update({str(self.root/p['path']):p['sha256'] for p in (contract,review,runtime,archive)})
        actual=b.verify_execution_contract(self.root,contract,cfg,{'files':files},[self.row])
        self.assertEqual(set(actual['dependencies']),set(k for group in groups.values() for k in group))
        cached=b._VERIFIED_DEPENDENCIES[(str(self.root.resolve()),contract['sha256'])]
        self.assertIsInstance(cached,MappingProxyType)
        with self.assertRaises(TypeError):cached['unexpected']='a'*64
        with patch.object(b,'normalized',side_effect=AssertionError('must reuse verified map')):
            self.assertIs(b.observer_dependencies(self.root,contract),cached)
            self.assertIs(b.observer_dependencies(self.root,contract),cached)
        original=(self.root/contract['path']).read_bytes()
        (self.root/contract['path']).write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'immutable project pointer changed'):
            b.observer_dependencies(self.root,contract)
        (self.root/contract['path']).write_bytes(original)
        # An empty guessed source glob or a new late package file is not a
        # certificate that all baseline policy/WM code was frozen.
        extra=self.root/'repo/packages/physmani_wm/physmani_wm/late.py';extra.write_text('# unbound late source\n')
        with self.assertRaisesRegex(ValueError,'inventory'):
            b.verify_execution_contract(self.root,contract,cfg,{'files':files},[self.row])
        extra.unlink()
        (self.root/archived[0]['path']).write_text('# historical source tampered\n')
        with self.assertRaisesRegex(ValueError,'immutable project pointer changed'):
            b.verify_execution_contract(self.root,contract,cfg,{'files':files},[self.row])


class ComponentRawChainTests(unittest.TestCase):
    """Consumer boundary has independent real projection/closure tests above.

    Use one generated complete275 raw ledger and mock only the separately
    audited closure to exercise every original case identity and receipt.
    """
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=b.ROOT/'runtime'); self.root=Path(self.temp.name)
        self.early=config(); self.final=deepcopy(self.early); self.final['context_checkpoint']={'path':'later.pt','sha256':'f'*64}
        self.roster=[]
        for task in b.TASK_SPECS:
            n=18 if task in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed') else 19 if task=='insert_onto_rotating_peg_high_speed' else 20
            self.roster += [dict(task=task,raw_episode_id='episode'+str(i),path='/source/'+task+'/'+str(i)) for i in range(n)]
        roster=put(self.root,'roster.json',dict(split='validation',episodes=self.roster))
        ec=put(self.root,'early_config.json',self.early); el=put(self.root,'early_lock.json',{'files':{'dummy':'a'*64}})
        self.final_config=put(self.root,'final_config.json',self.final)
        deps={}
        for name in ('queue.py','child.py','our_mani/multitask/baseline_component.py','epoch_99999.pth','python-executable'):
            path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('frozen source/data\n'); deps[str(path)]=b.file_sha(path)
        self.early['baseline_checkpoint']=str(self.root/'epoch_99999.pth'); self.final['baseline_checkpoint']=self.early['baseline_checkpoint']
        ec=put(self.root,'early_config.json',self.early); self.final_config=put(self.root,'final_config.json',self.final)
        self.final_lock=put(self.root,'final_lock.json',{'files':deps})
        self.contract=put(self.root,'execution_contract.json',{'frozen_utc':'2026-01-01T00:00:00+00:00'})
        early_files=dict(deps,**{str(self.root/ec['path']):ec['sha256'],str(self.root/roster['path']):roster['sha256']})
        el=put(self.root,'early_lock.json',{'files':early_files})
        plan=dict(version='r10_fresh_policy_queue_v1',created_utc='2026-01-02T00:00:00+00:00',split='validation',
            methods=['physmani'],replicas=1,source_episodes=275,expected_cases=275,config=ec,manifest=roster,frozen_inputs=el,
            parent_source=dict(path='queue.py',sha256=deps[str(self.root/'queue.py')]),
            child_source=dict(path='child.py',sha256=deps[str(self.root/'child.py')]),
            output='artifacts/framework/multitask_r10/early',jobs=[],official_TEST_used_for_fitting=False,
            case_isolation='fresh_simulator_process_per_episode')
        for row in self.roster:
            identity=b.case_identity(split='validation',task=row['task'],raw_episode_id=row['raw_episode_id']+'__policy_aug_r0',
                method='physmani',source_row=row,config_sha256=ec['sha256'],inputs_sha256=el['sha256'],
                parent_raw_episode_id=row['raw_episode_id'],replica=0)
            plan['jobs'].append(dict(job_id=identity['sha256'][:20],identity=identity,row=row,source_raw_episode_id=row['raw_episode_id'],
                replica=0,exploration_seed=None,record_demo=False))
        plan['sha256']=b.digest(plan); self.plan=put(self.root,'plan.json',plan); self.plan_value=plan
        self.paths=[]
        for job in plan['jobs']:
            row=job['row']; cid=job['job_id']; child_name=plan['output']+'/cases/'+cid+'/attempt0/case.json'
            reset=dict(complete=True,permission=self.early['development_reset_profile'],source_resampled=False,official_TEST_allowed=False)
            witness=dict(version=b.VERSION,execution_contract=self.contract,
                projected_config_sha256=b.digest(b.project_baseline_config(self.early)),full_config_identity_preserved=True,
                projection_helper_sha256=deps[str(self.root/'our_mani/multitask/baseline_component.py')],
                calibration_binding_sha256=b.digest({'fixture_row':row}),split='validation',method='physmani',complete=True,
                actual_python_executable=str(self.root/'python-executable'),
                loaded_native_files_sha256={str(self.root/'python-executable'):deps[str(self.root/'python-executable')]},
                shared_reset_receipt=reset,imported_dependency_files_sha256={str(self.root/'child.py'):deps[str(self.root/'child.py')]})
            record=native_completion_record(job['identity'],process_id=123,success=False,fatal=False,ticks=200,
                initial_conditions=dict(metric_only=True,policy_use_allowed=False,reset_receipt={'reset_recovery_receipt':reset},
                    baseline_execution_dependency_witness=witness),alignment_receipt=dict(clock=dict(ticks=200,dt=.05,budget_seconds=10.,simulation_seconds=10.),
                        actual_physics_seconds=10.,actual_task_seconds=10.,native_success_latched=False,native_terminal_failure_latched=False))
            record.update(video=None,cleanup_completed=True,baseline_executor=self.early['baseline_executor'],
                compute_seconds=100.,wall_seconds=150.,simulation_seconds=10.,evaluation_process_pid=123,
                source_row_sha256=job['identity']['source_row_sha256'],
                checkpoint_sha256=deps[str(self.root/'epoch_99999.pth')],config_sha256=ec['sha256'],manifest_sha256=roster['sha256'],
                frozen_input_lock=dict(path=str(self.root/el['path']),sha256=el['sha256']),entrypoint_sha256=plan['child_source']['sha256'])
            child=put(self.root,child_name,record); log=self.root/(plan['output']+'/cases/'+cid+'/native.log'); log.write_text('actual CPU fixture\n')
            command=['bash','scripts/run_aligned_gpu.sh','child.py','--config',ec['path'],'--manifest',roster['path'],
                '--frozen-inputs',el['path'],'--split','validation','--task',row['task'],'--raw-episode-id',row['raw_episode_id'],
                '--method','physmani','--replica','0','--out',str(Path(child_name).parent)]
            record['parent_case_process_receipt']=dict(child_path=child_name,child_sha256=child['sha256'],
                child_process_id=123,launcher_process_id=321,plan_sha256=plan['sha256'],parent_source_sha256=plan['parent_source']['sha256'],
                command=command,log_path=str(log.relative_to(self.root)),log_sha256=b.file_sha(log))
            self.paths.append(self.root/(plan['output']+'/results/'+cid+'.json'))
            put(self.root,str(self.paths[-1].relative_to(self.root)),record)
        put(self.root,plan['output']+'/status.json',dict(complete=True,plan_sha256=plan['sha256'],completed=275,expected=275,errors=[],stop_requested=False))
        self.roster_pointer=roster
        self.closure=dict(contract={'frozen_utc':'2026-01-01T00:00:00+00:00','execution_entrypoints':dict(
            parent_source=plan['parent_source'],child_source=plan['child_source'],launcher_source={'path':'scripts/run_aligned_gpu.sh','sha256':'e'*64})},
            dependencies=deps,mutable_state_permission=None,fingerprint={'sealed':'CPU fixture'})
    def tearDown(self):self.temp.cleanup()
    def run_audit(self):
        with patch.object(b,'verify_execution_contract',return_value=self.closure), \
             patch.object(b,'calibration_dependency',side_effect=lambda row:{'fixture_row':row}):
            return b.validate_baseline_component(self.root,self.plan,self.contract,self.final_config,self.final_lock,self.roster_pointer)

    def test_complete275_keeps_early_identity_and_never_grants_test_or_full_release(self):
        result=self.run_audit(); self.assertEqual(result['sources'],275)
        self.assertTrue(result['baseline_component_reusable']); self.assertFalse(result['formal_TEST_permitted'])
        self.assertFalse(result['formal_validation_release_issued']); self.assertFalse(result['raw_identity_rewritten'])
        self.assertEqual(result['actual_early_config'],self.plan_value['config']); self.assertEqual(result['requires_complete_five_method_VAL_cases'],1375)

    def test_raw_tamper_missing_witness_wrong_projection_or_demo_cannot_reuse(self):
        path=self.paths[0]; original=path.read_text()
        for mutation in ('raw','witness','projection','demo'):
            record=json.loads(original)
            if mutation=='raw':record['official_task_success']=True
            elif mutation=='witness':record['recorded_initial_conditions'].pop('baseline_execution_dependency_witness')
            elif mutation=='projection':record['recorded_initial_conditions']['baseline_execution_dependency_witness']['projected_config_sha256']='f'*64
            else:record['video']={'fake':'demo'}
            put(self.root,str(path.relative_to(self.root)),record)
            with self.subTest(mutation=mutation),self.assertRaises(Exception):self.run_audit()
            path.write_text(original)

    def test_missing_case_and_changed_final_source_or_unknown_config_refused(self):
        path=self.paths[0]; text=path.read_text(); path.unlink()
        with self.assertRaises(ValueError):self.run_audit()
        path.write_text(text)
        (self.root/'child.py').write_text('changed executable\n')
        with self.assertRaisesRegex(ValueError,'closure changed'):self.run_audit()
        (self.root/'child.py').write_text('frozen source/data\n')
        final=dict(self.final,unknown_new_feature=True); self.final_config=put(self.root,'final_config.json',final)
        with self.assertRaisesRegex(ValueError,'config changed'):self.run_audit()


    def test_plan_TEST_isolation_and_exact_source_job_are_mandatory(self):
        original=deepcopy(self.plan_value)
        for kind in ('TEST','isolation','job_source'):
            changed=deepcopy(original)
            if kind=='TEST':changed['official_TEST_used_for_fitting']=True
            elif kind=='isolation':changed['case_isolation']='reused_native_environment'
            else:changed['jobs'][0]['source_raw_episode_id']='different'
            changed.pop('sha256');changed['sha256']=b.digest(changed);self.plan=put(self.root,'plan.json',changed)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.run_audit()
        self.plan=put(self.root,'plan.json',original)

    def test_parent_wrapper_PID_unknown_flags_and_noncase_raw_basename_are_rejected(self):
        path=self.paths[0];original=json.loads(path.read_text())
        for kind in ('PID','flag','raw_basename'):
            changed=deepcopy(original);parent=changed['parent_case_process_receipt']
            if kind=='PID':parent['launcher_process_id']=0
            elif kind=='flag':parent['command']+=['--unreviewed-flag','enabled']
            else:
                old=self.root/parent['child_path'];new=old.with_name('not_case.json');new.write_bytes(old.read_bytes())
                parent['child_path']=str(new.relative_to(self.root));parent['child_sha256']=b.file_sha(new)
            put(self.root,str(path.relative_to(self.root)),changed)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.run_audit()
        put(self.root,str(path.relative_to(self.root)),original)


if __name__ == '__main__': unittest.main()
