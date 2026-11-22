import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p4devval',ROOT/'scripts/collect_p4_devval_teacher_20261103.py')
val=importlib.util.module_from_spec(spec);spec.loader.exec_module(val)
PLAN=ROOT/'config/framework/P4_teacher_devVAL_20261103_v1/plan.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class DevVALTeacherTests(unittest.TestCase):
    def test_materialized_plan_is_compiler_compatible_and_never_spawns_in_preflight(self):
        compiler_spec=importlib.util.spec_from_file_location('p4compiler',ROOT/'scripts/compile_p4_public_dataset_20261103.py')
        compiler=importlib.util.module_from_spec(compiler_spec);compiler_spec.loader.exec_module(compiler)
        plan,_,configs=val.validate_plan(PLAN)
        self.assertEqual(len(configs),8)
        for j in plan['jobs']:
            _,_,cfg=compiler.validate_assignment(j,'val')
            self.assertEqual(cfg['P4_collection_metadata']['split'],'val')
        with patch.object(val.subprocess,'Popen',side_effect=AssertionError('Native forbidden')):
            result=val.run(PLAN)
        self.assertFalse(result['Native_started'])
        self.assertTrue(result['TRAIN_parent_seed_and_initial_XY_disjoint'])

    def test_reused_TRAIN_parent_or_seed_is_rejected_before_any_case_read(self):
        plan=json.loads(PLAN.read_text());train=json.loads((ROOT/plan['TRAIN_plan']['path']).read_text())
        for field in ('parent_id','seed'):
            mutated=json.loads(json.dumps(plan));mutated['jobs'][0][field]=train['jobs'][0][field]
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'plan.json';write(path,mutated)
                with self.assertRaises(ValueError):val.validate_plan(path)

    def test_actual_active_visual_v42_worker_flag_is_rejected(self):
        plan=json.loads(PLAN.read_text());config=(ROOT/plan['jobs'][0]['config']).resolve()
        actual_read=Path.read_text
        def read_with_wrong_mode(path,*args,**kwargs):
            text=actual_read(path,*args,**kwargs)
            if path.resolve()==config:
                value=json.loads(text);value['public_grasp_active_visual_v42']=True
                return json.dumps(value)
            return text
        # Only parser input is injected; real config bytes/SHA stay untouched.
        with patch.object(Path,'read_text',read_with_wrong_mode),self.assertRaisesRegex(ValueError,'no public controller'):
            val.validate_plan(PLAN)

    def test_actual_execution_requires_model_manifest_before_any_spawn(self):
        with patch.object(val.subprocess,'Popen',side_effect=AssertionError('Native forbidden')):
            with self.assertRaises(ValueError):val.run(PLAN,execute=True,owner_path='unused')

    def test_model_freeze_is_TRAIN_only_and_detects_parameter_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);weight=root/'artifact/model.pt';weight.parent.mkdir();weight.write_bytes(b'frozenCPUfixture')
            manifest=dict(schema='P4_model_freeze_before_newdevVAL_v1',training_complete=True,fit_split='train',
                new_devVAL_used_for_fit=False,formal_TEST_used=False,checkpoint='artifact/model.pt',
                files_sha256={'artifact/model.pt':hashlib.sha256(weight.read_bytes()).hexdigest()})
            path=root/'model_freeze.json';write(path,manifest)
            with patch.object(val.shared,'ROOT',root):
                evidence,_=val.verify_model_freeze(path);self.assertTrue(evidence['model_is_frozen_but_not_used_by_truth_teacher'])
                manifest['new_devVAL_used_for_fit']=True;write(path,manifest)
                with self.assertRaises(ValueError):val.verify_model_freeze(path)
                manifest['new_devVAL_used_for_fit']=False;write(path,manifest);weight.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):val.verify_model_freeze(path)
                manifest['files_sha256']={'artifact/../model.pt':'x'};manifest['checkpoint']='artifact/../model.pt';write(path,manifest)
                with self.assertRaises(ValueError):val.verify_model_freeze(path)

    def test_all_sixteen_actual_TRAIN_exits_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);family=root/'runtime/framework/local_mpc_20261102'
            jobs=[dict(episode_id=str(k),parent_id='p'+str(k),split='train',path=f'runtime/framework/local_mpc_20261102/train/{k}') for k in range(16)]
            plan_path=root/'config/train.json';write(plan_path,dict(jobs=jobs))
            owner=family/'owner';write(owner/'result.json',dict(complete=True,episodes=16,results=jobs))
            for j in jobs:
                case=root/j['path'];write(case.parent/(case.name+'.owner.actual_exit.json'),dict(actual_process_returncode=0))
                write(case/'worker_exit.json',dict(exit_code=0))
            plan=dict(TRAIN_plan=dict(path='config/train.json',owner='runtime/framework/local_mpc_20261102/owner'))
            with patch.object(val,'ROOT',root),patch.object(val,'FAMILY',family),patch.object(val.shared,'ROOT',root):
                self.assertTrue(val.verify_TRAIN_reaped(plan)['all_16_actual_children_reaped'])
                write(root/jobs[-1]['path']/'worker_exit.json',dict(exit_code=1))
                with self.assertRaises(RuntimeError):val.verify_TRAIN_reaped(plan)

    def test_explicit_VAL_audit_never_calls_hardcoded_TRAIN_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);family=root/'runtime/framework/local_mpc_20261102';case=family/'case'
            config=dict(dataset_use='val',task='pick_moving_target_on_the_table',native_task_source_sha256='taskSHA',native_task_scene_sha256='sceneSHA')
            job=dict(split='val',scene='static',parent_id='valParent',episode_id='val0',path=str(case.relative_to(root)),
                evaluation='artifacts/framework/local_mpc_20261102/eval',frozen_GRU_checkpoint_sha256='GRU')
            write(case.parent/(case.name+'.owner.actual_exit.json'),dict(actual_process_returncode=0))
            write(case/'worker_exit.json',dict(exit_code=0));write(case/'result.json',dict(complete=True,source_unchanged=True))
            native={'third_party/RLBench/rlbench/tasks/'+config['task']+'.py':'taskSHA',
                    'third_party/RLBench/rlbench/task_ttms/'+config['task']+'.ttm':'sceneSHA'}
            write(case/'run_inputs.json',dict(config=config,TEST_read=False,scene='static',source_sha256={'x':'y'},native_sources_sha256=native))
            write(case/'initial_snapshot.json',dict(native_height_preserved=True,state=dict(object_position=[0,0,.775]),native_default_target_pose_before_override=[0,0,.775]))
            (case/'native_ticks.jsonl').write_text('')
            with patch.object(val,'ROOT',root),patch.object(val,'FAMILY',family),patch.object(val.shared,'ROOT',root), \
                 patch.object(val.shared,'audit_completed_job',side_effect=AssertionError('hardcodedTRAIN audit forbidden')), \
                 patch.object(val.shared,'validate_archived_sources',return_value=dict(verified=True)), \
                 patch.object(val.shared,'audit_public_capture',return_value=(dict(exactly_50ms=True),{})), \
                 patch.object(val.shared,'contact_labels',return_value=[]), \
                 patch('our_mani.local_mpc_control_20261103.native_task_evaluation.write_case_evaluation',return_value=dict(metrics={})), \
                 patch.object(val.shared,'sha',return_value='fixtureSHA'):
                result=val.audit_completed_job(job,config,expected_split='val')
                self.assertEqual(result['split'],'val');self.assertTrue(result['new_development_VAL_only'])
                self.assertFalse(result['official_VAL_or_TEST_read'])
                with self.assertRaises(ValueError):val.audit_completed_job(job,config,expected_split='train')


if __name__=='__main__':unittest.main()
