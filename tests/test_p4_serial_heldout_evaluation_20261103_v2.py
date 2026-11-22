"""Focused v2 transitive-identity/annotation contracts; no TEST materialization."""
import ast
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


api=imported('heldout_v2_contracts',ROOT/'scripts/run_p4_serial_heldout_evaluation_20261103_v2.py')
old=imported('heldout_v1_comparison',ROOT/'scripts/run_p4_serial_heldout_evaluation_20261103.py')
prep=imported('heldout_preparer_v2_contracts',ROOT/'scripts/prepare_p4_new_heldout_test_20261103_v2.py')
CKPT='artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
MODEL='artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/model_freeze_before_newdevVAL.json'


class V2Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.weight=self.root/CKPT;self.weight.parent.mkdir(parents=True)
        # Independent copies, never hardlinks to frozen production weights.
        shutil.copyfile(ROOT/CKPT,self.weight)
        self.put('source.py',{'synthetic_training_source':True})
        self.model=dict(schema='P4_model_freeze_before_newdevVAL_v1',training_complete=True,fit_split='train',
            new_devVAL_used_for_fit=False,formal_TEST_used=False,checkpoint=CKPT,
            files_sha256={CKPT:api.CHECKPOINT_SHA,'source.py':api.sha(self.root/'source.py')})
        self.put(MODEL,self.model)
        self.common={CKPT:api.CHECKPOINT_SHA,MODEL:api.sha(self.root/MODEL),'source.py':api.sha(self.root/'source.py')}
        self.jobs=[];self.configs=[]
        for i in range(8):
            job=dict(id=f'{i:02d}_val_static',parent_id=f'synthetic_parent_{i}',seed=261103100+i,
                split='val',scene='static',config=f'config/framework/student/{i}.json',path=f'{api.FAMILY}/student/{i}')
            cfg=dict(api.PROTOCOL,seed=job['seed'],dataset_use='val',candidate_transformer_checkpoint=CKPT,
                candidate_model_freeze_manifest=MODEL,gripper_velocity=.2,paired_student_v51_case=f'old_case/{i}')
            self.put(job['config'],cfg);self.jobs.append(job);self.configs.append(cfg)
            case=self.root/job['path'];case.mkdir(parents=True)
            self.put(job['path']+'.owner.json',dict(started_utc='2026-11-03T00:00:00+00:00',child_pid=123,
                argv=['--mode','p1','--scene','static','--config',job['config'],'--output',job['path']]))
            self.put(job['path']+'.owner.actual_exit.json',dict(started_utc='2026-11-03T00:00:00+00:00',
                finished_utc='2026-11-03T00:00:02+00:00',actual_process_returncode=0,output=str(case)))
            self.put(job['path']+'/worker_exit.json',dict(finished_utc='2026-11-03T00:00:01+00:00',exit_code=0,
                result_recorded=True,native_closed=True))
            self.put(job['path']+'/result.json',dict(complete=True,source_unchanged=True,Native_success=False))
            sources=dict(self.common,**{job['config']:api.sha(self.root/job['config'])})
            self.put(job['path']+'/run_inputs.json',dict(mode='p1',scene='static',config=cfg,source_sha256=sources,
                native_sources_sha256={'native.py':'a'*64},TEST_read=False,old_experiments_resumed=False,
                information_setting='public_candidate_transformer_v5_development'))
            for name in sources:
                target=case/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(self.root/name,target)
        self.val=self.put('config/framework/student_plan.json',dict(schema='explicit_v52_synthetic_reference',split='val',
            weights_frozen=True,jobs=self.jobs))  # Deliberately no redundant top checkpoint SHA.
        self.assigned=self.put('config/framework/assigned.json',dict(jobs=self.jobs))
        self.owner=f'{api.FAMILY}/student_owner'
        self.inputs=self.put(self.owner+'/inputs.json',dict(plan=str(self.val.relative_to(self.root)),plan_sha256=api.sha(self.val),runner_sha256='b'*64))
        self.plan=dict(preassigned_partitions=dict(devval=self.pointer(self.assigned)))
        self.freeze=dict(development_validation=dict(plan=self.pointer(self.val),plan_schema='explicit_v52_synthetic_reference',
            owner=self.owner,owner_inputs=self.pointer(self.inputs),runner_sha256='b'*64),common_controller_files_sha256=self.common,
            native_sources_sha256={'native.py':'a'*64},behavior_configuration=api.behavior_configuration(self.configs[0]))

    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path

    def pointer(self,path):return dict(path=str(path.relative_to(self.root)),sha256=api.sha(path))

    def refresh_plan(self,**updates):
        value=api.load(self.val);value.update(updates);self.val.write_text(json.dumps(value))
        self.freeze['development_validation']['plan']=self.pointer(self.val)
        inp=api.load(self.inputs);inp['plan_sha256']=api.sha(self.val);self.inputs.write_text(json.dumps(inp))
        self.freeze['development_validation']['owner_inputs']=self.pointer(self.inputs)

    def test_missing_top_field_requires_every_eight_complete_transitive_chains_no_writes(self):
        paths=[self.val,self.inputs,*[self.root/j['config']for j in self.jobs]]
        before={str(p):api.sha(p)for p in paths};calls=[];real=api.verify_case_checkpoint_chain
        def observed(root,job,*args):calls.append(job['id']);return real(root,job,*args)
        with mock.patch.object(api,'verify_case_checkpoint_chain',side_effect=observed), \
                mock.patch.object(api.subprocess,'Popen',side_effect=AssertionError('identity audit must not execute')):
            result=api.verify_devval(self.root,self.plan,self.freeze)
        self.assertTrue(result['complete']);self.assertEqual(result['actual_children_reaped'],8)
        self.assertFalse(result['top_level_checkpoint_SHA_present'])
        self.assertEqual(calls,[j['id']for j in self.jobs])
        self.assertTrue(all(r['model_identity']['config_run_inputs_archive_and_TRAIN_model_freeze_agree']for r in result['rows']))
        self.assertEqual(before,{str(p):api.sha(p)for p in paths})

    def test_present_top_field_must_match_even_when_all_leaf_evidence_is_correct(self):
        self.refresh_plan(checkpoint_sha256=api.CHECKPOINT_SHA)
        self.assertTrue(api.verify_devval(self.root,self.plan,self.freeze)['complete'])
        for bad in ('d'*64,None,True):
            self.refresh_plan(checkpoint_sha256=bad)
            with mock.patch.object(api,'audit_exit',side_effect=AssertionError('bad header must fail before case labels')):
                with self.assertRaisesRegex(RuntimeError,'top-level checkpoint SHA'):
                    api.verify_devval(self.root,self.plan,self.freeze)

    def test_one_missing_actual_exit_or_archived_model_leaf_cannot_pass(self):
        case=self.root/self.jobs[-1]['path'];actual=Path(str(case)+'.owner.actual_exit.json')
        saved=actual.read_bytes();actual.unlink()
        with self.assertRaises(FileNotFoundError):api.verify_devval(self.root,self.plan,self.freeze)
        actual.write_bytes(saved)
        (case/'source_snapshot'/CKPT).unlink()
        with self.assertRaises(FileNotFoundError):api.verify_devval(self.root,self.plan,self.freeze)

    def test_valid_hashes_do_not_substitute_for_TRAIN_only_completed_model_contract(self):
        model=deepcopy(self.model);model['fit_split']='val'
        self.put(MODEL,model);self.common[MODEL]=api.sha(self.root/MODEL)
        for job in self.jobs:
            case=self.root/job['path'];shutil.copyfile(self.root/MODEL,case/'source_snapshot'/MODEL)
            path=case/'run_inputs.json';inputs=api.load(path);inputs['source_sha256'][MODEL]=self.common[MODEL];path.write_text(json.dumps(inputs))
        with self.assertRaisesRegex(RuntimeError,'TRAIN-only model freeze'):
            api.verify_devval(self.root,self.plan,self.freeze)

    def test_only_named_pair_annotation_is_newly_excluded_real_behavior_remains_strict(self):
        newly_excluded=api.SCENARIO_AND_ANNOTATION_KEYS-old.SCENARIO_AND_ANNOTATION_KEYS
        self.assertEqual(newly_excluded,{'paired_student_v51_case'})
        a=deepcopy(self.configs[0]);b=deepcopy(a);b['paired_student_v51_case']='another_reference'
        self.assertEqual(api.behavior_configuration(a),api.behavior_configuration(b))
        for key,value in [('gripper_velocity',.3),('probe_terminal_at_fixed_deadline',True),('unreviewed_annotation','ignored?')]:
            changed=deepcopy(a);changed[key]=value
            self.assertNotEqual(api.behavior_configuration(a),api.behavior_configuration(changed))
        path=self.root/self.jobs[3]['config'];changed=api.load(path);changed['gripper_velocity']=.3;path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError,'different model/control'):
            api.verify_devval(self.root,self.plan,self.freeze)

    def test_source_equality_and_serial_STOP_actual_wait_protections_are_preserved(self):
        api.check_map(self.root,self.common)
        self.put('source.py',{'changed_working_tree':True})
        with self.assertRaisesRegex(RuntimeError,'frozen file changed'):api.check_map(self.root,self.common)
        for name in ('supervise_child','execute','requested_stop','propagate_stop','audit_exit'):
            self.assertEqual(ast.dump(ast.parse(inspect.getsource(getattr(api,name)))),
                             ast.dump(ast.parse(inspect.getsource(getattr(old,name)))))

    def test_partition_is_unchanged_and_preparer_has_no_reassignment_entry(self):
        path=ROOT/api.PARTITION_PATH
        self.assertEqual(api.sha(path),api.PARTITION_SHA)
        self.assertEqual(prep.fixed_jobs(),api.load(path)['jobs'])
        self.assertFalse(hasattr(prep,'assign_partition'))
        self.assertTrue(prep.RUNNER.endswith('_v2.py'));self.assertTrue(prep.SELF.endswith('_v2.py'))


if __name__=='__main__':unittest.main()
