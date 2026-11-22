"""Full original275 VAL plans; CPU mocked execution only, no native launch."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_multitask_recovery_runner import archive, put
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('VAL_initialization_runner',ROOT/'scripts/collect_multitask_validation_initialization_r10.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=r.collection
from our_mani.multitask import development_reset_recovery as physical


def fixture(root):
    base_path=ROOT/'artifacts/framework/multitask_r10/VAL_hold_collection_plan_v11.json'
    original=r.read(base_path)
    closure=r.read(ROOT/'artifacts/framework/multitask_r10/TRAIN_recovery_v12_initial_pair_setup/plan_bound.json')
    # Source-plan fixture is metadata, never a purported executed TRAIN plan.
    # It binds the current helper version without changing any frozen artifact.
    for name in closure['inputs']:
        if Path(name).suffix in ('.py','.sh'):closure['inputs'][name]=c.sha(name)
    physical_ptr=r.pointer(ROOT/'artifacts/framework/multitask_r10/development_physical_reset_permission_v3.json')
    RML_ptr=r.pointer(ROOT/'artifacts/framework/multitask_r10/RML_factory_hook_permission_v1.json')
    closure['protocol']['development_reset_profile']=physical_ptr
    closure['protocol']['RML_factory_hook']=RML_ptr
    policy=r.read(r.checked(physical_ptr))
    for name,checksum in policy['native_inputs_sha256'].items():
        closure['inputs'][str((Path(closure['protocol']['baseline_repo'])/name).resolve())]=checksum
    closure['protocol_path']=str(root/'closure_protocol.json');put(root/'closure_protocol.json',closure['protocol'])
    closure['inputs'][str(root/'closure_protocol.json')]=c.sha(root/'closure_protocol.json')
    closure.pop('plan_sha256');closure['plan_sha256']=c.canonical_sha(closure);put(root/'closure_plan.json',closure)
    new=r.prepare_plan(base_path,root/'closure_plan.json',physical_permission=physical_ptr,rml_permission=RML_ptr,
        protocol_output=root/'protocol.json',plan_output=root/'plan.json')
    execution=archive(root/'archive',root/'plan.json',root/'collection')
    train=ROOT/'artifacts/framework/protocol/train_manifest.json'
    val=ROOT/'artifacts/framework/protocol/validation_manifest.json'
    registry=dict(schema=r.REGISTRATION_SCHEMA,original_plan=r.pointer(base_path),original_collection=str((root/'original_unexecuted').relative_to(ROOT)),
        source_closure_plan=r.pointer(root/'closure_plan.json'),execution=execution,
        original_TRAIN_parents=r.pointer(train),original_VAL_parents=r.pointer(val))
    put(root/'registry.json',registry)
    return root/'plan.json',root/'registry.json',root/'collection',original,new


class VALInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=c.BASE);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_actual_full275_roster_profile_closure_and_native_dependencies_no_launch(self):
        with patch.object(r.subprocess,'Popen',side_effect=AssertionError('native forbidden')):
            plan,registry,output,old,new=fixture(self.root)
            context=r.verify_execution(plan,registry,output=output)
            from our_mani.multitask.validation_lineage import verify_validation_execution_registration
            independently_checked=verify_validation_execution_registration(ROOT,r.pointer(registry))
        self.assertEqual(new['cases'],old['cases']);self.assertEqual(len(new['cases']),275)
        self.assertEqual(new['augmentation_budget'],old['augmentation_budget'])
        self.assertEqual(set(new['protocol'])-set(old['protocol']),set(r.HOOK_KEYS))
        self.assertNotIn('recovery_input',new['protocol'])
        policy=physical.load_permission(new['protocol']['development_reset_profile'],native_repo=new['protocol']['baseline_repo'])
        for name,checksum in policy['native_inputs_sha256'].items():
            self.assertEqual(new['inputs'][str((Path(new['protocol']['baseline_repo'])/name).resolve())],checksum)

    def test_changed_case_teacher_seed_geometry_or_fake_recovery_rejected(self):
        original=r.read(ROOT/'artifacts/framework/multitask_r10/VAL_hold_collection_plan_v11.json')
        for kind in ('missing','teacher','ticks','seed','variation','geometry','recovery'):
            bad=deepcopy(original)
            bad['protocol'].update(development_reset_profile={'fixture':True},RML_factory_hook={'fixture':True})
            if kind=='missing':bad['cases'].pop()
            elif kind=='teacher':bad['cases'][0]['teacher_used']=True
            elif kind=='ticks':bad['cases'][0]['maximum_collection_ticks']=200
            elif kind=='seed':bad['protocol']['seed']+=1
            elif kind=='variation':bad['cases'][0]['source']['variation']+=1
            elif kind=='geometry':bad['cases'][0]['template_asset']=None
            else:bad['protocol']['recovery_input']={'invented':True}
            with self.subTest(kind=kind),self.assertRaises(ValueError):r.validate_transition(original,bad)

    def test_existing_old_attempt_or_completion_and_foreign_namespace_rejected(self):
        plan,registry,output,old,new=fixture(self.root)
        old_output=self.root/'original_unexecuted'
        (old_output/'infrastructure_attempts').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'previous native'):r.verify_execution(plan,registry)
        (old_output/'infrastructure_attempts').rmdir()
        canonical=c.case_directory(old_output,old['cases'][0])/'case.json';put(canonical,{'status':'completed'})
        with self.assertRaisesRegex(ValueError,'already has completions'):r.verify_execution(plan,registry)
        canonical.unlink()
        with self.assertRaisesRegex(ValueError,'escaped'):r.verify_execution(plan,registry,output=self.root/'foreign')

    def hooks(self,plan,captures):
        @contextmanager
        def scope():
            self.captures=captures
            yield
        return scope()

    def fake_collector(self,plan_path,case_id,output,guard_report):
        plan=r.read(plan_path);case=next(row for row in plan['cases'] if row['case_id']==case_id)
        reset=dict(version=physical.VERSION,permission=deepcopy(plan['protocol']['development_reset_profile']),complete=True,
            task=case['task'],split='validation',source=deepcopy(case['source']),replica=case['replica'],
            source_resampled=False,official_TEST_allowed=False,**r.lineage.expected_reset(case,plan['protocol']))
        self.captures['simulations'].append(SimpleNamespace(reset_recovery_receipt=reset))
        out=c.case_directory(output,case);put(out/'TRAIN_geometry.json',r.read(case['template_asset']))
        entry={key:case[key] for key in ('behavior','behavior_replica','maximum_collection_ticks','teacher_used',
            'context_primary_view','interaction_eligible','split','task','raw_episode_id','parent_raw_episode_id')}
        entry['source_kind']='VAL_native_generated_non_teacher'
        record=dict(case_id=case_id,plan_sha256=plan['plan_sha256'],status='completed',case_isolation='fresh_simulator_process_per_episode',
            process_id=os.getpid(),entry=entry,alignment=dict(clock=dict(ticks=0)),
            files_sha256={str((out/'TRAIN_geometry.json').relative_to(ROOT)):c.sha(out/'TRAIN_geometry.json')},
            teacher_actions_used=False,teacher_paths_requested=False,teacher_action_rows=0,policy_history_contains_native_GT=False,
            native_reset_receipt=dict(reset_recovery_receipt=deepcopy(reset)),metric_fixture_valid=False)
        c.write_json(out/'case.json',record);return record

    def test_child_same_seed_embedded_reset_raw_base_and_no_teacher_witness(self):
        plan,registry,output,old,new=fixture(self.root);case=old['cases'][0]
        attempt=output/'infrastructure_attempts'/case['case_id']/'0/output'
        with patch.object(r.owned,'scoped_owned_hooks',side_effect=self.hooks),patch.object(c,'collect_native_case',side_effect=self.fake_collector):
            record=r.collect_case(plan,case['case_id'],attempt,self.root/'guard',registry)
        out=c.case_directory(attempt,case);hooks=r.read(out/'development_initialization_hook_receipt.json')
        self.assertEqual(hooks['RML_path_recovery_receipts'],[]);self.assertFalse(hooks['teacher_actions_used'])
        self.assertNotIn('failed_parent_receipts',hooks);self.assertNotIn('recovery_execution',record)
        self.assertEqual(record['native_reset_receipt']['reset_recovery_receipt'],hooks['reset_recovery_receipts'][0])
        self.assertEqual(r.read(out/'base_collector_receipt.json')['metric_fixture_valid'],False)
        bad=deepcopy(record);bad['native_reset_receipt']['reset_recovery_receipt']['seed']+=1;put(out/'bad.json',bad)
        with self.assertRaisesRegex(ValueError,'same-source reset'):r.validate_case(out/'bad.json',r.verify_execution(plan,registry),case)

    def test_unexpected_RML_path_prevents_completion_and_keeps_raw_witness(self):
        plan,registry,output,old,new=fixture(self.root);case=old['cases'][0]
        attempt=output/'infrastructure_attempts'/case['case_id']/'0/output'
        def unexpected(*args):
            record=self.fake_collector(*args)
            self.captures['paths'].append(SimpleNamespace(recovery_receipt={'unexpected_teacher_path':True}))
            return record
        with patch.object(r.owned,'scoped_owned_hooks',side_effect=self.hooks),patch.object(c,'collect_native_case',side_effect=unexpected):
            with self.assertRaisesRegex(ValueError,'zero teacher/RML'):r.collect_case(plan,case['case_id'],attempt,self.root/'guard',registry)
        hooks=r.read(c.case_directory(attempt,case)/'development_initialization_hook_receipt.json')
        self.assertFalse(hooks['complete']);self.assertEqual(len(hooks['RML_path_recovery_receipts']),1)
        self.assertFalse((c.case_directory(output,case)/'case.json').exists())

    def test_fixed_subset_unknown_duplicate_or_existing_attempt_blocks_before_launch(self):
        plan,registry,output,old,new=fixture(self.root);case=old['cases'][0]
        for ids in ([],['unknown'],[case['case_id']]*2):
            with self.subTest(ids=ids),self.assertRaises(ValueError):r.verify_execution(plan,registry,case_ids=ids)
        (output/'infrastructure_attempts'/case['case_id']/'0').mkdir(parents=True)
        with patch.object(r.subprocess,'Popen',side_effect=AssertionError('native forbidden')),patch.object(c,'require_r09_release',return_value={}):
            with self.assertRaisesRegex(ValueError,'no automatic VAL retry'):
                r.run_queue(plan,output,self.root/'guard',registry,case_ids=[case['case_id']],workers=1)

    def test_mocked_own_parent_command_manifest_pending275_and_readonly_resume(self):
        plan,registry,output,old,new=fixture(self.root);case=old['cases'][0];calls=[]
        class Process:
            pid=os.getpid()+100000
            def __init__(self,command,**kwargs):
                calls.append(command)
                def arg(flag):return command[command.index(flag)+1]
                r.collect_case(arg('--plan'),arg('--case-id'),arg('--output'),arg('--guard-report'),arg('--execution-registration'))
            def wait(self,timeout):return 0
        with patch.object(r.owned,'scoped_owned_hooks',side_effect=self.hooks),patch.object(c,'collect_native_case',side_effect=self.fake_collector),\
             patch.object(r.subprocess,'Popen',Process),patch.object(c,'require_r09_release',return_value={}):
            merged=r.run_queue(plan,output,self.root/'guard',registry,case_ids=[case['case_id']],workers=1)
            canonical=c.case_directory(output,case)/'case.json';before=canonical.read_bytes()
            r.run_queue(plan,output,self.root/'guard',registry,case_ids=[case['case_id']],workers=1)
        self.assertEqual(len(calls),1);self.assertEqual(before,canonical.read_bytes())
        self.assertEqual(calls[0][2],str(ROOT/'scripts/collect_multitask_validation_initialization_r10.py'))
        self.assertEqual(merged['expected'],275);self.assertEqual(merged['completed'],1);self.assertFalse(merged['complete'])
        self.assertEqual(merged['shared_profile_execution_registration'],r.pointer(registry))
        parent=r.read(output/'infrastructure_attempts'/case['case_id']/'0/parent_receipt.json')
        self.assertTrue(parent['first_execution']);self.assertFalse(parent['automatic_retries'])
        record=r.read(canonical);child=r.read(ROOT/record['parent_process_receipt']['child_case_path'])
        self.assertEqual(child,{key:value for key,value in record.items() if key!='parent_process_receipt'})
        self.assertNotEqual(record['process_id'],parent['process_id'])
        self.assertEqual(record['process_id'],child['process_id'])


if __name__=='__main__':unittest.main()
