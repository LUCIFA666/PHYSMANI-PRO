"""Recovery-only scheduling tests: real archive/plan proofs, mocked execution."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_PERMISSION = ROOT/'artifacts/framework/multitask_r10/development_physical_reset_permission_v4.json'
spec = importlib.util.spec_from_file_location('owned_recovery_runner', ROOT/'scripts/collect_multitask_recovery_r10.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
c = r.collection
# Preload CPU dependencies before the forbidden-process assertion.
from our_mani.multitask import development_reset_recovery


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    return r.pointer(path)


def archive(root, plan_path, collection_path):
    plan = r.read(plan_path); entries = []
    for i, (name, checksum) in enumerate(plan['inputs'].items()):
        if Path(name).suffix in ('.py', '.sh'):
            target = root/'sources'/(str(i)+Path(name).suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(name).read_bytes())
            entries.append(dict(original_path=name, path=str(target.relative_to(ROOT)), sha256=checksum))
    snap = dict(version=r.lineage.SOURCE_SCHEMA, all_plan_inputs_verified_before_and_after=True,
        source_files_modified=False, case_records_or_labels_modified=False,
        plan=dict(path=str(plan_path.relative_to(ROOT)), sha256=c.sha(plan_path), internal_sha256=plan['plan_sha256']),
        source_count=len(entries), entries=entries)
    return dict(plan=r.pointer(plan_path), source_snapshot=put(root/'snapshot.json', snap),
                collection=str(collection_path.relative_to(ROOT)))


def fixture(root):
    actual = r.read(ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_plan_v11.json')
    base = deepcopy(actual)
    base['cases'] = deepcopy(actual['cases'][:2])
    base['created_utc'] = c.utc()
    base['protocol_path'] = str(root/'base_protocol.json')
    base['inputs'] = {}
    for relative in ('scripts/collect_multitask_training_r10.py', 'scripts/run_local_sim.sh',
                     'our_mani/multitask/simulation.py', 'our_mani/multitask/train_teacher.py'):
        base['inputs'][str(ROOT/relative)] = c.sha(ROOT/relative)
    permission = r.read(PHYSICAL_PERMISSION)
    for name, checksum in permission['native_inputs_sha256'].items():
        base['inputs'][str((Path(base['protocol']['baseline_repo'])/name).resolve())] = checksum
    put(root/'base_protocol.json', base['protocol'])
    base['inputs'][str(root/'base_protocol.json')] = c.sha(root/'base_protocol.json')
    base.pop('plan_sha256', None); base['plan_sha256'] = c.canonical_sha(base)
    put(root/'base_plan.json', base)
    old = archive(root/'old_archive', root/'base_plan.json', root/'old_collection')
    failed = []
    for case in base['cases']:
        path = root/'old_collection/infrastructure_attempts'/case['case_id']/'0/parent_receipt.json'
        failed.append(put(path, dict(case_id=case['case_id'], status='infrastructure_error', process_id=123,
            policy_failure_label_assigned=False, started_utc='2026-10-18T00:00:00+00:00',
            finished_utc='2026-10-18T00:00:01+00:00', command=['bash', str(ROOT/'scripts/run_local_sim.sh'),
                str(ROOT/'scripts/collect_multitask_training_r10.py'), '--collect-case', '--plan', str(root/'base_plan.json'),
                '--case-id', case['case_id'], '--output', str(path.parent/'output')])))
    rml = put(root/'rml_permission.json', r.rml_permission_value())
    physical = r.pointer(PHYSICAL_PERMISSION)
    inputs = put(root/'recovery_input.json', dict(schema=r.INPUT_SCHEMA, original_plan=old['plan'],
        failed_parent_receipts=failed, development_reset_profile=physical, RML_factory_hook=rml,
        case_ids=[case['case_id'] for case in base['cases']], infrastructure_only=True))
    newplan = r.prepare_recovery_plan(old, physical_permission=physical, rml_permission=rml,
        recovery_input=inputs, protocol_output=root/'new_protocol.json', plan_output=root/'new_plan.json')
    new = archive(root/'new_archive', root/'new_plan.json', root/'recovery_collection')
    old_sources = {k:v for k,v in base['inputs'].items() if Path(k).suffix in ('.py','.sh')}
    new_sources = {k:v for k,v in newplan['inputs'].items() if Path(k).suffix in ('.py','.sh')}
    changes = [dict(original_path=name, before_sha256=old_sources.get(name), after_sha256=new_sources.get(name))
        for name in sorted(set(old_sources)|set(new_sources)) if old_sources.get(name)!=new_sources.get(name)]
    declaration = dict(schema='r10_declared_infrastructure_only_acquisition_transition_v1',
        from_plan_sha256=base['plan_sha256'], to_plan_sha256=newplan['plan_sha256'], created_utc=c.utc(),
        reason='CPU fixture explicit infrastructure-only recovery', preserves_case_seed_variation_geometry_labels_budget=True,
        allowed_recovery_case_ids=[case['case_id'] for case in base['cases']], source_changes=changes,
        protocol_changes=[dict(json_pointer='/protocol/'+key, old=dict(present=False),
            new=dict(present=True, value=newplan['protocol'][key]), reason='explicit scoped hook/metadata',
            evidence=newplan['protocol'][key]) for key in r.HOOK_KEYS])
    new.update(predecessor_plan_sha256=base['plan_sha256'],
        infrastructure_change_declaration=put(root/'change.json', declaration))
    put(root/'registry.json', dict(schema=r.REGISTRATION_SCHEMA, original=old, recovery=new))
    return root/'new_plan.json', root/'registry.json', root/'recovery_collection', base, newplan


class RecoveryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=c.BASE)
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_actual_permissions_roundtrip_and_full_roster_preserved_without_native(self):
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('native forbidden')):
            plan, registry, output, old, new = fixture(self.root)
            context = r.verify_execution(plan, registry, output=output, case_ids=[old['cases'][0]['case_id']])
        self.assertEqual(new['cases'], old['cases'])
        self.assertEqual(new['augmentation_budget'], old['augmentation_budget'])
        self.assertEqual(new['original_demonstration_budget'], old['original_demonstration_budget'])
        self.assertEqual(len(context['allowed']), 2)
        self.assertTrue(all(str(ROOT/p) in new['inputs'] for p in r.OWNED_SOURCES))
        permission = r.read(PHYSICAL_PERMISSION)
        self.assertEqual(permission['version'], 'r10_shared_development_physical_init_separate_expert_waypoint_v4')
        self.assertEqual(permission['joint_interval_numerical_tolerance'], {'arm':1e-6, 'gripper':1e-5})
        self.assertEqual(permission['joint_interval_native_units'], {'arm':'radian', 'gripper':'metre'})
        self.assertFalse(permission['raw_joint_positions_clipped'])

    def test_original_canonical_recovery_mislabel_and_input_tamper_are_rejected(self):
        plan, registry, output, old, new = fixture(self.root)
        case = old['cases'][0]
        canonical = c.case_directory(self.root/'old_collection', case)/'case.json'
        put(canonical, {'status':'completed'})
        with self.assertRaisesRegex(ValueError, 'original canonical'): r.verify_execution(plan, registry)
        canonical.unlink()
        bad = r.read(registry); bad['schema'] = 'pretend_v11'
        put(self.root/'bad_registry.json', bad)
        with self.assertRaisesRegex(ValueError, 'registration'): r.verify_execution(plan, self.root/'bad_registry.json')
        # The recovery input is immutable even if the plan JSON itself is unchanged.
        (self.root/'recovery_input.json').write_text('{}')
        with self.assertRaises(ValueError): r.verify_execution(plan, registry)

    def test_fixed_failure_subset_no_empty_unknown_duplicate_or_output_escape(self):
        plan, registry, output, old, new = fixture(self.root)
        for ids in ([], ['unknown'], [old['cases'][0]['case_id']]*2):
            with self.subTest(ids=ids), self.assertRaises(ValueError): r.verify_execution(plan, registry, case_ids=ids)
        with self.assertRaisesRegex(ValueError, 'outside'): r.verify_execution(plan, registry, output=self.root/'elsewhere')

    def test_RML_permission_policy_numbers_and_evidence_are_exact(self):
        value = r.rml_permission_value()
        for mutation in ('steps', 'claim', 'source', 'extra'):
            bad = deepcopy(value)
            if mutation=='steps': bad['configuration']['maximum_virtual_steps']=20000
            elif mutation=='claim': bad['historical_fault_reproduced_in_native_diagnostic']=True
            elif mutation=='source': bad['inputs_sha256'][next(iter(bad['inputs_sha256']))]='0'*64
            else: bad['new_unbounded_hook']=True
            ptr = put(self.root/(mutation+'.json'), bad)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'permission'): r.load_rml_permission(ptr)

    def test_explicit_worker_command_names_our_runner_and_actual_new_plan(self):
        command = r.worker_command(self.root/'new.json', 'fixed', self.root/'out', self.root/'guard.json', self.root/'registry.json')
        self.assertEqual(command[2], str(ROOT/'scripts/collect_multitask_recovery_r10.py'))
        self.assertEqual(command[command.index('--plan')+1], str(self.root/'new.json'))
        self.assertEqual(command[command.index('--execution-registration')+1], str(self.root/'registry.json'))
        self.assertNotIn(str(ROOT/'scripts/collect_multitask_training_r10.py'), command)

    def test_unresolved_attempt_blocks_entire_queue_before_any_process(self):
        plan, registry, output, old, new = fixture(self.root)
        (output/'infrastructure_attempts'/old['cases'][1]['case_id']/'0').mkdir(parents=True)
        with patch.object(r.subprocess, 'Popen', side_effect=AssertionError('must not launch')), \
             patch.object(c, 'require_r09_release', return_value={'CPU_fixture':True}):
            with self.assertRaisesRegex(ValueError, 'no automatic recovery retry'):
                r.run_recovery_queue(plan, output, self.root/'guard', registry,
                    case_ids=[case['case_id'] for case in old['cases']])

    def test_scoped_hooks_restore_originals_and_bind_original_planner_alias(self):
        from our_mani.multitask import simulation, train_teacher, rml_path_recovery as adapter
        from our_mani.multitask.development_reset_recovery import SharedPhysicalDevelopmentSimulation
        plan, registry, output, old, new = fixture(self.root)
        base_sim, base_planner = simulation.AlignedDevelopmentSimulation, train_teacher.native_path_planner
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with r.scoped_owned_hooks(new, dict(simulations=[], paths=[])):
                self.assertTrue(issubclass(simulation.AlignedDevelopmentSimulation, SharedPhysicalDevelopmentSimulation))
                self.assertIsNot(train_teacher.native_path_planner, base_planner)
                self.assertIs(adapter.native_path_planner, base_planner)
                raise RuntimeError('fixture')
        self.assertIs(simulation.AlignedDevelopmentSimulation, base_sim)
        self.assertIs(train_teacher.native_path_planner, base_planner)
        with patch.object(train_teacher, 'native_path_planner', lambda env: None):
            with self.assertRaisesRegex(ValueError, 'original alias'):
                with r.scoped_owned_hooks(new, dict(simulations=[], paths=[])): pass


    def mocked_child_hooks(self, plan, captures):
        @contextmanager
        def hooks():
            captures['simulations'].append(SimpleNamespace(reset_recovery_receipt=dict(complete=True, CPU_fixture=True)))
            captures['paths'].append(SimpleNamespace(recovery_receipt=dict(close_complete=True, CPU_fixture=True)))
            yield
        return hooks()

    def fake_native_collector(self, plan_path, case_id, output, guard_report):
        plan = r.read(plan_path); case = next(row for row in plan['cases'] if row['case_id']==case_id)
        record = dict(case_id=case_id, plan_sha256=plan['plan_sha256'], status='completed',
            case_isolation='fresh_simulator_process_per_episode', process_id=os.getpid(),
            entry={key:case[key] for key in ('behavior','behavior_replica','maximum_collection_ticks',
                'teacher_used','context_primary_view','interaction_eligible')},
            alignment=dict(clock=dict(ticks=0)), files_sha256={},
            teacher_actions_used=case['teacher_used'], teacher_paths_requested=case['teacher_used'],
            unchanged_label_payload=dict(target_pose='CPU fixture, no native data', valid=False))
        record['entry'].update(split=case['split'],task=case['task'],raw_episode_id=case['raw_episode_id'],parent_raw_episode_id=case['parent_raw_episode_id'])
        c.write_json(c.case_directory(output, case)/'case.json', record)
        return record

    def test_mocked_complete_child_preserves_raw_collector_labels_and_binds_hooks(self):
        plan, registry, output, old, new = fixture(self.root); case=old['cases'][0]
        attempt_output=output/'infrastructure_attempts'/case['case_id']/'0/output'
        with patch.object(r, 'scoped_owned_hooks', side_effect=self.mocked_child_hooks), \
             patch.object(c, 'collect_native_case', side_effect=self.fake_native_collector), \
             patch.object(r.subprocess, 'Popen', side_effect=AssertionError('native forbidden')):
            record=r.collect_recovery_case(plan,case['case_id'],attempt_output,self.root/'guard',registry)
        out=c.case_directory(attempt_output,case)
        base=r.read(out/'base_collector_receipt.json'); hooks=r.read(out/'recovery_hook_receipt.json')
        self.assertEqual(base['unchanged_label_payload'],record['unchanged_label_payload'])
        self.assertEqual(base['plan_sha256'],new['plan_sha256'])
        self.assertTrue(hooks['complete']); self.assertFalse(hooks['new_attempt_is_original_v11'])
        self.assertEqual(hooks['failed_parent_receipts'],r.read(self.root/'recovery_input.json')['failed_parent_receipts'][:1])
        self.assertEqual(record['recovery_execution']['base_collector_receipt'],r.pointer(out/'base_collector_receipt.json'))
        tampered=deepcopy(record);tampered['unchanged_label_payload']['valid']=True
        put(out/'tampered.json',tampered)
        context=r.verify_execution(plan,registry)
        with self.assertRaisesRegex(ValueError,'base collector'):r.validate_recovery_result(out/'tampered.json',context,case)

    def test_child_failure_retains_hook_error_without_canonical_or_fake_old_rows(self):
        plan, registry, output, old, new=fixture(self.root); case=old['cases'][0]
        attempt_output=output/'infrastructure_attempts'/case['case_id']/'0/output'
        with patch.object(r, 'scoped_owned_hooks', side_effect=self.mocked_child_hooks), \
             patch.object(c, 'collect_native_case', side_effect=RuntimeError('owned failed source')):
            with self.assertRaisesRegex(RuntimeError,'owned failed source'):
                r.collect_recovery_case(plan,case['case_id'],attempt_output,self.root/'guard',registry)
        out=c.case_directory(attempt_output,case); hooks=r.read(out/'recovery_hook_receipt.json')
        self.assertFalse(hooks['complete']);self.assertEqual(hooks['error']['message'],'owned failed source')
        self.assertFalse(hooks['partial_historical_rows_reconstructed']);self.assertFalse((out/'case.json').exists())
        self.assertFalse((c.case_directory(output,case)/'case.json').exists())

    def test_mocked_parent_raw_child_canonical_chain_and_readonly_resume(self):
        plan,registry,output,old,new=fixture(self.root); case=old['cases'][0]; calls=[]
        class Process:
            pid=os.getpid()+100000
            def __init__(self,command,**kwargs):
                calls.append(command)
                def arg(key):return command[command.index(key)+1]
                r.collect_recovery_case(arg('--plan'),arg('--case-id'),arg('--output'),arg('--guard-report'),arg('--execution-registration'))
            def wait(self,timeout):return 0
        with patch.object(r, 'scoped_owned_hooks', side_effect=self.mocked_child_hooks), \
             patch.object(c, 'collect_native_case', side_effect=self.fake_native_collector), \
             patch.object(c, 'require_r09_release', return_value={'CPU_fixture':True}), \
             patch.object(r.subprocess, 'Popen', Process):
            merged=r.run_recovery_queue(plan,output,self.root/'guard',registry,case_ids=[case['case_id']],workers=1)
            canonical=c.case_directory(output,case)/'case.json'; before=canonical.read_bytes()
            resumed=r.run_recovery_queue(plan,output,self.root/'guard',registry,case_ids=[case['case_id']],workers=1)
        self.assertEqual(len(calls),1);self.assertEqual(canonical.read_bytes(),before)
        self.assertEqual(merged['completed'],1);self.assertFalse(merged['complete'])
        record=r.read(canonical);parent=r.read(output/'infrastructure_attempts'/case['case_id']/'0/parent_receipt.json')
        raw=r.read(ROOT/record['parent_process_receipt']['child_case_path'])
        self.assertEqual(raw,{key:value for key,value in record.items() if key!='parent_process_receipt'})
        self.assertEqual(parent['command'],calls[0]);self.assertFalse(parent['automatic_retries'])
        self.assertNotEqual(record['process_id'],parent['process_id'])
        self.assertEqual(record['process_id'],raw['process_id'])
        self.assertEqual(record['parent_process_receipt']['process_id'],parent['process_id'])
        self.assertFalse((c.case_directory(self.root/'old_collection',case)/'case.json').exists())


if __name__=='__main__': unittest.main()
