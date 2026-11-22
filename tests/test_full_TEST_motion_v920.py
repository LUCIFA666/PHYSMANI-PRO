"""Source-only and in-memory1920-case tests; no real experiment data is read."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = module_at('full_TEST_motion_v920_subject', PROJECT / 'scripts/summarize_full_TEST_motion_v920_r10.py')
fakes = module_at('full_TEST_motion_v920_851_fakes', PROJECT / 'tests/test_formal_TEST_motion_v851.py')


def source(name):
    return dict(path=name, sha256=m.PINS.get(name, m.sha_bytes(('CPU-fake-source:' + name).encode())))


def batch_metrics():
    return dict(schema='r10_scoped_frozen_lookup_metrics_v900', closed=True, complete=True,
        maps=1, declared_files=1, watched_directories=1, pointer_checks=1,
        full_graph_boundary_checks=2, used_path_end_checks=1, used_external=0,
        actual_SHA_checks=2, strong_stat_reuses=0, exact_JSON_payloads=0)


def batch_receipt(row, sources, *, check, case=None, driver=None):
    value = dict(schema=m.P.BATCH_SCHEMA, complete=True, check_only=check, entry='formal',
        original_entry=m.P.ORIGINAL_ENTRY,
        actual_outer_argv=row['argv'][2:] + (['--check'] if check else []),
        forwarded_original_argv=row['physical_argv'][2:] + (['--check'] if check else []),
        control=row['execution_entry']['batch_control'], lock=row['execution_entry']['batch_lock'],
        sources=sources, scientific_policy_changed=False, full_verify_inputs_calls_retained=True,
        native_entry_claim='original_function_called_under_explicit_v901_bootstrap',
        phases=[dict(phase=p, wall_seconds=.1, boundary_recheck_complete=True, metrics=batch_metrics())
            for p in ('original751_bindings', 'original742_prepare')])
    if not check:
        value.update(case=case, original_admission_driver_receipt=driver)
    return value


def memory_fixture(reducer, count=134, motion_override=False):
    """Extend the existing851 zero-tick fixture using the real pure902 writer.

    The fake906 ledger intentionally omits manifest/original-manifest/routes:
    those are read by the original source gate outside its EvidenceLedger.
    All runtime/case structures stay in memory, including the two distinct tiers.
    """
    with patch.object(fakes, 'source', source):
        initial = fakes.memory_fixture(reducer)
    data = initial.data

    def checksum(name):
        if name in m.PINS:
            return m.PINS[name]
        return m.digest(data[name]) if name in data else source(name)['sha256']

    def ptr(name):
        return dict(path=name, sha256=checksum(name))

    old_plan = data[m.PREVIOUS + '/plan.json']
    old_jobs = old_plan['jobs']
    clean = dict(complete=True, returncode=0, timed_out=False, interrupted=False, automatic_retry=False)
    old_completed = {}
    for i, row in enumerate(old_jobs):
        for kind in ('CPU', 'native'):
            leaf = m.PREVIOUS + '/' + kind + '/' + row['job_id']
            data[leaf + '/launch.json'].update(launcher_pid=10000 + i * 2 + int(kind == 'native'),
                launcher_start_ticks=20000 + i * 2 + int(kind == 'native'))
        terminal = data[m.PREVIOUS + '/native/' + row['job_id'] + '/terminal.json']
        terminal.update(native_success=False, motion_pass=False)
        if i < count:
            old_completed[row['job_id']] = deepcopy(terminal)
    data[m.PREVIOUS + '/status.json'] = dict(split='test', expected=1920, completed=old_completed,
        failed={}, active=[], checking=[], drain_requested=True, stop_requested=False)
    data[m.PREVIOUS + '/terminal.json'].update(complete=False, completed=count, drained=True,
        drain_requested=True, stop_requested=False)
    old_launch = data[m.PREVIOUS + '/launch.json']
    old_launch.update(max_native_slots=2, max_GPU_slots=1, max_CPU_preflight_slots=2,
        suite_lock_sha256=checksum(m.BASE.SUITE_LOCK),
        admission_driver=data[m.BASE.SUITE]['formal_case_driver'],
        admission_guard=data[m.BASE.SUITE]['formal_admission_guard'],
        native_runtime=data[m.BASE.SUITE]['author_outcome_protocol']['runtime_consumer'],
        original_native_runtime=data[m.BASE.SUITE]['author_outcome_protocol']['original_runtime'])
    data[m.PREVIOUS + '/DRAIN'] = dict(reason='CPU fake clean natural drain')
    data[m.OLD_CONTROLLER + '/status.json'] = dict(stage='requires_concrete_repair', automatic_retry=False)
    data[m.OLD_CONTROLLER + '/failure.json'] = dict(complete=False, user_stop=False, automatic_retry=False)

    all_ids = {r['job_id'] for r in old_jobs}
    completed_ids = sorted(old_completed)
    protocol = data[m.BASE.SUITE]['author_outcome_protocol']
    shared = {p: source(p) for p in (*m.P.COMMON_SOURCES, m.P.BATCH_ENTRY,
        'our_mani/multitask/frozen_lookup_batch_v900.py')}
    data[m.BATCH_CONTROL] = dict(schema='r10_scoped_frozen_lookup_bootstrap_control_v901', execution_enabled=True)
    data[m.BATCH_LOCK] = dict(schema='r10_scoped_frozen_lookup_bootstrap_inputs_v901', execution_enabled=True,
        files={p: v['sha256'] for p, v in shared.items()})
    entry = dict(kind='batched901', pointer=shared[m.P.BATCH_ENTRY],
        batch_control=ptr(m.BATCH_CONTROL), batch_lock=ptr(m.BATCH_LOCK))
    control = dict(schema=m.P.CONTROL_SCHEMA, split='test', expected_cases=1920, source_count=640,
        policy_sha256=m.POLICY, physical_policy_changed=False, user_authorized_resume=True,
        standing_authorization_reused=True, new_user_event_invented=False,
        completed_job_ids=completed_ids, remaining_job_ids=sorted(all_ids - set(completed_ids)),
        completed_key_sha256=m.digest(completed_ids), remaining_key_sha256=m.digest(sorted(all_ids - set(completed_ids))),
        suite=ptr(m.BASE.SUITE), suite_lock=ptr(m.BASE.SUITE_LOCK), source_inputs=ptr(m.BASE.SOURCE_LOCK),
        validation_release=ptr(m.BASE.VAL_RELEASE), author_outcome_protocol=protocol,
        resume_controller=source(m.CONTROLLER_SOURCE), resume_owner=source(m.OWNER), resume_auditor=source(m.AUDITOR),
        execution_entry=entry, execution_implementation=m.P.execution_implementation(entry, shared),
        resource_profile=dict(schema=m.P.PROFILE_SCHEMA, max_native_slots=4, max_GPU_slots=2,
            max_CPU_preflight_slots=2, ready_limit=8),
        resource_qualification=dict(path='runtime/framework/CPU_fake_qualification.json', sha256='a' * 64),
        **{k: data[m.BASE.CONTROL][k] for k in
            ('environment_manifest', 'environment_wrapper', 'environment_namespace_proof')})
    for key, filename in (('previous_launch', 'launch.json'), ('previous_plan', 'plan.json'),
                         ('previous_status', 'status.json'), ('previous_terminal', 'terminal.json'),
                         ('previous_DRAIN', 'DRAIN')):
        control[key] = ptr(m.PREVIOUS + '/' + filename)
    for key, filename in (('previous_controller_launch', 'launch.json'),
                         ('previous_controller_status', 'status.json'),
                         ('previous_controller_failure', 'failure.json')):
        control[key] = ptr(m.OLD_CONTROLLER + '/' + filename)
    data[m.CONTROL] = control
    resume_files = {str(PROJECT / p): source(p)['sha256'] for p in m.SOURCES}
    for name in (m.CONTROL, m.BATCH_CONTROL, m.BATCH_LOCK, *[p['path'] for k, p in control.items()
            if k.startswith('previous_') and isinstance(p, dict)]):
        resume_files[str(PROJECT / name)] = checksum(name)
    for row in old_jobs[:count]:
        leaf = m.PREVIOUS + '/'
        for kind, files in (('CPU', ('launch.json', 'terminal.json', 'CPU_preflight.json', 'stdout.log')),
                            ('native', ('launch.json', 'terminal.json'))):
            for filename in files:
                name = leaf + kind + '/' + row['job_id'] + '/' + filename
                resume_files[str(PROJECT / name)] = checksum(name)
        out = m.option(row['argv'], '--out')
        for filename in ('case.json', 'admission_driver_receipt.json'):
            name = out + '/' + filename
            resume_files[str(PROJECT / name)] = checksum(name)
    data[m.LOCK] = dict(schema=m.P.LOCK_SCHEMA, approved_for_execution=True, control=ptr(m.CONTROL), files=resume_files)
    cp, lp = ptr(m.CONTROL), ptr(m.LOCK)
    jobs = m.P.resumed_jobs(old_plan, control, entry)
    common = dict(split='test', policy_sha256=m.POLICY, author_outcome_protocol=protocol,
        execution_protocol_changed=True, resume_control=cp, resume_control_lock=lp,
        admission_driver=data[m.BASE.SUITE]['formal_case_driver'], admission_guard=data[m.BASE.SUITE]['formal_admission_guard'],
        native_runtime=protocol['runtime_consumer'], original_native_runtime=protocol['original_runtime'],
        resource_profile=control['resource_profile'], resource_qualification=control['resource_qualification'],
        execution_implementation=control['execution_implementation'], automatic_retry=False)
    owner_env = fakes.environment(702, control)
    owner_launch = dict(common, pid=702, start_ticks=703, source_sha256=source(m.OWNER)['sha256'],
        imported_completed_cases=count, execution_entry=entry, frozen_environment=owner_env,
        max_native_slots=4, max_GPU_slots=2, max_CPU_preflight_slots=2,
        suite_lock_sha256=ptr(m.BASE.SUITE_LOCK)['sha256'], split_source_lock_sha256=ptr(m.BASE.SOURCE_LOCK)['sha256'],
        source_manifest=ptr(m.BASE.MANIFEST), base_stage_owner=source(m.BASE.OWNER), batch_sources=shared)
    data[m.RUNTIME + '/launch.json'] = owner_launch
    data[m.RUNTIME + '/plan.json'] = dict(common, jobs=jobs, expected_cases=1920, source_outcome_filtering=False,
        imported_completed_cases=count, frozen_environment=owner_env)
    completed = {}
    cases, drivers, tiers, batches = [], [], [], []
    for i, (row, old_row) in enumerate(zip(jobs, old_jobs)):
        imported = row['imported_completed']
        runtime = row['evidence_runtime']
        out = m.option(row['physical_argv'], '--out')
        old_out = m.option(old_row['argv'], '--out')
        if not imported:
            data[out + '/case.json'] = deepcopy(data[old_out + '/case.json'])
            if motion_override and i == count:
                data[out + '/case.json'].setdefault('actor_metrics', {})['task_dispatch_v43'] = dict(
                    actual_observed_window_limits_passed=True)
            data[out + '/admission_driver_receipt.json'] = deepcopy(data[old_out + '/admission_driver_receipt.json'])
            data[out + '/admission_driver_receipt.json']['case'] = ptr(out + '/case.json')
        casep, driverp = ptr(out + '/case.json'), ptr(out + '/admission_driver_receipt.json')
        cases.append(casep); drivers.append(driverp)
        bindings = {k: row[k] for k in m.H.BINDINGS}
        case = data[casep['path']]
        if not imported:
            for kind in ('CPU', 'native'):
                check = kind == 'CPU'; leaf = runtime + '/' + kind + '/' + row['job_id']
                data[leaf + '/launch.json'] = dict(argv=row['argv'] + (['--check'] if check else []),
                    physical_argv=row['physical_argv'] + (['--check'] if check else []),
                    check_only=check, owner_pid=702, native_slots=0 if check else 1,
                    GPU_slots=0 if check else int(row['method'] == 'physmani'), execution_protocol_changed=True,
                    expected_identity_sha256=None if check else case['identity']['sha256'],
                    launcher_pid=10000 + i * 2 + int(not check), launcher_start_ticks=20000 + i * 2 + int(not check),
                    execution_entry=row['execution_entry'], inherited_frozen_environment=owner_env,
                    current_mount_namespace=owner_env['current_mount_namespace'], **bindings)
                data[leaf + '/terminal.json'] = dict(clean)
                if check:
                    cpu = dict(complete=True, native_started=False, GPU_started=False, actual_split='test',
                        execution_protocol_changed=True, identity=case['identity'], component_domain=case['component_domain'],
                        batch_execution=batch_receipt(row, shared, check=True), **bindings)
                    data[leaf + '/CPU_preflight.json'] = cpu
                    data[leaf + '/stdout.log'] = deepcopy(cpu)
                else:
                    bp = out + '/batch_driver_receipt.json'
                    data[bp] = batch_receipt(row, shared, check=False, case=casep, driver=driverp)
                    batches.append(ptr(bp))
                    data[leaf + '/terminal.json'].update(case=casep, admission_driver_receipt=driverp,
                        batch_driver_receipt=ptr(bp), native_success=False, motion_pass=m.recorded_motion_flag(case))
        record = deepcopy(data[runtime + '/native/' + row['job_id'] + '/terminal.json'])
        completed[row['job_id']] = record
        owner = old_launch if imported else owner_launch
        tiers.append(dict(job_id=row['job_id'], evidence_runtime=runtime, imported_completed=imported,
            actual_stage_owner_pid=owner['pid'], actual_stage_owner_start_ticks=owner['start_ticks'],
            previous_row_sha256=row['previous_row_sha256'], execution_entry=row['execution_entry'],
            case=casep, admission_driver_receipt=driverp))
    data[m.RUNTIME + '/status.json'] = dict(common, expected=1920, completed=completed, failed={}, active=[], checking=[],
        ready_jobs=[], CPU_ready=0, native_slots=0, GPU_slots=0, CPU_launched=1920-count, imported_completed=count,
        owner_error=None, stop_requested=False, drain_requested=False, frozen_environment=owner_env)
    data[m.RUNTIME + '/terminal.json'] = dict(common, complete=True, expected=1920, completed=1920, failed=0,
        owner_error=None, stop_requested=False, drain_requested=False, drained=False, CPU_ready_unexecuted=[],
        CPU_not_launched=0, imported_completed_cases=count, resumed_new_cases=1920-count,
        formal_release_requires_independent_audit=True, frozen_environment=owner_env, native_successes=0)
    controller = dict(source_sha256=source(m.CONTROLLER_SOURCE)['sha256'], pid=402, start_ticks=403,
        resume_control=cp, resume_control_lock=lp, automatic_retry=False, frozen_environment=fakes.environment(402, control))
    data[m.CONTROLLER + '/launch.json'] = controller
    data[m.CONTROLLER + '/status.json'] = dict(stage='complete',automatic_retry=False,
        frozen_environment=controller['frozen_environment'])
    args = []
    for key, flag in (('suite', '--suite'), ('suite_lock', '--suite-lock'), ('source_inputs', '--source-inputs'),
                      ('validation_release', '--validation-release')):
        args.extend((flag, control[key]['path']))
    args += ['--runtime', m.RUNTIME, '--out', m.ART+'/test', '--resume-control', m.CONTROL,
        '--resume-lock', m.LOCK, '--resume-lock-sha256', lp['sha256']]
    audit_args = ['--resume-control', m.CONTROL, '--resume-lock', m.LOCK, '--resume-lock-sha256', lp['sha256']]
    for stage, script, number, tail in ((m.OWNER_STAGE, m.OWNER, 702, args), (m.AUDIT_STAGE, m.AUDITOR, 802, audit_args)):
        data[stage + '/launch.json'] = dict(argv=[str(PROJECT/'.conda-env/bin/python'), '-B', '-u', script, *tail],
            source_sha256=source(script)['sha256'], pid=number, start_ticks=number+1, automatic_retry=False,
            inherited_frozen_environment=controller['frozen_environment'], current_mount_namespace='mnt:[2]')
        data[stage + '/terminal.json'] = dict(complete=True, returncode=0, automatic_retry=False)
    owner_launch['argv'] = data[m.OWNER_STAGE + '/launch.json']['argv'][3:]
    result = fakes.result_fixture()
    result.update(schema=m.RESULT_SCHEMA, source_sha256=source(m.AUDITOR)['sha256'], resume_control=cp, resume_control_lock=lp,
        source_input_lock=control['source_inputs'], validation_release=control['validation_release'], author_outcome_protocol=protocol,
        resource_profile=control['resource_profile'], resource_qualification=control['resource_qualification'],
        execution_implementation=control['execution_implementation'], used_evidence_rechecked_at_close=True,
        old753_results_not_rerun=True, independent_composite_audit=True, original754_single_owner_result_claimed=False,
        automatic_retry=False, admission_driver=data[m.BASE.SUITE]['formal_case_driver'],
        admission_guard=data[m.BASE.SUITE]['formal_admission_guard'], base_result_auditor=data[m.BASE.SUITE]['formal_result_auditor'],
        cases=cases, admission_driver_receipts=drivers, evidence_tiers=tiers, batch_driver_receipts=batches,
        auditor_invocation=dict(pid=802, start_ticks=803, source=source(m.AUDITOR),
            argv=data[m.AUDIT_STAGE+'/launch.json']['argv'][3:], inherited_physical_stage_identity_claimed=False),
        frozen_environment=fakes.environment(802, control), actual_stage_owners=[
            dict(runtime=m.PREVIOUS, launch=control['previous_launch'], terminal=control['previous_terminal'],
                pid=700, start_ticks=701, completed=count,
                resource_limits={k: old_launch[k] for k in ('max_native_slots', 'max_GPU_slots', 'max_CPU_preflight_slots')}),
            dict(runtime=m.RUNTIME, launch=ptr(m.RUNTIME+'/launch.json'), terminal=ptr(m.RUNTIME+'/terminal.json'),
                pid=702, start_ticks=703, completed=1920-count, resource_profile=control['resource_profile'])])
    excluded = {m.BASE.MANIFEST, m.BASE.ORIGINAL_MANIFEST, m.BASE.ROUTES, m.RESULT}
    result['inputs_sha256'] = {str(PROJECT/name): checksum(name) for name in data
        if name not in excluded and name != m.BASE.RESULT}
    result['inputs_sha256'].update({str(PROJECT/name): source(name)['sha256'] for name in m.SOURCES})
    data[m.RESULT] = result
    data[m.CONTROLLER+'/terminal.json'] = dict(complete=True, automatic_retry=False, TEST_results=ptr(m.RESULT))
    data[m.AUDIT_STAGE+'/stdout.log'] = dict(complete=True, schema=m.RESULT_SCHEMA, split='test', cases=1920,
        out=m.RESULT, motion_is_diagnostic_only=True, native_successes_by_method=result['native_successes_by_method'])

    class FakeReader:
        root = PROJECT
        lookup = m.Reader.lookup
        def __init__(self):
            self.inputs = {}; self.evidence = {}; self.frozen = {}; self.original = {}; self.resumed = {}
            self.opened = []; self.data = data; self.closed = False
        def check(self, name, actual, kwargs):
            expected = [kwargs[k] for k in ('expected',) if kwargs.get(k) is not None]
            for k in ('evidence', 'frozen', 'original', 'resumed'):
                if kwargs.get(k):
                    expected.append(self.lookup(getattr(self, k), name))
            m.require(all(x == actual for x in expected), 'input SHA changed: '+name)
            m.require(name not in self.inputs or self.inputs[name] == actual, 'input changed during fake read: '+name)
        def raw(self, name, **kwargs):
            self.opened.append(name)
            raw = (PROJECT/name).read_bytes() if name in m.PINS else ('CPU-fake-source:'+name).encode()
            actual = m.sha_bytes(raw); self.check(name, actual, kwargs); self.inputs[name] = actual
            return raw
        def json(self, name, **kwargs):
            self.opened.append(name); actual = checksum(name); self.check(name, actual, kwargs)
            self.inputs[name] = actual; return data[name]
        def pointer(self, name):
            return dict(path=name, sha256=self.inputs[name])
        def pointed(self, value, **kwargs):
            m.pointer(value); return self.json(value['path'], expected=value['sha256'], **kwargs)
        def close(self):
            self.closed = True
    return FakeReader()


class FullTESTMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reducer = m.H.load_reducer(m.Reader(PROJECT))

    def fixture(self):
        return memory_fixture(self.reducer)

    def collect(self, reader):
        return m.collect(reader, m.digest(reader.data[m.RESULT]), resume_lock_sha=m.digest(reader.data[m.LOCK]))

    def test_full1920_two_tiers_preserve_false_zero_tick_and_unknown(self):
        reader = self.fixture(); rows, result, subset, reducer = self.collect(reader)
        self.assertEqual(len(rows), 1920)
        self.assertEqual(Counter(r['evidence_runtime'] for r in rows), {m.PREVIOUS:134, m.RUNTIME:1786})
        self.assertEqual(Counter(r['method'] for r in rows), {method:640 for method in m.METHODS})
        self.assertEqual(len({(r['task'], r['source_raw_episode_id']) for r in rows}), 640)
        self.assertTrue(all(r['official_task_success'] is False for r in rows))
        self.assertTrue(all(r['recorded_window_diagnostics']['combined_window_checks_passed'] is False for r in rows))
        self.assertTrue(all(r['metrics']['peak_window_joint_acceleration_rad_s2'] is None for r in rows))
        self.assertEqual(len(result['batch_driver_receipts']), 1786)
        self.assertTrue(reader.closed)
        for method in m.METHODS:
            summary = reducer.summarize([r for r in rows if r['method'] == method])
            self.assertEqual(summary['case_count'], 640)

    def test_actor_motion_precedence_is_status_consistency_not_filter(self):
        reader=memory_fixture(self.reducer,motion_override=True)
        rows, *_=self.collect(reader)
        row=reader.data[m.RUNTIME+'/plan.json']['jobs'][134]
        record=reader.data[m.RUNTIME+'/status.json']['completed'][row['job_id']]
        case=reader.data[m.option(row['physical_argv'],'--out')+'/case.json']
        self.assertIs(record['motion_pass'],True)
        self.assertIs(case['all_stage_counted_joint_motion']['all_stage_actual_window_checks_passed'],False)
        self.assertEqual(len(rows),1920)
        self.assertTrue(all(r['official_task_success'] is False for r in rows))

    def test_motion_flag_exact_actor_feed_primary_mirror_precedence(self):
        case=dict(actor_metrics=dict(task_dispatch_v43=dict(actual_observed_window_limits_passed=False),
            continuous_reference_feedforward=dict(observed_motion_qualification_failed=False,observed_events=[{}])),
            all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=True))
        self.assertIs(m.recorded_motion_flag(case),False)
        del case['actor_metrics']['task_dispatch_v43']
        self.assertIs(m.recorded_motion_flag(case),True)
        case['actor_metrics']['continuous_reference_feedforward']['observed_events']=[]
        self.assertIs(m.recorded_motion_flag(case),True)
        case['all_stage_counted_joint_motion']={}
        case['alignment_receipt']=dict(all_stage_counted_joint_motion_v191=dict(all_stage_actual_window_checks_passed=False))
        self.assertIs(m.recorded_motion_flag(case),False)
        self.assertIsNone(m.recorded_motion_flag({}))

    def test_missing_final907_refuses_before_result_source_or_cases(self):
        reader = self.fixture(); reader.data[m.CONTROLLER+'/terminal.json']['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete907'):
            self.collect(reader)
        self.assertNotIn(m.RESULT, reader.opened)
        self.assertNotIn(m.BASE.MANIFEST, reader.opened)
        self.assertFalse(any(p.endswith('/case.json') for p in reader.opened))

    def test_907_noncomplete_status_refused_before_result_source_or_cases(self):
        reader=self.fixture();reader.data[m.CONTROLLER+'/status.json']['stage']='independent_TEST_results'
        with self.assertRaisesRegex(ValueError,'actual907 status must be complete'):
            self.collect(reader)
        self.assertNotIn(m.RESULT,reader.opened)
        self.assertNotIn(m.BASE.MANIFEST,reader.opened)
        self.assertFalse(any(p.endswith('/case.json') for p in reader.opened))

    def test_reporting_helper_absent_old_maps_is_honestly_disclosed(self):
        reader=self.fixture();rows,*_=self.collect(reader)
        self.assertEqual(len(rows),1920)
        self.assertEqual(len(reader.reporting_dependencies),1)
        proof=reader.reporting_dependencies[0]
        self.assertIs(proof['original_physical_input_lock_member'],False)
        self.assertIs(proof['original_TEST_suite_input_lock_member'],False)
        self.assertIs(proof['old_input_lock_amended'],False)
        self.assertIs(proof['proof_of_original_physical_freeze_membership_claimed'],False)

    def test_reporting_helper_present_old_conflict_is_not_silently_ignored(self):
        name,checksum=next(iter(m.REPORTING_DEPENDENCIES.items()))
        class Reader:
            root=PROJECT
            original={}
            frozen={}
            def raw(self,path,**kwargs):
                self.read=(path,kwargs)
        for which in ('original','frozen'):
            for key in (name,str(PROJECT/name)):
                reader=Reader();reader.original={};reader.frozen={}
                setattr(reader,which,{key:'0'*64})
                with self.subTest(which=which,key=key),self.assertRaisesRegex(ValueError,'old declaration conflicts'):
                    m.verify_reporting_dependencies(reader)
        reader=Reader();reader.original={name:checksum};reader.frozen={str(PROJECT/name):checksum}
        records=m.verify_reporting_dependencies(reader)
        self.assertTrue(records[0]['original_physical_input_lock_member'])
        self.assertTrue(records[0]['original_TEST_suite_input_lock_member'])

    def test_incomplete902_owner_refuses_before_result_or_source(self):
        reader = self.fixture(); reader.data[m.RUNTIME+'/terminal.json']['completed'] = 1919
        with self.assertRaisesRegex(ValueError, 'full1920 mixed owner terminal'):
            self.collect(reader)
        self.assertNotIn(m.RESULT, reader.opened)
        self.assertNotIn(m.BASE.MANIFEST, reader.opened)

    def test_changed_owner_protocol_rejected_before_result(self):
        reader=self.fixture()
        reader.data[m.RUNTIME+'/status.json']['author_outcome_protocol']={'forged':True}
        with self.assertRaisesRegex(ValueError,'immutable902 owner bindings'):
            self.collect(reader)
        self.assertNotIn(m.RESULT,reader.opened)

    def test_row_driver_cannot_differ_from_released_original(self):
        reader=self.fixture()
        reader.data[m.RUNTIME+'/plan.json']['jobs'][134]['admission_driver']=source('scripts/forged_driver.py')
        with self.assertRaises(ValueError):
            self.collect(reader)
        self.assertFalse(any(p.endswith('/case.json') for p in reader.opened))

    def test_missing_or_wrong_external_sha_refuses_before_reads(self):
        reader = self.fixture()
        with self.assertRaisesRegex(ValueError, 'externally pinned'):
            m.collect(reader, 'a'*64)
        self.assertEqual(reader.opened, [])
        with self.assertRaisesRegex(ValueError, 'input SHA changed'):
            m.collect(reader, 'a'*64, resume_lock_sha='b'*64)
        self.assertEqual(reader.opened, [m.LOCK])

    def test_partial_or_fabricated906_result_rejected_before_manifest(self):
        for key, value in (('completed_method_cases',1919), ('independent_composite_audit',False),
                           ('original754_single_owner_result_claimed',True)):
            reader = self.fixture(); reader.data[m.RESULT][key] = value
            reader.data[m.CONTROLLER+'/terminal.json']['TEST_results']['sha256'] = m.digest(reader.data[m.RESULT])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.collect(reader)
            self.assertNotIn(m.BASE.MANIFEST, reader.opened)

    def test_906_ledger_does_not_claim_source_gate_manifest_or_routes(self):
        reader = self.fixture(); evidence = reader.data[m.RESULT]['inputs_sha256']
        for name in (m.BASE.MANIFEST, m.BASE.ORIGINAL_MANIFEST, m.BASE.ROUTES):
            self.assertNotIn(str(PROJECT/name), evidence)
        rows, *_ = self.collect(reader)
        self.assertEqual(len(rows),1920)

    def test_old753_cannot_gain_new_namespace_or901_evidence(self):
        reader = self.fixture(); row = reader.data[m.RUNTIME+'/plan.json']['jobs'][0]
        launch = reader.data[m.PREVIOUS+'/native/'+row['job_id']+'/launch.json']
        launch['current_mount_namespace'] = 'mnt:[2]'
        with self.assertRaises(ValueError):
            self.collect(reader)

    def test_new_cpu_forwarded_argv_or_closed_metrics_cannot_be_forged(self):
        for field in ('forwarded_original_argv', 'closed'):
            reader = self.fixture(); row = reader.data[m.RUNTIME+'/plan.json']['jobs'][134]
            cpu = reader.data[m.RUNTIME+'/CPU/'+row['job_id']+'/CPU_preflight.json']
            if field == 'closed': cpu['batch_execution']['phases'][0]['metrics']['closed'] = False
            else: cpu['batch_execution'][field] = ['forged']
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.collect(reader)

    def test_duplicate_tier_or_changed_counted_false_rejected(self):
        reader = self.fixture(); reader.data[m.RESULT]['evidence_tiers'][-1] = deepcopy(reader.data[m.RESULT]['evidence_tiers'][-2])
        reader.data[m.CONTROLLER+'/terminal.json']['TEST_results']['sha256'] = m.digest(reader.data[m.RESULT])
        with self.assertRaisesRegex(ValueError, 'exact unique mixed evidence tier'):
            self.collect(reader)

    def test_main_failure_never_creates_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root/m.OUTPUT).parent.mkdir(parents=True)
            with patch.object(m,'ROOT',root), patch.object(m,'Reader',return_value=object()), \
                    patch.object(m,'collect',side_effect=ValueError('CPU fake incomplete release')):
                with self.assertRaisesRegex(ValueError,'CPU fake incomplete'):
                    m.main(['--test-results-sha256','a'*64,'--resume-lock-sha256','b'*64])
            self.assertFalse((root/m.OUTPUT).exists())

    def test_existing_output_is_preserved_before_reader(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); target=root/m.OUTPUT; target.parent.mkdir(parents=True); target.write_text('CPU fake existing')
            with patch.object(m,'ROOT',root), patch.object(m,'Reader') as factory:
                with self.assertRaisesRegex(ValueError,'never overwrite'):
                    m.main(['--test-results-sha256','a'*64,'--resume-lock-sha256','b'*64])
                factory.assert_not_called()
            self.assertEqual(target.read_text(),'CPU fake existing')

    def test_reader_denies_raw_model_TRAIN_and_symlink_before_open(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); reader=m.Reader(root)
            for path in ('/UNOPENED_TEST/reset.pkl','data/test.hdf5','models/weights.pt',
                         'artifacts/framework/protocol/train_manifest.json'):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError,'non-evidence'):
                    reader.raw(path)
            target=root/'unopened.pkl'; target.write_text('CPU fake only')
            alias=root/m.BASE.MANIFEST; alias.parent.mkdir(parents=True); alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'symlink aliases'):
                reader.raw(m.BASE.MANIFEST)

    def test_reader_enforces_used_bytes_and_end_stat_without_graph_walk(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); target=root/m.CONTROL; target.parent.mkdir(parents=True); target.write_bytes(b'{}')
            reader=m.Reader(root); reader.resumed={m.CONTROL:m.sha_bytes(b'{}')}
            self.assertEqual(reader.json(m.CONTROL,resumed=True),{})
            target.write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError,'stat changed'):
                reader.close()


if __name__ == '__main__':
    unittest.main()
