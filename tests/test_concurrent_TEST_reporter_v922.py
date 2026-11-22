"""CPU-only synthetic1920 metadata. Never reads real TEST/case/raw/model data."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/report_concurrent_TEST_halfhour_v922_r10.py'
spec=importlib.util.spec_from_file_location('observer922_fake',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def ptr(name):return dict(path=name,sha256='a'*64)


def grid():
    rows=[]
    for task in m.TASK_LABELS:
        for raw in range(40):
            for method in m.METHODS:
                name=task+'--'+str(raw)+'--'+method
                out='artifacts/framework/multitask_r10/old_TEST/'+task+'/episode'+str(raw)+'/'+method
                rows.append(dict(job_id=name,task=task,method=method,raw='episode'+str(raw),worker_sha256='b'*64,
                    admission_driver=ptr('scripts/run_formal_case_author_outcome_v751_r10.py'),
                    argv=['bash','scripts/run_aligned_gpu.sh' if method=='physmani' else 'scripts/run_local_sim.sh',
                        'scripts/run_formal_case_author_outcome_v751_r10.py','--split','test','--out',out]))
    return dict(split='test',expected_cases=1920,jobs=rows)


def resumed(old,count=3):
    old_ids=[r['job_id'] for r in old['jobs'][:count]]
    entry=dict(kind='batched901',pointer=ptr('scripts/run_batched_formal_driver_v901_r10.py'),
        batch_control=ptr('artifacts/framework/multitask_r10/TEST_batch_control.json'),batch_lock=ptr('artifacts/framework/multitask_r10/TEST_batch_lock.json'))
    control=dict(completed_job_ids=old_ids,remaining_job_ids=[r['job_id'] for r in old['jobs'][count:]],execution_entry=entry)
    result=deepcopy(old)
    for row,previous in zip(result['jobs'],old['jobs']):
        adopt=row['job_id'] in old_ids
        row.update(evidence_runtime=m.OLD_RUNTIME if adopt else m.RUNTIME,imported_completed=adopt,
            previous_row_sha256=m.digest(previous),previous_worker_sha256=previous['worker_sha256'])
        if adopt:row.update(physical_argv=deepcopy(previous['argv']),execution_entry=dict(kind='original751',pointer=previous['admission_driver']))
        else:
            inner=deepcopy(previous['argv']);inner[-1]=str(Path('artifacts/framework/multitask_r10/formal_TEST_concurrent_resume_v902/test')/row['task']/row['raw']/row['method'])
            outer=[*inner[:2],entry['pointer']['path'],'--batch-control',entry['batch_control']['path'],
                '--batch-lock',entry['batch_lock']['path'],'--batch-lock-sha256',entry['batch_lock']['sha256'],'--entry','formal','--',*inner[3:]]
            row.update(physical_argv=inner,argv=outer,worker_sha256=entry['pointer']['sha256'],execution_entry=entry)
    return result,control


def complete(row,success=False):
    argv=row['physical_argv'];out=Path(argv[argv.index('--out')+1])
    return dict(complete=True,native_success=success,returncode=0,timed_out=False,interrupted=False,automatic_retry=False,
        case=ptr(str(out/'case.json')),admission_driver_receipt=ptr(str(out/'admission_driver_receipt.json')))


class GridTests(unittest.TestCase):
    def test_exact_grid_and_imported_plus_new_false_are_counted_once(self):
        old=grid();plan,control=resumed(old);jobs,imported=m.plan_provenance(plan,old,control)
        records={row['job_id']:complete(row) for row in plan['jobs'][:6]}
        old_status=dict(completed={key:records[key] for key in imported})
        status=dict(split='test',expected=1920,completed=records,failed={},active=[],checking=[])
        result=m.count_status(status,jobs,old_status,imported)
        self.assertEqual(sum(result['completed'].values()),6)
        self.assertEqual((result['imported_completed'],result['new_completed']),(3,3))
        self.assertEqual(sum(result['success'].values()),0)

    def test_duplicate_raw_mismatched_sources_wrong_method_or_partial_grid_refused(self):
        for mutation in ('duplicate_raw','mismatch','wrong_method','partial','duplicate_key'):
            plan=grid()
            if mutation=='duplicate_raw':plan['jobs'][3]['raw']='episode0'
            elif mutation=='mismatch':plan['jobs'][0]['raw']='different_original'
            elif mutation=='wrong_method':plan['jobs'][0]['method']='value'
            elif mutation=='partial':plan['jobs'].pop()
            else:plan['jobs'][1]['job_id']=plan['jobs'][0]['job_id']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.plan_grid(plan)

    def test_changed_imported_tier_or_row_or_physical_entry_refused(self):
        for mutation in ('tier','adopt','argv','sha'):
            old=grid();plan,control=resumed(old);row=plan['jobs'][0]
            if mutation=='tier':row['evidence_runtime']=m.RUNTIME
            elif mutation=='adopt':row['imported_completed']=False
            elif mutation=='argv':row['physical_argv'][-1]='other'
            else:row['previous_row_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.plan_provenance(plan,old,control)

    def test_unclean_complete_unknown_or_rewritten_imported_false_never_counts(self):
        for mutation in ('returncode','missing_timeout','interrupted','retry','error','unknown','rewrite'):
            old=grid();plan,control=resumed(old);jobs,imported=m.plan_provenance(plan,old,control)
            records={r['job_id']:complete(r) for r in plan['jobs'][:3]};old_status=dict(completed=deepcopy(records))
            first=next(iter(records))
            if mutation=='returncode':records[first]['returncode']=1
            elif mutation=='missing_timeout':records[first].pop('timed_out')
            elif mutation=='interrupted':records[first]['interrupted']=True
            elif mutation=='retry':records[first]['automatic_retry']=True
            elif mutation=='error':records[first]['error']='censored'
            elif mutation=='unknown':records['foreign']=complete(plan['jobs'][4])
            else:records[first]['native_success']=True
            # For clean-field negatives use a new row, so failure is not just old-record inequality.
            if mutation not in ('unknown','rewrite'):
                records=deepcopy(old_status['completed']);key=plan['jobs'][3]['job_id'];records[key]=complete(plan['jobs'][3])
                if mutation=='returncode':records[key]['returncode']=1
                elif mutation=='missing_timeout':records[key].pop('timed_out')
                elif mutation=='interrupted':records[key]['interrupted']=True
                elif mutation=='retry':records[key]['automatic_retry']=True
                else:records[key]['error']='censored'
            status=dict(split='test',expected=1920,completed=records,failed={},active=[],checking=[])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.count_status(status,jobs,old_status,imported)


class MetadataTests(unittest.TestCase):
    def test_actual907_python_B_u_prefix_is_preserved_when_matching_self_argv(self):
        inner=[m.AUDITOR_SOURCE,'--resume-control',m.CONTROL]
        launch=dict(argv=[str(m.ROOT/'.conda-env/bin/python'),'-B','-u',*inner])
        self.assertEqual(m.stage_inner(launch,m.AUDITOR_SOURCE),inner)
        for argv in ([str(m.ROOT/'.conda-env/bin/python'),'-u',*inner],
                     ['bash','scripts/wrapper.sh',*inner],
                     [str(m.ROOT/'.conda-env/bin/python'),'-B','-u',m.OWNER_SOURCE]):
            with self.subTest(argv=argv),self.assertRaises(ValueError):m.stage_inner(dict(argv=argv),m.AUDITOR_SOURCE)

    def test_forbidden_raw_model_manifest_and_arbitrary_environment_are_rejected_before_open(self):
        def forbidden(name):self.fail('forbidden read callback: '+name)
        reader=m.Metadata(read_bytes=forbidden)
        for name in ('models/model.json','artifacts/framework/multitask_r10/models/model.json',
                     'artifacts/framework/protocol/test_manifest.json','data/a.hdf5','outputs/case.json'):
            with self.subTest(name=name),self.assertRaises(ValueError):reader.bytes(name)
            with self.subTest(pointer=name),self.assertRaises(ValueError):reader.permit_pointer(ptr(name))

    def test_status_is_one_exact_bytes_snapshot_even_if_callback_later_changes(self):
        values=[b'{"tick":1}',b'{"tick":2}'];calls=[]
        def read(name):calls.append(name);return values[min(len(calls)-1,1)]
        reader=m.Metadata(read_bytes=read);name=m.RUNTIME+'/status.json'
        self.assertEqual(reader.json(name),{'tick':1})
        self.assertEqual(reader.json(name),{'tick':1})
        self.assertEqual(reader.checksums[name],hashlib.sha256(values[0]).hexdigest())
        self.assertEqual(len(calls),1)

    def test_STOP_precedes_every_metadata_read(self):
        reader=m.Metadata(stop=lambda:True,read_bytes=lambda name:self.fail('STOP opened metadata'))
        with self.assertRaises(m.legacy.ObservationStopped):reader.json(m.VAL_RELEASE)

    def test_bad_sha_and_source_path_alias_refused(self):
        reader=m.Metadata(read_bytes=lambda name:b'{}')
        with self.assertRaises(ValueError):reader.pointer(dict(path=m.VAL_RELEASE,sha256=None))
        with self.assertRaises(ValueError):reader.bytes(m.VAL_RELEASE,'0'*64)


class ReportTests(unittest.TestCase):
    def test_seed_keeps_830_phase_and_uses_history_after_final_None(self):
        config=dict(legacy_schedule={key:ptr(m.OLD_REPORT+'/'+name) for key,name in
            (('status','status.json'),('chat_ready','chat_ready.json'),('history','history.jsonl'))})
        payload={m.OLD_REPORT+'/status.json':b'{"next_update_utc":null}',m.OLD_REPORT+'/chat_ready.json':b'{"next_update_utc":null}',
            m.OLD_REPORT+'/history.jsonl':b'{"next_update_utc":"2026-10-21T15:38:21.384754+00:00"}\n{"next_update_utc":null}\n'}
        for key,p in config['legacy_schedule'].items():p['sha256']=m.sha(payload[p['path']])
        actual=m.schedule_seed(config,m.Metadata(read_bytes=lambda name:payload[name]))
        self.assertEqual(actual,m.legacy.parse_utc('2026-10-21T15:38:21.384754+00:00'))
        self.assertEqual(m.legacy.advance_deadline(actual,actual+3601,1800),actual+5400)

    def test_new_txt_only_and_chat_ready_never_acknowledges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'TEST_结果更新.txt';old.write_text('original report')
            data=m.empty_data();data.update(admitted=True,plan_valid=True,imported_completed=2,new_completed=1)
            key=next(iter(m.TASK_LABELS));data['completed'][key,'cv_geometry']=3
            data['success'][key,'cv_geometry']=1
            old_history=[dict(summary='old830 actual summary',origin='original830')]
            with patch.object(m,'ROOT',root),patch.object(m.legacy,'ROOT',root):
                item=m.save_report(data,None,old_history,next_due=10000,reason='fake',now=9000)
            self.assertEqual(old.read_text(),'original report')
            self.assertTrue((root/m.OUT).exists())
            self.assertEqual(item['completed'],3)
            self.assertEqual(item['resumed_new_completed'],1)
            self.assertFalse(item['TEST_case_files_read'])
            self.assertFalse((root/m.STATE/'chat_ack.json').exists())
            chat=json.loads((root/m.STATE/'chat_ready.json').read_bytes())
            self.assertNotIn('sent',chat);self.assertNotIn('delivered',chat)
            self.assertIn('old830 actual summary',(root/m.OUT).read_text())
            # Inherited history remains an external source, not a forged new snapshot.
            self.assertEqual(len((root/m.STATE/'history.jsonl').read_text().splitlines()),1)


def final_fixture():
    """Invent the metadata written by902/906/907, without their physical code."""
    old_plan=grid();plan,control=resumed(old_plan)
    jobs,imported=m.plan_provenance(plan,old_plan,control)
    records={row['job_id']:complete(row,success=False) for row in plan['jobs']}
    old_status=dict(completed={key:records[key] for key in imported})
    status=dict(split='test',expected=1920,completed=records,failed={},active=[],checking=[],
        ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0)
    data=m.count_status(status,jobs,old_status,imported)
    config={key:ptr(name) for key,name in m.FIXED_METADATA.items()}
    config.update(policy_sha256='e'*64,author_outcome_protocol=dict(invented='unchanged'),
        source_bindings={name:ptr(name) for name in m.SOURCES},resume_control=ptr(m.CONTROL),
        resume_control_lock=ptr(m.CONTROL_LOCK),test_source_inputs=ptr(m.SOURCE_INPUTS),
        validation_release=ptr(m.VAL_RELEASE),resource_qualification=ptr(m.QUALIFICATION))

    def environment(pid):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,
            checked_OS_files=6418,all_current_OS_targets_full_byte_SHA_verified=True,
            host_PID_network_namespace_preserved=True,proc_or_dev_replaced=False,
            task_parameters_or_strategy_changed=False,project_writable=True,
            wrapper=config['environment_wrapper'],manifest=config['environment_manifest'],
            namespace_qualification=config['environment_namespace_proof'],
            current_mount_namespace='mnt:[invented_private]',mount_namespace='mnt:[invented_private]',
            reference_mount_namespace='mnt:[invented_host]',qualified_host_mount_namespace='mnt:[invented_host]',
            observer_pid=pid,pid=pid,observer_start_ticks=pid*10,outer_wrapper_pid=8000,
            outer_wrapper_start_ticks=80000,current_OS_target_SHA_map_sha256='a'*64,
            restored_four_target_SHA256={name:'a'*64 for name in m.legacy.RESTORED_TARGETS},source_locks={})

    controller=dict(frozen_environment=environment(9070),stage='complete')
    owner_launch=dict(pid=9020,start_ticks=90200)
    old_launch=dict(pid=7530,start_ticks=75300)
    terminal=dict(policy_sha256=config['policy_sha256'],author_outcome_protocol=config['author_outcome_protocol'],
        complete=True,expected=1920,completed=1920,failed=0,stop_requested=False,automatic_retry=False,
        CPU_ready_unexecuted=[],CPU_not_launched=0,owner_error=None,drain_requested=False,drained=False,
        imported_completed_cases=len(imported),resumed_new_cases=1920-len(imported))
    values={m.CONTROL:{},m.CONTROL_LOCK:{},m.VAL_RELEASE:{},m.SOURCE_INPUTS:{},m.QUALIFICATION:{},
        m.OWNER_SOURCE:{},m.AUDITOR_SOURCE:{},m.RUNTIME+'/launch.json':owner_launch,
        m.RUNTIME+'/plan.json':plan,m.RUNTIME+'/status.json':status,m.RUNTIME+'/terminal.json':terminal,
        m.OLD_RUNTIME+'/launch.json':old_launch,m.OLD_RUNTIME+'/plan.json':old_plan,
        m.OLD_RUNTIME+'/status.json':old_status,m.OLD_RUNTIME+'/terminal.json':{}}
    checksums={name:m.sha(json.dumps(value,sort_keys=True).encode()) for name,value in values.items()}
    def point(name):return dict(path=name,sha256=checksums[name])
    control.update(previous_launch=point(m.OLD_RUNTIME+'/launch.json'),
        previous_terminal=point(m.OLD_RUNTIME+'/terminal.json'))
    leaf=m.CONTROLLER+'/stages/independent_TEST_results'
    argv=[str(m.ROOT/'.conda-env/bin/python'),'-B','-u',m.AUDITOR_SOURCE,'--invented']
    values.update({leaf+'/launch.json':dict(source_sha256=config['source_bindings'][m.AUDITOR_SOURCE]['sha256'],
        inherited_frozen_environment=controller['frozen_environment'],
        current_mount_namespace='mnt:[invented_private]',pid=9060,start_ticks=90600,argv=argv),
        leaf+'/terminal.json':dict(complete=True,returncode=0,automatic_retry=False),
        leaf+'/stdout.log':dict(complete=True,split='test',cases=1920,out=m.RESULTS)})
    result=dict(schema='r10_composite_concurrent_resumed_formal_TEST_results_v906',complete=True,split='test',
        source_cases=640,method_cases=1920,completed_method_cases=1920,
        policy_sha256=config['policy_sha256'],author_outcome_protocol=config['author_outcome_protocol'],
        methods=list(m.METHODS),tasks=list(m.TASK_LABELS),TEST_used_for_fitting=False,
        motion_is_diagnostic_only=True,native_task_success_unchanged=True,
        source_sha256=config['source_bindings'][m.AUDITOR_SOURCE]['sha256'],
        resume_control=config['resume_control'],resume_control_lock=config['resume_control_lock'],
        source_input_lock=config['test_source_inputs'],validation_release=config['validation_release'],
        resource_profile=m.PROFILE,resource_qualification=config['resource_qualification'],
        source_outcome_filtering=False,native_failures_retained=True,Value_NGFF_FASTER_used=False,
        independent_composite_audit=True,
        cases=[records[row['job_id']]['case'] for row in plan['jobs']],
        admission_driver_receipts=[records[row['job_id']]['admission_driver_receipt'] for row in plan['jobs']],
        auditor_invocation=dict(pid=9060,start_ticks=90600,
            source=config['source_bindings'][m.AUDITOR_SOURCE],argv=argv[3:]),
        frozen_environment=environment(9060),
        inputs_sha256={str(m.ROOT/name):checksum for name,checksum in checksums.items()},
        actual_stage_owners=[dict(runtime=m.OLD_RUNTIME,launch=control['previous_launch'],
            terminal=control['previous_terminal'],pid=7530,start_ticks=75300,completed=len(imported)),
            dict(runtime=m.RUNTIME,launch=point(m.RUNTIME+'/launch.json'),terminal=point(m.RUNTIME+'/terminal.json'),
                pid=9020,start_ticks=90200,completed=1920-len(imported))],
        evidence_tiers=[dict(job_id=row['job_id'],**{key:row[key] for key in
            ('evidence_runtime','imported_completed','previous_row_sha256','execution_entry')},
            case=records[row['job_id']]['case'],admission_driver_receipt=records[row['job_id']]['admission_driver_receipt'])
            for row in plan['jobs']],
        native_successes_by_method={method:0 for method in m.METHODS},
        per_task=[dict(task=task,methods={method:dict(episodes=40,successes=0) for method in m.METHODS})
            for task in m.TASK_LABELS])
    values[m.RESULTS]=result
    data.update(config=config,controller=controller)
    return dict(data=data,jobs=jobs,terminal=terminal,owner_launch=owner_launch,control=control,
        values=values,result=result,preload=tuple(checksums),audit_leaf=leaf)


def final_reader(fixture):
    """Freeze one exact byte snapshot per allowed fake path, as collect does."""
    payload={name:json.dumps(value,sort_keys=True).encode() for name,value in fixture['values'].items()}
    if m.RESULTS in payload:
        payload[m.CONTROLLER+'/terminal.json']=json.dumps(dict(complete=True,automatic_retry=False,
            TEST_results=dict(path=m.RESULTS,sha256=m.sha(payload[m.RESULTS])))).encode()
    reads=[]
    def read(name):
        if name not in payload:raise FileNotFoundError(name)
        if name not in m.base_allowed():raise AssertionError('not reporter metadata: '+name)
        reads.append(name);return payload[name]
    reader=m.Metadata(read_bytes=read)
    for name in fixture['preload']:reader.bytes(name)
    return reader,reads


def source_fixture(*,val_permitted=True,qualification_complete=True):
    """Small observer/control bytes only; no case, source640 or owner reads."""
    payload={name:('invented source '+name).encode() for name in m.SOURCES}
    def point(name):return dict(path=name,sha256=m.sha(payload[name]))
    protocol=dict(schema='r10_author_action_outcome_protocol_v741',implementation=point(m.SOURCES[8]),
        runtime_consumer=point(m.SOURCES[9]),original_runtime=point(m.SOURCES[10]),
        failure_sampling='current_native_result_at_explicit_action_end',success_sampling='original_per_tick_success_latch',
        pose_action_max_ticks=10,public_hold_max_ticks=1,budget_partial_action_consumed=True,
        native_RML_parameters_unchanged=True,block_native_fail_conditions=0,block_extra_task_success_calls=0)
    policy='e'*64
    def put(name,value):payload[name]=json.dumps(value,sort_keys=True).encode()
    for key,name in m.FIXED_METADATA.items():
        if key=='environment_wrapper':payload[name]=b'invented820 wrapper source'
        else:put(name,{})
    put(m.FIXED_METADATA['standing_continuation_record'],dict(continue_until_user_stop=True,user_stop_received=False))
    put(m.FIXED_METADATA['concurrency_authorization_record'],dict(user_authorized_concurrency_change=True))
    for name in ('status.json','chat_ready.json'):
        put(m.OLD_REPORT+'/'+name,dict(next_update_utc='2026-10-21T15:38:21.384754+00:00'))
    payload[m.OLD_REPORT+'/history.jsonl']=b'{"summary":"invented830","next_update_utc":"2026-10-21T15:38:21.384754+00:00"}\n'
    put(m.SOURCE_INPUTS,dict(must_not_be_opened_by_load_or_failed_gates=True))
    release=dict(schema='r10_full16_common_author_RML_validation_release_v531',complete=True,split='validation',
        source_cases=315,method_cases=945,completed_method_cases=945,policy_sha256=policy,
        author_outcome_protocol=protocol,methods=list(m.METHODS),tasks=list(m.TASK_LABELS),
        TEST_used_for_fitting=False,motion_is_diagnostic_only=True,native_task_success_unchanged=True,
        formal_test_permitted=val_permitted,TEST_used=False,
        source_sha256=point(m.SOURCES[6])['sha256'],cases=[{} for _ in range(945)],
        admission_driver_receipts=[{} for _ in range(945)])
    put(m.VAL_RELEASE,release)
    implementation=dict(schema='invented common TRAIN/formal implementation')
    qualification=dict(schema='r10_concurrent_TRAIN_resource_qualification_v902',complete=qualification_complete,
        split='train',resource_profile=deepcopy(m.PROFILE),execution_entry=implementation,
        policy_sha256=policy,author_outcome_protocol=protocol,case_count=6,infrastructure_failed=0,
        cleanup_completed=True,resource_bounds_verified=True,overlap_verified=True,independent_audit_complete=True,
        observed_native_peak=4,observed_GPU_peak=2,formal_score_contribution=False,TEST_read=False,
        source_sha256=point(m.QUALIFIER_SOURCE)['sha256'])
    put(m.QUALIFICATION,qualification)
    fixed={key:point(name) for key,name in m.FIXED_METADATA.items()}
    control=dict(schema='r10_concurrent_resumed_formal_TEST_control_v902',expected_cases=1920,source_count=640,
        split='test',resource_profile=deepcopy(m.PROFILE),physical_policy_changed=False,user_authorized_resume=True,
        standing_authorization_reused=True,new_user_event_invented=False,policy_sha256=policy,
        author_outcome_protocol=protocol,execution_implementation=implementation,
        resource_qualification=point(m.QUALIFICATION),resource_qualifier=point(m.QUALIFIER_SOURCE),
        resume_controller=point(m.CONTROLLER_SOURCE),resume_owner=point(m.OWNER_SOURCE),resume_auditor=point(m.AUDITOR_SOURCE),
        validation_release=point(m.VAL_RELEASE),source_inputs=point(m.SOURCE_INPUTS),**fixed)
    put(m.CONTROL,control)
    execution_lock=dict(schema='r10_concurrent_resumed_formal_TEST_inputs_v902',approved_for_execution=True,
        files={name:m.sha(payload[name]) for name in (m.CONTROL,m.CONTROLLER_SOURCE,m.OWNER_SOURCE,m.AUDITOR_SOURCE)})
    put(m.CONTROL_LOCK,execution_lock)
    config=dict(schema=m.SCHEMA,protocol_status='formal_TEST_concurrent',runtime=m.RUNTIME,controller=m.CONTROLLER,
        test_results=m.RESULTS,resume_control=point(m.CONTROL),resume_control_lock=point(m.CONTROL_LOCK),
        validation_release=point(m.VAL_RELEASE),test_source_inputs=point(m.SOURCE_INPUTS),
        resource_qualification=point(m.QUALIFICATION),resource_qualifier=point(m.QUALIFIER_SOURCE),
        policy_sha256=policy,author_outcome_protocol=protocol,source_bindings={name:point(name) for name in m.SOURCES},
        legacy_schedule={key:point(m.OLD_REPORT+'/'+name) for key,name in
            (('status','status.json'),('chat_ready','chat_ready.json'),('history','history.jsonl'))},**fixed)
    frozen=dict(schema=m.LOCK_SCHEMA,approved_for_observation=True,
        files={name:m.sha(raw) for name,raw in payload.items()})
    return dict(config=config,frozen=frozen,payload=payload)


def source_reader(fixture,*,allow=None):
    payload=dict(fixture['payload'])
    config_raw=json.dumps(fixture['config'],sort_keys=True).encode()
    frozen=deepcopy(fixture['frozen']);frozen['files'][m.STATE+'/source.json']=m.sha(config_raw)
    frozen_raw=json.dumps(frozen,sort_keys=True).encode()
    payload.update({m.STATE+'/source.json':config_raw,m.STATE+'/source_inputs.json':frozen_raw})
    reads=[]
    def read(name):
        if allow is not None and name not in allow:raise AssertionError('gate opened forbidden metadata: '+name)
        reads.append(name)
        if name not in payload:raise FileNotFoundError(name)
        return payload[name]
    return m.sha(frozen_raw),read,reads


class SourceGateTests(unittest.TestCase):
    LOAD_ONLY={m.STATE+'/source.json',m.STATE+'/source_inputs.json',*m.SOURCES}

    def load(self,fixture,allow=None):
        checksum,read,reads=source_reader(fixture,allow=self.LOAD_ONLY if allow is None else allow)
        return m.load_source(checksum,read_bytes=read),reads

    def test_load_source_reads_only_exact_code_config_and_small_observer_lock(self):
        fixture=source_fixture();bundle,reads=self.load(fixture)
        self.assertEqual(set(reads),self.LOAD_ONLY)
        self.assertEqual(bundle['config']['policy_sha256'],'e'*64)
        for key,name in m.FIXED_METADATA.items():self.assertEqual(bundle['config'][key]['path'],name)
        self.assertNotIn(m.SOURCE_INPUTS,reads)
        self.assertNotIn(m.RESULTS,reads)
        self.assertFalse(any(name.startswith(m.RUNTIME+'/') for name in reads))

    def test_dynamic_or_model_manifest_extra_lock_entries_fail_before_open(self):
        for name in (m.RUNTIME+'/status.json',m.CONTROLLER+'/status.json',m.RESULTS,
                     'artifacts/framework/multitask_r10/models/model.json',
                     'artifacts/framework/protocol/test_manifest.json','outputs/case.json'):
            fixture=source_fixture();fixture['frozen']['files'][name]='a'*64
            with self.subTest(name=name),self.assertRaises(ValueError):self.load(fixture)

    def test_each_fixed_metadata_field_refuses_arbitrary_JSON_or_wrong_fixed_path(self):
        for key in m.FIXED_METADATA:
            fixture=source_fixture();fixture['config'][key]=ptr('artifacts/framework/multitask_r10/models/model.json')
            with self.subTest(field=key),self.assertRaisesRegex(ValueError,'wrong fixed metadata source'):
                self.load(fixture)
        fixture=source_fixture()
        fixture['config']['standing_continuation_record']=deepcopy(fixture['config']['concurrency_authorization_record'])
        with self.assertRaisesRegex(ValueError,'wrong fixed metadata source'):self.load(fixture)

    def test_changed_source_bytes_or_missing_required_source_lock_entry_refused(self):
        fixture=source_fixture();fixture['payload'][m.AUDITOR_SOURCE]=b'changed fake auditor source'
        with self.assertRaisesRegex(ValueError,'frozen metadata SHA changed'):self.load(fixture)
        fixture=source_fixture();fixture['frozen']['files'].pop(m.AUDITOR_SOURCE)
        with self.assertRaises((ValueError,KeyError)):self.load(fixture)
        fixture=source_fixture();fixture['config']['source_bindings'].pop(m.AUDITOR_SOURCE)
        with self.assertRaisesRegex(ValueError,'complete small source/control binding map'):self.load(fixture)

    def test_failed_VAL_gate_cannot_read_control_source640_or_new_owner(self):
        fixture=source_fixture(val_permitted=False);bundle,_=self.load(fixture)
        allowed=self.LOAD_ONLY|{m.VAL_RELEASE}
        _,read,reads=source_reader(fixture,allow=allowed)
        with self.assertRaisesRegex(ValueError,'actual complete824 VAL release'):
            m.collect(bundle,read_bytes=read)
        self.assertIn(m.VAL_RELEASE,reads)
        self.assertNotIn(m.CONTROL,reads)
        self.assertNotIn(m.SOURCE_INPUTS,reads)
        self.assertFalse(any(name.startswith(m.RUNTIME+'/') for name in reads))

    def test_failed_resource_qualification_cannot_read_controller_or_new_owner_or_source640(self):
        fixture=source_fixture(qualification_complete=False);bundle,_=self.load(fixture)
        allowed=self.LOAD_ONLY|{m.VAL_RELEASE,m.CONTROL,m.CONTROL_LOCK,m.QUALIFICATION,
            m.FIXED_METADATA['standing_continuation_record'],m.FIXED_METADATA['concurrency_authorization_record']}
        _,read,reads=source_reader(fixture,allow=allowed)
        with self.assertRaisesRegex(ValueError,'real independent six-TRAIN-case resource qualification'):
            m.collect(bundle,read_bytes=read)
        self.assertIn(m.QUALIFICATION,reads)
        self.assertNotIn(m.SOURCE_INPUTS,reads)
        self.assertFalse(any(name.startswith(m.RUNTIME+'/') or name.startswith(m.CONTROLLER+'/') for name in reads))


class FinalGateTests(unittest.TestCase):
    def gate(self,fixture,reader):
        return m.final_gate(fixture['data'],fixture['jobs'],fixture['terminal'],
            fixture['owner_launch'],fixture['control'],reader)

    def test_full1920_writer_shaped_metadata_releases_all_native_false_without_case_reads(self):
        fixture=final_fixture();reader,reads=final_reader(fixture)
        self.assertTrue(self.gate(fixture,reader))
        self.assertEqual(sum(fixture['data']['completed'].values()),1920)
        self.assertEqual(sum(fixture['data']['success'].values()),0)
        self.assertEqual((fixture['data']['imported_completed'],fixture['data']['new_completed']),(3,1917))
        self.assertIn(m.RESULTS,reads)
        self.assertFalse(any(Path(name).name in ('case.json','admission_driver_receipt.json') for name in reads))

    def test_old_audit_cannot_release_changed_status_or_missing_input_SHA(self):
        for mutation in ('current_status_changed','wrong_status_SHA','missing_VAL_SHA'):
            fixture=final_fixture()
            if mutation=='current_status_changed':
                fixture['values'][m.RUNTIME+'/status.json']['updated_utc']='invented later status bytes'
            elif mutation=='wrong_status_SHA':
                fixture['result']['inputs_sha256'][str(m.ROOT/m.RUNTIME/'status.json')]='0'*64
            else:fixture['result']['inputs_sha256'].pop(str(m.ROOT/m.VAL_RELEASE))
            reader,_=final_reader(fixture)
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'exact old/new metadata bytes'):
                self.gate(fixture,reader)

    def test_wrong_source_or_relabelled_tier_or_case_pointer_never_releases(self):
        for mutation in ('auditor_source','imported_runtime','imported_flag','case','driver'):
            fixture=final_fixture();result=fixture['result']
            if mutation=='auditor_source':result['source_sha256']='0'*64
            elif mutation=='imported_runtime':result['evidence_tiers'][0]['evidence_runtime']=m.RUNTIME
            elif mutation=='imported_flag':result['evidence_tiers'][0]['imported_completed']=False
            else:
                field='case' if mutation=='case' else 'admission_driver_receipt'
                result['evidence_tiers'][0][field]=ptr('artifacts/framework/multitask_r10/unrelated/'+field+'.json')
            reader,_=final_reader(fixture)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.gate(fixture,reader)

    def test_nonzero_or_missing_actual_audit_stage_prevents_release(self):
        for mutation in ('nonzero','incomplete','missing_launch','missing_terminal','missing_stdout'):
            fixture=final_fixture();leaf=fixture['audit_leaf'];values=fixture['values']
            if mutation=='nonzero':values[leaf+'/terminal.json']['returncode']=1
            elif mutation=='incomplete':values[leaf+'/terminal.json']['complete']=False
            else:
                filename={'missing_launch':'launch.json','missing_terminal':'terminal.json','missing_stdout':'stdout.log'}[mutation]
                values.pop(leaf+'/'+filename)
            reader,_=final_reader(fixture)
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,FileNotFoundError)):
                self.gate(fixture,reader)

    def test_actual_owner_or_auditor_PID_start_and_count_must_match(self):
        for mutation in ('old_PID','new_start','new_count','auditor_PID'):
            fixture=final_fixture();result=fixture['result']
            if mutation=='old_PID':result['actual_stage_owners'][0]['pid']+=1
            elif mutation=='new_start':result['actual_stage_owners'][1]['start_ticks']+=1
            elif mutation=='new_count':result['actual_stage_owners'][1]['completed']-=1
            else:result['auditor_invocation']['pid']+=1
            reader,_=final_reader(fixture)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.gate(fixture,reader)

    def test_complete_owner_without_controller_final_still_is_not_released(self):
        fixture=final_fixture();fixture['values'].pop(m.RESULTS)
        reader,_=final_reader(fixture)
        self.assertFalse(self.gate(fixture,reader))


def legacy_collect_fixture(*, change=None, stale_launch_pointer=False):
    """Real legacy writer shape: the plan omits protocol, its launch records it.

    All values are invented in memory; no experiment outputs are inspected.
    Rebinding a negative launch models an already-frozen but invalid record,
    while stale_launch_pointer exercises exact byte-SHA enforcement separately.
    """
    fixture=source_fixture();payload=fixture['payload'];config=fixture['config']
    immutable=set(fixture['frozen']['files'])
    def put(name,value):payload[name]=json.dumps(value,sort_keys=True).encode()
    def point(name):return dict(path=name,sha256=m.sha(payload[name]))
    policy=config['policy_sha256'];protocol=config['author_outcome_protocol']
    old=grid();old['policy_sha256']=policy
    plan,partition=resumed(old)
    plan.update(policy_sha256=policy,author_outcome_protocol=protocol)
    records={row['job_id']:complete(row) for row in plan['jobs'][:6]}
    old_status=dict(completed={key:records[key] for key in partition['completed_job_ids']})
    old_launch=dict(pid=7530,start_ticks=75300,policy_sha256=policy,author_outcome_protocol=protocol,
        source_sha256=config['source_bindings'][m.SOURCES[11]]['sha256'])
    old_launch_name=m.OLD_RUNTIME+'/launch.json'
    put(old_launch_name,old_launch);launch_pointer=point(old_launch_name)
    if change is not None:change(old,old_launch)
    put(old_launch_name,old_launch)
    put(m.OLD_RUNTIME+'/plan.json',old);put(m.OLD_RUNTIME+'/status.json',old_status)
    source_lock=dict(schema='r10_full16_post_admission_split_source_lock_v1',split='test',source_count=640,
        policy_sha256=policy,TEST_used_for_fitting=False,source_content_used_for_policy_fitting=False,
        complete_VAL_release_observed=True,source_manifest_sha256='c'*64,
        source_lock_preparer_sha256=config['source_bindings'][m.SOURCES[7]]['sha256'])
    put(m.SOURCE_INPUTS,source_lock)
    control=json.loads(payload[m.CONTROL])
    control.update(partition,previous_plan=point(m.OLD_RUNTIME+'/plan.json'),
        previous_status=point(m.OLD_RUNTIME+'/status.json'),
        previous_launch=launch_pointer if stale_launch_pointer else point(old_launch_name),
        source_inputs=point(m.SOURCE_INPUTS))
    put(m.CONTROL,control)
    execution_lock=json.loads(payload[m.CONTROL_LOCK])
    execution_lock['files'][m.CONTROL]=point(m.CONTROL)['sha256']
    execution_lock['files'][old_launch_name]=control['previous_launch']['sha256']
    put(m.CONTROL_LOCK,execution_lock)
    config.update(resume_control=point(m.CONTROL),resume_control_lock=point(m.CONTROL_LOCK),
        test_source_inputs=point(m.SOURCE_INPUTS))
    def environment(pid):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,
            checked_OS_files=6418,all_current_OS_targets_full_byte_SHA_verified=True,
            host_PID_network_namespace_preserved=True,proc_or_dev_replaced=False,
            task_parameters_or_strategy_changed=False,project_writable=True,
            wrapper=config['environment_wrapper'],manifest=config['environment_manifest'],
            namespace_qualification=config['environment_namespace_proof'],
            current_mount_namespace='mnt:[invented_private]',mount_namespace='mnt:[invented_private]',
            reference_mount_namespace='mnt:[invented_host]',qualified_host_mount_namespace='mnt:[invented_host]',
            observer_pid=pid,pid=pid,observer_start_ticks=pid*10,outer_wrapper_pid=8000,
            outer_wrapper_start_ticks=80000,current_OS_target_SHA_map_sha256='a'*64,
            restored_four_target_SHA256={name:'a'*64 for name in m.legacy.RESTORED_TARGETS},source_locks={})
    env=environment(9070)
    put(m.CONTROLLER+'/launch.json',dict(pid=9070,start_ticks=90700,frozen_environment=env,
        source_sha256=config['source_bindings'][m.CONTROLLER_SOURCE]['sha256'],
        resume_control=config['resume_control'],resume_control_lock=config['resume_control_lock']))
    put(m.CONTROLLER+'/status.json',dict(stage='formal_TEST_1920',frozen_environment=env,automatic_retry=False))
    owner_launch=dict(pid=9020,start_ticks=90200,policy_sha256=policy,author_outcome_protocol=protocol,
        source_sha256=config['source_bindings'][m.OWNER_SOURCE]['sha256'],
        split_source_lock_sha256=config['test_source_inputs']['sha256'],
        source_manifest=dict(sha256=source_lock['source_manifest_sha256']),resource_profile=m.PROFILE,
        resume_control=config['resume_control'],resume_control_lock=config['resume_control_lock'],
        frozen_environment=environment(9020),argv=[m.OWNER_SOURCE,'--invented'])
    put(m.RUNTIME+'/launch.json',owner_launch);put(m.RUNTIME+'/plan.json',plan)
    put(m.RUNTIME+'/status.json',dict(split='test',expected=1920,policy_sha256=policy,
        author_outcome_protocol=protocol,completed=records,failed={},active=[],checking=[]))
    put(m.CONTROLLER+'/stages/formal_TEST_1920/launch.json',dict(pid=9020,start_ticks=90200,
        source_sha256=owner_launch['source_sha256'],inherited_frozen_environment=env,
        current_mount_namespace=env['current_mount_namespace'],
        argv=[str(m.ROOT/'.conda-env/bin/python'),'-B','-u',*owner_launch['argv']]))
    fixture['frozen']['files']={name:m.sha(payload[name]) for name in immutable}
    return fixture


class LegacyPolicyBindingTests(unittest.TestCase):
    def collect_fixture(self,fixture):
        checksum,read,reads=source_reader(fixture)
        bundle=m.load_source(checksum,read_bytes=read)
        return m.collect(bundle,read_bytes=read),reads

    def test_actual_legacy_plan_omits_protocol_and_frozen_launch_supplies_it(self):
        fixture=legacy_collect_fixture()
        self.assertNotIn('author_outcome_protocol',json.loads(fixture['payload'][m.OLD_RUNTIME+'/plan.json']))
        result,reads=self.collect_fixture(fixture)
        self.assertTrue(result['admitted']);self.assertTrue(result['plan_valid'])
        self.assertEqual(sum(result['completed'].values()),6)
        self.assertEqual(sum(result['success'].values()),0)
        self.assertEqual((result['imported_completed'],result['new_completed']),(3,3))
        self.assertFalse(result['release_complete'])
        self.assertIn(m.OLD_RUNTIME+'/launch.json',reads)
        self.assertFalse(any(Path(name).name in ('case.json','admission_driver_receipt.json') for name in reads))

    def test_missing_or_wrong_frozen_legacy_launch_protocol_is_rejected(self):
        for mutation in ('missing','wrong'):
            def change(plan,launch):
                if mutation=='missing':launch.pop('author_outcome_protocol')
                else:launch['author_outcome_protocol']=dict(wrong='protocol')
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'policy/protocol'):
                self.collect_fixture(legacy_collect_fixture(change=change))

    def test_missing_or_wrong_legacy_launch_source_SHA_is_rejected(self):
        for mutation in ('missing','wrong'):
            def change(plan,launch):
                if mutation=='missing':launch.pop('source_sha256')
                else:launch['source_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'old753 launch/plan policy source'):
                self.collect_fixture(legacy_collect_fixture(change=change))

    def test_legacy_plan_policy_is_still_required_and_cannot_disagree(self):
        for mutation in ('missing','wrong'):
            def change(plan,launch):
                if mutation=='missing':plan.pop('policy_sha256')
                else:plan['policy_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'old753 launch/plan policy source'):
                self.collect_fixture(legacy_collect_fixture(change=change))

    def test_wrong_legacy_launch_policy_is_rejected_even_if_plan_is_correct(self):
        def change(plan,launch):launch['policy_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'policy/protocol'):
            self.collect_fixture(legacy_collect_fixture(change=change))

    def test_changed_legacy_launch_bytes_cannot_rebind_frozen_control_pointer(self):
        def change(plan,launch):launch['invented_later_edit']='changed bytes'
        with self.assertRaisesRegex(ValueError,'frozen metadata SHA changed'):
            self.collect_fixture(legacy_collect_fixture(change=change,stale_launch_pointer=True))

    def test_wrong_previous_launch_path_is_rejected_before_open(self):
        fixture=legacy_collect_fixture()
        control=json.loads(fixture['payload'][m.CONTROL])
        control['previous_launch']=ptr(m.RUNTIME+'/launch.json')
        fixture['payload'][m.CONTROL]=json.dumps(control,sort_keys=True).encode()
        checksum=m.sha(fixture['payload'][m.CONTROL])
        fixture['config']['resume_control']['sha256']=checksum
        lock=json.loads(fixture['payload'][m.CONTROL_LOCK]);lock['files'][m.CONTROL]=checksum
        fixture['payload'][m.CONTROL_LOCK]=json.dumps(lock,sort_keys=True).encode()
        fixture['config']['resume_control_lock']['sha256']=m.sha(fixture['payload'][m.CONTROL_LOCK])
        for name in (m.CONTROL,m.CONTROL_LOCK):fixture['frozen']['files'][name]=m.sha(fixture['payload'][name])
        # The wrong path is already allowed for the new owner but must not be
        # accepted as evidence for the exact previous753 launch.
        with self.assertRaises(ValueError):self.collect_fixture(fixture)


if __name__=='__main__':unittest.main()
