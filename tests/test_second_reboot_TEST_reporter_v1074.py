"""Invented metadata/temp-only reporter checks; no experiment admission runs."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import report_second_reboot_TEST_halfhour_v1074_r10 as m
from our_mani.multitask import composite_TEST_reboot_resume_v1070 as core
PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixture1070_for1074',Path(__file__).with_name('test_composite_TEST_reboot_resume_v1070.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def ptr(name):return dict(path=name,sha256='a'*64)


class Fixture:
    def __init__(self,full=False,extra=2):
        self.store={};self.reads=[];self.root=Path('/tmp/invented_report1074_metadata')
        c,parent,status,audit,*_=f.fixture();self.control=c;self.parent_status=status
        for name in m.SOURCES:self.store[name]=(PROJECT/name).read_bytes()
        self.bindings={name:self.point(name) for name in m.SOURCES}
        c.update(suite_lock=ptr('CPU_only/suite_lock.json'),author_outcome_protocol=dict(schema='r10_author_action_outcome_protocol_v741',native_RML_parameters_unchanged=True,pose_action_max_ticks=10,public_hold_max_ticks=1),resume_owner=self.bindings[m.OWNER_SOURCE],resume_auditor=self.bindings[m.AUDITOR_SOURCE],
            result_auditor=self.bindings[m.AUDITOR_SOURCE],resume_controller=self.bindings[m.CONTROLLER_SOURCE],
            resource_profile=dict(schema='r10_frozen_concurrent_native_resource_profile_v1031',max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4,ready_limit=16),
            resource_capacity_source='qualified1031',resource_profile_within_capacity=True,resource_profile_changed=True,
            resource_qualification=ptr('runtime/framework/concurrent_TRAIN_resource_qualification_v1031.json'),
            resource_qualifier=ptr('scripts/audit_concurrent_TRAIN_resource_canary_v1031_r10.py'),
            environment_manifest=ptr('CPU_only/manifest.json'),environment_wrapper=ptr('CPU_only/wrapper.py'),
            environment_namespace_proof=ptr('CPU_only/namespace.json'),source_inputs=ptr('CPU_only/source_lock.json'),
            validation_release=ptr('CPU_only/VAL_release.json'),execution_implementation={'CPU':'same901'})
        c['resource_capacity_envelope']=dict(c['resource_profile'],max_native_slots=12,max_GPU_slots=5,max_CPU_preflight_slots=6,ready_limit=24);c['previous_resource_profile']=deepcopy(c['resource_capacity_envelope'])
        c['previous_resource_qualification']=deepcopy(c['resource_qualification'])
        for row in parent['jobs']:
            if row['job_id'] in status['completed']:status['completed'][row['job_id']].update(self.pointers(row))
        self.put(m.OLD+'/plan.json',parent);self.put(m.OLD+'/status.json',status)
        audit.update(source=self.bindings[m.AUDIT_SOURCE]);self.put(m.REBOOT,audit);self.boot_id=audit['current_boot_id']
        self.put(m.BOOT,dict(schema='reboot_hardware_boot_point_observation_v1068',boot_id=self.boot_id,btime_unix=1792670782,
            uptime_seconds=556,clock_ticks_per_second=100,old_owner_created_before_current_boot=True,old_start_ticks_impossible_in_current_boot=True))
        proof=dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v954',complete=True,source=self.bindings[m.CHECK_SOURCE],
            Native_started=False,GPU_started=False,admission_or_TEST_started=False,declared_model_case_raw_bytes_read=False,
            scientific_inputs_or_old_locks_modified=False,original_v900_topology_constructor_passed=True,
            original_full_topology_final_check_passed=True,parsed_suite_lock_is_exact_hashed_payload=True,
            topology_is_point_observation_not_continuous=True,suite_lock=c['suite_lock'],declared_files=10,
            unique_canonical_paths=10,watched_directories=4,actual_environment=self.env(954,9540,c))
        self.put(m.RECHECK,proof)
        for key,name in [('previous_plan',m.OLD+'/plan.json'),('previous_status',m.OLD+'/status.json'),
                ('reboot_audit',m.REBOOT),('host_reboot_observation',m.BOOT),('environment_recheck',m.RECHECK)]:c[key]=self.point(name)
        self.put(m.CONTROL,c);self.store[m.CONTROL_LOCK]=b'{"large_frozen_science_map":"never parsed or traversed"}'
        cfgptr=dict(resume_control=self.point(m.CONTROL),resume_control_lock=self.point(m.CONTROL_LOCK))
        self.common=dict(split='test',policy_sha256=c['policy_sha256'],author_outcome_protocol=c['author_outcome_protocol'],
            automatic_retry=False,resource_profile=c['resource_profile'],reboot_audit=self.point(m.REBOOT),boot_id=self.boot_id,
            infrastructure_recovery=True,**cfgptr)
        rows=core.resumed_jobs(parent,c,c['execution_entry'])
        self.plan=dict(self.common,jobs=rows,expected_cases=1920,imported_completed_cases=1419,source_outcome_filtering=False)
        self.owner=dict(self.common,pid=1071,start_ticks=10710,source_sha256=self.bindings[m.OWNER_SOURCE]['sha256'],
            argv=[m.OWNER_SOURCE,'--resume-control',m.CONTROL],frozen_environment=self.env(1071,10710,c))
        self.ctl=dict(pid=1073,start_ticks=10730,source_sha256=self.bindings[m.CONTROLLER_SOURCE]['sha256'],automatic_retry=False,
            frozen_environment=self.env(1073,10730,c),**cfgptr)
        done={k:deepcopy(status['completed'][k]) for k in c['completed_job_ids']}
        remaining=[r for r in rows if r['job_id'] not in done]
        for i,row in enumerate(remaining if full else remaining[:extra]):
            done[row['job_id']]=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False,
                native_success=bool(i%2),motion_pass=None if i%3==0 else False,**self.pointers(row))
        self.status=dict(self.common,expected=1920,completed=done,failed={},active=[],checking=[],ready_jobs=[],
            CPU_ready=0,native_slots=0,GPU_slots=0,owner_error=None,drain_requested=False,stop_requested=False,updated_utc='synthetic',imported_completed=1419)
        for name,value in [('launch',self.owner),('plan',self.plan),('status',self.status)]:self.put(m.RUNTIME+'/'+name+'.json',value)
        self.put(m.CONTROLLER+'/launch.json',self.ctl);self.put(m.CONTROLLER+'/status.json',dict(stage='complete' if full else 'formal_TEST_1920'))
        self.put(m.CONTROLLER+'/stages/formal_TEST_1920/launch.json',self.stage(m.OWNER_SOURCE,1071,10710,self.owner['argv']))
        oldlaunch=dict(pid=10600,start_ticks=156000,source_sha256=self.bindings[m.LEGACY_IDENTITY_SOURCE]['sha256'],created_utc='2026-10-22T11:55:37+00:00',observer_only=True,Native_started=False)
        self.put(m.LEGACY+'/launch.json',oldlaunch)
        self.cfg=dict(schema=m.SCHEMA,runtime=m.RUNTIME,controller=m.CONTROLLER,test_results=m.RESULTS,source_bindings=self.bindings)
        self.cfg.update({key:self.point(name) for key,name in m.BINDINGS.items()});self.freeze()
        if full:self.final()
        self.reads.clear()
    def put(self,name,value):self.store[name]=m.encoded(value);return self.point(name)
    def point(self,name):return dict(path=name,sha256=m.sha(self.store[name]))
    def read(self,name):
        self.reads.append(name)
        if name not in self.store:raise FileNotFoundError(name)
        return self.store[name]
    def freeze(self):
        self.put(m.STATE+'/source.json',self.cfg)
        pointers=[self.point(m.STATE+'/source.json'),*self.bindings.values(),*(self.cfg[k] for k in m.BINDINGS)]
        self.put(m.STATE+'/source_inputs.json',dict(schema=m.LOCK_SCHEMA,approved_for_observation=True,files={p['path']:p['sha256'] for p in pointers}))
    def bundle(self):return m.load_source(self.point(m.STATE+'/source_inputs.json')['sha256'],root=self.root,read_bytes=self.read)
    def collect(self):return m.collect(self.bundle(),root=self.root,read_bytes=self.read,alive=lambda _:False,boot_id=self.boot_id)
    def pointers(self,row):
        out=core.option(row['physical_argv'],'--out')
        return dict(case=ptr(out+'/case.json'),admission_driver_receipt=ptr(out+'/admission_driver_receipt.json'))
    def env(self,pid,start,c):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,checked_OS_files=6418,
            all_current_OS_targets_full_byte_SHA_verified=True,observer_pid=pid,observer_start_ticks=start,
            current_mount_namespace='mnt:[synthetic]',manifest=c['environment_manifest'],wrapper=c['environment_wrapper'],
            namespace_qualification=c['environment_namespace_proof'])
    def stage(self,source,pid,start,args):
        return dict(pid=pid,start_ticks=start,source_sha256=self.bindings[source]['sha256'],automatic_retry=False,
            argv=[str(self.root/'.conda-env/bin/python'),'-B','-u',*args],inherited_frozen_environment=self.ctl['frozen_environment'])
    def final(self):
        terminal=dict(self.common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,stop_requested=False,
            drain_requested=False,drained=False,CPU_not_launched=0,CPU_ready_unexecuted=[],imported_completed_cases=1419,resumed_new_cases=501)
        self.put(m.RUNTIME+'/terminal.json',terminal)
        for stage in ('formal_TEST_1920','independent_TEST_results'):
            self.put(m.CONTROLLER+'/stages/'+stage+'/terminal.json',dict(complete=True,returncode=0,automatic_retry=False))
        args=[m.AUDITOR_SOURCE,'--resume-control',m.CONTROL]
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/launch.json',self.stage(m.AUDITOR_SOURCE,1072,10720,args))
        totals={method:sum(record['native_success'] is True for key,record in self.status['completed'].items()
            if next(row for row in self.plan['jobs'] if row['job_id']==key)['method']==method) for method in m.METHODS}
        records=[self.status['completed'][row['job_id']] for row in self.plan['jobs']]
        fields=('resource_profile','resource_qualification','resource_qualifier','resource_capacity_envelope','resource_profile_within_capacity',
            'resource_capacity_source','resource_profile_changed','previous_resource_profile','previous_resource_qualification','execution_implementation','environment_recheck')
        result=dict(schema=m.RESULT_SCHEMA,complete=True,split='test',source_cases=640,method_cases=1920,completed_method_cases=1920,
            methods=list(m.METHODS),tasks=list(core.TASKS),source_sha256=self.bindings[m.AUDITOR_SOURCE]['sha256'],
            policy_sha256=self.control['policy_sha256'],author_outcome_protocol=self.control['author_outcome_protocol'],
            resume_control=self.cfg['resume_control'],resume_control_lock=self.cfg['resume_control_lock'],source_input_lock=self.control['source_inputs'],
            validation_release=self.control['validation_release'],reboot_audit=self.cfg['reboot_audit'],host_reboot_observation=self.cfg['host_reboot_observation'],
            independent_composite_audit=True,used_evidence_rechecked_at_close=True,native_failures_retained=True,motion_unknown_retained=True,
            zero_tick_cases_retained=True,motion_is_diagnostic_only=True,adopted_completed_results_not_rerun=True,
            automatic_retry=False,source_outcome_filtering=False,Value_NGFF_FASTER_used=False,native_successes_by_method=totals,
            auditor_invocation=dict(pid=1072,start_ticks=10720,source=self.bindings[m.AUDITOR_SOURCE],argv=args),
            cases=[r['case'] for r in records],admission_driver_receipts=[r['admission_driver_receipt'] for r in records],
            evidence_tiers=[dict(job_id=row['job_id'],evidence_runtime=row['evidence_runtime'],case=record['case'],
                admission_driver_receipt=record['admission_driver_receipt'],**{k:row[k] for k in ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','resume_lineage_v1055','resume_lineage_v1070')})
                for row,record in zip(self.plan['jobs'],records)],**{k:self.control[k] for k in fields})
        counts=Counter(row['evidence_runtime'] for row in self.plan['jobs'])
        result['actual_stage_owners']=[dict(runtime=runtime,completed=counts[runtime],pid=1071 if runtime==m.RUNTIME else i+1,
            start_ticks=10710 if runtime==m.RUNTIME else i+100,launch=self.point(m.RUNTIME+'/launch.json') if runtime==m.RUNTIME else ptr(runtime+'/launch.json'),
            terminal=None if runtime in m.INTERRUPTED else self.point(m.RUNTIME+'/terminal.json') if runtime==m.RUNTIME else ptr(runtime+'/terminal.json')) for i,runtime in enumerate(m.TIERS)]
        self.put(m.RESULTS,result);self.put(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=self.point(m.RESULTS)))
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/stdout.log',dict(complete=True,schema=m.RESULT_SCHEMA,split='test',cases=1920,out=m.RESULTS,native_successes_by_method=totals))


class ReporterTests(unittest.TestCase):
    def test_default_inactive_and_external_pin_required(self):
        with patch.object(m,'load_source',side_effect=AssertionError('inactive')),redirect_stdout(io.StringIO()) as out:m.main([])
        self.assertIn('"active": false',out.getvalue())
        with self.assertRaises(ValueError):m.load_source(None,read_bytes=lambda _:b'')
    def test_fixed_phase_never_resets_and_boundary_is_strictly_future(self):
        for now in (0,501.384754,501.4,1800,1792668215.123):
            due=m.phase_deadline(now);self.assertGreater(due,now);self.assertTrue(m.is_phase(due));self.assertLessEqual(due-now,1800.000001)
    def test_metadata_only_1421_runtime_counts_nativefalse_and_motionunknown(self):
        fixture=Fixture();data=fixture.collect()
        self.assertTrue(data['admitted']);self.assertFalse(data['release_complete'])
        self.assertEqual(sum(data['completed'].values()),1421);self.assertGreater(sum(data['unknown_motion'].values()),0)
        self.assertLess(sum(data['success'].values()),1421)
        self.assertTrue(set(fixture.reads)<=m.allowed_names())
        self.assertFalse(any(name.endswith('/case.json') or '/CPU/' in name or '/native/' in name for name in fixture.reads))
        self.assertTrue(data['legacy_interruption']['interrupted_by_host_reboot'])
    def test_full_status_without_independent_terminal_stays_partial(self):
        fixture=Fixture(full=True);fixture.put(m.CONTROLLER+'/status.json',dict(stage='formal_TEST_1920'))
        for name in (m.CONTROLLER+'/terminal.json',m.RESULTS):fixture.store.pop(name)
        data=fixture.collect();self.assertEqual(sum(data['completed'].values()),1920);self.assertFalse(data['release_complete'])
    def test_actual_full_independent_terminal_grid_passes(self):self.assertTrue(Fixture(full=True).collect()['release_complete'])
    def test_malformed_final_result_or_stage_cannot_claim_release(self):
        for mode in ('partial','wrong_schema','wrong_pointer','bad_stage','invented_oldterminal','invented_first_oldterminal'):
            fixture=Fixture(full=True)
            if mode=='bad_stage':fixture.put(m.CONTROLLER+'/stages/independent_TEST_results/terminal.json',dict(complete=True,returncode=1,automatic_retry=False))
            else:
                result=json.loads(fixture.store[m.RESULTS])
                if mode=='partial':result['completed_method_cases']=1919
                elif mode=='wrong_schema':result['schema']='partial_saved_status'
                elif mode=='wrong_pointer':result['cases'][0]=ptr('outside/case.json')
                else:
                    runtime=m.FIFTH if mode=='invented_first_oldterminal' else m.OLD
                    next(x for x in result['actual_stage_owners'] if x['runtime']==runtime)['terminal']=ptr(runtime+'/terminal.json')
                fixture.put(m.RESULTS,result);fixture.put(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=fixture.point(m.RESULTS)))
            with self.subTest(mode=mode),self.assertRaises(ValueError):fixture.collect()
    def test_whitelist_refuses_raw_case_model_and_pointer_expansion(self):
        reader=m.Metadata(read_bytes=lambda _:self.fail('read forbidden bytes'))
        for name in ('data/raw.hdf5','artifacts/model.pt',m.OLD+'/native/episode/case.json','../secrets'):
            with self.subTest(name=name),self.assertRaises(ValueError):reader.bytes(name)
        fixture=Fixture();fixture.cfg['owner_plan']=ptr('data/raw.hdf5');fixture.freeze()
        with self.assertRaises(ValueError):fixture.bundle()
    def test_legacy_reboot_requires_real_dead_identity_and_same_boot(self):
        fixture=Fixture();bundle=fixture.bundle()
        with self.assertRaises(ValueError):m.collect(bundle,root=fixture.root,read_bytes=fixture.read,alive=lambda _:True,boot_id=fixture.boot_id)
        with self.assertRaises(ValueError):m.collect(bundle,root=fixture.root,read_bytes=fixture.read,alive=lambda _:False,boot_id='another boot')
    def test_legacy_direct_start_ticks_and_identity_source_are_not_old_helper_or_checkpoint(self):
        for mode in ('missing_ticks','helper_source','post_boot_creation','zero_pid'):
            fixture=Fixture();value=json.loads(fixture.store[m.LEGACY+'/launch.json'])
            if mode=='missing_ticks':value.pop('start_ticks')
            elif mode=='helper_source':value['source_sha256']=fixture.bindings[m.HELPER_SOURCE]['sha256']
            elif mode=='post_boot_creation':value['created_utc']='2026-10-22T12:10:00+00:00'
            else:value['pid']=0
            fixture.put(m.LEGACY+'/launch.json',value)
            fixture.cfg['legacy_reporter_launch']=fixture.point(m.LEGACY+'/launch.json');fixture.freeze()
            with self.subTest(mode=mode),self.assertRaises(ValueError):fixture.collect()
        fixture=Fixture();fixture.cfg['legacy_reporter_checkpoint']=ptr('runtime/framework/latest_executor_continuation_v1039.json');fixture.freeze()
        with self.assertRaises(ValueError):fixture.bundle()
        reader=m.Metadata(read_bytes=lambda _:self.fail('identity source must never execute'))
        with self.assertRaises(ValueError):m.load_module(reader,m.LEGACY_IDENTITY_SOURCE)
        self.assertNotIn('runtime/framework/latest_executor_continuation_v1039.json',m.allowed_names())

    def test_actual_lower_quota_keeps_full1031_capacity_and_science_lock_is_bytes_only(self):
        fixture=Fixture();fixture.store[m.CONTROL_LOCK]=b'non-JSON full scientific input map stays opaque'
        fixture.cfg['resume_control_lock']=fixture.point(m.CONTROL_LOCK)
        for name in (m.RUNTIME+'/launch.json',m.RUNTIME+'/plan.json',m.RUNTIME+'/status.json',m.CONTROLLER+'/launch.json'):
            value=json.loads(fixture.store[name]);value['resume_control_lock']=fixture.cfg['resume_control_lock'];fixture.put(name,value)
        for key,name in (('owner_launch',m.RUNTIME+'/launch.json'),('owner_plan',m.RUNTIME+'/plan.json'),('controller_launch',m.CONTROLLER+'/launch.json')):
            fixture.cfg[key]=fixture.point(name)
        fixture.freeze();data=fixture.collect()
        self.assertEqual(data['profile']['schema'],'r10_frozen_concurrent_native_resource_profile_v1031')
        self.assertEqual(tuple(data['profile'][k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')),(8,3,4,16))
        self.assertEqual(tuple(data['resource_capacity_envelope'][k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')),(12,5,6,24))
        self.assertEqual(data['legacy_interruption']['identity'],dict(pid=10600,start_ticks=156000))
        self.assertEqual(data['legacy_interruption']['runtime'],m.LEGACY)
        self.assertTrue(all('/case.json' not in name and '/CPU/' not in name and '/native/' not in name for name in fixture.reads))

    def test_unknown_observation_is_none_and_ready_has_no_auto_ack(self):
        fixture=Fixture();bundle=fixture.bundle()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=m.empty_data('unreadable current status')
            history=m.save_report(data,bundle,[],now=1000,next_due=m.phase_deadline(1000),reason='CPU temp',root=root)
            self.assertIsNone(history[0]['total_completed']);self.assertTrue(all(x['completed'] is None for x in history[0]['methods'].values()))
            ready=json.loads((root/m.STATE/'chat_ready.json').read_text());self.assertFalse(ready['chat_sent']);self.assertFalse(ready['delivery_acknowledged'])
            self.assertFalse((root/m.STATE/'chat_ack.json').exists());self.assertFalse((root/m.LEGACY).exists())
    def test_snapshot_writes_only_new_outputs_and_preserves_input_status(self):
        fixture=Fixture();data=fixture.collect();before=deepcopy(fixture.store)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);history=m.save_report(data,fixture.bundle(),[],now=1000,next_due=m.phase_deadline(1000),reason='CPU temp',root=root)
            self.assertEqual(history[0]['total_completed'],1421);self.assertIn('NativeFalse',(root/m.OUT).read_text())
            self.assertEqual(fixture.store,before);self.assertFalse((root/'TEST_16任务进度.txt').exists())
            self.assertTrue((root/m.STATE/'snapshots').is_dir())
    def test_native_unknown_remains_unknown_and_cannot_enter_full_release(self):
        for full in (False,True):
            fixture=Fixture(full=full)
            row=next(r for r in fixture.plan['jobs'] if not r['imported_completed'] and r['job_id'] in fixture.status['completed'])
            fixture.status['completed'][row['job_id']]['native_success']=None
            fixture.put(m.RUNTIME+'/status.json',fixture.status)
            if full:
                with self.assertRaises(ValueError):fixture.collect()
            else:
                data=fixture.collect();self.assertEqual(sum(data['unknown_native'].values()),1)
                self.assertFalse(data['release_complete']);self.assertNotIn(m.RESULTS,fixture.reads)
    def test_small_lock_cannot_expand_to_science_graph(self):
        fixture=Fixture();lock=json.loads(fixture.store[m.STATE+'/source_inputs.json'])
        lock['files']['data/raw.hdf5']='b'*64;fixture.put(m.STATE+'/source_inputs.json',lock)
        with self.assertRaises(ValueError):fixture.bundle()
        self.assertNotIn('data/raw.hdf5',fixture.reads)
    def test_metadata_alias_is_refused_before_reading_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=root/'unlisted_data';target.write_bytes(b'never read')
            source=root/m.CONTROL;source.parent.mkdir(parents=True);source.symlink_to(target)
            with self.assertRaises(ValueError):m.Metadata(root=root).bytes(m.CONTROL)
    def test_observed_infrastructure_error_and_draining_remain_admitted_and_visible(self):
        fixture=Fixture();before_success=sum(v['native_success'] is True for v in fixture.status['completed'].values())
        pending=next(r for r in fixture.plan['jobs'] if r['job_id'] not in fixture.status['completed'])
        fixture.status.update(failed={pending['job_id']:dict(complete=False,automatic_retry=False,stage='native')},
            owner_error='synthetic IO failure',drain_requested=True,stop_requested=True)
        fixture.put(m.RUNTIME+'/status.json',fixture.status)
        fixture.put(m.CONTROLLER+'/status.json',dict(stage='draining_for_concrete_repair',error='synthetic stage repair'))
        data=fixture.collect();self.assertTrue(data['admitted']);self.assertFalse(data['release_complete'])
        self.assertEqual(data['failed'],1);self.assertEqual(sum(data['completed'].values()),1421)
        self.assertEqual(sum(data['success'].values()),before_success)
        for text in ('基础设施失败 1','owner_error: synthetic IO failure','drain_requested=True','stop_requested=True','draining_for_concrete_repair'):
            self.assertTrue(any(text in error for error in data['errors']))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);history=m.save_report(data,fixture.bundle(),[],now=1000,next_due=m.phase_deadline(1000),reason='CPU temp infra',root=root)
            text=(root/m.OUT).read_text()
            for value in ('本轮infra：1','owner_error：synthetic IO failure','drain：True','stop：True','draining_for_concrete_repair'):
                self.assertIn(value,text)
            self.assertEqual(history[0]['current1071_infrastructure_failed'],1)
    def test_only_own_stop_or_global_user_stop(self):
        self.assertEqual(set(m.STOP_PATHS),{'runtime/framework/multitask_r10_STOP',m.STATE+'/STOP'})


if __name__=='__main__':unittest.main()
