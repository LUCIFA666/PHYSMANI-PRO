"""CPU/temp synthetic four-tier reporter, with no actual experiment IO or APIs."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
m=module('report1014_fakes',PROJECT/'scripts/report_composite_TEST_halfhour_v1014_r10.py')
base=module('report959_CPU_fixture',PROJECT/'tests/test_composite_TEST_reporter_v959.py')
CORE_BYTES=(PROJECT/m.CORE).read_bytes()
PARENT_CORE_BYTES=(PROJECT/m.PARENT_CORE).read_bytes()
ptr=base.ptr
outcome=base.outcome

class Fixture:
    def __init__(self,*,full=False,adopted=1200,extra=2):
        old=base.Fixture(extra=adopted-584)
        self.store=deepcopy(old.store);self.reads=[]
        for name in m.SOURCES:
            self.store[name]=CORE_BYTES if name==m.CORE else PARENT_CORE_BYTES if name==m.PARENT_CORE else self.store.get(name,('# synthetic '+name+'\n').encode())
        self.bindings={name:self.point(name) for name in m.SOURCES}
        self.policy=old.policy;self.protocol=old.protocol;self.base=old.base;self.entry=old.entry
        self.original=deepcopy(old.original);self.middle=deepcopy(old.parent)
        middle_control=json.loads(self.store[m.MIDDLE_CONTROL]);middle_control.update(schema='r10_concurrent_resumed_formal_TEST_control_v902',split='test',author_outcome_protocol=self.protocol)
        self.put(m.MIDDLE_CONTROL,middle_control)
        middle_lock=json.loads(self.store[m.MIDDLE_LOCK]);middle_lock.update(schema='r10_concurrent_resumed_formal_TEST_inputs_v902',approved_for_execution=True)
        middle_lock['files'][m.MIDDLE_CONTROL]=self.point(m.MIDDLE_CONTROL)['sha256'];self.put(m.MIDDLE_LOCK,middle_lock)
        old_control=deepcopy(old.control);old_control.update(previous_resume_control=self.point(m.MIDDLE_CONTROL),previous_resume_control_lock=self.point(m.MIDDLE_LOCK))
        self.put(m.OLD_CONTROL,old_control);self.old_control=old_control
        old_lock=json.loads(self.store[m.OLD_LOCK]);old_lock['files'].update(middle_lock['files'])
        for name in (m.OLD_CONTROL,m.MIDDLE_CONTROL,m.MIDDLE_LOCK):old_lock['files'][name]=self.point(name)['sha256']
        self.put(m.OLD_LOCK,old_lock)
        self.parent=dict(launch=deepcopy(old.owner),plan=deepcopy(old.plan),status=deepcopy(old.status))
        self.parent['terminal']=dict(old.common,complete=False,expected=1920,completed=adopted,failed=0,drained=True,
            drain_requested=True,stop_requested=False,automatic_retry=False,policy_sha256=self.policy)
        self.parent['status']['drain_requested']=True
        for value in self.parent.values():
            value.update(resume_control=self.point(m.OLD_CONTROL),resume_control_lock=self.point(m.OLD_LOCK))
        for runtime,records in ((m.ORIGINAL,self.original),(m.MIDDLE,self.middle),(m.PARENT,self.parent)):
            for key,value in records.items():self.put(runtime+'/'+key+'.json',value)
        core=m.load_core(m.Metadata(read_bytes=self.read),dict(source_bindings=self.bindings))
        ids=[r['job_id'] for r in self.parent['plan']['jobs']]
        self.control=dict(deepcopy(old_control),schema=core.CONTROL_SCHEMA,previous_runtime=m.PARENT,
            previous_controller_runtime='runtime/framework/ordered_composite_TEST_v958',
            previous_resume_control=self.point(m.OLD_CONTROL),previous_resume_control_lock=self.point(m.OLD_LOCK),
            resource_profile=deepcopy(m.PROFILE),resource_profile_changed=True,previous_resource_profile=deepcopy(old_control['resource_profile']),
            previous_resource_qualification=deepcopy(old_control['resource_qualification']),resource_qualification=ptr('runtime/framework/synthetic1005.json'),
            completed_job_ids=ids[:adopted],failed_job_ids=[],remaining_job_ids=ids[adopted:],
            resume_owner=self.bindings[m.OWNER_SOURCE],resume_auditor=self.bindings[m.AUDITOR_SOURCE],resume_controller=self.bindings[m.CONTROLLER_SOURCE])
        for key in ('completed','failed','remaining'):self.control[key+'_key_sha256']=m.digest(sorted(self.control[key+'_job_ids']))
        for prefix,runtime,records in (('original_',m.ORIGINAL,self.original),('previous_',m.PARENT,self.parent)):
            for key in records:self.control[prefix+key]=self.point(runtime+'/'+key+'.json')
        self.put(m.CONTROL,self.control)
        pointers=[self.point(m.CONTROL),self.point(m.OLD_CONTROL),self.point(m.OLD_LOCK),*self.bindings.values(),
            *(self.control['previous_'+k] for k in self.parent),*(self.control['original_'+k] for k in self.original)]
        files=deepcopy(old_lock['files']);files.update({p['path']:p['sha256'] for p in pointers})
        self.put(m.CONTROL_LOCK,dict(schema=core.LOCK_SCHEMA,approved_for_execution=True,control=self.point(m.CONTROL),files=files))
        self.retirement()
        self.config=dict(schema=m.SCHEMA,protocol_status='formal_TEST_composite',runtime=m.RUNTIME,controller=m.CONTROLLER,
            test_results=m.RESULTS,source_bindings=self.bindings,policy_sha256=self.policy,author_outcome_protocol=self.protocol,
            resume_control=self.point(m.CONTROL),resume_control_lock=self.point(m.CONTROL_LOCK),legacy_retirement=self.point(m.RETIREMENT),environment_recheck=self.point(m.RECHECK))
        self.freeze_config();self.env=self.environment(301,3001)
        self.common=dict(self.base,split='test',resume_control=self.config['resume_control'],resume_control_lock=self.config['resume_control_lock'],
            resource_profile=m.PROFILE,frozen_environment=self.environment(302,3002),
            **{k:self.control[k] for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck','resource_profile_changed','previous_resource_profile','previous_resource_qualification')})
        self.plan=dict(self.common,expected_cases=1920,imported_completed_cases=adopted,source_outcome_filtering=False,
            jobs=core.resumed_jobs(self.parent['plan'],self.control,self.entry))
        self.owner=dict(self.common,pid=302,start_ticks=3002,source_sha256=self.bindings[m.OWNER_SOURCE]['sha256'],
            argv=[m.OWNER_SOURCE,'--resume-control',m.CONTROL],base_stage_owner=self.bindings[m.ORIGINAL_SOURCE],parent_stage_owner=self.bindings[m.PARENT_SOURCE],
            imported_completed_cases=adopted,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4)
        count=1920 if full else adopted+extra;records={r['job_id']:outcome(r) for r in self.plan['jobs'][:count]}
        records.update(deepcopy(self.parent['status']['completed']))
        self.status=dict(self.common,expected=1920,completed=records,failed={},active=[],checking=[],ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0,
            owner_error=None,drain_requested=False,stop_requested=False,updated_utc='synthetic',imported_completed=adopted)
        for key,value in (('launch',self.owner),('plan',self.plan),('status',self.status)):self.put(m.RUNTIME+'/'+key+'.json',value)
        self.put(m.CONTROLLER+'/launch.json',dict(pid=301,start_ticks=3001,source_sha256=self.bindings[m.CONTROLLER_SOURCE]['sha256'],resume_control=self.config['resume_control'],
            resume_control_lock=self.config['resume_control_lock'],frozen_environment=self.env,automatic_retry=False))
        self.put(m.CONTROLLER+'/status.json',dict(stage='complete' if full else 'formal_TEST_1920',frozen_environment=self.env,automatic_retry=False))
        self.put(m.CONTROLLER+'/stages/formal_TEST_1920/launch.json',self.stage(m.OWNER_SOURCE,302,3002,self.owner['argv']))
        if full:self.complete_result()
        self.reads.clear()

    def point(self,name):return dict(path=name,sha256=m.sha(self.store[name]))
    def put(self,name,value):self.store[name]=m.encoded(value);return self.point(name)
    def read(self,name):
        self.reads.append(name)
        if name not in self.store:raise FileNotFoundError(name)
        return self.store[name]
    def environment(self,pid,start):
        return dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,checked_OS_files=6418,
            all_current_OS_targets_full_byte_SHA_verified=True,observer_pid=pid,observer_start_ticks=start,current_mount_namespace='mnt:[synthetic]',
            manifest=self.control['environment_manifest'],wrapper=self.control['environment_wrapper'],namespace_qualification=self.control['environment_namespace_proof'])
    def stage(self,source,pid,start,argv):
        return dict(pid=pid,start_ticks=start,source_sha256=self.bindings[source]['sha256'],argv=[str(m.ROOT/'.conda-env/bin/python'),'-B','-u',*argv],
            inherited_frozen_environment=self.env,current_mount_namespace=self.env['current_mount_namespace'],automatic_retry=False)
    def retirement(self):
        for key,name in m.RETIREMENT_FILES.items():
            if name not in self.store:self.store[name]=b'{"summary":"actual959 historical902 infra6"}\n' if key=='history' else b'{}\n'
        self.put(m.RETIREMENT_FILES['launch'],dict(pid=99,source_sha256=self.point(m.LEGACY_SOURCE)['sha256'],
            source_config_sha256=self.point(m.RETIREMENT_FILES['source_config'])['sha256'],source_lock_sha256=self.point(m.RETIREMENT_FILES['source_lock'])['sha256']))
        receipt=dict(schema=m.RETIREMENT_SCHEMA,complete=True,observer_only=True,private_stop_only=True,exit_observed=True,experiment_STOP_written=False,
            session_id=123,exit_returncode=0,observed_exit_utc='2026-10-22T08:30:00Z',pid=99,start_ticks=999,
            **{k:self.point(n) for k,n in m.RETIREMENT_FILES.items()})
        self.put(m.RETIREMENT,receipt)
    def freeze_config(self):
        self.put(m.STATE+'/source.json',self.config)
        points=[self.point(m.STATE+'/source.json'),*self.bindings.values(),*(self.config[k] for k in ('resume_control','resume_control_lock','legacy_retirement','environment_recheck'))]
        self.put(m.STATE+'/source_inputs.json',dict(schema=m.LOCK_SCHEMA,approved_for_observation=True,files={p['path']:p['sha256'] for p in points}))
    def bundle(self):return m.load_source(self.point(m.STATE+'/source_inputs.json')['sha256'],read_bytes=self.read)
    def collect(self):return m.collect(self.bundle(),read_bytes=self.read)
    def complete_result(self):
        n=len(self.control['completed_job_ids']);terminal=dict(self.common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,stop_requested=False,
            drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,imported_completed_cases=n,resumed_new_cases=1920-n)
        self.put(m.RUNTIME+'/terminal.json',terminal);clean=dict(complete=True,returncode=0,finished_utc='synthetic',automatic_retry=False)
        for stage in ('formal_TEST_1920','independent_TEST_results'):self.put(m.CONTROLLER+'/stages/'+stage+'/terminal.json',clean)
        argv=[m.AUDITOR_SOURCE,'--resume-control',m.CONTROL];self.put(m.CONTROLLER+'/stages/independent_TEST_results/launch.json',self.stage(m.AUDITOR_SOURCE,303,3003,argv))
        owner_data={m.ORIGINAL:self.original,m.MIDDLE:self.middle,m.PARENT:self.parent,m.RUNTIME:dict(launch=self.owner,terminal=terminal)}
        tiers=[];cases=[];drivers=[];counts=Counter(r['evidence_runtime'] for r in self.plan['jobs'])
        for row in self.plan['jobs']:
            record=self.status['completed'][row['job_id']];owner=owner_data[row['evidence_runtime']]['launch']
            tiers.append(dict(job_id=row['job_id'],**{k:row[k] for k in ('evidence_runtime','imported_completed','resume_lineage','resume_lineage_v1010','previous_row_sha256','execution_entry')},
                actual_stage_owner_pid=owner['pid'],actual_stage_owner_start_ticks=owner['start_ticks'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],
                case=record['case'],admission_driver_receipt=record['admission_driver_receipt'],batch_driver_receipt=record.get('batch_driver_receipt')))
            cases.append(record['case']);drivers.append(record['admission_driver_receipt'])
        result=dict(self.base,schema='r10_four_tier_composite_formal_TEST_results_v1012',complete=True,split='test',source_cases=640,method_cases=1920,completed_method_cases=1920,
            methods=list(m.METHODS),tasks=list(m.TASK_LABELS),source_sha256=self.bindings[m.AUDITOR_SOURCE]['sha256'],resume_control=self.config['resume_control'],resume_control_lock=self.config['resume_control_lock'],
            source_input_lock=self.control['source_inputs'],validation_release=self.control['validation_release'],resource_profile=m.PROFILE,resource_qualification=self.control['resource_qualification'],
            execution_implementation=self.control['execution_implementation'],environment_recheck=self.control['environment_recheck'],independent_composite_audit=True,used_evidence_rechecked_at_close=True,
            native_failures_retained=True,motion_unknown_retained=True,zero_tick_cases_retained=True,motion_is_diagnostic_only=True,old_completed_results_not_rerun=True,
            resource_profile_changed=True,previous_resource_profile=self.control['previous_resource_profile'],previous_resource_qualification=self.control['previous_resource_qualification'],
            source_outcome_filtering=False,Value_NGFF_FASTER_used=False,old907_or958_successful_terminal_claimed=False,original754_906_or957_complete_result_claimed=False,
            auditor_invocation=dict(pid=303,start_ticks=3003,source=self.bindings[m.AUDITOR_SOURCE],argv=argv),frozen_environment=self.environment(303,3003),
            actual_stage_owners=[dict(runtime=r,launch=self.point(r+'/launch.json'),terminal=self.point(r+'/terminal.json'),pid=g['launch']['pid'],
                start_ticks=g['launch']['start_ticks'],completed=counts[r]) for r,g in owner_data.items()],cases=cases,admission_driver_receipts=drivers,evidence_tiers=tiers,
            native_successes_by_method={method:0 for method in m.METHODS},per_task=[dict(task=t,methods={method:dict(episodes=40,successes=0) for method in m.METHODS}) for t in m.TASK_LABELS],
            historical_infrastructure_attempts=[dict(runtime=r,job_ids=c['failed_job_ids'],previous_status=c['previous_status'],previous_terminal=c['previous_terminal'],
                counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True) for r,c in ((m.MIDDLE,self.old_control),(m.PARENT,self.control))],
            inputs_sha256={name:m.sha(raw) for name,raw in self.store.items()})
        self.put(m.RESULTS,result);self.put(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=self.point(m.RESULTS)))
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/stdout.log',dict(complete=True,schema=result['schema'],split='test',cases=1920,native_successes_by_method=result['native_successes_by_method'],motion_is_diagnostic_only=True,out=m.RESULTS))

class ReporterTests(unittest.TestCase):
    def test_partial_four_real_tiers_adopt_each_historical_result_once_and_separate_infra(self):
        f=Fixture();data=f.collect();item=m.snapshot(data,now=1000,next_due=m.phase_deadline(1000),reason='CPU')
        self.assertEqual(item['total_completed'],1202);self.assertEqual(item['imported_completed'],1200);self.assertEqual(item['new_completed'],2)
        self.assertEqual(item['evidence_tiers'],{m.ORIGINAL:134,m.MIDDLE:450,m.PARENT:616,m.RUNTIME:2})
        self.assertEqual(item['current1011_infrastructure_failed'],0);self.assertEqual(item['historical902_infrastructure_failed'],6)
        self.assertEqual(item['historical956_infrastructure_failed'],0);self.assertFalse(data['release_complete'])
        self.assertNotIn(m.RESULTS,f.reads);self.assertFalse(any('case.json' in p for p in f.reads))
    def test_full1920_all_False_None_motion_four_tiers_is_release(self):
        for adopted in (1200,1207):
            f=Fixture(full=True,adopted=adopted);data=f.collect();self.assertTrue(data['release_complete'])
            self.assertEqual(sum(data['completed'].values()),1920);self.assertEqual(sum(data['success'].values()),0)
            self.assertEqual(sum(data['unknown_motion'].values()),1920);self.assertFalse(any(p.endswith('case.json') for p in f.reads))
    def test_current_native_None_not_zero_and_nativeFalse_not_infrastructure(self):
        f=Fixture();key=f.plan['jobs'][1200]['job_id'];f.status['completed'][key].update(native_success=None,motion_pass=False);f.put(m.RUNTIME+'/status.json',f.status)
        data=f.collect();self.assertEqual(sum(data['unknown_native'].values()),1);self.assertEqual(sum(data['motion_false'].values()),1);self.assertEqual(data['failed'],0)
    def test_source_pointer_not_launch_pointer_and_observer_receipts_are_not_flattened(self):
        f=Fixture();f.owner['parent_stage_owner']=f.control['previous_launch'];f.put(m.RUNTIME+'/launch.json',f.owner)
        with self.assertRaises(ValueError):f.collect()
        f=Fixture();f.owner['frozen_environment']['checked_utc']='own-check';f.plan['frozen_environment']=deepcopy(f.owner['frozen_environment']);f.status['frozen_environment']=deepcopy(f.owner['frozen_environment'])
        for k,v in (('launch',f.owner),('plan',f.plan),('status',f.status)):f.put(m.RUNTIME+'/'+k+'.json',v)
        self.assertTrue(f.collect()['admitted'])
    def test_old959_real_root_retirement_no_self_terminal_or_launch_start_fabrication(self):
        f=Fixture();self.assertNotIn('start_ticks',json.loads(f.store[m.LEGACY+'/launch.json']));self.assertTrue(f.collect()['admitted'])
        f=Fixture();ret=json.loads(f.store[m.RETIREMENT]);ret['exit_observed']=False;f.put(m.RETIREMENT,ret);f.config['legacy_retirement']=f.point(m.RETIREMENT);f.freeze_config()
        with self.assertRaises(ValueError):f.collect()
    def test_whole956_parent_SHA_both_lineages_and_imported_terminal_change_rejected(self):
        for kind in ('lineage1010','lineage955','oldsha','imported','record'):
            f=Fixture();row=f.plan['jobs'][600]
            if kind=='lineage1010':row['resume_lineage_v1010']['parent_row_sha256']='f'*64
            elif kind=='lineage955':row['resume_lineage']['parent_row_sha256']='f'*64
            elif kind=='oldsha':row['previous_row_sha256']='f'*64
            elif kind=='imported':row['evidence_runtime']=m.RUNTIME
            else:f.status['completed'][row['job_id']]['native_success']=True
            f.put(m.RUNTIME+'/plan.json',f.plan);f.put(m.RUNTIME+'/status.json',f.status)
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.collect()
    def test_current_failed_attempt_separate_from_history_and_completed(self):
        f=Fixture();key=f.plan['jobs'][1202]['job_id'];f.status['failed'][key]=dict(complete=False,automatic_retry=False,stage='native');f.put(m.RUNTIME+'/status.json',f.status)
        data=f.collect();self.assertEqual(data['failed'],1);self.assertEqual(data['historical_infrastructure_failed'],6);self.assertEqual(data['historical956_infrastructure_failed'],0)
    def test_unfinished_owner_or_controller_never_opens_final_result(self):
        for kind in ('partial','stage','owner'):
            f=Fixture(full=True)
            if kind=='partial':f.put(m.CONTROLLER+'/status.json',dict(stage='formal_TEST_1920',frozen_environment=f.env,automatic_retry=False))
            elif kind=='stage':f.put(m.CONTROLLER+'/stages/independent_TEST_results/terminal.json',dict(complete=False,returncode=1,automatic_retry=False))
            else:t=json.loads(f.store[m.RUNTIME+'/terminal.json']);t['CPU_not_launched']=1;f.put(m.RUNTIME+'/terminal.json',t)
            if kind=='partial':self.assertFalse(f.collect()['release_complete'])
            else:
                with self.assertRaises(ValueError):f.collect()
            self.assertNotIn(m.RESULTS,f.reads)
    def test_full_auditor_PID_four_owner_tier_order_SHA_and_profile_mismatch_rejected(self):
        for kind in ('PID','owner','lineage','SHA','profile','history'):
            f=Fixture(full=True);r=json.loads(f.store[m.RESULTS])
            if kind=='PID':r['auditor_invocation']['pid']+=1
            elif kind=='owner':r['actual_stage_owners'].pop()
            elif kind=='lineage':r['evidence_tiers'][600]['resume_lineage_v1010']['parent_row_sha256']='f'*64
            elif kind=='profile':r['resource_profile']={}
            elif kind=='history':r['historical_infrastructure_attempts'][0]['runtime']=m.PARENT
            else:f.store[m.RESULTS]+=b' '; 
            if kind!='SHA':f.put(m.RESULTS,r);t=json.loads(f.store[m.CONTROLLER+'/terminal.json']);t['TEST_results']=f.point(m.RESULTS);f.put(m.CONTROLLER+'/terminal.json',t)
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.collect()
    def test_phase_is_original_microsecond_phase_not_activation_plus1800(self):
        for text in ('2026-10-22T08:00:00+00:00','2026-10-22T08:09:00+00:00','2026-10-22T08:38:21.384754+00:00'):
            now=datetime.fromisoformat(text).timestamp();due=m.phase_deadline(now);self.assertTrue(m.is_phase(due));self.assertGreater(due,now)
        self.assertEqual(datetime.fromtimestamp(m.phase_deadline(datetime.fromisoformat('2026-10-22T08:09:00+00:00').timestamp()),timezone.utc).isoformat(),'2026-10-22T08:38:21.384754+00:00')
    def test_forbidden_paths_stop_and_alias_fail_before_open(self):
        read=lambda *_:self.fail('outside allowlist/STOP callback invoked')
        for path in ('case.json','data/TEST/raw.h5','runtime/framework/train_inventory.json','model.pt'):
            with self.assertRaises(ValueError):m.Metadata(read_bytes=read).bytes(path)
        with self.assertRaises(m.ObservationStopped):m.Metadata(stop=lambda:True,read_bytes=read).bytes(m.CONTROL)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/m.CONTROL;p.parent.mkdir(parents=True);target=p.parent/'other.json';target.write_text('{}');p.symlink_to(target)
            with self.assertRaises(ValueError):m.Metadata(root=root).bytes(m.CONTROL)
    def test_unknown_admission_failure_never_fabricates_zero_current_count_or_hides_history(self):
        f=Fixture();data=f.collect();bad=m.observation_error(data,ValueError('CPU changed'));item=m.snapshot(bad,now=1000,next_due=m.phase_deadline(1000),reason='bad')
        txt,_=m.render(bad,item,[]);self.assertIsNone(item['total_completed']);self.assertIn('未知',txt);self.assertIn('[959真实历史]',txt)
        self.assertEqual(item['historical902_infrastructure_failed'],6);self.assertEqual(item['historical956_infrastructure_failed'],0)
    def test_legacy959_snapshot_history_shows_real_counts_and_956_label(self):
        f=Fixture();data=f.collect();data['legacy_history']=[dict(origin='actual959',record=dict(created_utc='CPU-history',total_completed=1202,
            current956_infrastructure_failed=0,historical902_infrastructure_failed=6,
            methods={method:dict(successes=1,completed=400) for method in m.METHODS}))]
        item=m.snapshot(data,now=1000,next_due=m.phase_deadline(1000),reason='CPU');txt,_=m.render(data,item,[])
        self.assertIn('[959真实历史] CPU-history 完成1202/1920',txt);self.assertIn('当时956 infra 0',txt);self.assertIn('历史902失败尝试 6',txt)
    def test_small_lock_no_graph_expansion_and_single_snapshot_no_status_reread(self):
        f=Fixture();bundle=f.bundle();f.collect();self.assertEqual(f.reads.count(m.RUNTIME+'/status.json'),1)
        lock=json.loads(f.store[m.STATE+'/source_inputs.json']);lock['files']['data/model.pt']='f'*64;f.put(m.STATE+'/source_inputs.json',lock)
        with self.assertRaises(ValueError):f.bundle()
    def test_temp_save_only_fresh1014_no_ack_or_old_modification(self):
        f=Fixture();data=f.collect()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);m.require_fresh_observer(root);h=m.save_report(data,f.bundle(),[],now=1000,next_due=m.phase_deadline(1000),reason='CPU',root=root)
            self.assertEqual(len(h),1);self.assertTrue((root/m.OUT).exists());self.assertFalse(any('ack' in p.name.lower() for p in root.rglob('*')))
            self.assertFalse((root/m.LEGACY).exists());self.assertFalse((root/m.RUNTIME).exists())
            with self.assertRaises(ValueError):m.require_fresh_observer(root)
    def test_default_main_is_inert_and_prepared_controls_do_not_prevent_fresh_activation(self):
        with patch.object(m,'load_source',side_effect=AssertionError('must remain inactive')):self.assertEqual(m.main([]),0)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);state=root/m.STATE;state.mkdir(parents=True)
            for name in ('source.json','source_inputs.json','legacy959_retirement.json'):(state/name).write_text('{}')
            m.require_fresh_observer(root)

if __name__=='__main__':unittest.main()
