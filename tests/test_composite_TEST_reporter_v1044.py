"""Pure CPU/temp five-tier observer fixtures; never read experiment metadata."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module('report1044_CPU_fakes', PROJECT/'scripts/report_composite_TEST_halfhour_v1044_r10.py')
base = module('report1014_CPU_fixture_for1044', PROJECT/'tests/test_composite_TEST_reporter_v1014.py')
CORE_BYTES = (PROJECT/m.CORE).read_bytes()
PREVIOUS_CORE_BYTES = (PROJECT/m.PREVIOUS_CORE).read_bytes()
PARENT_CORE_BYTES = (PROJECT/m.PARENT_CORE).read_bytes()
ptr, outcome = base.ptr, base.outcome


class Fixture:
    def __init__(self, *, full=False, adopted=1394, extra=2, profile=None, capacity_source='qualified1031'):
        if not 1266 < adopted < 1920:
            raise ValueError('synthetic current parent must include a genuine fourth tier')
        old = base.Fixture(adopted=1266, extra=adopted-1266)
        self.store, self.reads = deepcopy(old.store), []
        for name in m.SOURCES:
            self.store[name] = (CORE_BYTES if name == m.CORE else PREVIOUS_CORE_BYTES if name == m.PREVIOUS_CORE
                else PARENT_CORE_BYTES if name == m.PARENT_CORE else self.store.get(name, ('# synthetic '+name+'\n').encode()))
        self.bindings = {name:self.point(name) for name in m.SOURCES}
        self.policy, self.protocol, self.base, self.entry = old.policy, old.protocol, old.base, old.entry
        self.original, self.middle, self.parent = deepcopy(old.original), deepcopy(old.middle), deepcopy(old.parent)
        self.old_control = deepcopy(old.old_control)
        self.previous_control = deepcopy(old.control)
        self.previous = dict(launch=deepcopy(old.owner), plan=deepcopy(old.plan), status=deepcopy(old.status))
        self.previous['status']['drain_requested'] = True
        self.previous['terminal'] = dict(old.common, complete=False, expected=1920, completed=adopted, failed=0,
            drained=True, drain_requested=True, stop_requested=False, automatic_retry=False, policy_sha256=self.policy)
        for key, value in self.previous.items():
            self.put(m.PREVIOUS+'/'+key+'.json', value)
        core = m.load_core(m.Metadata(read_bytes=self.read), dict(source_bindings=self.bindings))
        ids = [r['job_id'] for r in self.previous['plan']['jobs']]
        capacity = deepcopy(m.PROFILE if capacity_source == 'qualified1005_fallback' else m.CAPACITY)
        actual = deepcopy(capacity if profile is None else profile)
        version = '1005' if capacity_source == 'qualified1005_fallback' else '1031'
        self.control = dict(deepcopy(self.previous_control), schema=core.CONTROL_SCHEMA,
            previous_runtime=m.PREVIOUS, previous_controller_runtime='runtime/framework/ordered_composite_TEST_v1013',
            previous_resume_control=self.point(m.PREVIOUS_CONTROL), previous_resume_control_lock=self.point(m.PREVIOUS_LOCK),
            resource_profile=actual, resource_capacity_envelope=capacity, resource_capacity_source=capacity_source,
            resource_profile_within_capacity=True, resource_profile_changed=actual != self.previous_control['resource_profile'],
            previous_resource_profile=deepcopy(self.previous_control['resource_profile']),
            previous_resource_qualification=deepcopy(self.previous_control['resource_qualification']),
            resource_qualification=ptr('runtime/framework/concurrent_TRAIN_resource_qualification_v'+version+'.json'),
            resource_qualifier=ptr('scripts/audit_concurrent_TRAIN_resource_canary_v'+version+'_r10.py'),
            completed_job_ids=ids[:adopted], failed_job_ids=[], remaining_job_ids=ids[adopted:],
            resume_owner=self.bindings[m.OWNER_SOURCE], resume_auditor=self.bindings[m.AUDITOR_SOURCE],
            resume_controller=self.bindings[m.CONTROLLER_SOURCE])
        for key in ('completed', 'failed', 'remaining'):
            self.control[key+'_key_sha256'] = m.digest(sorted(self.control[key+'_job_ids']))
        for key in self.previous:
            self.control['previous_'+key] = self.point(m.PREVIOUS+'/'+key+'.json')
        self.put(m.CONTROL, self.control)
        previous_lock = json.loads(self.store[m.PREVIOUS_LOCK])
        pointers = [self.point(m.CONTROL), self.point(m.PREVIOUS_CONTROL), self.point(m.PREVIOUS_LOCK), *self.bindings.values(),
            *(self.control['previous_'+k] for k in self.previous), *(self.control['original_'+k] for k in self.original)]
        files = deepcopy(previous_lock['files']); files.update({p['path']:p['sha256'] for p in pointers})
        self.put(m.CONTROL_LOCK, dict(schema=core.LOCK_SCHEMA, approved_for_execution=True, control=self.point(m.CONTROL), files=files))
        self.retirement()
        self.config = dict(schema=m.SCHEMA, protocol_status='formal_TEST_composite', runtime=m.RUNTIME, controller=m.CONTROLLER,
            test_results=m.RESULTS, source_bindings=self.bindings, policy_sha256=self.policy, author_outcome_protocol=self.protocol,
            resume_control=self.point(m.CONTROL), resume_control_lock=self.point(m.CONTROL_LOCK), legacy_retirement=self.point(m.RETIREMENT),
            environment_recheck=self.point(m.RECHECK))
        self.freeze_config(); self.env = self.environment(401, 4001)
        self.common = dict(self.base, split='test', resume_control=self.config['resume_control'], resume_control_lock=self.config['resume_control_lock'],
            resource_profile=actual, frozen_environment=self.environment(402, 4002),
            **{k:self.control[k] for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck',
                'resource_profile_changed','previous_resource_profile','previous_resource_qualification',
                'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source')})
        self.plan = dict(self.common, expected_cases=1920, imported_completed_cases=adopted, source_outcome_filtering=False,
            jobs=core.resumed_jobs(self.previous['plan'], self.control, self.entry))
        self.owner = dict(self.common, pid=402, start_ticks=4002, source_sha256=self.bindings[m.OWNER_SOURCE]['sha256'],
            argv=[m.OWNER_SOURCE,'--resume-control',m.CONTROL], base_stage_owner=self.bindings[m.ORIGINAL_SOURCE],
            parent_stage_owner=self.bindings[m.PREVIOUS_SOURCE], imported_completed_cases=adopted,
            **{k:actual[k] for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')})
        count = 1920 if full else adopted+extra
        records = {r['job_id']:outcome(r) for r in self.plan['jobs'][:count]}
        records.update(deepcopy(self.previous['status']['completed']))
        self.status = dict(self.common, expected=1920, completed=records, failed={}, active=[], checking=[], ready_jobs=[],
            CPU_ready=0, native_slots=0, GPU_slots=0, owner_error=None, drain_requested=False, stop_requested=False,
            updated_utc='synthetic', imported_completed=adopted)
        for key, value in (('launch',self.owner), ('plan',self.plan), ('status',self.status)):
            self.put(m.RUNTIME+'/'+key+'.json', value)
        self.put(m.CONTROLLER+'/launch.json', dict(pid=401,start_ticks=4001,source_sha256=self.bindings[m.CONTROLLER_SOURCE]['sha256'],
            resume_control=self.config['resume_control'],resume_control_lock=self.config['resume_control_lock'],frozen_environment=self.env,automatic_retry=False))
        self.put(m.CONTROLLER+'/status.json', dict(stage='complete' if full else 'formal_TEST_1920',frozen_environment=self.env,automatic_retry=False))
        self.put(m.CONTROLLER+'/stages/formal_TEST_1920/launch.json', self.stage(m.OWNER_SOURCE,402,4002,self.owner['argv']))
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
        return dict(pid=pid,start_ticks=start,source_sha256=self.bindings[source]['sha256'],
            argv=[str(m.ROOT/'.conda-env/bin/python'),'-B','-u',*argv],inherited_frozen_environment=self.env,
            current_mount_namespace=self.env['current_mount_namespace'],automatic_retry=False)
    def retirement(self):
        for key,name in m.RETIREMENT_FILES.items():
            if name not in self.store:
                self.store[name]=b'{"summary":"synthetic1014 observed historical902 infra6"}\n' if key=='history' else b'{}\n'
        self.put(m.RETIREMENT_FILES['launch'],dict(pid=98,source_sha256=self.point(m.LEGACY_SOURCE)['sha256'],
            source_config_sha256=self.point(m.RETIREMENT_FILES['source_config'])['sha256'],source_lock_sha256=self.point(m.RETIREMENT_FILES['source_lock'])['sha256']))
        self.put(m.RETIREMENT,dict(schema=m.RETIREMENT_SCHEMA,complete=True,observer_only=True,private_stop_only=True,exit_observed=True,
            experiment_STOP_written=False,session_id=123,exit_returncode=0,observed_exit_utc='synthetic',pid=98,start_ticks=998,
            **{k:self.point(n) for k,n in m.RETIREMENT_FILES.items()}))
    def freeze_config(self):
        self.put(m.STATE+'/source.json',self.config)
        points=[self.point(m.STATE+'/source.json'),*self.bindings.values(),*(self.config[k] for k in
            ('resume_control','resume_control_lock','legacy_retirement','environment_recheck'))]
        self.put(m.STATE+'/source_inputs.json',dict(schema=m.LOCK_SCHEMA,approved_for_observation=True,files={p['path']:p['sha256'] for p in points}))
    def bundle(self):return m.load_source(self.point(m.STATE+'/source_inputs.json')['sha256'],read_bytes=self.read)
    def collect(self):return m.collect(self.bundle(),read_bytes=self.read)
    def complete_result(self):
        n=len(self.control['completed_job_ids'])
        terminal=dict(self.common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,stop_requested=False,
            drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,imported_completed_cases=n,resumed_new_cases=1920-n)
        self.put(m.RUNTIME+'/terminal.json',terminal)
        for stage in ('formal_TEST_1920','independent_TEST_results'):
            self.put(m.CONTROLLER+'/stages/'+stage+'/terminal.json',dict(complete=True,returncode=0,finished_utc='synthetic',automatic_retry=False))
        argv=[m.AUDITOR_SOURCE,'--resume-control',m.CONTROL]
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/launch.json',self.stage(m.AUDITOR_SOURCE,403,4003,argv))
        owner_data={m.ORIGINAL:self.original,m.MIDDLE:self.middle,m.PARENT:self.parent,m.PREVIOUS:self.previous,
            m.RUNTIME:dict(launch=self.owner,terminal=terminal)}
        counts=Counter(row['evidence_runtime'] for row in self.plan['jobs']);tiers=[];cases=[];drivers=[]
        for row in self.plan['jobs']:
            record=self.status['completed'][row['job_id']];owner=owner_data[row['evidence_runtime']]['launch']
            tiers.append(dict(job_id=row['job_id'],**{k:row[k] for k in ('evidence_runtime','imported_completed','resume_lineage',
                'resume_lineage_v1010','resume_lineage_v1040','previous_row_sha256','execution_entry')},actual_stage_owner_pid=owner['pid'],
                actual_stage_owner_start_ticks=owner['start_ticks'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],
                case=record['case'],admission_driver_receipt=record['admission_driver_receipt'],batch_driver_receipt=record.get('batch_driver_receipt')))
            cases.append(record['case']);drivers.append(record['admission_driver_receipt'])
        result=dict(self.base,schema='r10_five_tier_composite_formal_TEST_results_v1042',complete=True,split='test',source_cases=640,
            method_cases=1920,completed_method_cases=1920,methods=list(m.METHODS),tasks=list(m.TASK_LABELS),
            source_sha256=self.bindings[m.AUDITOR_SOURCE]['sha256'],resume_control=self.config['resume_control'],resume_control_lock=self.config['resume_control_lock'],
            source_input_lock=self.control['source_inputs'],validation_release=self.control['validation_release'],resource_profile=self.control['resource_profile'],
            resource_qualification=self.control['resource_qualification'],execution_implementation=self.control['execution_implementation'],
            environment_recheck=self.control['environment_recheck'],independent_composite_audit=True,used_evidence_rechecked_at_close=True,
            native_failures_retained=True,motion_unknown_retained=True,zero_tick_cases_retained=True,motion_is_diagnostic_only=True,
            old_completed_results_not_rerun=True,source_outcome_filtering=False,Value_NGFF_FASTER_used=False,automatic_retry=False,
            old907_or958_or1013_successful_terminal_claimed=False,original754_906_957_or1012_complete_result_claimed=False,
            **{k:self.control[k] for k in ('resource_profile_changed','previous_resource_profile','previous_resource_qualification',
                'resource_capacity_envelope','resource_profile_within_capacity','resource_capacity_source')},
            auditor_invocation=dict(pid=403,start_ticks=4003,source=self.bindings[m.AUDITOR_SOURCE],argv=argv),frozen_environment=self.environment(403,4003),
            actual_stage_owners=[dict(runtime=r,launch=self.point(r+'/launch.json'),terminal=self.point(r+'/terminal.json'),
                pid=g['launch']['pid'],start_ticks=g['launch']['start_ticks'],completed=counts[r]) for r,g in owner_data.items()],
            cases=cases,admission_driver_receipts=drivers,evidence_tiers=tiers,native_successes_by_method={method:0 for method in m.METHODS},
            per_task=[dict(task=t,methods={method:dict(episodes=40,successes=0) for method in m.METHODS}) for t in m.TASK_LABELS],
            historical_infrastructure_attempts=[dict(runtime=r,job_ids=c['failed_job_ids'],previous_status=c['previous_status'],
                previous_terminal=c['previous_terminal'],counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True)
                for r,c in ((m.MIDDLE,self.old_control),(m.PARENT,self.previous_control),(m.PREVIOUS,self.control))],
            inputs_sha256={name:m.sha(raw) for name,raw in self.store.items()})
        self.put(m.RESULTS,result)
        self.put(m.CONTROLLER+'/terminal.json',dict(complete=True,automatic_retry=False,TEST_results=self.point(m.RESULTS)))
        self.put(m.CONTROLLER+'/stages/independent_TEST_results/stdout.log',dict(complete=True,schema=result['schema'],split='test',cases=1920,
            native_successes_by_method=result['native_successes_by_method'],motion_is_diagnostic_only=True,out=m.RESULTS))


class ReporterTests(unittest.TestCase):
    def test_partial_dynamic_five_tiers_once_and_history_separate(self):
        for adopted in (1394,1407):
            f=Fixture(adopted=adopted);data=f.collect()
            item=m.snapshot(data,now=1000,next_due=m.phase_deadline(1000),reason='CPU')
            self.assertEqual((item['total_completed'],item['imported_completed'],item['new_completed']),(adopted+2,adopted,2))
            self.assertEqual(item['evidence_tiers'],{m.ORIGINAL:134,m.MIDDLE:450,m.PARENT:682,m.PREVIOUS:adopted-1266,m.RUNTIME:2})
            self.assertEqual((item['current1041_infrastructure_failed'],item['historical902_infrastructure_failed'],
                item['historical956_infrastructure_failed'],item['historical1011_infrastructure_failed']),(0,6,0,0))
            self.assertFalse(data['release_complete']);self.assertNotIn(m.RESULTS,f.reads)
            self.assertFalse(any('case.json' in p for p in f.reads))

    def test_full_all_nativeFalse_motionNone_five_tiers_is_final(self):
        f=Fixture(full=True);data=f.collect()
        self.assertTrue(data['release_complete']);self.assertEqual(sum(data['completed'].values()),1920)
        self.assertEqual(sum(data['success'].values()),0);self.assertEqual(sum(data['unknown_motion'].values()),1920)
        self.assertFalse(any('case.json' in p for p in f.reads))

    def test_lower_actual_quota_distinct_from_capacity_and_genuine_fallback_unchanged(self):
        lower=dict(m.CAPACITY,max_native_slots=10,max_GPU_slots=4,max_CPU_preflight_slots=5,ready_limit=20)
        for profile,source,changed in ((m.CAPACITY,'qualified1031',True),(lower,'qualified1031',True),
                (m.PROFILE,'qualified1005_fallback',False)):
            f=Fixture(full=True,profile=profile,capacity_source=source);data=f.collect()
            self.assertEqual(data['profile'],profile);self.assertIs(f.control['resource_profile_changed'],changed)
            self.assertTrue(data['release_complete'])
            item=m.snapshot(data,now=1000,next_due=m.phase_deadline(1000),reason='CPU');txt,_=m.render(data,item,[])
            self.assertIn('/'.join(str(profile[k]) for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots','ready_limit')),txt)
            self.assertEqual(item['resource_capacity_envelope'],m.CAPACITY if source=='qualified1031' else m.PROFILE)
            self.assertEqual(item['resource_capacity_source'],source)

    def test_invalid_capacity_alias_bool_or_overquota_refused(self):
        valid=Fixture().control
        for kind in ('too_many','bool','capacity','source','fallback_lower'):
            c=deepcopy(valid)
            if kind=='too_many':c['resource_profile']['max_native_slots']=13
            elif kind=='bool':c['resource_profile']['max_CPU_preflight_slots']=True
            elif kind=='capacity':c['resource_capacity_envelope']['ready_limit']=25
            elif kind=='source':c['resource_qualifier']['path']='scripts/unknown.py'
            else:c.update(resource_profile=dict(m.PROFILE,max_native_slots=7),resource_capacity_envelope=m.PROFILE,resource_capacity_source='qualified1005_fallback')
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.recorded_profiles(c)

    def test_current_unknown_native_is_not_nativeFalse_or_infra(self):
        f=Fixture();row=f.plan['jobs'][1394];f.status['completed'][row['job_id']].update(native_success=None,motion_pass=False)
        f.put(m.RUNTIME+'/status.json',f.status);data=f.collect()
        self.assertEqual(sum(data['unknown_native'].values()),1);self.assertEqual(sum(data['motion_false'].values()),1)
        self.assertEqual(data['failed'],0);self.assertEqual(sum(data['success'].values()),0)

    def test_history_or_new_lineage_forgery_and_wrong_owner_source_refused(self):
        for kind in ('lineage','oldrow','record','owner'):
            f=Fixture()
            if kind=='lineage':f.plan['jobs'][1300]['resume_lineage_v1040']['parent_row_sha256']='f'*64
            elif kind=='oldrow':f.plan['jobs'][600]['resume_lineage_v1010']['parent_row_sha256']='f'*64
            elif kind=='record':f.status['completed'][f.plan['jobs'][1300]['job_id']]['native_success']=True
            else:f.owner['parent_stage_owner']=f.control['previous_launch']
            for k,v in (('plan',f.plan),('status',f.status),('launch',f.owner)):f.put(m.RUNTIME+'/'+k+'.json',v)
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.collect()

    def test_current_failed_attempt_not_history_or_clean_nativeFalse(self):
        f=Fixture();row=f.plan['jobs'][1396];f.status['failed'][row['job_id']]=dict(complete=False,automatic_retry=False)
        f.put(m.RUNTIME+'/status.json',f.status);data=f.collect()
        self.assertEqual(data['failed'],1);self.assertEqual(data['historical_infrastructure_failed'],6)
        self.assertFalse(data['release_complete'])

    def test_no_final_result_read_before_owner_controller_and_stage_clean(self):
        for kind in ('partial','stage','owner'):
            f=Fixture(full=True)
            if kind=='partial':f.put(m.CONTROLLER+'/status.json',dict(stage='formal_TEST_1920',frozen_environment=f.env,automatic_retry=False))
            elif kind=='stage':f.put(m.CONTROLLER+'/stages/independent_TEST_results/terminal.json',dict(complete=False,returncode=1,automatic_retry=False))
            else:t=json.loads(f.store[m.RUNTIME+'/terminal.json']);t['CPU_not_launched']=1;f.put(m.RUNTIME+'/terminal.json',t)
            if kind=='partial':self.assertFalse(f.collect()['release_complete'])
            else:
                with self.assertRaises(ValueError):f.collect()
            self.assertNotIn(m.RESULTS,f.reads)

    def test_final_wrong_source_tier_history_capacity_flags_or_actual_PID_refused(self):
        for kind in ('PID','owner','lineage','SHA','capacity','history','flag','totals'):
            f=Fixture(full=True);r=json.loads(f.store[m.RESULTS])
            if kind=='PID':r['auditor_invocation']['pid']+=1
            elif kind=='owner':r['actual_stage_owners'].pop()
            elif kind=='lineage':r['evidence_tiers'][1300]['resume_lineage_v1040']['parent_row_sha256']='f'*64
            elif kind=='capacity':r['resource_capacity_envelope']={}
            elif kind=='history':r['historical_infrastructure_attempts'].pop()
            elif kind=='flag':r['old907_or958_or1013_successful_terminal_claimed']=True
            elif kind=='totals':r['native_successes_by_method']['physmani']=1
            else:f.store[m.RESULTS]+=b' '
            if kind!='SHA':
                f.put(m.RESULTS,r);t=json.loads(f.store[m.CONTROLLER+'/terminal.json']);t['TEST_results']=f.point(m.RESULTS);f.put(m.CONTROLLER+'/terminal.json',t)
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.collect()

    def test_private1014_retirement_no_self_terminal_or_fabricated_start(self):
        f=Fixture();self.assertNotIn('start_ticks',json.loads(f.store[m.LEGACY+'/launch.json']))
        self.assertNotIn(m.LEGACY+'/terminal.json',f.store);self.assertTrue(f.collect()['admitted'])
        f=Fixture();ret=json.loads(f.store[m.RETIREMENT]);ret['experiment_STOP_written']=True
        f.put(m.RETIREMENT,ret);f.config['legacy_retirement']=f.point(m.RETIREMENT);f.freeze_config()
        with self.assertRaises(ValueError):f.collect()

    def test_original_halfhour_phase_survives_activation_and_missed_deadlines(self):
        for text in ('2026-10-22T10:00:00+00:00','2026-10-22T10:09:00+00:00','2026-10-22T10:38:21.384754+00:00'):
            now=datetime.fromisoformat(text).timestamp();due=m.phase_deadline(now)
            self.assertTrue(m.is_phase(due));self.assertGreater(due,now)
        now=datetime.fromisoformat('2026-10-22T10:09:00+00:00').timestamp()
        self.assertEqual(datetime.fromtimestamp(m.phase_deadline(now),timezone.utc).isoformat(),'2026-10-22T10:38:21.384754+00:00')

    def test_forbidden_data_stop_and_alias_rejected_before_read(self):
        read=lambda *_:self.fail('forbidden data or STOP invoked reader')
        for path in ('case.json','model.pt','data/TEST/raw.h5','artifacts/framework/protocol/test_manifest.json'):
            with self.assertRaises(ValueError):m.Metadata(read_bytes=read).bytes(path)
        with self.assertRaises(m.ObservationStopped):m.Metadata(stop=lambda:True,read_bytes=read).bytes(m.CONTROL)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/m.CONTROL;path.parent.mkdir(parents=True);target=path.parent/'target.json';target.write_text('{}');path.symlink_to(target)
            with self.assertRaises(ValueError):m.Metadata(root=root).bytes(m.CONTROL)

    def test_unknown_error_preserves_frozen_history_but_not_current_counts(self):
        f=Fixture();data=f.collect();bad=m.observation_error(data,ValueError('metadata changed'))
        item=m.snapshot(bad,now=1000,next_due=m.phase_deadline(1000),reason='error');txt,_=m.render(bad,item,[])
        self.assertIsNone(item['total_completed']);self.assertIsNone(item['resource_profile']);self.assertIn('未知',txt)
        self.assertIn('[1014真实历史]',txt);self.assertEqual(item['historical902_infrastructure_failed'],6)

    def test_small_lock_exact_read_once_and_source_SHA(self):
        f=Fixture();f.collect();self.assertEqual(f.reads.count(m.RUNTIME+'/status.json'),1)
        lock=json.loads(f.store[m.STATE+'/source_inputs.json']);lock['files']['model.pt']='f'*64;f.put(m.STATE+'/source_inputs.json',lock)
        with self.assertRaises(ValueError):f.bundle()
        f=Fixture();f.store[m.OWNER_SOURCE]+=b'changed'
        with self.assertRaises(ValueError):f.bundle()

    def test_temp_save_only_own_files_fresh_snapshots_chat_ready_without_ack(self):
        f=Fixture();data=f.collect()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);m.require_fresh_observer(root)
            history=m.save_report(data,f.bundle(),[],now=1000,next_due=m.phase_deadline(1000),reason='CPU',root=root)
            self.assertEqual(len(history),1);self.assertTrue((root/m.OUT).exists())
            self.assertFalse(any('ack' in p.name.lower() for p in root.rglob('*')))
            self.assertFalse((root/m.LEGACY).exists());self.assertFalse((root/m.RUNTIME).exists())
            status=json.loads((root/m.STATE/'status.json').read_text());ready=json.loads((root/m.STATE/'chat_ready.json').read_text())
            self.assertFalse(ready['chat_sent']);self.assertFalse(ready['delivery_acknowledged'])
            self.assertEqual(len(status['input_snapshots']),2)
            with self.assertRaises(ValueError):m.require_fresh_observer(root)

    def test_default_main_inert_without_source_or_resource_access(self):
        with patch.object(m,'load_source',side_effect=AssertionError('must stay inactive')),patch.object(m.Path,'exists',side_effect=AssertionError('no OS probe')):
            self.assertEqual(m.main([]),0)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);state=root/m.STATE;state.mkdir(parents=True)
            for leaf in ('source.json','source_inputs.json','legacy1014_retirement.json'):(state/leaf).write_text('{}')
            m.require_fresh_observer(root)


if __name__=='__main__':unittest.main()
