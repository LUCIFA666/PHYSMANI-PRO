"""Independent synthetic interval/GPU/complete-writer audit. No real experiments."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resource1005_cpu',ROOT/'scripts/audit_concurrent_TRAIN_resource_canary_v1005_r10.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('fixtures1005_audit',ROOT/'tests/test_concurrent_TRAIN_resource_canary_v1005.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);r=f.m
PROFILE=r.PROFILE


def fixture():
    rows,_=f.selected();rows=r.ordered_native_rows(rows)
    launches,terminals,CPU_launches,CPU_terminals,cases={},{},{},{},{}
    for i,row in enumerate(rows):
        key=row['job_id']
        begin=(100+i if i<3 else 140+i) if i<8 else (380 if i==8 else 310+i)
        end=(360 if i==0 else 600) if i<3 or i==8 else (300+i if i<8 else 500+i)
        launches[key]=dict(launcher_pid=2000+i,launcher_start_ticks=3000+i,started_monotonic_ns=begin)
        terminals[key]=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False,
            launcher_pid=2000+i,launcher_start_ticks=3000+i,started_monotonic_ns=begin,finished_monotonic_ns=end,native_success=False)
        cb=(1+i if i<4 else 16+i) if i<8 else 170+i-8
        ce=(15 if i<4 else 35) if i<8 else 270+i-8
        CPU_launches[key]=dict(launcher_pid=4000+i,launcher_start_ticks=5000+i,started_monotonic_ns=cb,
            check_only=True,native_slots=0,GPU_slots=0)
        CPU_terminals[key]=dict(complete=True,returncode=0,timed_out=False,interrupted=False,automatic_retry=False,
            launcher_pid=4000+i,launcher_start_ticks=5000+i,started_monotonic_ns=cb,finished_monotonic_ns=ce)
        cases[key]=dict(process_id=1000+i,official_task_success=False)
    live=[dict(job_id=row['job_id'],method=row['method'],pid=launches[row['job_id']]['launcher_pid'],
        start_ticks=launches[row['job_id']]['launcher_start_ticks']) for row in rows[:8]]
    events=[dict(monotonic_ns=200,active=live,native_slots=8,GPU_slots=3),dict(monotonic_ns=700,active=[],native_slots=0,GPU_slots=0)]
    apps=[]
    for i in range(3):
        key=rows[i]['job_id'];apps.append(dict(pid=1000+i,pid_start_ticks=6000+i,job_id=key,
            launcher_pid=2000+i,launcher_start_ticks=3000+i,used_memory_mib=1800,gpu_uuid='GPU-fake',
            ancestor_chain=[dict(pid=1000+i,start_ticks=6000+i,ppid=2000+i,state='R'),
                            dict(pid=2000+i,start_ticks=3000+i,ppid=1,state='S')]))
    samples=[dict(monotonic_ns=200,sampled_until_monotonic_ns=201,GPU_queries_atomic_snapshot_claimed=False,GPU_observation_complete=True,
        gpu=dict(uuid='GPU-fake',memory_used_mib=7000,memory_total_mib=16000,utilization_gpu_percent=70),compute_apps=apps)]
    ready_sample=deepcopy(samples[0]);ready_sample.update(monotonic_ns=130,sampled_until_monotonic_ns=131)
    readiness=dict(complete=True,job_ids=[row['job_id'] for row in rows[:3]],monotonic_ns=130,hardware_sample=ready_sample)
    events.insert(0,dict(event='contexts_ready',monotonic_ns=132,active=live[:3],native_slots=3,GPU_slots=3,context_readiness=readiness))
    samples.insert(0,ready_sample)
    return rows,launches,terminals,events,CPU_launches,CPU_terminals,cases,samples


def check(v):
    rows,nl,nt,events,cl,ct,cases,samples=v
    native=m.interval_audit(rows,nl,nt,events,PROFILE)
    cpu=m.CPU_interval_audit(rows,cl,ct,PROFILE)
    joint=m.joint_interval_audit(rows,native,cpu,rows[:8])
    hardware=m.hardware_audit(samples,rows,nl,nt,cases,joint_intervals=(native['intervals'],cpu['intervals']))
    return native,cpu,joint,hardware


class IntervalHardwareTests(unittest.TestCase):
    def test_all_false_joint_real_intervals_and_three_GPU_main_contexts(self):
        native,cpu,joint,hardware=check(fixture())
        self.assertEqual((native['observed_native_peak'],native['observed_GPU_peak'],cpu['observed_CPU_preflight_peak']),(8,3,4))
        self.assertTrue(joint['complete']);self.assertTrue(hardware['actual_three_compute_contexts_during_joint8_3_4_observed'])
        self.assertEqual(len(hardware['witness']['jobs']),3)
        self.assertFalse(hardware['continuous_hardware_bound_claimed'])

    def test_separate_peaks_without_shared_interval_are_rejected(self):
        v=fixture()
        for row in v[0][8:]:
            k=row['job_id'];v[4][k]['started_monotonic_ns']=40;v[5][k].update(started_monotonic_ns=40,finished_monotonic_ns=80)
        with self.assertRaises(ValueError):check(v)

    def test_hidden_ninth_native_or_fifth_CPU_rejected_by_full_interval_sweep(self):
        for kind in ('native','CPU'):
            v=fixture();k=v[0][-1]['job_id']
            if kind=='native':v[1][k]['started_monotonic_ns']=110;v[2][k]['started_monotonic_ns']=110
            else:v[4][k]['started_monotonic_ns']=5;v[5][k]['started_monotonic_ns']=5
            with self.subTest(kind=kind),self.assertRaises(ValueError):check(v)

    def test_missing_duplicate_bool_PID_or_wrong_actual_interval_refuses(self):
        for kind in ('missing','duplicate','bool','terminal','events'):
            v=fixture();k=v[0][0]['job_id']
            if kind=='missing':v[0].pop()
            elif kind=='duplicate':v[0][-1]=deepcopy(v[0][0])
            elif kind=='bool':v[1][k]['launcher_pid']=True
            elif kind=='terminal':v[2][k]['launcher_start_ticks']+=1
            else:v[3][0]['monotonic_ns']=99
            with self.subTest(kind=kind),self.assertRaises(ValueError):check(v)

    def test_GPU_unknown_low_headroom_false_primary_or_ancestry_fail_closed(self):
        for kind in ('unknown','headroom','only2','wrongprimary','ancestry','outside_joint','fourthowned'):
            v=fixture();sample=v[-1][-1]
            if kind=='unknown':sample['GPU_observation_complete']=False
            elif kind=='headroom':sample['gpu']['memory_used_mib']=15000
            elif kind=='only2':sample['compute_apps'].pop()
            elif kind=='wrongprimary':v[-2][v[0][0]['job_id']]['process_id']=9999
            elif kind=='ancestry':sample['compute_apps'][0]['ancestor_chain'][0]['ppid']=1
            elif kind=='outside_joint':sample.update(monotonic_ns=275,sampled_until_monotonic_ns=276)
            else:
                app=deepcopy(sample['compute_apps'][0]);app.update(pid=99999,job_id=v[0][8]['job_id'])
                sample['compute_apps'].append(app)
            with self.subTest(kind=kind),self.assertRaises(ValueError):check(v)

    def test_full_GPU_query_window_must_fit_inside_joint_lifetimes(self):
        v=fixture();v[-1][-1]['sampled_until_monotonic_ns']=271
        with self.assertRaises(ValueError):check(v)

    def test_normal_GPU_start_exit_during_nonwitness_query_is_diagnostic_only(self):
        v=fixture();sample=deepcopy(v[-1][-1]);sample.update(monotonic_ns=350,sampled_until_monotonic_ns=390)
        row=v[0][8];launch=v[1][row['job_id']]
        sample['compute_apps'].append(dict(pid=1008,pid_start_ticks=6008,job_id=row['job_id'],
            launcher_pid=launch['launcher_pid'],launcher_start_ticks=launch['launcher_start_ticks'],
            used_memory_mib=1800,gpu_uuid='GPU-fake',ancestor_chain=[
                dict(pid=1008,start_ticks=6008,ppid=launch['launcher_pid'],state='R'),
                dict(pid=launch['launcher_pid'],start_ticks=launch['launcher_start_ticks'],ppid=1,state='S')]))
        v[-1].append(sample);result=check(v)[3]
        self.assertEqual(result['witness']['monotonic_ns'],200)
        self.assertEqual(result['valid_samples'],3)

    def test_external_GPU_job_is_not_hidden_or_killed_but_never_qualifies(self):
        v=fixture();v[-1][-1]['compute_apps'].append(dict(pid=9999,pid_start_ticks=999,job_id=None,used_memory_mib=3000,gpu_uuid='GPU-fake'))
        result=check(v)[3];self.assertEqual(len(result['witness']['jobs']),3)

    def test_equal_end_start_is_not_false_overlap(self):
        v=fixture()
        for row in v[0][8:]:
            key=row['job_id'];v[4][key]['started_monotonic_ns']=303;v[5][key].update(started_monotonic_ns=303,finished_monotonic_ns=304)
        with self.assertRaises(ValueError):check(v)


class FullSyntheticAuditTests(unittest.TestCase):
    def build(self,root,*,cleanup=True,self_signed=False,changed_sidecar=False,fake_formal_policy=False):
        rows,nl,nt,events,cl,ct,cases,samples=fixture();runtime=r.RUNTIME
        def write(name,value,raw=False):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if raw else json.dumps(value,sort_keys=True).encode())
            return point(name)
        def point(name):return dict(path=str(name),sha256=r.sha((root/name).read_bytes()))
        write(r.SOURCE,b'# synthetic runner',True);write(m.SOURCE,b'# synthetic independent auditor',True)
        binding=dict(control=write('artifacts/control.json',{}),lock=write('artifacts/inputs.json',{}),files={})
        control=dict(inventory=write('runtime/fake_TRAIN_inventory.json',{}),original_TRAIN_plan=write('artifacts/original_plan.json',{}),
            previous_CPU_receipts=[],resource_profile=PROFILE,execution_implementation={'sources':{}},policy_sha256='a'*64,
            author_outcome_protocol={'schema':'original_protocol'},common_RML_profile_sha256='b'*64,
            training_bootstrap=dict(control=write('artifacts/train_control.json',{}),lock=write('artifacts/train_lock.json',{})))
        plan=dict(schema=r.PLAN_SCHEMA,rows=rows,inputs=binding,resource_profile=PROFILE,
            **{k:control[k] for k in ('execution_implementation','policy_sha256','author_outcome_protocol','common_RML_profile_sha256','training_bootstrap')})
        pp=write(runtime+'/plan.json',plan);owners={}
        for phase,pid in (('CPU',100),('native',200)):
            env=dict(observer_pid=pid,observer_start_ticks=pid+1,current_mount_namespace='mnt:'+phase)
            argv=[r.SOURCE,'--control',binding['control']['path'],'--lock',binding['lock']['path'],'--lock-sha256',binding['lock']['sha256']]
            if phase=='native':argv.append('--execute-native')
            owners[phase]=dict(pid=pid,start_ticks=pid+1,argv=argv,source_sha256=point(r.SOURCE)['sha256'],
                control=binding['control'],lock=binding['lock'],plan=pp,resource_inventory={'empty':True},frozen_environment=env)
            write(runtime+'/'+phase+'_owner_launch.json',owners[phase])
        metrics=dict(schema='r10_scoped_frozen_lookup_metrics_v900',closed=True,complete=True,maps=1,declared_files=1,
            watched_directories=1,pointer_checks=2,full_graph_boundary_checks=2,used_path_end_checks=1,used_external=0,
            actual_SHA_checks=3,strong_stat_reuses=1,exact_JSON_payloads=0)
        cpu_results,late_results,native_results,case_pointers={},{},{},{}
        for i,row in enumerate(rows):
            key=row['job_id']
            for phase in ('CPU','native'):
                checkonly=phase=='CPU';suffix=['--check'] if checkonly else [];leaf=runtime+'/'+phase+'/'+key
                owner=owners['CPU' if checkonly and i<8 else 'native'];origin=cl[key] if checkonly else nl[key]
                launch=dict(origin,argv=row['argv']+suffix,physical_argv=row['physical_argv']+suffix,
                    owner_pid=owner['pid'],owner_start_ticks=owner['start_ticks'],check_only=checkonly,
                    native_slots=int(not checkonly),GPU_slots=int(not checkonly and row['method']=='physmani'),
                    inherited_frozen_environment=owner['frozen_environment'],current_mount_namespace=owner['frozen_environment']['current_mount_namespace'])
                lp=write(leaf+'/launch.json',launch);terminal=dict(ct[key] if checkonly else nt[key],launch=lp)
                batch=dict(schema='r10_scoped_frozen_lookup_bootstrap_receipt_v901',complete=True,entry='train',check_only=checkonly,
                    actual_outer_argv=row['argv'][2:]+suffix,forwarded_original_argv=row['physical_argv'][2:]+suffix,
                    control=control['training_bootstrap']['control'],lock=control['training_bootstrap']['lock'],sources={},
                    scientific_policy_changed=False,full_verify_inputs_calls_retained=True,
                    phases=[dict(phase='original742_prepare',boundary_recheck_complete=True,metrics=metrics)])
                if checkonly:
                    proof=dict(complete=True,native_started=False,GPU_started=False,identity=row['expected_identity'],batch_execution=batch)
                    terminal['CPU_preflight']=write(leaf+'/CPU_preflight.json',proof);write(leaf+'/stdout.log',proof)
                    (cpu_results if i<8 else late_results)[key]=terminal
                else:
                    case=dict(cases[key],identity=row['expected_identity'],split='train',cleanup_completed=cleanup,
                        component_domain={'purpose':'train_qualification'},common_author_RML_parameters={'original':True},
                        Value_NGFF_FASTER_used=False,TEST_used_for_fitting=False,author_outcome_protocol=control['author_outcome_protocol'],
                        full16_frozen_execution=dict(policy_sha256=control['policy_sha256'] if fake_formal_policy else None,source_row_sha256=row['source_row_sha256']))
                    cp=write(row['out']+'/case.json',case);batch['case']=cp;terminal['case']=cp
                    terminal['batch_driver_receipt']=write(row['out']+'/batch_driver_receipt.json',batch)
                    if changed_sidecar:terminal['batch_driver_receipt']['sha256']='0'*64
                    write(leaf+'/stdout.log',b'actual Native diagnostics need not be JSON\n',True);native_results[key]=terminal
                write(leaf+'/stderr.log',b'',True);tp=write(leaf+'/terminal.json',terminal)
                if not checkonly:case_pointers[key]=tp
        write(runtime+'/CPU_owner_terminal.json',dict(complete=True,native_started=False,GPU_started=False,results=cpu_results,
            pid=100,start_ticks=101,source_sha256=point(r.SOURCE)['sha256'],control=binding['control'],lock=binding['lock'],
            launch=point(runtime+'/CPU_owner_launch.json')))
        ep=write(runtime+'/native_resource_events.json',{'events':events})
        sp=write(runtime+'/native_resource_samples.jsonl',b'\n'.join(json.dumps(s).encode() for s in samples)+b'\n',True)
        summary=dict(schema=r.SUMMARY_SCHEMA,complete=True,independent_audit_complete=self_signed,resource_bounds_verified=False,
            TEST_read=False,formal_score_contribution=False,training_weights_updated=False,infrastructure_failed=0,
            not_launched_job_ids=[],case_count=12,context_readiness_observed=True,qualification_blocked_reason=None,cases=case_pointers,results=native_results,late_CPU_results=late_results,
            resource_profile=PROFILE,plan=pp,owner=point(runtime+'/native_owner_launch.json'),resource_events=ep,resource_samples=sp,
            final_resource_inventory={'empty':True},**{k:plan[k] for k in ('execution_implementation','policy_sha256','author_outcome_protocol','common_RML_profile_sha256')})
        summary_ptr=write(runtime+'/run_summary.json',summary)
        write(runtime+'/native_owner_terminal.json',dict(complete=True,independent_audit_complete=False,automatic_retry=False,
            pid=200,start_ticks=201,source_sha256=point(r.SOURCE)['sha256'],launch=point(runtime+'/native_owner_launch.json'),
            run_summary=summary_ptr,resource_events=ep,resource_samples=sp))
        runner=SimpleNamespace(RUNTIME=runtime,SOURCE=r.SOURCE,PLAN_SCHEMA=r.PLAN_SCHEMA,SUMMARY_SCHEMA=r.SUMMARY_SCHEMA,PROFILE=PROFILE,
            selected_rows=lambda *args:rows,ordered_native_rows=lambda _:rows,read_pointer=lambda p:json.loads((root/p['path']).read_bytes()),
            point=point,alive=lambda _:False,option=lambda argv,k:argv[argv.index(k)+1],
            require_empty_inventory=lambda v:m.require(v=={'empty':True},'empty inventory required'),
            require_CPU_receipt=lambda row,proof,c:m.require(proof['identity']==row['expected_identity'],'CPU identity differs'))
        return control,binding,runner,rows

    def execute_fake(self,root,control,binding,runner):
        with patch.object(m,'ROOT',root),patch('our_mani.multitask.evaluation.validate_completed_case') as case, \
             patch('our_mani.multitask.author_outcome_receipt_v749.require_sampling_case') as outcome, \
             patch('our_mani.multitask.common_rml_protocol_v530.require_attestation') as rml, \
             patch('scripts.audit_environment_resumed_formal_VAL_v824_r10.require_environment_receipt',side_effect=lambda x,*a,**k:x):
            result=m.audit(control,binding,runner)
            self.assertEqual((case.call_count,outcome.call_count,rml.call_count),(12,12,12))
        return result

    def test_full_writer_all_twelve_false_and_real_owner_tiers_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,b,runner,rows=self.build(root);result=self.execute_fake(root,c,b,runner)
        self.assertTrue(result['complete']);self.assertTrue(result['independent_audit_complete'])
        self.assertEqual(result['case_count'],12);self.assertEqual(result['native_successes_diagnostic'],0)
        self.assertTrue(result['joint_supply_overlap']['complete']);self.assertEqual(result['actual_CPU_preflight_peak'],4)
        self.assertFalse(result['raw_manifest_model_contents_read']);self.assertFalse(result['TEST_read'])

    def test_wrong_sidecar_self_signed_or_missing_cleanup_never_qualifies(self):
        for kwargs in ({'changed_sidecar':True},{'self_signed':True},{'cleanup':False},{'fake_formal_policy':True}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                root=Path(d);c,b,runner,_=self.build(root,**kwargs)
                with self.assertRaises(ValueError):self.execute_fake(root,c,b,runner)

    def test_old_owner_for_late_CPU_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,b,runner,rows=self.build(root)
            leaf=root/r.RUNTIME/'CPU'/rows[8]['job_id']/'launch.json';v=json.loads(leaf.read_text());v['owner_pid']=100
            leaf.write_text(json.dumps(v))
            with self.assertRaises(ValueError):self.execute_fake(root,c,b,runner)


if __name__=='__main__':unittest.main()
