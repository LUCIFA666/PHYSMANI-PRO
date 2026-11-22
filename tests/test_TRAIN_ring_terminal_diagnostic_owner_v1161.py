"""CPU-only serial diagnostic owner contracts, using synthetic children."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from scripts import run_TRAIN_ring_terminal_diagnostic_owner_v1161_r10 as owner


def roster():
    return dict(split='train',episodes=[dict(task='insert_onto_rotating_peg_high_speed',raw_episode_id=r)
        for r in ('episode102','episode103','episode105','episode108','episode999')])


class DiagnosticOwner(unittest.TestCase):
    def test_default_inert_and_fixed_same_parent_two_geometry_methods(self):
        with patch.object(owner,'prepare',side_effect=AssertionError('no prepare')),\
             patch.object(owner,'execute',side_effect=AssertionError('no execute')),\
             patch('sys.stdout',new_callable=io.StringIO) as out:owner.main([])
        self.assertFalse(json.loads(out.getvalue())['Native_started'])
        jobs=owner.build_jobs(roster())
        self.assertEqual([(j['task'],j['raw'],j['method']) for j in jobs],list(owner.CANARIES))
        self.assertEqual(len(jobs),2);self.assertEqual({j['raw'] for j in jobs},{'episode102'})
        self.assertEqual(len({j['out'] for j in jobs}),2)
        for bad in ('test','missing','duplicate'):
            value=roster()
            if bad=='test':value['split']='test'
            if bad=='missing':value['episodes'].pop(0)
            if bad=='duplicate':value['episodes'][1]=deepcopy(value['episodes'][0])
            with self.assertRaises(ValueError):owner.build_jobs(value)
        argv=owner.command(jobs[0]);self.assertEqual(owner.command(jobs[0],True),argv+['--check'])
        self.assertEqual(argv[argv.index('--diagnostic-lock')+1],owner.DIAGNOSTIC_LOCK)

    def test_prepare_only_alias_is_explicit_and_never_executes(self):
        with patch.object(owner,'prepare',return_value={'Native_started':False}) as prepared,\
             patch.object(owner,'execute',side_effect=AssertionError('CPU preparation cannot execute')):
            self.assertFalse(owner.main(['--prepare-only'])['Native_started']);prepared.assert_called_once_with()

    def test_pending_lock_or_source_pins_refuse_before_hash_or_import(self):
        with patch.object(owner,'DIAGNOSTIC_LOCK_SHA',None),patch.object(owner,'sha',side_effect=AssertionError('pending must be inert')):
            with self.assertRaisesRegex(ValueError,'pending1160'):owner.overlay_module()

    def predecessor(self,root,bad=None):
        owner.fresh(root/owner.PREDECESSOR_PLAN,dict(jobs=[dict(task=t,raw=r,method=m) for t,r,m in owner.PREDECESSOR_CANARIES],source_overlay_lock={}))
        source=root/owner.PREDECESSOR_SOURCE;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('synthetic source')
        plan_sha=owner.sha(root/owner.PREDECESSOR_PLAN);source_sha=owner.sha(source)
        ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])+(0 if bad=='live' else 1)
        launch=dict(pid=os.getpid(),start_ticks=ticks,plan=dict(path=owner.PREDECESSOR_PLAN,sha256=plan_sha))
        rows=[]
        def ref(name):
            path=owner.PREDECESSOR+'/synthetic/'+name+'.json';owner.fresh(root/path,dict(synthetic=name))
            return dict(path=path,sha256=owner.sha(root/path))
        for index,(task,raw,method) in enumerate(owner.PREDECESSOR_CANARIES):
            rows.append(dict(task=task,raw=raw,method=method,case=ref(str(index)+'case'),owned_hooks=ref(str(index)+'hooks'),
                CPU_check={k:ref(str(index)+'cpu'+k) for k in ('launch','terminal','sidecar')},
                native={k:ref(str(index)+'native'+k) for k in ('launch','terminal','sidecar')}))
        status=dict(stage='complete',active=None,failed=[],completed=rows)
        terminal=dict(complete=True,completed=4,infrastructure_failed=0,automatic_retry=False)
        owner.fresh(root/(owner.PREDECESSOR+'/launch.json'),launch)
        owner.fresh(root/(owner.PREDECESSOR+'/terminal.json'),terminal)
        completion=dict(schema=owner.EXIT_SCHEMA,actual_tool_exit_observed=True,exec_session=owner.PREDECESSOR_EXEC_SESSION,returncode=0,
            owner_pid=os.getpid(),owner_start_ticks=ticks,owner_alive=False,natural_completion=True,
            native_cases_complete=4,completed_case_infrastructure_failures=0,plan=launch['plan'],
            terminal=dict(path=owner.PREDECESSOR+'/terminal.json',sha256=owner.sha(root/(owner.PREDECESSOR+'/terminal.json'))))
        if bad=='running':status['stage']='running'
        if bad=='active':status['active']={'child':'live'}
        if bad=='count':status['completed']=rows[:-1]
        if bad=='order':status['completed']=list(reversed(rows))
        if bad=='unobserved':completion['actual_tool_exit_observed']=False
        if bad=='returncode':completion['returncode']=1
        if bad=='boolcode':completion['returncode']=False
        if bad=='session':completion['exec_session']+=1
        if bad=='old_schema':completion['schema']='r10_actual_natural_block_preclose_comparison_owner_exit_v1155'
        if bad=='pointer':status['completed'][0]['case']['sha256']='changed'
        owner.fresh(root/(owner.PREDECESSOR+'/status.json'),status)
        owner.fresh(root/owner.PREDECESSOR_COMPLETION,completion)
        return plan_sha,source_sha

    def test_actual_four_completed_dead_predecessor_and_root_observed_exit_required(self):
        for bad in (None,'live','running','active','count','order','unobserved','returncode','boolcode','session','old_schema','pointer'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);plan_sha,source_sha=self.predecessor(root,bad)
                with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),\
                     patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha),patch.object(owner,'predecessor_record_integrity',return_value=None):
                    if bad is None:self.assertTrue(owner.predecessor_complete()['natural_completion'])
                    else:
                        with self.assertRaises(ValueError):owner.predecessor_complete()

    def test_predecessor_result_integrity_is_checked_before_new_overlay(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);plan_sha,source_sha=self.predecessor(root)
            with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),\
                 patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha),\
                 patch.object(owner,'predecessor_record_integrity',side_effect=ValueError('changed prior completed case')),\
                 patch.object(owner,'overlay_module',side_effect=AssertionError('do not activate')):
                with self.assertRaisesRegex(ValueError,'changed prior'):owner.prepare()
            self.assertFalse((root/owner.PLAN).exists())

    def test_predecessor_real_verifier_delegated_once_and_original_row_required(self):
        from scripts import run_TRAIN_ring_terminal_comparison_v1158_r10 as previous
        task,raw,method=owner.PREDECESSOR_CANARIES[0]
        row=dict(task=task,raw=raw,method=method,CPU_check={},native={});job=dict(row);lock={'source':'synthetic'}
        with patch.object(previous,'completed_case',return_value=row) as called:
            owner.predecessor_record_integrity(row,[job],lock);called.assert_called_once_with(job,{}, {},lock)
        with patch.object(previous,'completed_case',return_value=dict(row,native_success=True)):
            with self.assertRaisesRegex(ValueError,'integrity changed'):owner.predecessor_record_integrity(row,[job],lock)

    def test_protocol_requires_actual_canonical1157_parent_not_old1140(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source='our_mani/multitask/train_high_ring_diagnostic_v1160.py'
            for name in (owner.WORKER,source):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic CPU source')
            lock=dict(path=owner.DIAGNOSTIC_LOCK,sha256='locked')
            protocol=dict(schema=owner.DIAGNOSTIC_SCHEMA,diagnostic_source_overlay_lock=lock,
                parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
                actual_worker=dict(path=owner.WORKER,sha256=owner.sha(root/owner.WORKER)),
                actual_preparer=dict(path=source,sha256=owner.sha(root/source)),diagnostic_only=True,
                formal_score_contribution=False,formal_VAL_TEST_qualification=False,
                inherited_1157_execution_protocol=dict(schema='r10_full16_fixed_TRAIN_ring_terminal_overlay_v1157',
                    source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
                    changed_factor=owner.PARENT_FACTOR,
                    core_factory=dict(module='our_mani.multitask.author_ring_terminal_goal_integration_v1156')))
            with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SOURCE=source)):
                owner.validate_protocol(owner.build_jobs(roster())[0],protocol,lock)
                protocol['inherited_1157_execution_protocol']['core_factory']['module']='our_mani.multitask.author_ring_release_integration_v1140'
                with self.assertRaises(ValueError):owner.validate_protocol(owner.build_jobs(roster())[0],protocol,lock)

    def test_quality_delegates_three_streams_and_keeps_original_false_and_exceptions(self):
        from our_mani.multitask import train_high_ring_diagnostic_v1160 as actual
        from tests.test_train_high_ring_diagnostic_v1160 import quality_fixture
        events=quality_fixture();row=owner.build_jobs(roster())[0]
        with patch.object(owner,'overlay_module',return_value=actual):
            quality=owner.diagnostic_quality(row,events,ticks=3,terminal=False)
            self.assertTrue(quality['complete']);self.assertEqual(quality['original_plan_exceptions'],1)
            for stream in ('passive_native_conditions','public_execution_trace','readiness_probe'):
                broken=deepcopy(events)
                if stream=='passive_native_conditions':broken[stream]['dropped_events']=1
                elif stream=='public_execution_trace':broken[stream]['trace_receipt']['frame_dropped']=1
                else:broken[stream]['readiness_probe']['recording_errors']=1
                with self.assertRaises(ValueError):owner.diagnostic_quality(row,broken,ticks=3,terminal=False)

    def test_prepare_cannot_activate_or_write_before_predecessor_drains(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)),\
             patch.object(owner,'predecessor_complete',side_effect=ValueError('still running')),\
             patch.object(owner,'overlay_module',side_effect=AssertionError('cannot activate')):
            with self.assertRaises(ValueError):owner.prepare()
            self.assertFalse((Path(directory)/owner.PLAN).exists())

    def test_scoped_review_is_exact_two_sources_and_current_diagnostic_lock(self):
        for bad in (None,'released','sources','lock','exit'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);ref=dict(path=owner.DIAGNOSTIC_LOCK,sha256='source lock')
                previous=dict(actual_exec_completion=dict(path=owner.PREDECESSOR_COMPLETION,sha256='exit'))
                value=dict(schema=owner.REVIEW_SCHEMA,review_complete=True,release_bounded_two=True,reviewer='root',
                    fixed_TRAIN_sources=deepcopy(owner.FIXED_SOURCES),exec_completion=previous['actual_exec_completion'],
                    diagnostic_lock=ref,diagnostic_only=True)
                if bad=='released':value['release_bounded_two']=False
                if bad=='sources':value['fixed_TRAIN_sources'].reverse()
                if bad=='lock':value['diagnostic_lock']={}
                if bad=='exit':value['exec_completion']={}
                owner.fresh(root/owner.ROOT_REVIEW,value)
                with patch.object(owner,'ROOT',root):
                    if bad is None:self.assertEqual(owner.root_review(ref,previous)['path'],owner.ROOT_REVIEW)
                    else:
                        with self.assertRaises(ValueError):owner.root_review(ref,previous)

    def test_serial_cpu_native_each_and_all_false_retained_no_retry(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);calls=[];plan=dict(jobs=owner.build_jobs(roster()),source_overlay_lock={})
            owner.fresh(root/owner.PLAN,plan);checksum=owner.sha(root/owner.PLAN)
            def child(row,check,status):
                calls.append((row['method'],check));status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native')
                return dict(sidecar={'path':'synthetic','sha256':'synthetic'})
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),\
                 patch.object(owner,'predecessor_complete',return_value={}),\
                 patch.object(owner,'child_phase',side_effect=child),\
                 patch.object(owner,'completed_case',side_effect=lambda row,*args:dict(job_id=row['job_id'],native_success=False)),\
                 patch('sys.stdout',new_callable=io.StringIO):owner.execute(checksum)
            self.assertEqual(calls,[(method,check) for method in owner.METHODS for check in (True,False)])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text())
            self.assertEqual(status['stage'],'complete');self.assertEqual(len(status['completed']),2)
            self.assertEqual(status['failed'],[]);self.assertTrue(all(not r['native_success'] for r in status['completed']))

    def test_failure_never_invents_native_failure_or_retries_or_next_case(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);plan=dict(jobs=owner.build_jobs(roster()),source_overlay_lock={})
            owner.fresh(root/owner.PLAN,plan);checksum=owner.sha(root/owner.PLAN);calls=[]
            def child(row,check,status):
                calls.append(check);status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native')
                if not check:raise RuntimeError('actual infrastructure exception')
                return {}
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),\
                 patch.object(owner,'predecessor_complete',return_value={}),patch.object(owner,'child_phase',side_effect=child):
                with self.assertRaises(RuntimeError):owner.execute(checksum)
            self.assertEqual(calls,[True,False]);value=json.loads((root/owner.RUNTIME/'failure.json').read_text())
            self.assertFalse(value['failed'][0]['native_failure_inferred']);self.assertFalse((root/owner.RUNTIME/'terminal.json').exists())

    def test_quality_is_delegated_to_frozen_diagnostic_collector_without_policy_recall(self):
        validator=Mock(return_value={'complete':True})
        with patch.object(owner,'overlay_module',return_value=NS(require_passive_quality=validator)):
            self.assertEqual(owner.diagnostic_quality({},dict(events=[]),ticks=200,terminal=False),{'complete':True})
        validator.assert_called_once_with(dict(events=[]),ticks=200,terminal=False)

    def test_child_process_namespace_and_return_code_are_real_and_status_keeps_only_pointers(self):
        for bad in (None,'returncode','boolcode','outerpid','innerpid','factor'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);row=owner.build_jobs(roster())[0]
                ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
                status=dict(completed=[],failed=[],active=None,stage='running',source_overlay_lock={'path':'lock','sha256':'lock'})
                process=dict(pid=98765,start_ticks=456)
                value=dict(schema='synthetic diagnostic schema',complete=True,source_overlay_lock=status['source_overlay_lock'],
                    parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
                    diagnostic_only=True,formal_score_contribution=False,process=process,
                    selection=dict(task=row['task'],method=row['method'],raw_episode_id=row['raw']),
                    frozen_OS_environment=dict(outer_wrapper_pid=os.getpid(),outer_wrapper_start_ticks=ticks,
                        observer_pid=98765,observer_start_ticks=456),
                    execution=dict(mode='CPU_check',complete=True,native_started=False,GPU_started=False,
                        native_execution_ready=True,actual_split='train',execution_protocol=dict(factor='correct')))
                if bad=='outerpid':value['frozen_OS_environment']['outer_wrapper_pid']+=1
                if bad=='innerpid':value['frozen_OS_environment']['observer_start_ticks']+=1
                if bad=='factor':value['execution']['execution_protocol']['factor']='wrong'
                owner.fresh(root/(row['runtime']+'/CPU_check/worker_sidecar.json'),value)
                code=False if bad=='boolcode' else (3 if bad=='returncode' else 0)
                child=NS(pid=os.getpid(),wait=lambda:code,poll=lambda:code)
                def validate(row,protocol,lock):
                    owner.require(protocol==dict(factor='correct'),'actual diagnostic protocol required')
                with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=value['schema'])),\
                     patch.object(owner,'validate_protocol',side_effect=validate),patch.object(owner.subprocess,'Popen',return_value=child) as spawn:
                    if bad is None:
                        result=owner.child_phase(row,True,status)
                        self.assertEqual(set(result),{'launch','terminal','sidecar'})
                        self.assertNotIn('execution',result);self.assertNotIn('process',result)
                    else:
                        with self.assertRaises(ValueError):owner.child_phase(row,True,status)
                self.assertEqual(spawn.call_count,1)

    def test_completed_case_reloads_sidecars_and_retains_native_false_with_terminal_aware_quality(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);row=owner.build_jobs(roster())[0]
            inherited_protocol=dict(schema='r10_full16_fixed_TRAIN_ring_terminal_overlay_v1157')
            inherited=dict(sha256='a'*64,execution_protocol_schema=inherited_protocol['schema'],
                execution_protocol_sha256=owner.digest(inherited_protocol),source_overlay_lock_sha256=owner.PARENT_LOCK_SHA)
            protocol=dict(inherited_1157_identity_sha256=inherited['sha256'],inherited_1157_execution_protocol=inherited_protocol)
            identity=dict(sha256='identity',inherited_1157_identity_sha256=inherited['sha256'],
                execution_protocol_schema=owner.DIAGNOSTIC_SCHEMA,execution_protocol_sha256=owner.digest(protocol),
                source_overlay_lock_sha256='lock')
            lock=dict(path='lock',sha256='lock')
            case=dict(version=owner.DIAGNOSTIC_SCHEMA+'_case',complete=True,cleanup_completed=True,split='train',
                task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],process_id=123,
                official_task_success=False,native_terminal_failure=False,source_overlay_lock=lock,
                parent_source_overlay_lock=dict(path=owner.PARENT_LOCK,sha256=owner.PARENT_LOCK_SHA),
                diagnostic_only=True,infrastructure_error=None,formal_score_contribution=False,
                original_745_execution_acceptance_granted=False,identity=identity,execution_protocol=protocol,
                counted_clock=dict(ticks=4,dt=.05,seconds=.2),
                alignment_receipt=dict(clock=dict(ticks=4,dt=.05,simulation_seconds=.2),outcomes_are_policy_observations=False),
                native_outcome_sampling=dict(last_tick=4,active_action=None,aborted=False,completed_actions=[
                    dict(start_tick=0,end_tick=4,ticks=4,max_ticks=10,complete=True)]))
            name=row['out']+'/case.json';owner.fresh(root/name,case)
            case_ref=dict(path=name,sha256=owner.sha(root/name))
            cpu_name=row['runtime']+'/CPU_check/worker_sidecar.json';native_name=row['runtime']+'/native/worker_sidecar.json'
            owner.fresh(root/cpu_name,dict(execution=dict(identity=identity,execution_protocol=protocol,inherited_1157_identity=inherited)))
            owner.fresh(root/native_name,dict(process=dict(pid=123),execution=dict(identity=identity,case=case_ref,
                native_success=False,native_failure=False,diagnostic_events=dict(events=[]))))
            owner.fresh(root/(row['out']+'/owned_hook_receipt.json'),dict(actual_split='train',reset=[dict(complete=True)],paths=[]))
            cpu=dict(sidecar=dict(path=cpu_name,sha256=owner.sha(root/cpu_name)))
            native=dict(sidecar=dict(path=native_name,sha256=owner.sha(root/native_name)))
            for phase_name,record in (('CPU_check',cpu),('native',native)):
                for key in ('launch','terminal'):
                    name=row['runtime']+'/'+phase_name+'/'+key+'.json';owner.fresh(root/name,dict(synthetic=key))
                    record[key]=dict(path=name,sha256=owner.sha(root/name))
            quality=Mock(return_value=dict(complete=True))
            with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=owner.DIAGNOSTIC_SCHEMA)),\
                 patch.object(owner,'validate_protocol'),patch.object(owner,'diagnostic_quality',quality):
                result=owner.completed_case(row,cpu,native,lock)
            self.assertFalse(result['native_success']);self.assertFalse(result['native_terminal_failure'])
            self.assertEqual(result['CPU_check'],cpu);self.assertEqual(result['native'],native)
            quality.assert_called_once_with(row,dict(events=[]),ticks=4,terminal=False)
            (root/cpu_name).write_text('{}')
            with patch.object(owner,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'evidence changed'):
                    owner.completed_case(row,cpu,native,lock)


if __name__=='__main__':
    unittest.main()
