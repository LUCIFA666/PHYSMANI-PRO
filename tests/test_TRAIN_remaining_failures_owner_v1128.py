"""CPU-only bounded owner checks; every child and predecessor is synthetic."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_TRAIN_remaining_failures_v1128_r10 as owner
from our_mani.multitask.train_ring_success_tap_v1125 import PassiveRingSuccessTap
from our_mani.multitask.train_buzz_failure_tap_v1105 import PassiveBuzzFailureTap
from our_mani.multitask.train_public_execution_trace_v1126 import public_execution_trace_scope
from tests.test_train_ring_success_tap_v1125 import fixture as ring_fixture
from tests.test_train_buzz_failure_tap_v1105 import fixtures as buzz_fixture
from tests import test_train_public_execution_trace_v1126 as trace_fixture


def roster():
    first={'insert_onto_rotating_peg':['episode101','episode102','episode105','episode112'],
        'beat_the_rotating_buzz':['episode10','episode104','episode105','episode107']}
    return dict(split='train',episodes=[dict(task=t,raw_episode_id=r) for t,rows in first.items() for r in rows+['episode999']])


def observed_events(row, *, planner_error=False):
    if row['kind']=='ring':
        task,_=ring_fixture()
        with PassiveRingSuccessTap(row['task'],row['method']) as tap:task.success()
    else:
        task,env=buzz_fixture(fail=True)
        with PassiveBuzzFailureTap(row['task'],row['method']) as tap:env._tick('arm')
    fixture=trace_fixture.Tests().fixture();public={}
    with public_execution_trace_scope(fixture.runtime,public):
        fixture.runtime._geometry(None,fixture.env,[]);fixture.frontend.observe({},trace_fixture.robot())
        if planner_error:fixture.service.error=RuntimeError('original planner result')
        try:fixture.service.plan_path([0.,0.,1.])
        except RuntimeError:
            if not planner_error:raise
    return dict(schema=owner.DIAGNOSTIC_SCHEMA+'_observers',passive_native_conditions=tap.receipt(),
        public_execution_trace=public,observer_finalization_errors=[],factory_restored=True,
        writer_restored=True,parent_annotation_restored=True,restored=True)


class BoundedOwner(unittest.TestCase):
    def test_default_inert_and_exact_two_first_four_sources_without_queue(self):
        with patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')),\
             patch.object(owner,'prepare',side_effect=AssertionError('no prepare')),\
             patch.object(owner,'execute',side_effect=AssertionError('no execute')),\
             patch('sys.stdout',new_callable=io.StringIO) as output:owner.main([])
        self.assertFalse(json.loads(output.getvalue())['Native_started'])
        jobs=owner.build_jobs(roster())
        self.assertEqual([(r['task'],r['raw'],r['method']) for r in jobs],list(owner.CANARIES))
        self.assertEqual(len(jobs),2);self.assertEqual(jobs[0]['kind'],'ring');self.assertEqual(jobs[1]['kind'],'buzz')
        self.assertEqual(len({r['out'] for r in jobs}),2);self.assertEqual(len({r['runtime'] for r in jobs}),2)
        self.assertTrue(all('/observed_v1128/' in r['out'] and '/observed_v1128/' in r['runtime'] for r in jobs))
        self.assertEqual(owner.FIXED_SOURCES,[dict(task=r['task'],raw_episode_id=r['raw'],method=r['method']) for r in jobs])

    def test_invalid_roster_and_exact_diagnostic_cli(self):
        for bad in ('split','missing','duplicate'):
            value=roster()
            if bad=='split':value['split']='test'
            if bad=='missing':value['episodes']=[r for r in value['episodes'] if r['raw_episode_id']!='episode101']
            if bad=='duplicate':value['episodes'][1]=deepcopy(value['episodes'][0])
            with self.assertRaises(ValueError):owner.build_jobs(value)
        row=owner.build_jobs(roster())[0];native=owner.command(row)
        self.assertEqual(owner.command(row,True),native+['--check'])
        self.assertIn(owner.WORKER,native)
        self.assertEqual(native[native.index('--diagnostic-lock')+1],owner.DIAGNOSTIC_LOCK)
        self.assertNotIn('--overlay-lock',native)

    def predecessor_fixture(self, root, *, bad=None):
        owner.fresh(root/owner.PREDECESSOR_PLAN,{'synthetic':'plan'})
        path=root/owner.PREDECESSOR_SOURCE;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source')
        plan_sha=owner.sha(root/owner.PREDECESSOR_PLAN);source_sha=owner.sha(path)
        ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])+(0 if bad=='live' else 1)
        launch=dict(pid=os.getpid(),start_ticks=ticks,plan=dict(path=owner.PREDECESSOR_PLAN,sha256=plan_sha))
        completed=[]
        for index,(task,raw,method) in enumerate(owner.PREDECESSOR_CANARIES):
            def ref(label):
                name=owner.PREDECESSOR+f'/synthetic/{index}_{label}.json'
                owner.fresh(root/name,{'synthetic':label});return dict(path=name,sha256=owner.sha(root/name))
            completed.append(dict(task=task,raw=raw,method=method,case=ref('case'),owned_hooks=ref('hooks'),
                CPU_check={k:ref('cpu_'+k) for k in ('launch','terminal','sidecar')},
                native={k:ref('native_'+k) for k in ('launch','terminal','sidecar')}))
        status=dict(stage='complete',active=None,failed=[],completed=completed)
        terminal=dict(complete=True,completed=4,infrastructure_failed=0,automatic_retry=False)
        owner.fresh(root/(owner.PREDECESSOR+'/launch.json'),launch)
        owner.fresh(root/(owner.PREDECESSOR+'/terminal.json'),terminal)
        completion=dict(schema=owner.EXIT_SCHEMA,actual_tool_exit_observed=True,returncode=0,
            owner_pid=launch['pid'],owner_start_ticks=ticks,owner_alive=False,natural_completion=True,
            native_cases_complete=4,completed_case_infrastructure_failures=0,plan=launch['plan'],
            terminal=dict(path=owner.PREDECESSOR+'/terminal.json',sha256=owner.sha(root/(owner.PREDECESSOR+'/terminal.json'))))
        if bad=='running':status['stage']='running'
        if bad=='active':status['active']={'unexpected':'child'}
        if bad=='failed':status['failed']=[{'infrastructure':'error'}]
        if bad=='order':status['completed']=list(reversed(completed))
        if bad=='unobserved':completion['actual_tool_exit_observed']=False
        if bad=='returncode':completion['returncode']=1
        if bad=='boolcode':completion['returncode']=False
        if bad=='unnatural':completion['natural_completion']=False
        if bad=='terminalsha':completion['terminal']['sha256']='changed'
        if bad=='casesha':status['completed'][0]['case']['sha256']='changed'
        owner.fresh(root/(owner.PREDECESSOR+'/status.json'),status)
        if bad!='missing_exit':owner.fresh(root/owner.PREDECESSOR_COMPLETION,completion)
        return plan_sha,source_sha

    def test_predecessor_requires_real_natural_complete_and_dead_identity_and_tool_exit0(self):
        for bad in (None,'live','running','active','failed','order','unobserved','returncode','boolcode','unnatural','terminalsha','casesha','missing_exit'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);plan_sha,source_sha=self.predecessor_fixture(root,bad=bad)
                with patch.object(owner,'ROOT',root),patch.object(owner,'PREDECESSOR_PLAN_SHA',plan_sha),\
                     patch.object(owner,'PREDECESSOR_SOURCE_SHA',source_sha):
                    if bad is None:self.assertTrue(owner.predecessor_complete()['natural_completion'])
                    else:
                        with self.assertRaises((ValueError,FileNotFoundError)):owner.predecessor_complete()

    def test_root_review_binds_exact_order_sources_exec_and_diagnostic_lock(self):
        for bad in (None,'order','exec','lock','released','reviewer'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);ref={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diagnostic'}
                predecessor={'actual_exec_completion':{'path':owner.PREDECESSOR_COMPLETION,'sha256':'exit'}}
                review=dict(schema=owner.REVIEW_SCHEMA,review_complete=True,release_bounded_two=True,reviewer='root',
                    fixed_TRAIN_sources=deepcopy(owner.FIXED_SOURCES),exec_completion=predecessor['actual_exec_completion'],
                    diagnostic_lock=ref,diagnostic_only=True)
                if bad=='order':review['fixed_TRAIN_sources'].reverse()
                if bad=='exec':review['exec_completion']={'sha256':'wrong'}
                if bad=='lock':review['diagnostic_lock']={'sha256':'wrong'}
                if bad=='released':review['release_bounded_two']=False
                if bad=='reviewer':review['reviewer']='unreviewed'
                owner.fresh(root/owner.ROOT_REVIEW,review)
                with patch.object(owner,'ROOT',root):
                    if bad is None:self.assertEqual(owner.root_review(ref,predecessor)['path'],owner.ROOT_REVIEW)
                    else:
                        with self.assertRaises(ValueError):owner.root_review(ref,predecessor)

    def test_prepare_blocked_before_overlay_or_writes_while_predecessor_runs(self):
        with TemporaryDirectory() as directory,patch.object(owner,'ROOT',Path(directory)),\
             patch.object(owner,'predecessor_complete',side_effect=ValueError('still running')),\
             patch.object(owner,'overlay_module',side_effect=AssertionError('no diagnostic activation')),\
             patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no child')):
            with self.assertRaisesRegex(ValueError,'still running'):owner.prepare()
            self.assertFalse((Path(directory)/owner.PLAN).exists())

    def test_actual_passive_streams_are_accepted_including_original_planner_exception(self):
        for row in owner.build_jobs(roster()):
            for planner_error in (False,True):
                result=owner.diagnostic_quality(row,observed_events(row,planner_error=planner_error))
                self.assertTrue(result['complete']);self.assertGreater(result['public_frames'],0)
                self.assertGreater(result['original_plans'],0)
                self.assertEqual(result['original_plan_exceptions'],int(planner_error))

    def test_zero_capture_missing_ring_leaf_or_bad_quality_never_counts_as_native_failure(self):
        row=owner.build_jobs(roster())[0]
        for bad in ('profilezero','emptyprofile','leaf','ordinal','profiledrop','profileerror','innerdrop',
                    'frameszero','planszero','tracedrop','traceerror','truncation','restore'):
            with self.subTest(bad=bad):
                events=observed_events(row);profile=events['passive_native_conditions'];trace=events['public_execution_trace']['trace_receipt']
                if bad=='profilezero':profile['original_success_returns_observed']=0
                if bad=='emptyprofile':profile['events']=[]
                if bad=='leaf':next(e for e in profile['events'] if e['event']=='original_task_success_return')['leaf_returns'].pop()
                if bad=='ordinal':
                    for e in profile['events']:
                        if e.get('registered_leaf_ordinal')==3:e['registered_leaf_ordinal']=2
                if bad=='profiledrop':profile['dropped_events']=1
                if bad=='profileerror':profile['observer_errors']=1
                if bad=='innerdrop':profile['inner_sequence_drops']=1
                if bad=='frameszero':trace['frames']=[];trace['frame_returns']=trace['frame_calls']=0
                if bad=='planszero':trace['plans']=[];trace['plan_calls']=trace['plan_returns']=0
                if bad=='tracedrop':trace['plan_dropped']=1
                if bad=='traceerror':trace['recording_errors']=1
                if bad=='truncation':trace['plan_value_truncations']=1
                if bad=='restore':events['restored']=False
                with self.assertRaises(ValueError):owner.diagnostic_quality(row,events)

    def plan_fixture(self,root):
        plan=dict(jobs=owner.build_jobs(roster()),source_overlay_lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diag'},remaining_queue=[])
        owner.fresh(root/owner.PLAN,plan);return plan,owner.sha(root/owner.PLAN)

    def test_execute_serial_two_then_natural_end_no_retries_and_preserves_false(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def child(row,check,status):
                status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native');calls.append((row['job_id'],check));return {'check':check}
            def complete(row,cpu,native,lock):
                self.assertTrue(cpu['check']);self.assertFalse(native['check']);return dict(job_id=row['job_id'],native_success=False)
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}),\
                 patch.object(owner,'child_phase',side_effect=child),patch.object(owner,'completed_case',side_effect=complete),\
                 patch.object(owner.subprocess,'Popen',side_effect=AssertionError('no real child')),patch('sys.stdout',new_callable=io.StringIO):
                plan,checksum=self.plan_fixture(root);owner.execute(checksum)
            self.assertEqual(calls,[(r['job_id'],check) for r in plan['jobs'] for check in (True,False)])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text());self.assertEqual(status['stage'],'complete')
            self.assertEqual(len(status['completed']),2);self.assertEqual(status['failed'],[])
            self.assertTrue(all(item['native_success'] is False for item in status['completed']))
            self.assertEqual(json.loads((root/owner.RUNTIME/'terminal.json').read_text())['completed'],2)

    def test_repeat_wrongsha_and_failed_collection_stop_without_retry(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}):
                plan,checksum=self.plan_fixture(root)
                with self.assertRaises(ValueError):owner.execute('wrongsha')
                owner.fresh(root/owner.RUNTIME/'launch.json',{'already':'launched'})
                with self.assertRaises(ValueError):owner.execute(checksum)
        with TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def child(row,check,status):
                status['active']=dict(job_id=row['job_id'],stage='CPU_check' if check else 'native');calls.append(check);return {}
            with patch.object(owner,'ROOT',root),patch.object(owner,'validate'),patch.object(owner,'predecessor_complete',return_value={}),\
                 patch.object(owner,'child_phase',side_effect=child),patch.object(owner,'completed_case',side_effect=ValueError('diagnostic collection incomplete')):
                plan,checksum=self.plan_fixture(root)
                with self.assertRaisesRegex(ValueError,'diagnostic collection'):owner.execute(checksum)
            self.assertEqual(calls,[True,False])
            status=json.loads((root/owner.RUNTIME/'status.json').read_text())
            self.assertEqual(status['completed'],[]);self.assertFalse(status['failed'][0]['native_failure_inferred'])
            self.assertEqual(status['failed'][0]['stage'],'result_review');self.assertFalse((root/owner.RUNTIME/'terminal.json').exists())

    def worker_fixture(self,root):
        row=owner.build_jobs(roster())[0];ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        status=dict(completed=[],failed=[],active=None,stage='running',source_overlay_lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diag'})
        value=dict(schema=owner.DIAGNOSTIC_SCHEMA,complete=True,source_overlay_lock=status['source_overlay_lock'],
            parent_source_overlay_lock={'path':'parentlock','sha256':'parentsha'},diagnostic_only=True,formal_score_contribution=False,
            selection=dict(task=row['task'],method=row['method'],raw_episode_id=row['raw']),process=dict(pid=9876,start_ticks=123),
            frozen_OS_environment=dict(outer_wrapper_pid=os.getpid(),outer_wrapper_start_ticks=ticks,observer_pid=9876,observer_start_ticks=123),
            execution=dict(mode='CPU_check',complete=True,native_started=False,GPU_started=False,actual_split='train',
                native_execution_ready=True,observers_installed=False,
                execution_protocol=dict(schema=owner.DIAGNOSTIC_SCHEMA,diagnostic_source_overlay_lock=status['source_overlay_lock'],
                    parent_source_overlay_lock={'path':'parentlock','sha256':'parentsha'})))
        name=row['runtime']+'/CPU_check/worker_sidecar.json';owner.fresh(root/name,value)
        return row,status,value,name

    def test_child_actual_pid_returncode_and_sidecar_namespace_binding(self):
        for bad in (None,'returncode','outerpid','observerpid','installed','started'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);row,status,value,name=self.worker_fixture(root)
                if bad=='outerpid':value['frozen_OS_environment']['outer_wrapper_pid']+=1
                if bad=='observerpid':value['frozen_OS_environment']['observer_start_ticks']+=1
                if bad=='installed':value['execution']['observers_installed']=True
                if bad=='started':value['execution']['native_started']=True
                (root/name).write_text(json.dumps(value));code=7 if bad=='returncode' else 0
                child=NS(pid=os.getpid(),wait=lambda:code,poll=lambda:code)
                with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=owner.DIAGNOSTIC_SCHEMA,PARENT_LOCK='parentlock',PARENT_LOCK_SHA='parentsha')),\
                     patch.object(owner.subprocess,'Popen',return_value=child) as spawn:
                    if bad is None:self.assertEqual(owner.child_phase(row,True,status)['process']['pid'],9876)
                    else:
                        with self.assertRaises(ValueError):owner.child_phase(row,True,status)
                self.assertEqual(spawn.call_count,1)
                terminal=json.loads((root/owner.RUNTIME/'launches'/row['job_id']/'CPU_check/terminal.json').read_text())
                self.assertEqual(terminal['returncode'],code);self.assertEqual(terminal['pid'],os.getpid())

    def test_interrupted_child_exit_remains_unknown_not_normal_terminal(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);row=owner.build_jobs(roster())[0];status=dict(completed=[],failed=[],stage='running')
            def interrupted():raise KeyboardInterrupt('synthetic interruption')
            child=NS(pid=os.getpid(),wait=interrupted,poll=lambda:None)
            with patch.object(owner,'ROOT',root),patch.object(owner.subprocess,'Popen',return_value=child):
                with self.assertRaises(KeyboardInterrupt):owner.child_phase(row,False,status)
            leaf=root/owner.RUNTIME/'launches'/row['job_id']/'native'
            interruption=json.loads((leaf/'owner_interruption.json').read_text())
            self.assertTrue(interruption['child_exit_unknown']);self.assertFalse(interruption['normal_child_terminal_claimed'])
            self.assertFalse((leaf/'terminal.json').exists())

    def test_completed_case_retains_native_false_and_checks_case_identity_hash_cleanup_and_quality(self):
        for bad in (None,'casehash','identity','cleanup','profilemissing'):
            with self.subTest(bad=bad),TemporaryDirectory() as directory:
                root=Path(directory);row=owner.build_jobs(roster())[0]
                lock={'path':owner.DIAGNOSTIC_LOCK,'sha256':'diag'};identity={'sha256':'identity'};protocol={'schema':owner.DIAGNOSTIC_SCHEMA}
                case=dict(version=owner.DIAGNOSTIC_SCHEMA+'_case',complete=True,cleanup_completed=bad!='cleanup',split='train',
                    task=row['task'],method=row['method'],source_raw_episode_id=row['raw'],process_id=9876,
                    official_task_success=False,native_terminal_failure=False,source_overlay_lock=lock,formal_score_contribution=False,
                    original_745_execution_acceptance_granted=False,parent_source_overlay_lock={'path':'parentlock','sha256':'parentsha'},
                    diagnostic_only=True,identity=identity,execution_protocol=protocol,infrastructure_error=None)
                owner.fresh(root/row['out']/'case.json',case)
                owner.fresh(root/row['out']/'owned_hook_receipt.json',dict(actual_split='train',reset=[{'complete':True}],paths=[{'close_complete':True}]))
                with patch.object(owner,'ROOT',root),patch.object(owner,'overlay_module',return_value=NS(SCHEMA=owner.DIAGNOSTIC_SCHEMA,PARENT_LOCK='parentlock',PARENT_LOCK_SHA='parentsha')):
                    cpu={'execution':dict(identity=identity,execution_protocol=protocol)}
                    events=observed_events(row)
                    if bad=='profilemissing':events['passive_native_conditions']['events']=[]
                    native=dict(process={'pid':9876},execution=dict(identity={'sha256':'changed'} if bad=='identity' else identity,
                        case=owner.pointer(row['out']+'/case.json'),native_success=False,native_failure=False,diagnostic_events=events))
                    if bad=='casehash':native['execution']['case']['sha256']='changed'
                    if bad is None:self.assertIs(owner.completed_case(row,cpu,native,lock)['native_success'],False)
                    else:
                        with self.assertRaises(ValueError):owner.completed_case(row,cpu,native,lock)


if __name__=='__main__':unittest.main()
