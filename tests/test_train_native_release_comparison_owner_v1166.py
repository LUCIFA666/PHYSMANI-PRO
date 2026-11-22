"""Bounded5 owner contracts: temporary JSON, mocked launches, no Native app."""
from copy import deepcopy
import io,json,os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import run_TRAIN_native_release_comparison_owner_v1166_r10 as m

PROJECT=Path(__file__).resolve().parents[1]


def grid():
    tasks={t:r for t,r,_ in m.CANARIES}
    roster=dict(split='train',episodes=[dict(task=t,raw_episode_id=r)for t,first in tasks.items()
        for r in (first,'second','third','fourth')])
    routes=[dict(task=t,method=method,GPU_required=False,configuration=dict(path='configs/'+t+'.json',sha256='config'))
            for t in tasks for method in m.METHODS]
    routes.extend(dict(task='other'+str(i),method='cv_geometry',GPU_required=False,configuration={})for i in range(48-len(routes)))
    return roster,dict(routes=routes)


def sample(task,success=False,fatal=False):
    outcome=dict(success=success,terminate=success or fatal,terminal_failure=fatal)
    action=dict(action_index=0,label='geometry_execute',start_tick=0,end_tick=4,ticks=4,max_ticks=10,
        complete=True,sampled_outcome=outcome,current_native_result=outcome)
    receipt=dict(schema='r10_explicit_author_action_outcome_sampling_v741',completed_actions=[action],last_tick=4,
        active_action=None,aborted=False,aborted_actions=[],global_tick_modulus_used=False,success_sampled_each_tick=True,
        failure_sampled_at_explicit_action_end=True,reset_failure_is_diagnostic_only=True,reset_success_semantics_preserved=True,
        sampled_outcome=outcome,initial_native_result=dict(success=False,terminate=False,terminal_failure=False),intermediate_failure_samples=0)
    return dict(version=m.CANDIDATE_SCHEMA+'_case',inherited_consumer_record_version='r10_full16_explicit_author_action_outcome_v744',
        task=task,method='context_geometry',complete=True,cleanup_completed=True,infrastructure_error=None,official_task_success=success,native_terminal_failure=fatal,
        counted_clock=dict(ticks=4,dt=.05,seconds=.2),native_outcome_sampling=receipt,
        author_outcome_protocol=dict(schema='r10_author_action_outcome_protocol_v741'),
        alignment_receipt=dict(clock=dict(ticks=4,dt=.05,simulation_seconds=.2),native_success_latched=success,
            native_terminal_failure_latched=fatal,native_terminal_latched=success or fatal,native_outcome=outcome,
            native_terminal_tick=4 if success or fatal else None,native_outcome_sampling=deepcopy(receipt)),
        all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=False))


class Tests(unittest.TestCase):
    def jobs(self):return m.build_jobs(*grid())

    def protocol(self,root,row,lock):
        for name in (m.WORKER,m.PREPARER):
            p=root/name
            if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CPU source fixture')
        component=dict(integration=dict(schema=m.CORE_SCHEMA,source_sha256=m.FROZEN_PINS[m.CORE_SOURCE]),
            task_rule=dict(task=row['task'],source_sha256=m.FROZEN_PINS[m.CORE_SOURCE],changed_factor=m.CHANGED_FACTOR,
                controlled_phase='deposit'if row['task']in m.BASKETS else'release'))
        if row['task']in m.BUZZ_TASKS:component['unchanged_Buzz_approach_rule']=dict(task=row['task'])
        return dict(schema=m.CANDIDATE_SCHEMA,source_overlay_lock=lock,
            actual_worker=m.pointer(m.WORKER),actual_preparer=m.pointer(m.PREPARER),execution_branch='core_selected_integration',
            comparison_parent_source_overlay_lock=m.COMPARISON_PARENT_LOCK,core_changed_relative_to1154=True,
            block_changed_relative_to1154=False,baseline_changed_relative_to1154=False,formal_score_contribution=False,
            formal_VAL_TEST_qualification=False,changed_factor=m.CHANGED_FACTOR,
            original_route=dict(author_outcome_protocol=dict(schema='r10_author_action_outcome_protocol_v741')),
            core_factory=dict(module=m.CORE_MODULE,function='integrated_components',source=m.CORE_SOURCE,sha256=m.FROZEN_PINS[m.CORE_SOURCE]),
            public_phase_component_binding=component)

    def evidence(self,root,row,success=False,fatal=False):
        lock=dict(path=m.SOURCE_LOCK,sha256='l'*64);protocol=self.protocol(root,row,lock);identity={'sha256':'CPU fixture'}
        case=sample(row['task'],success,fatal)
        case.update(split='train',source_raw_episode_id=row['raw'],method=row['method'],identity=identity,
            execution_protocol=protocol,source_overlay_lock=lock,process_id=12345,formal_score_contribution=False,
            original_745_execution_acceptance_granted=False)
        m.fresh(root/row['out']/'case.json',case);case_ref=m.pointer(row['out']+'/case.json')
        hooks=dict(actual_split='train',paths=[],reset=[dict(complete=True)],shared_development_hooks_applied=True)
        m.fresh(root/row['out']/'owned_hook_receipt.json',hooks)
        sides=[]
        for check,phase in [(True,'CPU_check'),(False,'native')]:
            execution=dict(complete=True,identity=identity,execution_protocol=protocol)
            if not check:execution.update(case=case_ref,native_success=success,native_failure=fatal)
            side=dict(execution=execution,process=dict(pid=12345))
            refs={}
            for key,value in [('launch',{}),('terminal',dict(returncode=0)),('sidecar',side)]:
                name='evidence/'+row['job_id']+'/'+phase+'/'+key+'.json';m.fresh(root/name,value);refs[key]=m.pointer(name)
            sides.append(refs)
        return case,lock,*sides

    def test_default_inert(self):
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('launch')),patch.object(m,'prepare',side_effect=AssertionError('prepare')),patch('sys.stdout',new_callable=io.StringIO)as out:
            m.main([])
        self.assertFalse(json.loads(out.getvalue())['Native_started'])

    def test_exact_five_four_parents_and_fresh_namespaces(self):
        rows=self.jobs();self.assertEqual([(r['task'],r['raw'],r['method'])for r in rows],list(m.CANARIES))
        self.assertEqual(len(rows),5);self.assertEqual(len({(r['task'],r['raw'])for r in rows}),4)
        self.assertEqual(len({r['out']for r in rows}),5);self.assertEqual(len({r['runtime']for r in rows}),5)
        self.assertTrue(all('/matched_v1166/'in r['out']for r in rows))

    def test_actual_TRAIN_first_sources_and_route_config_SHA(self):
        roster=json.loads((PROJECT/'artifacts/framework/protocol/train_manifest.json').read_text())
        routes=json.loads((PROJECT/'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/routes.json').read_text())
        rows=m.build_jobs(roster,routes)
        self.assertEqual([r['raw']for r in rows],['episode104','episode104','episode0','episode10','episode102'])
        for row in rows:self.assertEqual(m.sha(PROJECT/row['config']),row['config_sha256'])

    def test_split_replacement_duplicate_or_gpu_route_rejected(self):
        for mutation in ('split','first','duplicate','route','gpu'):
            roster,routes=grid()
            if mutation=='split':roster['split']='validation'
            elif mutation=='first':roster['episodes'][0]['raw_episode_id']='replacement'
            elif mutation=='duplicate':roster['episodes'][1]=deepcopy(roster['episodes'][0])
            elif mutation=='route':routes['routes'][-1]=deepcopy(routes['routes'][0])
            else:routes['routes'][0]['GPU_required']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.build_jobs(roster,routes)

    def test_explicit_stage_flags_and_actual_prior_stop(self):
        row=self.jobs()[0];cpu=m.command(row,True);native=m.command(row,False)
        for argv in (cpu,native):
            self.assertIn(m.WRAPPER,argv);self.assertIn(m.SIM,argv);self.assertIn(m.WORKER,argv)
            self.assertEqual(argv[argv.index('--split')+1],'train');self.assertEqual(argv[argv.index('--overlay-lock')+1],m.SOURCE_LOCK)
        self.assertIn('--check',cpu);self.assertNotIn('--execute',cpu);self.assertNotIn('--prior-stop',cpu)
        self.assertIn('--execute',native);self.assertNotIn('--check',native)
        self.assertEqual(native[native.index('--prior-stop')+1],m.PRIOR_STOP)

    def test_pending_real_lock_cannot_prepare_or_execute(self):
        with patch.object(m,'SOURCE_LOCK_SHA',None),self.assertRaisesRegex(ValueError,'pending'):m.overlay_module()

    def test_delegates_real_controlled_stop_without_fake192_completion(self):
        ptr=dict(path=m.PRIOR_STOP,sha256=m.PRIOR_STOP_SHA);fn=NS(require_prior_stop=lambda root,name:ptr)
        with patch.object(m,'overlay_module',return_value=fn):out=m.predecessor_complete()
        self.assertTrue(out['incomplete_prior_cohort']);self.assertFalse(out['prior192_readiness_granted'])
        with patch.object(m,'overlay_module',return_value=NS(require_prior_stop=lambda *a:{'path':m.PRIOR_STOP,'sha256':'bad'})),self.assertRaises(ValueError):m.predecessor_complete()

    def test_root_technical_review_binds_five_and_prior_stop(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
            lock={'path':m.SOURCE_LOCK,'sha256':'l'*64};prior={'prior_controlled_stop':{'path':m.PRIOR_STOP,'sha256':m.PRIOR_STOP_SHA}}
            value=dict(schema=m.REVIEW_SCHEMA,review_complete=True,release_bounded_five=True,reviewer='root',fixed_TRAIN_sources=m.FIXED_SOURCES,
                source_overlay_lock=lock,prior_controlled_stop=prior['prior_controlled_stop'],development_regression_only=True,formal192_readiness=False,changed_factor=m.CHANGED_FACTOR)
            m.fresh(Path(temp)/m.ROOT_REVIEW,value);self.assertEqual(m.root_review(lock,prior),m.pointer(m.ROOT_REVIEW))
            value['formal192_readiness']=True;(Path(temp)/m.ROOT_REVIEW).write_text(json.dumps(value))
            with self.assertRaises(ValueError):m.root_review(lock,prior)

    def test_original_741_clock_accepts_false_true_and_fatal_without_mutation(self):
        for success,fatal in [(False,False),(True,False),(False,True)]:
            c=sample(m.BASKETS[0],success,fatal);before=deepcopy(c);out=m.validate_clock_and_sampler(c)
            self.assertEqual(out['counted_clock']['ticks'],4);self.assertEqual(c,before)

    def test_clock_and_old_header_rejected(self):
        for mutation in ('ticks','seconds','version'):
            c=sample(m.BASKETS[0])
            if mutation=='ticks':c['counted_clock']['ticks']=5
            elif mutation=='seconds':c['counted_clock']['seconds']=.3
            else:c['version']='r10_full16_explicit_author_action_outcome_v744'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.validate_clock_and_sampler(c)

    def test_complete_false_fatal_and_motionfalse_retained_compact(self):
        for success,fatal in [(False,False),(False,True),(True,False)]:
            with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
                row=self.jobs()[0];case,lock,cpu,native=self.evidence(Path(temp),row,success,fatal)
                result=m.completed_case(row,cpu,native,lock)
                self.assertIs(result['native_success'],success);self.assertIs(result['native_terminal_failure'],fatal);self.assertIs(result['motion_pass'],False)
                self.assertNotIn('execution',result);self.assertNotIn('geometry_intents',result)

    def test_all_six_phase_references_are_rehashed(self):
        for stage in ('CPU_check','native'):
            for field in ('launch','terminal','sidecar'):
                with self.subTest(stage=stage,field=field),TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
                    row=self.jobs()[0];case,lock,cpu,native=self.evidence(Path(temp),row)
                    target=cpu if stage=='CPU_check'else native;(Path(temp)/target[field]['path']).write_text('{} changed')
                    with self.assertRaises(ValueError):m.completed_case(row,cpu,native,lock)

    def test_wrong_case_identity_header_native_type_or_reset_rejected(self):
        for mutation in ('identity','version','fatal_type','reset','cleanup'):
            with self.subTest(mutation=mutation),TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
                row=self.jobs()[0];case,lock,cpu,native=self.evidence(Path(temp),row)
                if mutation=='reset':
                    p=Path(temp)/row['out']/'owned_hook_receipt.json';h=json.loads(p.read_text());h['reset'].append({'complete':True});p.write_text(json.dumps(h))
                else:
                    if mutation=='identity':case['identity']={'sha256':'changed'}
                    elif mutation=='version':case['version']='old'
                    elif mutation=='fatal_type':case['native_terminal_failure']=0
                    else:case['cleanup_completed']=False
                    (Path(temp)/row['out']/'case.json').write_text(json.dumps(case))
                with self.assertRaises(ValueError):m.completed_case(row,cpu,native,lock)

    def test_task_phase_binding_includes_Buzz1130_and_rejects_ring_factor(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
            for row in self.jobs():
                protocol=self.protocol(Path(temp),row,{})
                m.validate_protocol(row,protocol,{})
                protocol['changed_factor']='author_ring_terminal_canonical_waypoint'
                with self.assertRaises(ValueError):m.validate_protocol(row,protocol,{})

    def test_mock_child_actual_PID_worker_namespace_and_explicit_mode(self):
        ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        for check in (True,False):
            with self.subTest(check=check),TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
                root=Path(temp);row=self.jobs()[0];lock=dict(path=m.SOURCE_LOCK,sha256='l'*64)
                protocol=self.protocol(root,row,lock);phase='CPU_check'if check else'native'
                execution=dict(complete=True,mode='CPU_check'if check else'new_protocol_native_TRAIN',
                    native_started=not check,GPU_started=False,actual_split='train',native_execution_ready=True,
                    native_false_preserved=True,execution_protocol=protocol)
                side=dict(schema=m.CANDIDATE_SCHEMA,complete=True,execution=execution,source_overlay_lock=lock,
                    formal_score_contribution=False,selection=dict(task=row['task'],raw_episode_id=row['raw'],method=row['method']),
                    process=dict(pid=12345,start_ticks=10),actual_worker=m.pointer(m.WORKER),actual_preparer=m.pointer(m.PREPARER),
                    prior_controlled_stop=None if check else dict(path=m.PRIOR_STOP,sha256=m.PRIOR_STOP_SHA),
                    frozen_OS_environment=dict(complete=True,outer_wrapper_pid=os.getpid(),outer_wrapper_start_ticks=ticks,
                        observer_pid=12345,observer_start_ticks=10))
                m.fresh(root/row['runtime']/phase/'worker_sidecar.json',side)
                child=NS(pid=os.getpid(),wait=lambda:0,poll=lambda:0)
                with patch.object(m.subprocess,'Popen',return_value=child)as popen:
                    refs=m.child_phase(row,check,dict(source_overlay_lock=lock))
                self.assertEqual(popen.call_count,1);self.assertEqual(popen.call_args.args[0],m.command(row,check))
                self.assertEqual(set(refs),{'launch','terminal','sidecar'})
                terminal=json.loads((root/refs['terminal']['path']).read_text());self.assertEqual(terminal['returncode'],0)
                self.assertEqual(terminal['start_ticks'],ticks)

    def test_child_nonzero_return_is_infrastructure_and_preserves_terminal(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
            row=self.jobs()[0];child=NS(pid=os.getpid(),wait=lambda:7,poll=lambda:7)
            with patch.object(m.subprocess,'Popen',return_value=child)as popen,self.assertRaisesRegex(ValueError,'infrastructure'):
                m.child_phase(row,True,dict(source_overlay_lock={}))
            self.assertEqual(popen.call_count,1)
            terminal=Path(temp)/m.RUNTIME/'launches'/row['job_id']/'CPU_check'/'terminal.json'
            self.assertEqual(json.loads(terminal.read_text())['returncode'],7)

    def run_mock_owner(self,root,*,stop_after_cpu=False,infra=False):
        rows=self.jobs();plan=dict(jobs=rows,source_overlay_lock={});m.fresh(root/m.PLAN,plan);calls=[]
        def child(row,check,status):
            calls.append((row['job_id'],check));status['active']={'job_id':row['job_id'],'stage':'CPU_check'if check else'native'}
            if infra and not check:raise ValueError('original child infra')
            return {'CPU_fixture':True}
        def completed(row,cpu,native,lock):return dict(job_id=row['job_id'],native_success=False,native_terminal_failure=False,motion_pass=False)
        stop=iter([False,True])if stop_after_cpu else None
        with patch.object(m,'validate'),patch.object(m,'predecessor_complete',return_value={'controlled':True}),\
             patch.object(m,'child_phase',side_effect=child),patch.object(m,'completed_case',side_effect=completed),\
             patch.object(m,'_stop_requested',side_effect=(lambda:next(stop))if stop else(lambda:False)),patch('sys.stdout',new_callable=io.StringIO):
            if stop_after_cpu or infra:
                with self.assertRaises((InterruptedError,ValueError)):m.execute(m.sha(root/m.PLAN))
            else:m.execute(m.sha(root/m.PLAN))
        return calls,json.loads((root/m.RUNTIME/'status.json').read_text())

    def test_serial_exact_ten_phases_no_retry_and_all_five_false_kept(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
            calls,status=self.run_mock_owner(Path(temp));self.assertEqual([c[1]for c in calls],[True,False]*5)
            self.assertEqual(status['stage'],'complete');self.assertEqual(len(status['completed']),5)
            self.assertTrue(all(r['native_success']is False for r in status['completed']))
            self.assertFalse((Path(temp)/m.RUNTIME/'exec_completion.json').exists())
            with patch.object(m,'validate'),patch.object(m,'predecessor_complete',return_value={}):
                with self.assertRaises(ValueError):m.execute(m.sha(Path(temp)/m.PLAN))

    def test_STOP_after_CPU_starts_no_Native_and_does_not_kill_child(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)),patch.object(m.subprocess,'Popen',side_effect=AssertionError('real process')):
            calls,status=self.run_mock_owner(Path(temp),stop_after_cpu=True)
            self.assertEqual(len(calls),1);self.assertTrue(calls[0][1]);self.assertEqual(status['completed'],[])
            self.assertFalse((Path(temp)/m.RUNTIME/'terminal.json').exists())

    def test_infrastructure_failure_stops_without_retry_or_result_relabel(self):
        with TemporaryDirectory()as temp,patch.object(m,'ROOT',Path(temp)):
            calls,status=self.run_mock_owner(Path(temp),infra=True)
            self.assertEqual(len(calls),2);self.assertEqual(status['stage'],'failed');self.assertEqual(status['completed'],[])
            self.assertFalse(status['failed'][0]['native_failure_inferred'])

    def test_only_parent_flock_and_no_self_asserted_tool_exit(self):
        text=(PROJECT/m.SOURCE).read_text();self.assertNotIn('reserved_resources_after_TEST',text)
        self.assertIn('fcntl.flock',text);self.assertNotIn("fresh(base/'exec_completion.json'",text)


if __name__=='__main__':unittest.main()
