import copy
import json
from pathlib import Path
import tempfile
import unittest
import io
from unittest.mock import patch

from our_mani.multitask import gpu_budget as budget
from scripts import run_project_GPU_budget_r10 as launcher
from scripts.run_PhysMani_task_VAL_under_budget_r10 import selected_command

ROOT=Path(__file__).resolve().parents[1]


class FakeControl:
    def __init__(self):self.calls=[];self.clients=[];self.apps='';self.fail_add=False;self.bad_remove=False
    def __call__(self,argv,*,env=None,input=None,timeout=30):
        self.calls.append(dict(argv=argv,env=env,input=input))
        stdout='';returncode=0
        if argv[0]=='nvidia-smi':
            stdout=('GPU-CPU-fixture, NVIDIA GeForce RTX 5090, 595.99.02, Default\n'
                    if 'uuid,name,driver_version,compute_mode' in argv[1] else self.apps)
        elif input=='get_server_list\n':stdout='901\n'
        elif input and input.startswith('sm_partition add '):
            stdout=('Failed to fulfill request\n' if self.fail_add else 'Partition GPU-CPU-fixture/ABC123 created\n')
        elif input=='ps\n':
            stdout='PID ID SERVER DEVICE NAMESPACE COMMAND ATTRIBUTES\n'+''.join(
                f'{pid} 1 901 GPU-CPU-fixture 402 python\n' for pid in self.clients)
        elif input and input.startswith('sm_partition rm '):
            stdout='Partition ABC123 '+('in use' if self.bad_remove else 'removed')+'\n'
        return dict(argv=list(argv),input=input,returncode=returncode,stdout=stdout,stderr='')


class ProjectGPUBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='b')
        self.work=Path(self.temp.name).resolve()
        self.fake=FakeControl()
        self.namespace=ROOT/'runtime/mps'/self.work.name
        self.metadata=self.work/'metadata'
        self.extra=self.work/'execution.json';self.extra.write_text('{}')
        self.prepared=budget.prepare(ROOT,self.metadata,self.namespace,
            ['/usr/bin/true'],[self.extra],run=self.fake)
        self.profile=json.loads((self.metadata/'profile.json').read_text())

    def tearDown(self):
        if self.namespace.exists():
            import shutil;shutil.rmtree(self.namespace)
        self.temp.cleanup()

    def process(self,pid):
        env={key:self.controller.env[key] for key in ('CUDA_VISIBLE_DEVICES',)+budget.MPS_KEYS if key in self.controller.env}
        return dict(pid=pid,parent_pid=1,start_ticks=1234+pid,argv=['CPU fixture'],MPS_environment=env,
            executable=str((budget.CONTROL if pid==900 else budget.SERVER).resolve()))

    def create(self):
        self.controller=budget.MPSBudgetController(ROOT,self.profile,self.work/'execution',run=self.fake,
            process_probe=self.process,daemon_discovery=lambda ns:[self.process(900)],existing_MPS=lambda:[])
        return self.controller

    def test_prepare_is_metadata_only_no_daemon_or_context(self):
        self.assertFalse(self.namespace.exists())
        self.assertTrue(self.profile['prepared_only'])
        self.assertFalse(self.profile['actual_MPS_or_workload_executed'])
        self.assertEqual(len(self.fake.calls),1)
        self.assertEqual(self.fake.calls[0]['argv'][0],'nvidia-smi')

    def test_exact_lifecycle_inherits_one_static_partition_and_cleans(self):
        controller=self.create();env=controller.start()
        self.assertEqual(env['CUDA_MPS_SM_PARTITION'],'GPU-CPU-fixture/ABC123')
        self.assertNotIn('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',env)
        self.fake.clients=[902]
        self.assertEqual(controller.validate_clients({902})[0]['pid'],902)
        self.fake.clients=[];controller.close();controller.close()
        self.assertEqual(sum(c['input']=='quit\n' for c in self.fake.calls),1)
        self.assertTrue(controller.closed)
        self.assertFalse(any('-c' in c['argv'] for c in self.fake.calls))

    def test_foreign_client_is_rejected_not_terminated_or_shutdown(self):
        controller=self.create();controller.start();self.fake.clients=[999]
        with self.assertRaisesRegex(ValueError,'foreign client'):controller.validate_clients({902})
        with self.assertRaisesRegex(ValueError,'clients remain'):controller.close()
        self.assertFalse(any(c['input']=='quit\n' for c in self.fake.calls))
        self.assertFalse(any('terminate_client' in (c['input'] or '') for c in self.fake.calls))

    def test_client_wrong_partition_and_compute_bypass_rejected(self):
        controller=self.create();controller.start();self.fake.clients=[902]
        original=controller.process_probe
        def wrong(pid):
            identity=original(pid);identity['MPS_environment']['CUDA_MPS_SM_PARTITION']='foreign';return identity
        controller.process_probe=wrong
        with self.assertRaisesRegex(ValueError,'inherit'):controller.validate_clients({902})
        self.fake.apps='999, unknown\n'
        with self.assertRaisesRegex(ValueError,'bypasses'):controller.verify_compute_processes({902},owned_pids={902,999})
        self.fake.clients=[];controller.close()

    def test_failed_partition_create_cleans_only_own_empty_daemon(self):
        controller=self.create();self.fake.fail_add=True
        with self.assertRaisesRegex(ValueError,'partition ID'):controller.start()
        controller.close()
        self.assertTrue(any(c['input']=='quit\n' for c in self.fake.calls))
        self.assertFalse(any((c['input'] or '').startswith('sm_partition rm') for c in self.fake.calls))

    def test_remove_returnzero_error_text_cannot_claim_cleanup(self):
        controller=self.create();controller.start();self.fake.bad_remove=True
        with self.assertRaisesRegex(ValueError,'removal'):controller.close()
        self.assertFalse(controller.closed)
        self.assertFalse(any(c['input']=='quit\n' for c in self.fake.calls))

    def test_unknown_existing_MPS_refuses_but_foreign_CUDA_is_untouched(self):
        controller=self.create();controller.existing_MPS=lambda:[999]
        with self.assertRaisesRegex(ValueError,'existing MPS'):controller.start()
        self.assertFalse(self.namespace.exists())
        controller.existing_MPS=lambda:[];self.fake.apps='999, unknown\n'
        controller.start();self.assertTrue(self.namespace.exists())
        self.assertEqual(controller.verify_compute_processes(set(),owned_pids=set()),[999])
        observation=controller.events[-1]['value']
        self.assertEqual(observation['foreign_observed_pids'],[999])
        self.assertTrue(observation['foreign_processes_untouched'])
        controller.close()

    def test_budget_widened_or_incomplete_source_closure_rejected(self):
        for kind in ('chunks','SM','source'):
            profile=copy.deepcopy(self.profile)
            if kind=='chunks':profile['chunks']=13
            elif kind=='SM':profile['partition_SM_count']=104
            else:profile['inputs']=[r for r in profile['inputs'] if r['path']!=str(budget.CONTROL.resolve())]
            with self.assertRaises(ValueError):budget.validate_profile(profile,ROOT)

    def test_symlink_and_long_namespace_rejected_before_write(self):
        outside=self.work/'outside';outside.mkdir();link=ROOT/'runtime/mps'/('l'+self.work.name)
        link.symlink_to(outside,target_is_directory=True)
        try:
            with self.assertRaises(ValueError):budget.canonical_inside(ROOT,link/'child')
        finally:link.unlink()
        profile=copy.deepcopy(self.profile);profile['namespace']=str(ROOT/'runtime/mps'/('x'*90))
        with self.assertRaisesRegex(ValueError,'short'):budget.validate_profile(profile,ROOT)

    def test_approval_exact_argv_final_inputs_and_new_contract(self):
        approval=dict(schema=budget.APPROVAL_SCHEMA,approved=True,budget_profile=self.prepared['profile'],
            execution_environment_contract=self.prepared['execution_environment_contract'],
            allowed_command_argv=['/usr/bin/true'],execution_inputs=[budget.pointer(self.extra)])
        path=self.work/'ROOT_approval.json';path.write_text(json.dumps(approval))
        budget.verify_approval(ROOT,self.metadata/'profile.json',path)
        for change in ('argv','inputs','approval'):
            altered=copy.deepcopy(approval)
            if change=='argv':altered['allowed_command_argv']+=['unknown']
            elif change=='inputs':altered['execution_inputs']=[]
            else:altered['approved']=False
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):budget.verify_approval(ROOT,self.metadata/'profile.json',path)

    def test_changed_execution_input_fails_without_launch(self):
        self.extra.write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError,'changed'):budget.validate_profile(self.profile,ROOT)
        self.assertFalse(self.namespace.exists())

    def test_context_handshake_requires_actual96_and_always_releases(self):
        for count in (96,104):
            controller=self.create() if count==96 else self.controller
            controller.env['CUDA_MPS_SM_PARTITION']='GPU-CPU-fixture/ABC123'
            identity=self.process(902)
            class Process:
                pid=902;returncode=0
                stdout=io.StringIO(json.dumps(dict(status='ready',identity=identity,actual_SM_count=count,
                    no_kernels_models_or_simulator=True))+'\n')
                def communicate(inner,content,timeout):
                    inner.release=content;return '{"context_destroyed":true}\n',''
            process=Process()
            with patch.object(launcher.subprocess,'Popen',return_value=process),\
                    patch.object(launcher,'process_identity',return_value=identity),\
                    patch.object(launcher,'identity_is_live',return_value=False),\
                    patch.object(launcher.select,'select',return_value=([process.stdout],[],[])),\
                    patch.object(controller,'validate_clients',return_value=[identity]),\
                    patch.object(controller,'verify_compute_processes',return_value=[902]):
                if count==96:launcher.handshake(controller,self.profile)
                else:
                    with self.assertRaisesRegex(ValueError,'exact96SM'):launcher.handshake(controller,self.profile)
            self.assertEqual(process.release,'\n')

    def test_task_selection_keeps_full275_and_executes_original_queue(self):
        plan=ROOT/'artifacts/framework/multitask_r10/PhysMani_early_VAL_approved_v12/queue_plan.json'
        argv,jobs=selected_command(plan,['moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed'],2)
        original=json.loads(plan.read_text())
        self.assertEqual(len(original['jobs']),275)
        self.assertEqual(jobs,[row for row in original['jobs'] if row['row']['task'] in
            ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')])
        self.assertEqual(argv[1],str(ROOT/'scripts/evaluate_physmani_projected_VAL_r10.py'))
        self.assertEqual(argv.count('--queue-job-id'),len(jobs))
        with self.assertRaises(AssertionError):selected_command(plan,['unknown'],2)


if __name__=='__main__':unittest.main()
