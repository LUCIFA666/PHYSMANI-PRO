"""Synthetic CPU raw-chain fixtures; never a native/resource qualification."""
import copy
import json
from pathlib import Path
import unittest

from our_mani.multitask import gpu_budget as budget
from our_mani.multitask.gpu_budget_audit import validate_budget_execution
from tests import test_project_GPU_budget_r10 as fixtures


class BudgetHistoryTests(unittest.TestCase):
    def process(self,pid):return fixtures.ProjectGPUBudgetTests.process(self,pid)
    def create(self):return fixtures.ProjectGPUBudgetTests.create(self)
    def tearDown(self):return fixtures.ProjectGPUBudgetTests.tearDown(self)
    def setUp(self):
        fixtures.ProjectGPUBudgetTests.setUp(self)
        controller=self.create();controller.start()
        profile_path=self.metadata/'profile.json'
        approval=dict(schema=budget.APPROVAL_SCHEMA,approved=True,budget_profile=budget.pointer(profile_path),
            execution_environment_contract=self.prepared['execution_environment_contract'],
            allowed_command_argv=self.profile['allowed_command_argv'],execution_inputs=[budget.pointer(self.extra)])
        apath=self.work/'CPU_ROOT_approval_fixture.json';apath.write_text(json.dumps(approval))
        _,approval_refs=budget.verify_approval(budget.ROOT if hasattr(budget,'ROOT') else Path(__file__).resolve().parents[1],profile_path,apath)
        context=self.process(902)
        controller.event('context_probe_process',dict(argv=['CPU fixture'],identity=context))
        controller.event('actual_context_probe',dict(status='ready',identity=context,actual_SM_count=96,
            no_kernels_models_or_simulator=True))
        self.fake.clients=[902];controller.validate_clients({902});controller.verify_compute_processes({902})
        controller.event('context_probe_cleanup',dict(returncode=0,stdout=json.dumps(dict(status='completed',
            context_destroyed=True,kernel_launches=0)),stderr='',identity_no_longer_live=True))
        child=self.process(903)
        controller.event('actual_workload_process',dict(identity=child,command=self.profile['allowed_command_argv'],
            original_workload_arguments_preserved=True))
        self.fake.clients=[904];identities=controller.validate_clients({904});controller.verify_compute_processes({904})
        controller.event('actual_workload_exit',dict(identity=child,returncode=0,identity_no_longer_live=True))
        self.fake.clients=[];controller.close()
        log=self.namespace/'l/server.log';log.write_text('Synthetic CPU fixture only; no daemon was launched.\n')
        self.receipt=dict(schema=budget.RECEIPT_SCHEMA,complete=True,budget_profile=budget.pointer(profile_path),
            approval=approval_refs,actual_workload_process=child,workload_returncode=0,
            observed_actual_CUDA_clients=identities,infrastructure_error=None,cleanup_error=None,
            automatic_retry=False,policy_failure_claimed=False,
            original_policy_data_model_clock_physics_and_release_gates_replaced=False,
            runtime_events=[budget.pointer(p) for p in sorted((controller.receipt_dir/'events').glob('*.json'))],
            actual_MPS_log_files=[budget.pointer(log)])
        self.receipt_path=controller.receipt_dir/'receipt.json'
        self.flush()

    def flush(self):self.receipt_path.write_text(json.dumps(self.receipt))

    def check(self,required=(904,)):
        return validate_budget_execution(Path(__file__).resolve().parents[1],budget.pointer(self.receipt_path),
            self.prepared['profile'],required_case_process_ids=required)

    def change_event(self,kind,transform):
        for ref in self.receipt['runtime_events']:
            path=Path(ref['path']);row=json.loads(path.read_text())
            if row['kind']==kind:
                transform(row['value']);path.write_text(json.dumps(row));ref.update(budget.pointer(path));self.flush();return
        self.fail('missing fixture event '+kind)

    def test_history_positive_and_actual_environment_contract(self):
        result=self.check()
        self.assertTrue(result['valid'])
        self.assertEqual(result['CUDA_SM_count'],96)
        self.assertEqual(result['environment_contract']['sha256'],self.prepared['execution_environment_contract']['sha256'])

    def test_history_wrong96_context_is_rejected_even_with_complete_true(self):
        self.change_event('actual_context_probe',lambda value:value.update(actual_SM_count=104))
        with self.assertRaisesRegex(ValueError,'context96'):self.check()

    def test_history_missing_native_case_client_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'case PID'):self.check(required=(999,))

    def test_history_forged_cleanup_not_accepted(self):
        self.change_event('cleanup',lambda value:value.update(partition_removed=False))
        with self.assertRaisesRegex(ValueError,'cleanup'):self.check()

    def test_history_unknown_resource_command_not_accepted(self):
        self.change_event('control_command',lambda value:value.update(input='sm_partition add GPU-CPU-fixture 13\n'))
        with self.assertRaisesRegex(ValueError,'unknown resource'):self.check()

    def test_history_client_environment_or_log_bytes_rejected(self):
        self.change_event('actual_owned_client_identities',lambda rows:rows[0]['MPS_environment'].update(CUDA_MPS_SM_PARTITION='foreign'))
        with self.assertRaisesRegex(ValueError,'environment'):self.check()


if __name__=='__main__':unittest.main(verbosity=2)
