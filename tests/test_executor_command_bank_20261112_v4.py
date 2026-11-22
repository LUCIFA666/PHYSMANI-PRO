import unittest
from types import SimpleNamespace
from our_mani.executor_contract_repair_20261112_v4.controller import execute_ranked_references


class CommandBankTests(unittest.TestCase):
    def refs(self):
        return [SimpleNamespace(timestamp=0.,position=(float(i),0,1),quaternion=(0,0,0,1),opening_intent=0.,selected_index=i)
                for i in range(3)]

    def test_rejected_primary_tries_next_existing_proposal(self):
        calls=[]
        def execute(action,**kwargs):
            calls.append(action.position[0])
            return dict(status='hardware_clearance_rejected'if len(calls)==1 else'chunk_complete',
                        executed_ticks=0 if len(calls)==1 else 10)
        reference,receipt,attempts=execute_ranked_references(SimpleNamespace(execute=execute),self.refs())
        self.assertEqual(calls,[0.,1.]);self.assertEqual(reference.selected_index,1)
        self.assertEqual(receipt['executed_ticks'],10);self.assertEqual(len(attempts),2)

    def test_partial_execution_never_reuses_consumed_observation_for_next_proposal(self):
        calls=[]
        def execute(action,**kwargs):calls.append(action);return dict(status='control_barrier_infeasible',executed_ticks=3)
        _,receipt,attempts=execute_ranked_references(SimpleNamespace(execute=execute),self.refs())
        self.assertEqual(len(calls),1);self.assertEqual(receipt['executed_ticks'],3)

    def test_invalid_robot_state_is_not_a_candidate_specific_rejection(self):
        calls=[]
        def execute(action,**kwargs):calls.append(action);return dict(status='measured_start_outside_bounds',executed_ticks=0)
        execute_ranked_references(SimpleNamespace(execute=execute),self.refs())
        self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
