"""CPU public/dispatch/executor contracts, no real NN, Native, GPU or TEST."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.generic_reference_mpc_20261104_v1.contracts import WorldTCPReference, select_reference
from our_mani.generic_reference_mpc_20261104_v1.public_observation import pack_public_observation, validate_policy_packet
from our_mani.generic_reference_mpc_20261104_v1.native_backend import GenericTaskNativeBackend, make_executor
from our_mani.generic_reference_mpc_20261104_v1.reference_mpc import GenericReferenceMPC, TrackingConfig, qualify_first_action, actual_reference_batch
from our_mani.generic_reference_mpc_20261104_v1.runner import check, execute_reference_cycle


class Contracts(unittest.TestCase):
    def packet(self):
        return pack_public_observation(timestamp=0.,rgb=np.zeros((4,256,256,3),np.uint8),
            depth_m=np.ones((4,256,256),np.float32),K=np.tile(np.diag([-200.,-200.,1.]),(4,1,1)),
            camera_clips=np.tile([.01,4.],(4,1)),
            T=np.tile(np.eye(4),(4,1,1)),q=np.zeros(7),tcp_pose=[0,0,1,0,0,0,1])

    def proposals(self):
        return dict(world_tcp_positions=np.zeros((8,3)),world_tcp_rotations=np.tile(np.eye(3),(8,1,1)),
            opening_intent=np.full(8,.5),opening_logits=np.zeros(8),proposal_logits=np.array([0,3,3,0,0,0,0,0]),
            proposal_score_semantics='expert_command_demonstration_mixture_not_task_success',calibrated_success_probability=False,
            output_semantics='expert_command_intent_not_observed_state',handoff_tau_seconds=None,
            physical_handoff_qualified=False)

    def test_explicit_reference_selection_and_no_fake_tau(self):
        args=dict(timestamp=0.,policy_sha256='a'*64,e_sha256='b'*64)
        with self.assertRaises(ValueError): select_reference(self.proposals(),**args)
        ref=select_reference(self.proposals(),selection_rule='highest_demo_likelihood',**args)
        self.assertEqual(ref.selected_index,1)
        self.assertEqual(ref.selection_rule,'highest_demo_likelihood')
        bad=self.proposals();bad['handoff_tau_seconds']=.05
        with self.assertRaises(ValueError): select_reference(bad,selection_rule='highest_demo_likelihood',**args)

    def test_public_has_no_task_GT_open_or_known_operation_label(self):
        packet=self.packet();validate_policy_packet(packet)
        self.assertNotIn('task',packet);self.assertNotIn('operation_id',packet)
        self.assertEqual(packet['robot15'][-1],0.)
        self.assertFalse(packet['robot15_observed_mask'][-1])
        for field in ('held','contact','object_pose','Native_success','instance_mask','gripper_open'):
            bad=dict(packet,**{field:False})
            with self.assertRaises(ValueError):validate_policy_packet(bad)

    def test_calibration_and_real_robot_consistency(self):
        packet=self.packet();bad=dict(packet);bad['T']=packet['T'].copy();bad['T'][0,0,0]=2
        with self.assertRaises(ValueError):validate_policy_packet(bad)
        bad=dict(packet);bad['robot15']=packet['robot15'].copy();bad['robot15'][-1]=1
        with self.assertRaises(ValueError):validate_policy_packet(bad)

    def test_native_constructor_default_never_launches(self):
        with self.assertRaisesRegex(ValueError,'explicit Native'):
            GenericTaskNativeBackend('not_even_a_task','missing',seed=0,variation=0,budget_ticks=2)

    def test_task_generic_legal_graspable_dispatch_without_Cube_attribute(self):
        backend=object.__new__(GenericTaskNativeBackend)
        objects=[object(),object(),object()];attempts=[];actuated=[];released=[]
        backend.task=SimpleNamespace(get_graspable_objects=lambda:objects)
        backend.gripper=SimpleNamespace(grasp=lambda obj:attempts.append(obj),release=lambda:released.append(True),
            actuate=lambda target,velocity:actuated.append((target,velocity))or False)
        backend.requested_opening=1.;backend.gripper_velocity=.2
        backend._can_command=lambda:None;backend._clock_tuple=lambda:(0.,0.,0)
        backend.actuate_gripper(0.);backend.actuate_gripper(0.)
        self.assertEqual(attempts,objects);self.assertEqual(len(actuated),2)
        backend.actuate_gripper(1.);self.assertEqual(len(released),1)
        self.assertFalse(hasattr(backend.task,'target_block'))

    def test_original_executor_two_ticks_robot_only_and_completion_latch(self):
        class Backend:
            requested_opening=1.;time=0.;q=np.zeros(7);dq=np.zeros(7)
            def joint_limits(self):return dict(lower=np.full(7,-3),upper=np.full(7,3),velocity=np.full(7,2.))
            def get_joint_positions(self):return self.q.copy()
            def get_joint_velocities(self):return self.dq.copy()
            def set_joint_target_positions(self,q):self.target=np.asarray(q)
            def clock(self):return self.time
            def terminal(self):return False
            def remaining_ticks(self):return 200
            def actuate_gripper(self,opening):self.calls+=1;return True
            def tick(self,phase):self.time+=.05;self.q=self.target.copy()
            calls=0
        backend=Backend();executor,completed=make_executor(backend)
        receipt=executor.execute_joint_target(np.full(7,.1),opening=0.)
        self.assertEqual(receipt.executed_ticks,2);self.assertAlmostEqual(backend.time,.1)
        self.assertEqual(backend.calls,1) # completed command not restarted second tick
        self.assertEqual(completed.gripper_completion_receipt()['completed_same_target_ticks_not_restarted'],1)

    def test_endpoint_seven_path_and_stale_reference_rejected(self):
        class Backend:
            def clock(self):return .1
            def get_joint_positions(self):return np.zeros(7)
            def joint_limits(self):return dict(lower=np.full(7,-3),upper=np.full(7,3))
            def probe_joint_collision(self,q):self.probes.append(q.copy());return False
            probes=[]
        backend=Backend();state=dict(time=.1,q=np.zeros(7))
        ok,receipt=qualify_first_action(backend,state,dict(joint_goal=np.ones(7)))
        self.assertTrue(ok);self.assertEqual([p['alpha']for p in receipt['probes']],[1.,.125,.25,.375,.5,.625,.75,.875])
        self.assertEqual(len(backend.probes),8)
        with self.assertRaises(ValueError):qualify_first_action(backend,dict(state,time=0.),dict(joint_goal=np.ones(7)))

    def test_actual_executor_snapshot_only_and_all16_default_check(self):
        s=dict(time=.1,q=np.zeros(7),dq=np.zeros(7),actuator_state=dict(reference_valid=True,
            reference_clock=0.,reference_positions=np.zeros(7),reference_velocities=np.zeros(7),reference_accelerations=np.zeros(7),target_joints=np.ones(7)))
        with self.assertRaises(ValueError):actual_reference_batch(s,2)
        report=check()
        self.assertEqual(len(report['tasks']),16)
        self.assertTrue(report['implemented']['generated_development_native_reset'])
        self.assertFalse(report['native_execution_ready']);self.assertFalse(report['Native_started'])
        self.assertEqual(report['model_forward_calls'],0)

    def test_pending_future_open_intent_cannot_change_same100ms_motor_target(self):
        class Backend:
            def mechanical_state(self,h):return dict(time=0.,q=np.zeros(7),actuator_state=h)
            def clock(self):return 0.
            def get_joint_positions(self):return np.zeros(7)
            def joint_limits(self):return dict(lower=np.full(7,-3),upper=np.full(7,3))
            def probe_joint_collision(self,q):return False
            def metric_receipt(self):return {}
        class Executor:
            gripper_target=1.
            def state(self):return dict(gripper_target=self.gripper_target)
            def execute_joint_target(self,q,**kwargs):
                self.commands.append(kwargs);return SimpleNamespace(to_dict=lambda:dict(status='completed',gripper_done=True))
            commands=[]
        planner=object.__new__(GenericReferenceMPC);backend=Backend();executor=Executor()
        for intent in (0.,1.):
            ref=WorldTCPReference(0.,[1,2,3],[0,0,0,1],intent,0,'caller_selected','a'*64,'b'*64)
            with patch.object(GenericReferenceMPC,'plan',return_value=dict(action=dict(joint_goal=np.zeros(7),opening=1.))):
                receipt=execute_reference_cycle(backend,executor,planner,ref)
            self.assertFalse(receipt['opening_intent_issued'])
            self.assertEqual(executor.commands[-1]['opening'],1.)
        with patch.object(GenericReferenceMPC,'plan',return_value=dict(action=dict(joint_goal=np.zeros(7),opening=0.))):
            with self.assertRaisesRegex(ValueError,'Pending BC'):
                execute_reference_cycle(backend,executor,planner,ref)

    def test_finger_rollout_holds_actual_command_independent_of_NN_intent(self):
        from our_mani.local_mpc_p2_20261103.reference import ReferenceLimits
        from our_mani.local_mpc_20261102.executor import JointLimits
        targets=[]
        limits=JointLimits(lower=np.full(7,-3),upper=np.full(7,3),velocity=np.full(7,2),
            acceleration=np.full(7,12),jerk=np.full(7,1000))
        def finger(state,refs,*args,**kwargs):
            targets.append(refs['gripper_target'].copy())
            return dict(finger_open_amounts=state['finger_open_amounts'],feasible=np.ones(len(state['q']),bool))
        fixture=SimpleNamespace(config=TrackingConfig(),limits=ReferenceLimits.from_executor_limits(limits),
            joint_response=SimpleNamespace(predict_joint=lambda q,dq,*a,**k:dict(q=q,dq=dq)),
            finger_response=SimpleNamespace(predict_execution=finger),
            kinematics=lambda q:np.tile(np.eye(4),(len(q),1,1)))
        state=dict(time=0.,q=np.zeros(7),dq=np.zeros(7),finger_open_amounts=np.ones(2),
            actuator_state=dict(reference_valid=False,gripper_target=1.))
        costs=[]
        for intent in (0.,1.):
            ref=WorldTCPReference(0.,[0,0,0],[0,0,0,1],intent,0,'caller_selected','a'*64,'b'*64)
            costs.append(GenericReferenceMPC._rollout(fixture,state,np.zeros((2,8,7)),ref)[0])
        np.testing.assert_array_equal(costs[0],costs[1])
        self.assertTrue(all(np.all(t==1.) for t in targets))


if __name__=='__main__':unittest.main()
