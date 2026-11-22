"""CPU-only replay gates and original helper endpoint+7 path; no simulator."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from our_mani.local_mpc_20261103 import public_prefix_static_collision_replay_20261103_v2 as r
from our_mani.local_mpc_20261102 import worker as old
from our_mani.local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
REQ=r.ROOT/'artifacts/framework/local_mpc_20261102/V67_hold_C_first_action_static_collision_request_v1/request.json'


class FakeNative:
 def __init__(self,q,kin,limits):
  self.q=np.asarray(q).copy();self.dq=np.zeros(7);self.target=self.q.copy();self.kin=kin;self.limits=limits;self.time=0.;self.opening=1.;self.applied=[];self.probes=[];self.closed=False;self.observed=[]
  self.arm=SimpleNamespace(get_tip=lambda:SimpleNamespace(get_pose=self.pose));self.gripper=SimpleNamespace(get_open_amount=lambda:[self.opening,self.opening])
 def pose(self):
  T=self.kin(self.q);return np.r_[T[:3,3],matrix_quaternion(T[:3,:3])].tolist()
 def get_joint_positions(self):return self.q.copy()
 def get_joint_velocities(self):return self.dq.copy()
 def clock(self):return self.time
 def terminal(self):return self.closed or self.time>=.5
 def remaining_ticks(self):return 10-round(self.time/.05)
 def joint_limits(self):return self.limits
 def read_state(self):return{'opening':self.opening}
 def set_joint_target_positions(self,q):self.target=np.asarray(q).copy();self.applied.append(self.target.copy())
 def actuate_gripper(self,o):self.opening=o;return True
 def tick(self,phase):
  oldq=self.q.copy();self.q=self.target.copy();self.dq=(self.q-oldq)/.05;self.time+=.05;self.observed.append(r.robot(self))
 def probe_joint_collision(self,q):
  self.probes.append(np.asarray(q).copy());self.last_collision_probe=dict(valid=True,actual_q_qdot_servo_ik_and_target_native_state_preserved=True,physics_steps=0);return False
 def pop_tick_records(self):return[]
 def close(self):self.closed=True


class ReplayContracts(unittest.TestCase):
 def test_real_request_check_has_no_native_and_exact_prefix_lengths(self):
  with patch.object(old,'make_backend',side_effect=AssertionError('No Native')):
   report=r.check(REQ)
  self.assertFalse(report['ready']);self.assertFalse(report['Native_started']);self.assertEqual(report['NN_forward_calls'],0)
  self.assertEqual([a['prefix_ticks']for a in report['anchors']],[158,82,52])
  self.assertEqual([a['prefix_macros']for a in report['anchors']],[79,41,26])

 def test_robot_reference_h_and_requested_opening_failclosed(self):
  a=dict(timestamp=0.,tcp_position=[0.,0.,1.],tcp_quaternion=[0.,0.,0.,1.],q=[0.]*7,dq=[0.]*7,finger_open_amounts=[1.,1.]);r.match_robot(a,deepcopy(a))
  for key in('q','dq','tcp_position','finger_open_amounts'):
   bad=deepcopy(a);bad[key][0]+=.001
   with self.assertRaises(ValueError):r.match_robot(bad,a)
  h=dict(reference_valid=False,reference_positions=None,gripper_target=1.,gripper_done=None)
  with self.assertRaises(ValueError):r.match_h(dict(h,gripper_done=False),h)
  with self.assertRaises(ValueError):r.near([1.],[1.001],r.TOL['reference'],'reference')

 def test_mock_execute_branch_replays_two_ticks_and_queries_eight_never_applies_C(self):
  request=r.read(REQ);original=request['anchors'][0];kin=PoEKinematics(r.read(r.pointer(request['known_kinematics'])))
  limits=dict(lower=original['original_limits']['native_joint_position_lower'],upper=original['original_limits']['native_joint_position_upper'],velocity=original['original_limits']['native_joint_velocity_limits'])
  cfg=dict(joint_reference_acceleration=12.,joint_reference_jerk=1000.)
  q=np.asarray(original['public_robot']['q']);baseline=FakeNative(q,kin,limits);backend=CompletedGripperBackend(baseline);ex=old.make_executor(backend,cfg)
  h0=deepcopy(ex.state());robots={0.:r.robot(baseline)};goal=q+np.array([.0001,0,0,0,0,0,0]);receipt=ex.execute_joint_target(goal,opening=1.,phase='original_public_phase').to_dict();h1=deepcopy(ex.state());robots.update({round(x['timestamp'],9):x for x in baseline.observed})
  macro=dict(time=0.,actuator_state_before=h0,actuator_state_after=h1,executed_action=dict(joint_goal=goal.tolist(),opening=1.),executor=receipt)
  C=q+np.array([.0002,0,0,0,0,0,0]);T=kin(C)
  anchor=dict(id=r.ANCHORS[0],scene='static',timestamp=.1,public_robot=robots[.1],current_actuator_h=h1,C_first_command=dict(joint_goal=C.tolist(),opening=0.,static_FK=dict(tcp_matrix=T.tolist())))
  prepared=dict(anchor=anchor,config=cfg,prefix=[macro],ticks=receipt['ticks'],robots=robots);fresh=FakeNative(q,kin,limits)
  with tempfile.TemporaryDirectory()as temp,patch.object(r,'FAMILY',Path(temp)),patch.object(r,'check',return_value={'ready':True}),patch.object(r,'load_request',return_value=(request,[prepared],{})),patch.object(old,'environment_proof'),patch.object(old,'native_source_snapshot',return_value={'mock':'fixture'}),patch.object(old,'make_backend',return_value=fresh),patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'','DISPLAY':'fixture'}):
   result=r.execute(REQ,r.ANCHORS[0],Path(temp)/'newcase','unused','unused')
  self.assertTrue(result['complete']);self.assertTrue(result['collision_helper_qualified']);self.assertFalse(result['C_command_executed']);self.assertEqual(len(fresh.applied),2)
  self.assertEqual(len(fresh.probes),8);self.assertEqual(fresh.time,.1);self.assertTrue(fresh.closed)
  current=np.asarray(robots[.1]['q'])
  for actual,alpha in zip(fresh.probes,(1.,.125,.25,.375,.5,.625,.75,.875)):
   np.testing.assert_allclose(actual,(1-alpha)*current+alpha*C,rtol=0,atol=1e-15)
  self.assertFalse(any(np.array_equal(a,C)for a in fresh.applied))

if __name__=='__main__':unittest.main()
