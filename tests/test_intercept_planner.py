import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.intercept_planner import JointTrajectory,CandidateInterception
from our_mani.reachability import hermite
from our_mani.contracts import Pose,RobotState,ObjectState,Reachability
from our_mani.dynamics import ConstantVelocityDynamics
from pathlib import Path
import json


class CartesianRobot:
    name='unit_test_only'
    lower=np.full(6,-4.);upper=np.full(6,4.)
    velocity_limits=np.full(6,10.);acceleration_limits=np.full(6,200.)
    collision_query=None
    def inverse(self,pose,seed):return np.r_[pose.position,Rotation.from_matrix(pose.rotation.copy()).as_rotvec()]
    def forward(self,q):return Pose(q[:3],Rotation.from_rotvec(q[3:]).as_matrix())
    def jacobian(self,q):return np.eye(6)
    def collision(self,q,t):return False


class CollisionScene:
    def bind(self,forecast,now):pass
    def check(self,t):return False


class JointTrajectoryTests(unittest.TestCase):
    def test_endpoints_and_terminal_velocity(self):
        q0=np.array([.2,-.4]);v0=np.array([.1,.2]);q1=np.array([.7,.1]);v1=np.array([.3,-.1])
        plan=JointTrajectory(2.,3.5,q0,v0,q1,v1)
        for t,q,v in ((2.,q0,v0),(3.5,q1,v1)):
            a,b=plan.at(t);np.testing.assert_allclose(a,q);np.testing.assert_allclose(b,v)
        q,v,_,times=hermite(q0,v0,q1,v1,1.5,21,return_times=True)
        for i,t in enumerate(times):
            a,b=plan.at(2.+t);np.testing.assert_allclose(a,q[i],atol=1e-12);np.testing.assert_allclose(b,v[i],atol=1e-12)

    def test_interception_has_explicit_terminal_velocity_and_fresh_handover(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/framework/pipeline_v2_intercept.json').read_text())
        rotation=Rotation.from_quat([0,1,0,0]).as_matrix()
        def robot(t,position,velocity):
            pose=Pose(position,rotation)
            return RobotState(t,pose,velocity,np.zeros(3),np.r_[position,[0,np.pi,0]],np.r_[velocity,[0,0,0]])
        def obj(t):
            p=np.array([.2+.05*t,0,.775]);pose=Pose(p)
            return ObjectState('target',t,pose,p+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),[.05,0,0],[0,0,0])
        workflow=CandidateInterception(ConstantVelocityDynamics(),CartesianRobot(),CollisionScene(),Pose([.2,0,1.]),config,predictor='cv',ranking='geometry')
        workflow.stage='seek'
        first=workflow.step(robot(0,[.2,0,.9],[0,0,0]),obj(0),10.)
        self.assertEqual(first['command'],'joint_trajectory')
        np.testing.assert_allclose(workflow.goal.linear_velocity,[.05,0,0])
        self.assertEqual(workflow.handovers,0)
        t=workflow.goal.arrival_time
        pose,velocity=workflow.goal.region.in_world(obj(t).pose,obj(t).linear_velocity,obj(t).angular_velocity)
        result=workflow.step(robot(t,pose.position,velocity),obj(t),8.)
        self.assertEqual(workflow.handovers,1)
        self.assertEqual(result['command'],'local_capture')

    def test_missing_target_does_not_reuse_a_stale_interception(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/framework/pipeline_v2_intercept.json').read_text())
        workflow=CandidateInterception(ConstantVelocityDynamics(),CartesianRobot(),CollisionScene(),Pose([.2,0,1.]),config,predictor='cv',ranking='geometry')
        robot=RobotState(1.,Pose([.2,0,.9]),[0,0,0],[0,0,0],np.zeros(6),np.zeros(6))
        result=workflow.step(robot,None,8.)
        self.assertEqual(result['command'],'hold')
        self.assertIsNone(workflow.goal)

    def test_value_ranking_selects_highest_probability_after_feasibility(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/framework/pipeline_v2_intercept.json').read_text())
        rotation=Rotation.from_quat([0,1,0,0]).as_matrix()
        position=np.array([.2,0,.9])
        robot=RobotState(0.,Pose(position,rotation),[0,0,0],[0,0,0],np.r_[position,[0,np.pi,0]],np.zeros(6))
        p=np.array([.2,0,.775])
        obj=ObjectState('target',0.,Pose(p),p+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),[.05,0,0],[0,0,0])
        value=lambda **state:float(state['object_pose'].position[0])
        workflow=CandidateInterception(ConstantVelocityDynamics(),CartesianRobot(),CollisionScene(),Pose([.2,0,1.]),config,predictor='cv',ranking='value',value=value)
        workflow.stage='seek'
        workflow.step(robot,obj,10.)
        self.assertAlmostEqual(workflow.goal.arrival_time,1.05)
        self.assertAlmostEqual(workflow.goal.region.value,.2+.05*1.05)

    def test_local_observation_gap_rebuilds_history_without_crashing(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/framework/pipeline_v2_intercept.json').read_text())
        workflow=CandidateInterception(ConstantVelocityDynamics(),CartesianRobot(),CollisionScene(),Pose([.2,0,1.]),config,predictor='context',ranking='geometry')
        class MissingHistory:
            def command(self,robot,obj):raise ValueError('context requires five or more increasing causal observations')
        workflow.stage='local';workflow.capture_start=0.;workflow.local=MissingHistory()
        robot=RobotState(1.,Pose([.2,0,.85]),[0,0,0],[0,0,0],np.zeros(6),np.zeros(6))
        p=np.array([.2,0,.775]);obj=ObjectState('target',1.,Pose(p),p+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),[0,0,0],[0,0,0])
        result=workflow.step(robot,obj,8.)
        self.assertEqual(result['command'],'hold')
        self.assertEqual(result['reason'],'local_context_history_rebuilding')
        self.assertEqual(workflow.stage,'local')

    def test_small_forecast_change_preserves_committed_arrival(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/framework/pipeline_v2_intercept.json').read_text())
        rotation=Rotation.from_quat([0,1,0,0]).as_matrix();position=np.array([.2,0,.9])
        def robot(t):return RobotState(t,Pose(position,rotation),[0,0,0],[0,0,0],np.r_[position,[0,np.pi,0]],np.zeros(6))
        def obj(t):
            p=np.array([.2+.05*t,0,.775])
            return ObjectState('target',t,Pose(p),p+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),[.05,0,0],[0,0,0])
        value=lambda **state:float(state['object_pose'].position[0])
        workflow=CandidateInterception(ConstantVelocityDynamics(),CartesianRobot(),CollisionScene(),Pose([.2,0,1.]),config,predictor='cv',ranking='value',value=value)
        workflow.stage='seek';workflow.step(robot(0.),obj(0.),10.)
        arrival=workflow.goal.arrival_time
        workflow.reachability.check=lambda *a,**k:Reachability(False,'temporary_servo_state_correction_infeasible')
        workflow.step(robot(.1),obj(.1),9.9)
        self.assertEqual(workflow.goal.arrival_time,arrival)
        self.assertEqual(workflow.plan.end,arrival)


if __name__=='__main__':unittest.main()
