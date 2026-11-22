"""Synthetic public-only decision/observer contracts; no Native or fitting."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,tempfile
from unittest.mock import patch
import unittest
import numpy as np

from our_mani.contracts import Pose,ObjectState,Forecast
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_20261102.mpc import MPCConfig,validate_candidate
from our_mani.local_mpc_20261102.terminal_mpc import TerminalCostConfig
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.public_state import PublicStateAssembler
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from our_mani.local_mpc_control_20261103.public_pregrasp import (
    ObservedPublicBackend,PublicPregraspMetric,PublicPregraspConfig,
    qualify_forecast_pregrasp_candidates,run_public_pregrasp)


def public_result(t,*,available=True):
    p=Pose([.2,0.,.775]);obj=ObjectState('target',t,p,p.position+np.array([[0,0,0],[.01,0,0],[0,.01,0]]),np.zeros(3),np.zeros(3))
    times=t+np.arange(17)*.05
    forecast=Forecast(('target',),times,np.tile(p.position,(17,1,1)),np.tile(np.eye(3),(17,1,1,1)),np.zeros((17,1,3)),np.zeros((17,1,3)))
    return dict(timestamp=t,mode='FREE',available=available,free_forecast_available=available,
        controlled_mode_available=False,object_observation=obj,goal_pose=Pose([.3,0.,1.]),forecast=forecast if available else None,
        rotation_observable=False,Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
        forecaster_is_injected_test_double=True,tracker_dependency_injected=True,goal_marker_dependency_injected=True)


def candidate(identity='c0',deadline=.5):
    return validate_candidate(dict(id=identity,object_tcp_position=[0,0,-.015],object_tcp_quaternion=[0,1,0,0],deadline=deadline,
        approach_direction=[0,0,1],pregrasp_offset=.07,opening=0.,required_opening=1.,allowed_relative_speed=.14))


def robot(t,p=(.2,0.,.83)):
    return PublicRobotSensors(t,p,[0,1,0,0],np.zeros(7),np.zeros(7),np.ones(2))


class RawBackend:
    def __init__(self,budget=14):self.ticks=0;self.budget=budget;self.q=np.zeros(7);self.requested=1.;self.captures=0;self.probes=[]
    def clock(self):return self.ticks*.05
    def remaining_ticks(self):return self.budget-self.ticks
    def terminal(self):raise AssertionError('Native outcome must not terminate public success')
    def read_state(self):
        now=self.clock();p=[.2,0.,1.] if now<.2-1e-8 else [.2,0.,.83]
        return dict(time=now,tcp_position=p,tcp_quaternion=[0,1,0,0],q=self.q.copy(),dq=np.zeros(7),
            finger_open_amounts=[.8,.7],requested_opening=self.requested,
            object_position=[999.,888.,777.],object_quaternion=[0,0,0,1],
            object_linear_velocity=[42.,43.,44.],object_angular_velocity=[1.,2.,3.],
            held=True,native_success=True,lift_goal_position=[555.,666.,777.],secret_handle=314159)
    def observe_rgbd(self):
        self.captures+=1
        return dict(timestamp=self.clock(),**{'observation/images':{c:np.zeros((2,2,3),np.uint8) for c in ('front','8','16','36')},
            'observation/depths':{c:np.ones((2,2),np.float32) for c in ('front','8','16','36')},
            'observation/calibrations':{c:CameraCalibration(np.eye(3),Pose([0,0,0])) for c in ('front','8','16','36')}})
    def tick(self,phase):self.ticks+=1;return self.read_state()
    def probe_joint_collision(self,q):self.probes.append(np.asarray(q).copy());return False
    def solve_ik(self,*args):raise AssertionError('Native IK forbidden')


class Bridge:
    def __init__(self,always_unavailable=False):self.times=[];self.always_unavailable=always_unavailable
    def observe(self,observation,sensors,*,close_requested):
        assert set(observation)=={'timestamp','observation/images','observation/depths','observation/calibrations'}
        assert set(vars(sensors))=={'timestamp','tcp_position','tcp_quaternion','q','dq','finger_open_amounts'}
        assert close_requested is False
        self.times.append(sensors.timestamp)
        return public_result(sensors.timestamp,available=len(self.times)>=5 and not self.always_unavailable)


class AssemblerSpy(PublicStateAssembler):
    def __init__(self):super().__init__(allow_test_doubles=True);self.observed=[];self.builds=[]
    def observe_robot(self,sensors):self.observed.append(sensors.timestamp);return super().observe_robot(sensors)
    def build(self,result,sensors,h,*,phase_state=None):
        self.builds.append((sensors.timestamp,deepcopy(h)))
        return super().build(result,sensors,h,phase_state=phase_state)


class Executor:
    def __init__(self,backend):
        self.backend=backend;self.commands=[];self.limits=object();self.config=SimpleNamespace(reference_frequency=6.)
        self.h=dict(reference_valid=False,reference_positions=None,reference_velocities=None,reference_accelerations=None,
            target_joints=None,reference_clock=None,gripper_target=.75,gripper_done=None,reference_is_measured_state=False)
    def state(self):return deepcopy(self.h)
    def execute_joint_target(self,q,*,opening,phase):
        self.commands.append(dict(q=np.asarray(q).copy(),opening=opening,phase=phase))
        self.backend.backend.q=np.asarray(q).copy();self.backend.backend.requested=opening
        self.backend.tick(phase);self.backend.tick(phase)
        self.h=dict(reference_valid=True,reference_positions=np.asarray(q).copy(),reference_velocities=np.zeros(7),
            reference_accelerations=np.zeros(7),target_joints=np.asarray(q).copy(),reference_clock=self.backend.clock(),
            gripper_target=opening,gripper_done=True,reference_is_measured_state=False)
        return dict(executed_ticks=2,gripper_target=opening,target_joints=np.asarray(q).copy())


class NumericModel:
    requires_joint_state=True;requires_joint_goal=True
    def initialize_batch(self,*args):raise AssertionError('test planner double only')
    def step_batch(self,*args):raise AssertionError('test planner double only')


class PoseAdapter:
    def __init__(self):self.positions=[]
    def solve(self,q,p,r):
        self.positions.append(np.asarray(p).copy());n=len(q)
        return dict(joint_goal=np.zeros((n,7)),joint_goal_feasible=np.ones(n,bool),ik_position_error=np.zeros(n),ik_angle_error=np.zeros(n))


class GoalHelper:
    def __init__(self,change_goal=False):self.adapter=PoseAdapter();self.change_goal=change_goal;self.action_calls=[]
    def qualify_action(self,backend,state,action,*,require_predicted_joint_goal):
        assert require_predicted_joint_goal is True
        self.action_calls.append(deepcopy(action));goal=np.asarray(action['joint_goal']).copy()
        if self.change_goal:goal+=1.
        return goal,None,dict(model_joint_goal_match=True,DLS_solves=0)


class Planner:
    opening=1.;seen=[]
    def __init__(self,*args,**kwargs):self.config=args[0]
    def plan(self,state,candidates,*,warm_start):
        type(self).seen.append(deepcopy(state))
        assert state['object_position'][0]<1. and state['held'] is False
        assert state['Native_object_state_used'] is False and state['Native_success_used'] is False
        assert state['phase_state']['phase']=='approach' and state['phase_state']['updated_at']==state['time']
        action=dict(tcp_position=state['tcp_position'].copy(),tcp_quaternion=state['tcp_quaternion'].copy(),
                    joint_goal=np.full(7,.1),opening=type(self).opening,nominal_kind=0)
        return dict(action=action,selected_candidate_id=candidates[0]['id'],phase_state=state['phase_state'],warm_start=None,
                    budget=dict(sequences=256,horizon=8),diagnostics=dict(diagnostic_only=True))


class PregraspTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.root=Path(self.directory.name)
        Planner.opening=1.;Planner.seen=[]
    def controller(self,*,always_unavailable=False,budget=14,change_goal=False):
        raw=RawBackend(budget);bridge=Bridge(always_unavailable);assembler=AssemblerSpy()
        observed=ObservedPublicBackend(raw,self.root/'public_shadow',bridge=bridge,assembler=assembler,save_frames=False)
        observed.sample();executor=Executor(observed);helper=GoalHelper(change_goal)
        factory_calls=[]
        def factory(*args,**kwargs):
            factory_calls.append(kwargs)
            c=MPCConfig(**kwargs['mpc_config'])
            return SimpleNamespace(config=c,execution_model=NumericModel(),planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig()),helper,{}
        return raw,bridge,assembler,observed,executor,helper,factory,factory_calls
    def run_case(self,*args,**kwargs):
        raw,bridge,assembler,observed,executor,helper,factory,calls=self.controller(*args,**kwargs)
        config=dict(simulation_seconds=.7,physics_dt=.05,control_dt=.1,horizon_seconds=.8,terminal_goal_cost=True,
                    closure_permitted=False,public_pregrasp_position_tolerance_m=.02,public_pregrasp_angle_tolerance_rad=.2,
                    public_pregrasp_dwell_seconds=.1)
        with patch('our_mani.local_mpc_control_20261103.public_pregrasp.build_response_planner',side_effect=factory),\
             patch('our_mani.local_mpc_control_20261103.public_pregrasp.ForecastJointGoalCEMPlanner',Planner):
            report=run_public_pregrasp(observed,executor,self.root,config)
        return report,raw,bridge,assembler,observed,executor,helper,calls

    def test_public_only_success_50ms_observer_100ms_fresh_h_and_exact_dispatch(self):
        report,raw,bridge,assembler,observer,executor,helper,calls=self.run_case()
        self.assertTrue(report['public_pregrasp_reached']);self.assertEqual(report['canary_stop_reason'],'public_pregrasp_reached')
        self.assertEqual(raw.ticks,6);self.assertEqual(raw.captures,7)
        np.testing.assert_allclose(bridge.times,np.arange(7)*.05)
        self.assertEqual(assembler.observed,bridge.times)
        self.assertEqual([x[0] for x in assembler.builds],[.1,.2])
        self.assertTrue(all(abs(t-h['reference_clock'])<1e-9 for t,h in assembler.builds))
        self.assertTrue(all(x['opening']==1. for x in executor.commands))
        np.testing.assert_array_equal(executor.commands[-1]['q'],helper.action_calls[-1]['joint_goal'])
        self.assertEqual(calls[0]['mpc_config']['approach_enter'],.02)
        self.assertFalse(report['Native_task_success_claimed']);self.assertFalse(report['Native_success_exit_used'])
        labels=[json.loads(x) for x in (self.root/'public_shadow/evaluation_labels.jsonl').read_text().splitlines()]
        self.assertTrue(all(x['held'] and x['native_success'] for x in labels))
        decisions=(self.root/'public_decisions.jsonl').read_text()
        self.assertNotIn('999.0',decisions);self.assertNotIn('555.0',decisions)
        self.assertTrue(Planner.seen);self.assertEqual(Planner.seen[0]['lift_goal_position'].tolist(),[.3,0.,1.])

    def test_selected_close_is_explicitly_rejected_not_rewritten_as_same_model_action(self):
        Planner.opening=0.
        report,_,_,_,_,executor,_,_=self.run_case()
        self.assertFalse(report['public_pregrasp_reached'])
        self.assertEqual(report['canary_stop_reason'],'closure_not_permitted_in_pregrasp_canary')
        self.assertTrue(all(c['opening']==1. for c in executor.commands))
        rows=[json.loads(x) for x in (self.root/'public_decisions.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['proposed_action']['opening'],0.)
        self.assertEqual(rows[-1]['executed_action']['opening'],1.)
        self.assertEqual(rows[-1]['action_rejected_reason'],'closure_not_permitted_in_pregrasp_canary')

    def test_unavailable_warms_real_ticks_open_without_GT_success_exit_or_planner(self):
        report,raw,_,_,_,executor,_,_=self.run_case(always_unavailable=True,budget=8)
        self.assertFalse(report['public_pregrasp_reached']);self.assertEqual(report['canary_stop_reason'],'budget_exhausted')
        self.assertEqual(report['planned_decisions'],0);self.assertEqual(raw.ticks,8)
        self.assertTrue(all(c['opening']==1. for c in executor.commands));self.assertEqual(Planner.seen,[])

    def test_modified_dispatch_goal_is_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError,'changed before execution'):self.run_case(change_goal=True)

    def test_metric_requires_contiguous_current_geometry_no_duplicate_or_historical_success(self):
        metric=PublicPregraspMetric();metric.set_candidate(candidate())
        sequence=[(0.,.83),(.05,.87),(.1,.83),(.15,.83),(.2,.83),(.25,.87)]
        passed=[]
        for t,z in sequence:metric.observe(public_result(t),robot(t,(.2,0.,z)));passed.append(metric.passed)
        self.assertEqual(passed,[False,False,False,False,True,False])
        before=metric.dwell;metric.observe(public_result(.25),robot(.25,(.2,0.,.83)));self.assertEqual(metric.dwell,before)
        with self.assertRaises(ValueError):PublicPregraspConfig(position_tolerance_m=.03)
        with self.assertRaises(ValueError):PublicPregraspConfig(dwell_seconds=.05)

    def test_GRU_tau_qualification_not_CV_and_expired_slots_retained_without_sampling_past(self):
        t=.2;result=public_result(t);times=result['forecast'].times
        p=result['forecast'].positions.copy();p[:,0,0]+=.1*(times-t)**2
        result['forecast']=Forecast(('target',),times,p,result['forecast'].rotations,np.zeros((17,1,3)),np.zeros((17,1,3)))
        bound=BoundForecastExecutionModel.from_public_bridge(NumericModel(),result)
        candidates=[candidate(f'c{i}',deadline=.1 if i==0 else .5+.3*(i%2)) for i in range(8)]
        helper=GoalHelper();raw=RawBackend();seen=[];sample=bound.sample
        def observed(at):seen.append(at);return sample(at)
        state=dict(time=t,q=np.zeros(7),object_position=[.2,0.,.775],object_quaternion=[0,0,0,1])
        with patch.object(bound,'sample',side_effect=observed):values,receipt=qualify_forecast_pregrasp_candidates(raw,state,candidates,bound,helper)
        self.assertNotIn(.1,seen);self.assertEqual(len(values),8)
        self.assertEqual(values[0]['deadline'],.1);self.assertFalse(values[0]['native_pose_qualified'])
        self.assertEqual(len(raw.probes),7)
        self.assertAlmostEqual(helper.adapter.positions[0][1,0],.2+.1*(.8-t)**2)
        self.assertFalse(receipt['CV_target_used']);self.assertFalse(receipt['deadline_shifted'])

    def test_executor_must_use_observed_backend_and_closure_config_cannot_enable(self):
        raw,_,_,observer,executor,_,factory,_=self.controller()
        executor.backend=raw
        with self.assertRaisesRegex(ValueError,'executor backend'):run_public_pregrasp(observer,executor,self.root,{})
        executor.backend=observer
        with self.assertRaisesRegex(ValueError,'closure cannot'):run_public_pregrasp(observer,executor,self.root,{'closure_permitted':True})


if __name__=='__main__':unittest.main()
