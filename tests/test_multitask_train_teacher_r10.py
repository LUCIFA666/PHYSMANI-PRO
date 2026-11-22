"""CPU boundaries for committed TRAIN supervision; no simulator imports."""
import unittest
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from our_mani.contracts import Pose,RobotState
from our_mani.multitask.train_teacher import (CommittedWaypointFollower,FollowerConfig,
    TeacherPathPlanningFailure,joint_backoff,execute_train_reference,NativeRMLPath)
from our_mani.multitask.train_teacher import plan_arm_path,native_path_planner
from tests.test_multitask_collection_r10 import collection,MockEnv,MockTracker,native_targets,teacher_label


def robot(time=0.,position=(0.,0.,0.),q=None):
    return RobotState(time,Pose(np.asarray(position)),np.zeros(3),np.zeros(3),
        np.zeros(7) if q is None else np.asarray(q),np.zeros(7))


def label(stage='wp0',phase='approach',opening=0.,goal=(0.,0.,0.)):
    value=teacher_label();value.update(stage=stage,phase=phase,opening=opening,
        tcp_xyzw=np.r_[goal,[0.,0.,0.,1.]],manipulated_role='ball');return value


class FakePath:
    def __init__(self,steps=2):self.steps=steps;self.calls=0;self.closed=False
    def step(self):self.calls+=1;return self.calls>=self.steps
    def get_executed_joint_position_action(self):return np.full(7,.8)
    def close(self):self.closed=True


def feedback(time,done=True,recovery=False):
    return dict(timestamp=time,measured_tcp_xyzw=Pose(np.zeros(3)).xyzw(),
        gripper_actuation_done=done,joint_limit_recovery_action=recovery)


class FollowerChecks(unittest.TestCase):
    def follower(self,teacher=None,path=None,settle=0,initial=1.):
        self.queries=[];self.path=FakePath() if path is None else path
        def query():self.queries.append(1);return label() if teacher is None else teacher()
        return CommittedWaypointFollower(query,lambda goal:self.path,
            FollowerConfig(.02,.10,.005,settle),initial_opening=initial)
    def command(self,follower,now=0.,position=(0.,0.,0.),q=None):
        return follower.command(robot(now,position,q),np.full(7,-1.),np.ones(7))

    def test_path_pending_and_gripper_pending_do_not_requery_or_close_early(self):
        f=self.follower()
        a=self.command(f);self.assertEqual(a['opening'],1.);self.assertEqual(a['execution_state'],'follow_path')
        b=self.command(f,.05,(.1,0,0));self.assertEqual(b['opening'],1.);self.assertTrue(b['path_done'])
        self.command(f,.10,(.1,0,0));self.assertEqual(self.path.calls,2);self.assertEqual(len(self.queries),1)
        c=self.command(f,.15);self.assertEqual(c['opening'],0.);self.assertTrue(c['measured_waypoint_arrival'])
        self.assertFalse(f.after_tick(feedback(.20,False)))
        self.command(f,.20);self.assertEqual(len(self.queries),1)
        self.assertTrue(f.after_tick(feedback(.25,True)))
        self.assertEqual(f.commands[0]['command_completed_at'],.25)
        self.assertEqual(f.commands[0]['completion_target_type'],'committed_world_waypoint')
        self.command(f,.25);self.assertEqual(len(self.queries),2);self.assertTrue(self.path.closed)

    def test_waypoint_is_committed_even_if_teacher_would_move(self):
        calls=[]
        def query():calls.append(1);return label(goal=(len(calls),0,0))
        f=self.follower(teacher=query,path=FakePath(1))
        self.command(f);a=self.command(f,.05)
        np.testing.assert_array_equal(a['label']['tcp_xyzw'][:3],[1,0,0]);self.assertEqual(len(calls),1)

    def test_q_recovery_pauses_path_and_raw_query_but_clock_age_advances(self):
        f=self.follower();self.command(f)
        a=self.command(f,.20,q=np.full(7,1.001))
        self.assertEqual(a['execution_state'],'public_joint_limit_recovery');self.assertEqual(self.path.calls,1)
        self.assertEqual(len(self.queries),1);self.assertAlmostEqual(a['command_age_seconds'],.2)
        np.testing.assert_allclose(a['joint_reference'],.995)
        self.assertFalse(f.after_tick(feedback(.25,True,True)));self.assertEqual(f.commands[0]['executed_tick_count'],0)
        f2=self.follower();a=self.command(f2,q=np.full(7,-1.001));self.assertIsNone(a['label']);self.assertFalse(self.queries)

    def test_release_settling_uses_explicit_counted_ticks(self):
        f=self.follower(teacher=lambda:label(opening=1.),path=FakePath(1),initial=0.,settle=2)
        a=self.command(f);self.assertEqual(a['opening'],1.)
        self.assertFalse(f.after_tick(feedback(.05)));self.assertEqual(f.state,'release_settle')
        held=self.command(f,.05);np.testing.assert_array_equal(held['joint_reference'],np.zeros(7))
        self.assertFalse(f.after_tick(feedback(.10)))
        self.command(f,.10);self.assertTrue(f.after_tick(feedback(.15)))
        self.assertEqual(len(self.queries),1);self.assertEqual(f.commands[0]['gripper_completed_at'],.05)

    def test_real_release_requires_exactly_ten_hold_ticks_and_initial_open_skips_them(self):
        f=self.follower(teacher=lambda:label(opening=1.),path=FakePath(1),initial=0.,settle=10)
        self.command(f);self.assertFalse(f.after_tick(feedback(.05)))
        for i in range(10):
            a=self.command(f,.05*(i+1));self.assertEqual(a['execution_state'],'release_settle')
            np.testing.assert_array_equal(a['joint_reference'],np.zeros(7))
            self.assertEqual(f.after_tick(feedback(.05*(i+2))),i==9)
        self.assertAlmostEqual(f.commands[0]['command_completed_at'],.55)
        initial=self.follower(teacher=lambda:label(opening=1.),path=FakePath(1),initial=1.,settle=10)
        self.command(initial);self.assertTrue(initial.after_tick(feedback(.05)))

    def test_planning_failure_is_preserved_without_retry_or_fake_completion(self):
        calls=[]
        def teacher():calls.append(1);return label()
        def planner(goal):raise TeacherPathPlanningFailure('no collision-free TRAIN path')
        f=CommittedWaypointFollower(teacher,planner,FollowerConfig(.02,.1,.005,0),initial_opening=1.)
        self.command(f);self.command(f,.05)
        self.assertEqual(len(calls),1);self.assertFalse(f.commands[0].get('command_completed_at'))
        f.finish('fixed_tick_budget',10.)
        self.assertEqual(f.commands[0]['cutoff_reason'],'fixed_tick_budget')

    def test_in_bound_q_is_not_projected_and_invalid_limits_rejected(self):
        self.assertIsNone(joint_backoff([.999],[-1.],[1.],.005))
        with self.assertRaises(ValueError):FollowerConfig(.02,.1,.005,-1)
        with self.assertRaises(Exception):joint_backoff([0.],[0.],[.001],.005)


class DirectExecutionChecks(unittest.TestCase):
    def environment(self):
        ticks=[];targets=[];grasped=[];released=[]
        arm=SimpleNamespace(get_joint_positions=lambda:[0.]*7,set_joint_target_positions=lambda q:targets.append(q),
            get_tip=lambda:SimpleNamespace(get_pose=lambda:Pose(np.zeros(3)).xyzw()))
        fingers=[1.,1.]
        def actuate(opening,velocity):fingers[:]=[opening]*2;return True
        gripper=SimpleNamespace(get_open_amount=lambda:fingers.copy(),actuate=actuate,
            release=lambda:released.append(1),grasp=lambda body:grasped.append(body))
        env=SimpleNamespace(development_split='train',arm=arm,gripper=gripper,
            robot_model=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7)),
            task=SimpleNamespace(t=3.,get_graspable_objects=lambda:['body']),time=0.,_requested_opening=1.,
            ledger=SimpleNamespace(grasp_requests=0,release_requests=0),protocol={'gripper_velocity':.2},
            _before_action=lambda:None)
        def tick(phase):ticks.append(phase);env.task.t+=.05;env.time+=.05
        env._tick=tick
        return env,ticks,targets,grasped,released

    def test_direct_reference_position_clip_one_tick_and_once_close(self):
        env,ticks,targets,grasped,released=self.environment()
        a=execute_train_reference(env,np.full(7,2.),0.,phase='teacher')
        b=execute_train_reference(env,np.full(7,.8),0.,phase='teacher')
        np.testing.assert_array_equal(a['applied_joint_reference'],np.ones(7))
        np.testing.assert_array_equal(b['applied_joint_reference'],np.full(7,.8))
        self.assertEqual(len(ticks),2);self.assertEqual(grasped,['body']);self.assertEqual(env.ledger.grasp_requests,1)
        self.assertTrue(a['gripper_actuation_done']);np.testing.assert_array_equal(a['gripper_open_amount_before'],[1.,1.])
        self.assertEqual(a['timestamp'],.05)
        np.testing.assert_array_equal(a['gripper_open_amount_after'],[0.,0.]);self.assertFalse(a['velocity_reference_increment_clipped'])
        execute_train_reference(env,np.zeros(7),1.,phase='release');self.assertEqual(len(released),1)
        env.development_split='validation'
        with self.assertRaises(Exception):execute_train_reference(env,np.zeros(7),1.,phase='forbidden')

    def test_single_point_path_has_no_native_RML_initialization(self):
        native=SimpleNamespace(_path_points=np.zeros(7),_num_joints=7,_rml_handle=None,
            get_executed_joint_position_action=lambda:native._joint_position_action)
        path=NativeRMLPath(native);self.assertTrue(path.step());np.testing.assert_array_equal(path.get_executed_joint_position_action(),np.zeros(7))
        path.close()
        with self.assertRaises(Exception):path.step()


class PureArmPlanningChecks(unittest.TestCase):
    def test_algorithm_enum_matches_real_readonly_author_API(self):
        config=json.loads((collection.BASE/'common_training_smoke_config_v5_DRAFT.json').read_text())
        path=Path(config['baseline_repo'])/'third_party/PyRep/pyrep/const.py'
        tree=ast.parse(path.read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ConfigurationPathAlgorithms')
        self.assertTrue(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RRTConnect'
            for t in n.targets) for n in cls.body))
    def test_family_parameters_and_collision_fallback_preserve_full_path(self):
        class NoPath(Exception):pass
        for family,initial in {'moving_basketball_in_hoop':True,'push_moving_button':True,
            'insert_onto_rotating_peg':True,'put_rubbish_in_moving_bin':True,
            'beat_the_rotating_buzz':False,'place_cups_on_rotating_frame':False,
            'remove_cups_from_rotating_frame':False}.items():
            calls=[];raw=object()
            def path(position,**kw):
                calls.append(kw)
                if not kw['ignore_collisions']:raise NoPath()
                return raw
            actual=plan_arm_path(SimpleNamespace(get_path=path),Pose(np.zeros(3)),
                family=family,algorithm='RRTConnect',configuration_path_error=NoPath)
            self.assertIs(actual,raw);self.assertEqual([c['ignore_collisions'] for c in calls],
                [True] if initial else [False,True])
            for call in calls:
                for key,value in dict(trials=100,max_configs=10,max_time_ms=10,trials_per_goal=5,
                    relative_to=None,algorithm='RRTConnect').items():self.assertEqual(call[key],value)
        def fail(*args,**kw):raise NoPath('one allowed attempt')
        with self.assertRaises(NoPath):plan_arm_path(SimpleNamespace(get_path=fail),Pose(np.zeros(3)),
            family='insert_onto_rotating_peg',algorithm='RRTConnect',configuration_path_error=NoPath)

    def test_task_wrapper_forbidden_and_native_clock_tamper_rejected(self):
        class NoPath(Exception):pass
        class IKFailure(Exception):pass
        class InvalidAction(Exception):pass
        modules={'pyrep.errors':SimpleNamespace(ConfigurationPathError=NoPath,IKError=IKFailure),
            'pyrep.const':SimpleNamespace(ConfigurationPathAlgorithms=SimpleNamespace(RRTConnect='RRTConnect')),
            'rlbench.backend.exceptions':SimpleNamespace(InvalidActionError=InvalidAction)}
        env=SimpleNamespace(development_split='train',spec=SimpleNamespace(family='push_moving_button'),
            clock=0.,task=SimpleNamespace(t=3.),ledger=SimpleNamespace(ticks=0),
            robot_model=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7)))
        env.physics_time=lambda:env.clock
        def forbidden(*args):raise AssertionError('hidden-physics task wrapper called')
        env.task.get_path=forbidden
        native=SimpleNamespace(_path_points=np.zeros(7),_num_joints=7,_rml_handle=None)
        env.arm=SimpleNamespace(get_joint_positions=lambda:np.zeros(7),
            get_joint_target_positions=lambda:np.zeros(7),
            get_tip=lambda:SimpleNamespace(get_pose=lambda:Pose(np.zeros(3)).xyzw()),
            get_path=lambda *args,**kw:native)
        with patch.dict('sys.modules',modules):
            planner=native_path_planner(env)
            self.assertIs(planner(Pose(np.zeros(3))).path,native)
            def clock_violation(*args,**kw):env.clock+=.05;return native
            env.arm.get_path=clock_violation
            with self.assertRaisesRegex(Exception,'advanced physics'):planner(Pose(np.zeros(3)))


class CommittedCollectionChecks(unittest.TestCase):
    def test_committed_collection_keeps_raw_close_and_issued_close_distinct(self):
        env=MockEnv(2);env.robot_model=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7))
        f=CommittedWaypointFollower(lambda:label('wp1','acquire',0.),lambda goal:FakePath(2),
            FollowerConfig(.02,.1,.005,0),initial_opening=1.)
        def executor(reference,opening,phase):
            env.advance(Pose(np.zeros(3)),opening)
            return dict(feedback(env.physics_time()),applied_joint_reference=reference,
                gripper_open_amount_before=np.ones(2),gripper_open_amount_after=np.full(2,opening))
        a,_=collection.collect_episode(env,MockTracker(),split='train',maximum_ticks=200,
            behavior=collection.TEACHER_BEHAVIOR,follower=f,reference_executor=executor,
            mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),targets=native_targets)
        np.testing.assert_array_equal(a['teacher_desired_opening'][:2],[0.,0.])
        np.testing.assert_array_equal(a['teacher_applied_opening'][:2],[1.,0.])
        np.testing.assert_array_equal(a['teacher_command_is_new'],[True,False,False])
        np.testing.assert_array_equal(a['teacher_opening'],a['teacher_applied_opening'])
        self.assertEqual(len(f.commands),1);self.assertFalse(a['teacher_action_valid'][-1])

    def test_declared_teacher_hold_inventory_uses_distinct_ids_and_one_parent(self):
        folder_base=collection.BASE/'cpu_collection_checks';folder_base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder_base) as folder:
            folder=Path(folder)
            config=collection.BASE/'common_training_smoke_config_v5_DRAFT.json'
            plan=collection.build_plan(collection.BASE/'demonstrations/manifest.json',config,
                folder/'plan.json',sources_per_task=1)
            self.assertEqual(len(plan['cases']),28)
            self.assertEqual(len({c['parent_source_group'] for c in plan['cases']}),14)
            self.assertEqual(len({(c['task'],c['raw_episode_id']) for c in plan['cases']}),28)
            self.assertEqual(plan['augmentation_budget']['maximum_physics_ticks_per_parent'],260)
            self.assertEqual(plan['augmentation_budget']['behaviors'][collection.TRAIN_HOLD_BEHAVIOR]['replica_indices'],[0])
            self.assertIn('experiment_plan',plan)
            self.assertEqual(collection.verify_plan(folder/'plan.json')['plan_sha256'],plan['plan_sha256'])
            empty=collection.merge_completed(folder/'plan.json',folder/'empty')
            self.assertEqual(empty['expected'],28);self.assertEqual(empty['source_episode_count'],0)
            self.assertFalse(empty['complete']);self.assertTrue(empty['no_teacher_fallback_for_missing_hold_parent'])
            bad=json.loads(config.read_text());bad['collection_behavior_budgets']['train'][collection.TEACHER_BEHAVIOR]['replica_indices']=[1]
            with self.assertRaises(collection.CollectionError):collection.validate_collection_protocol(bad)
            with self.assertRaises(collection.CollectionError):
                collection.build_plan(collection.BASE/'demonstrations/manifest.json',config,
                    folder/'wrong.json',sources_per_task=1,replicas=2)

    def test_hold_is_non_teacher_and_stops_at_fixed_budget_or_terminal(self):
        env=MockEnv(100)
        a,_=collection.collect_episode(env,MockTracker(),split='train',maximum_ticks=60,
            behavior=collection.TRAIN_HOLD_BEHAVIOR,mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),
            targets=native_targets)
        self.assertEqual(len(a['times']),61);self.assertFalse(a['teacher_action_valid'].any())
        self.assertFalse(a['teacher_command_is_new'].any());self.assertEqual(env.ledger.ticks,60)
        with self.assertRaises(collection.CollectionError):
            collection.collect_episode(MockEnv(),MockTracker(),split='train',maximum_ticks=60,
                behavior=collection.TRAIN_HOLD_BEHAVIOR,teacher=teacher_label,
                mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),targets=native_targets)

    def test_completed_ordered_phase_transition_and_native_final_success_are_separate_events(self):
        env=MockEnv(2);env.robot_model=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7))
        env.__class__=type('SuccessEnv',(MockEnv,),{'native_outcome':property(lambda self:
            SimpleNamespace(terminate=self.ledger.ticks>=2,success=self.ledger.ticks>=2))})
        stages=iter([label('wp0','approach',1.),label('wp1','acquire',1.)])
        f=CommittedWaypointFollower(lambda:next(stages),lambda goal:FakePath(1),FollowerConfig(.02,.1,.005,0),initial_opening=1.)
        def executor(reference,opening,phase):
            env.advance(Pose(np.zeros(3)),opening)
            return dict(feedback(env.physics_time()),applied_joint_reference=reference,
                gripper_open_amount_before=np.ones(2),gripper_open_amount_after=np.ones(2))
        events=[]
        a,_=collection.collect_episode(env,MockTracker(),split='train',maximum_ticks=200,
            behavior=collection.TEACHER_BEHAVIOR,follower=f,reference_executor=executor,
            mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),targets=native_targets,
            phase_order=('approach','acquire'),final_required_phase='acquire',phase_completion_events=events)
        self.assertEqual(len(events),2);self.assertEqual(events[0]['eligible_command_ids'],[0]);self.assertEqual(events[1]['eligible_command_ids'],[1])
        self.assertEqual(events[0]['evidence'],'completed_command_then_ordered_teacher_phase_transition')
        self.assertEqual(events[1]['evidence'],'TRAIN_native_success_during_executed_final_required_phase')
        self.assertTrue(a['teacher_command_completed'][:2].all());self.assertFalse(a['teacher_action_valid'][-1])
        self.assertEqual(events[0]['origin_reference_motion_mode'],'external')
        self.assertEqual(events[0]['current_reference_native_pose_xyzw'][:3],[666.,666.,666.])
        self.assertFalse(events[0]['committed_world_command_residual_is_current_body_residual'])

if __name__=='__main__':unittest.main()
