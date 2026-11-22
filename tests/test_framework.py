"""Contract and closed-loop transition tests. Test doubles are never benchmark methods."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import ObjectState, RobotState, Pose, HandoverRegion
from our_mani.perception import CameraCalibration, CausalTracker, fuse_views
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.reachability import TimedReachability
from our_mani.planning import RegionSelector
from our_mani.control import HandoverGate, VisualMPC
from our_mani.pipeline import ManipulationPipeline, Stage
from our_mani.faster import ActionBuffer, FasterApproach, CONDITIONING_VERSION, ACTION_SPACE, validate_metadata
from our_mani.evaluation import EpisodeResult, compare
from our_mani.execution import FeasibleExecutor, CartesianActionLimiter
from our_mani.capture_workflow import CaptureWorkflow
from our_mani.forecast_collision import ForecastCollisionScene
from our_mani.local_capture import LocalConfig


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("our_mani_faster_inputs", ROOT/"third_party/FASTER/src/openpi/policies/our_mani_policy.py")
inputs_module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = inputs_module
spec.loader.exec_module(inputs_module)


class CartesianTestRobot:
    name = "unit_test_cartesian_model_not_benchmark_robot"
    lower, upper = np.full(6, -2.), np.full(6, 2.)
    velocity_limits, acceleration_limits = np.full(6, 10.), np.full(6, 200.)
    collision_at = None

    def inverse(self, pose, seed):
        return np.r_[pose.position, Rotation.from_matrix(pose.rotation).as_rotvec()]

    def forward(self, joints):
        return Pose(joints[:3], Rotation.from_rotvec(joints[3:]).as_matrix())

    def jacobian(self, joints):
        return np.eye(6)

    def collision(self, joints, future_offset):
        return self.collision_at is not None and joints[0] > self.collision_at


def object_state(t=0., position=(.2, 0., 0.), velocity=(.05, 0., 0.), rotation=None, **kwargs):
    local = np.array([[-.02, -.02, 0.], [.02, -.02, 0.], [0., .04, 0.], [0., 0., .02]])
    local -= local.mean(0)
    pose = Pose(position, np.eye(3) if rotation is None else rotation)
    return ObjectState("target", t, pose, pose.apply(local), velocity, [0., 0., 0.], **kwargs)


def robot_state(t=0., position=(0., 0., 0.), velocity=(0., 0., 0.), **kwargs):
    return RobotState(t, Pose(position), velocity, [0., 0., 0.], np.r_[position, [0., 0., 0.]],
                      np.r_[velocity, [0., 0., 0.]], **kwargs)


def region(**kwargs):
    args = dict(region_id="capture", object_id="target", local_pose=Pose([0., 0., 0.]), radii=[.05, .05, .05])
    args.update(kwargs)
    return HandoverRegion(**args)


def make_goal():
    forecast = ConstantVelocityDynamics().predict([object_state()], horizon=1.2)
    return RegionSelector(TimedReachability(CartesianTestRobot())).select(forecast, [region()], robot_state(), 0.)


def actions(n):
    return np.tile(np.array([.2, 0., .1, 0., 0., 0., 1., 1.], np.float32), (n, 1))


class GeometryTests(unittest.TestCase):
    def test_depth_scale_and_world_extrinsic(self):
        c = CameraCalibration(np.eye(3), Pose([1., 2., 3.]), depth_scale=.001)
        cloud = c.backproject(np.full((2, 2), 1000), np.ones((2, 2), bool))
        np.testing.assert_allclose(cloud, [[1., 2., 4.], [2., 2., 4.], [1., 3., 4.], [2., 3., 4.]])

    def test_camera_timestamp_rejected(self):
        c = CameraCalibration(np.eye(3), Pose([0., 0., 0.]))
        with self.assertRaisesRegex(ValueError, "synchronization"):
            fuse_views([(c, .2, np.ones((2, 2)), np.ones((2, 2)))], 0.)

    def test_signed_simulator_camera_intrinsics_are_preserved(self):
        c = CameraCalibration(np.diag([-1., -1., 1.]), Pose([0., 0., 0.]))
        points = c.backproject(np.ones((2, 2)), np.ones((2, 2), bool))
        np.testing.assert_allclose(points[-1], [-1., -1., 1.])

    def test_tracker_recovers_causal_velocity_and_rotation(self):
        tracker = CausalTracker("target")
        cloud = object_state().points_world.copy()
        cloud -= cloud.mean(0)
        v, w = np.array([.12, -.03, .01]), np.array([0., 0., .4])
        for index in range(5):
            t = index*.05
            p = cloud@Rotation.from_rotvec(t*w).as_matrix().T + v*t
            state = tracker.update(p, t, corresponded=True)
        np.testing.assert_allclose(state.linear_velocity, v, atol=1e-10)
        np.testing.assert_allclose(state.angular_velocity, w, atol=1e-10)
        with self.assertRaises(ValueError):
            tracker.update(p, t, corresponded=True)

    def test_rotating_region_has_surface_velocity(self):
        reg = region(local_pose=Pose([.1, 0., 0.]))
        obj = Pose([1., 0., 0.], Rotation.from_euler("z", 90, degrees=True).as_matrix())
        p, v = reg.in_world(obj, np.zeros(3), np.array([0., 0., 2.]))
        np.testing.assert_allclose(p.position, [1., .1, 0.], atol=1e-12)
        np.testing.assert_allclose(v, [-.2, 0., 0.], atol=1e-12)

    def test_forecast_interpolates_rotation_and_rejects_extrapolation(self):
        obj = object_state()
        object.__setattr__(obj, "angular_velocity", np.array([0., 0., 1.]))
        f = ConstantVelocityDynamics().predict([obj], .3, .1)
        p, v, w = f.at("target", .15)
        np.testing.assert_allclose(p.position, obj.pose.position+.15*obj.linear_velocity)
        self.assertAlmostEqual(Rotation.from_matrix(p.rotation).magnitude(), .15)
        with self.assertRaises(ValueError):
            f.at("target", -.1)


class ReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.model = CartesianTestRobot()
        self.checker = TimedReachability(self.model)

    def test_feasible_terminal_twist(self):
        check = self.checker.check(robot_state(), Pose([.2, 0., 0.]), [.05, 0., 0.], [0., 0., 0.], .5)
        self.assertTrue(check.feasible, check.reason)
        self.assertAlmostEqual(check.joint_velocities[0], .05)

    def test_arrival_and_collision_filters(self):
        check = self.checker.check(robot_state(), Pose([.2, 0., 0.]), [0., 0., 0.], [0., 0., 0.], .001)
        self.assertFalse(check.feasible)
        self.model.collision_at = .1
        check = self.checker.check(robot_state(), Pose([.2, 0., 0.]), [0., 0., 0.], [0., 0., 0.], .5)
        self.assertEqual(check.reason, "trajectory_collision")

    def test_terminal_velocity_constraint(self):
        check = self.checker.check(robot_state(), Pose([.2, 0., 0.]), [100., 0., 0.], [0., 0., 0.], .5)
        self.assertEqual(check.reason, "terminal_joint_speed_limit")

    def test_value_cannot_override_feasibility(self):
        f = ConstantVelocityDynamics().predict([object_state()], 1.2)
        selector = RegionSelector(self.checker)
        impossible = region(region_id="impossible", local_pose=Pose([4., 0., 0.]), value=.99)
        valid = region(value=.3)
        goal = selector.select(f, [impossible, valid], robot_state(), 0.)
        self.assertEqual(goal.region.region_id, "capture")
        self.assertGreater(sum(selector.last_diagnostics["rejections"].values()), 0)

    def test_value_conditions_on_candidate_terminal_velocity(self):
        f = ConstantVelocityDynamics().predict([object_state()], 1.2)
        def value(**candidate):
            return .9 if candidate['robot_velocity'][0] < .04 else .1
        selector = RegionSelector(self.checker, success_value=value,
                                  relative_velocity_options=((-0.05, 0., 0.), (0.05, 0., 0.)))
        goal = selector.select(f, [region()], robot_state(), 0.)
        self.assertIsNotNone(goal)
        self.assertAlmostEqual(goal.linear_velocity[0], 0.)
        self.assertAlmostEqual(goal.region.value, .9)

    def test_out_of_domain_value_never_authorizes_capture(self):
        f = ConstantVelocityDynamics().predict([object_state()], 1.2)
        selector = RegionSelector(self.checker, success_value=lambda **kw: float('nan'))
        self.assertIsNone(selector.select(f, [region()], robot_state(), 0.))
        self.assertGreater(selector.last_diagnostics['rejections']['value_out_of_domain_or_invalid'], 0)

    def test_between_sample_position_extremum_cannot_cross_joint_limit(self):
        self.model.lower = np.full(6, -.15)
        self.model.upper = np.full(6, .15)
        checker = TimedReachability(self.model, minimum_samples=2)
        check = checker.check(robot_state(velocity=(2., 0., 0.)), Pose([0., 0., 0.]),
                              [2., 0., 0.], [0., 0., 0.], 1.)
        self.assertEqual(check.reason, 'joint_position_limit')


class PipelineTests(unittest.TestCase):
    def pipeline(self):
        self.invalidations = 0
        def invalidate():
            self.invalidations += 1
        return ManipulationPipeline(ConstantVelocityDynamics(), RegionSelector(TimedReachability(CartesianTestRobot())),
                                    VisualMPC(), [region()], Pose([0., 0., 0.]), invalidate_actions=invalidate)

    def test_recheck_contact_and_stable_lift(self):
        p = self.pipeline()
        d = p.step([object_state()], robot_state(), 0.)
        self.assertEqual(d.command, "request_faster")
        t = p.goal.arrival_time
        x = .2+.05*t
        d = p.step([object_state(t, (x, 0., 0.))], robot_state(t, (x, 0., 0.), (.05, 0., 0.)), t)
        self.assertEqual(d.stage, Stage.CLOSE)
        d = p.step([object_state(t+.05, (x, 0., 0.))], robot_state(t+.05, (x, 0., 0.), gripper_open=False, contact=True), t+.05)
        self.assertEqual(d.stage, Stage.LIFT)
        for delta in (.1, .2, .3, .5):
            now = t+delta
            obj = object_state(now, (x, 0., .1), (0., 0., 0.))
            robot = robot_state(now, (x, 0., .1), gripper_open=False, contact=True, task_success=True)
            d = p.step([obj], robot, now)
        self.assertEqual(d.stage, Stage.SUCCESS)
        self.assertGreaterEqual(self.invalidations, 2)

    def test_target_change_cancels_previous_approach(self):
        p = self.pipeline()
        p.step([object_state()], robot_state(), 0.)
        d = p.step([object_state(.05, (.7, 0., 0.))], robot_state(.05), .05)
        self.assertEqual(d.command, "hold_and_replan")
        self.assertIsNone(p.goal)

    def test_stale_observation_never_authorizes_handover(self):
        p = self.pipeline()
        p.step([object_state()], robot_state(), 0.)
        d = p.step([object_state()], robot_state(.5), .5)
        self.assertEqual(d.command, "hold_and_replan")

    def test_moving_too_fast_prevents_handover(self):
        goal = make_goal()
        t = goal.arrival_time
        obj = object_state(t, goal.pose.position)
        check = HandoverGate().check(goal, obj, robot_state(t, goal.pose.position, (2., 0., 0.)), t)
        self.assertFalse(check.allowed)

    def test_mpc_progress_and_acceleration_bound(self):
        robot = robot_state()
        obj = object_state(0., (.08, 0., 0.), (.03, 0., 0.))
        pose, v, w, info = VisualMPC().command(robot, obj, Pose([0., 0., 0.]))
        self.assertGreater(v[0], 0.)
        self.assertLessEqual(np.linalg.norm(v), .1001)
        self.assertLess(np.linalg.norm(pose.position-obj.pose.position), .08)

    def test_async_forecast_rechecked_against_current_target(self):
        p = self.pipeline()
        forecast = ConstantVelocityDynamics().predict([object_state()], 1.2)
        d = p.step([object_state(.05, (.7, 0., 0.))], robot_state(.05), .05, forecast=forecast)
        self.assertEqual(d.command, 'hold_and_replan')
        self.assertIsNone(p.goal)

    def test_approach_executor_cannot_close_gripper(self):
        executor = FeasibleExecutor(TimedReachability(CartesianTestRobot()))
        action = np.array([0., 0., 0., 0., 0., 0., 1., 0.])
        command, reason = executor.prepare(action, robot_state(), now=0., approach=True)
        self.assertIsNotNone(command, reason)
        self.assertEqual(command.gripper_open, 1.)


class FasterTests(unittest.TestCase):
    def test_joint_filter_integrates_bounded_acceleration_before_collision_checks(self):
        model = CartesianTestRobot()
        executor = FeasibleExecutor(TimedReachability(model))
        raw = np.array([1.,0.,0.,0.,0.,0.,1.,0.])
        robot = robot_state()
        command, reason = executor.prepare_joint_limited(raw, robot, now=0.)
        self.assertIsNotNone(command, reason)
        self.assertLessEqual(command.evidence['maximum_acceleration_fraction'], .5+1e-8)
        self.assertGreater(command.joints[0], 0.)
        model.collision_at = .05
        blocked, reason = executor.prepare_joint_limited(raw, robot, now=0.)
        self.assertIsNone(blocked)
        self.assertEqual(reason, 'trajectory_collision')

    def test_large_neural_waypoint_is_rate_limited_before_joint_validation(self):
        robot = robot_state()
        raw = np.array([1., 0., 0., 0., 0., 0., 1., 0.])
        filtered = CartesianActionLimiter().filter(raw, robot)
        velocity = (filtered[:3]-robot.pose.position)/.05
        self.assertAlmostEqual(velocity[0], .1)
        self.assertLessEqual(np.linalg.norm(velocity-robot.linear_velocity), 2.*.05+1e-9)
        command, reason = FeasibleExecutor(TimedReachability(CartesianTestRobot())).prepare(filtered, robot, now=0.)
        self.assertIsNotNone(command, reason)
        self.assertEqual(command.gripper_open, 1.)

    def test_old_goal_and_duplicate_final_cannot_execute(self):
        goal = make_goal()
        buffer = ActionBuffer()
        buffer.activate(goal.goal_id)
        request = buffer.request(goal, start_tick=0)
        self.assertEqual(buffer.accept(request, actions(3), first_index=0, now=.01), 3)
        buffer.pop(0, now=.02)
        buffer.accept(request, actions(3), first_index=0, now=.03)
        self.assertIsNotNone(buffer.pop(1, now=.04))
        buffer.invalidate()
        self.assertEqual(buffer.accept(request, actions(3), first_index=0, now=.05), 0)
        self.assertIsNone(buffer.pop(2, now=.06))

    def test_expired_buffered_action_is_not_executed(self):
        goal = make_goal()
        buffer = ActionBuffer(max_observation_age=.1)
        buffer.activate(goal.goal_id)
        request = buffer.request(goal, start_tick=0)
        buffer.accept(request, actions(1), first_index=0, now=.01)
        self.assertIsNone(buffer.pop(0, now=.2))

    def test_metadata_rejects_generic_pretrained_pi05(self):
        with self.assertRaises(ValueError):
            validate_metadata({"model": "pi05_base"})

    def test_official_partial_protocol_excludes_prefix_and_unfinished_tail(self):
        goal = make_goal()
        buffer = ActionBuffer()
        buffer.activate(goal.goal_id)
        request = buffer.request(goal, start_tick=10)
        buffer.accept(request, actions(2), first_index=0, now=0.)
        class ScriptedProtocolTestClient:
            def get_server_metadata(self):
                return {"our_mani_conditioning": CONDITIONING_VERSION, "action_space": ACTION_SPACE,
                        "has_trained": True, "checkpoint_sha256": "unit_test_double"}
            def infer_streaming(self, observation, on_actions_ready):
                assert observation["delay"] == 2
                chunk = actions(1)
                chunk[0, 0] = .33
                on_actions_ready(chunk)
                final = actions(10)
                final[2] = chunk[0]
                return {"actions": final}
        approach = FasterApproach(ScriptedProtocolTestClient(), buffer, clock=lambda: .01)
        try:
            self.assertTrue(approach.submit({"timestamp": 0.}, goal, now=0., start_tick=10, delay=2))
            approach.future.result(timeout=2.)
            self.assertIsNotNone(buffer.pop(10, now=.02))
            self.assertIsNotNone(buffer.pop(11, now=.02))
            self.assertAlmostEqual(float(buffer.pop(12, now=.02)[0]), .33, places=5)
            self.assertIsNone(buffer.pop(13, now=.02))
        finally:
            approach.close()

    def test_transform_preserves_goal_prefix_and_all_four_cameras(self):
        images = {c: np.full((8, 8, 3), index, np.uint8) for index, c in enumerate(inputs_module.CAMERAS)}
        goal = {"pose": [0., 0., .1, 0., 0., 0., 1.], "linear_velocity": [.1, 0., 0.],
                "angular_velocity": [0., 0., .2], "radii": [.03, .04, .05], "remaining_seconds": .4}
        result = inputs_module.OurManiInputs()({"observation/images": images, "observation/state": actions(1)[0],
                      "handover_goal": goal, "action_prefix": actions(2), "delay": 2, "prompt": "pick"})
        self.assertIn("arrival_s=0.400", result["prompt"])
        self.assertEqual(result["action_prefix"].shape, (2, 8))
        self.assertEqual(result["delay"], 2)
        np.testing.assert_equal(result["image"]["base_0_rgb"][0, 0], [0, 0, 0])
        np.testing.assert_equal(result["image"]["base_0_rgb"][0, -1], [1, 1, 1])
        np.testing.assert_equal(result["image"]["base_0_rgb"][-1, 0], [2, 2, 2])
        np.testing.assert_equal(result["image"]["base_0_rgb"][-1, -1], [3, 3, 3])


class WorkflowSafetyTests(unittest.TestCase):
    def workflow(self, success_value=None):
        buffer = ActionBuffer()
        selector = RegionSelector(TimedReachability(CartesianTestRobot()), success_value=success_value,
                                  minimum_success_probability=.5)
        # No network or policy benchmark result is produced by these tests.
        class PassiveApproach:
            def close(self): pass
            def submit(self, *args, **kwargs): return False
        workflow = CaptureWorkflow(ConstantVelocityDynamics(), selector, [region()], PassiveApproach(), buffer,
                                   Pose([.2,0,.2]), LocalConfig())
        workflow.goal, workflow.stage = make_goal(), Stage.APPROACH
        buffer.activate(workflow.goal.goal_id)
        self.addCleanup(workflow.close)
        return workflow, buffer

    def test_current_low_value_invalidates_queued_approach_and_prevents_capture(self):
        workflow, buffer = self.workflow(success_value=lambda **_: .2)
        goal = workflow.goal
        t = goal.arrival_time
        request = buffer.request(goal, start_tick=1, observation_time=t)
        buffer.accept(request, actions(1), first_index=0, now=t)
        robot = robot_state(t, goal.pose.position, goal.linear_velocity)
        obj = object_state(t, goal.pose.position, goal.linear_velocity)
        result = workflow.step({'timestamp':t}, robot, obj, tick=0)
        self.assertEqual(result.command, 'hold_and_replan')
        self.assertIsNone(workflow.local)
        self.assertIsNone(buffer.pop(1, now=t))

    def test_fresh_but_unsynchronized_target_cannot_authorize_handover(self):
        workflow, _ = self.workflow()
        t = workflow.goal.arrival_time
        obj = object_state(t-.1, workflow.goal.pose.position)
        result = workflow.step({'timestamp':t}, robot_state(t), obj, tick=0)
        self.assertEqual(result.command, 'hold_and_replan')
        self.assertIsNone(workflow.goal)

    def test_fresh_capture_window_is_not_cut_short_by_forecast_grid_end(self):
        from dataclasses import replace
        workflow, _ = self.workflow()
        workflow.goal = replace(workflow.goal, arrival_time=float(workflow.goal.forecast.times[-1]))
        t = workflow.goal.arrival_time+.1
        obj = object_state(t, (.3,0,0))
        robot = robot_state(t, (.3,0,0), obj.linear_velocity)
        result = workflow.step({'timestamp':t}, robot, obj, tick=0)
        self.assertEqual(result.stage, Stage.LOCAL)
        self.assertEqual(result.command, 'local_capture')

    def test_forecast_collision_probe_restores_scene_even_on_backend_error(self):
        class Shape:
            def __init__(self): self.pose = [9.,0.,0.,0.,0.,0.,1.]
            def get_pose(self): return list(self.pose)
            def set_pose(self, pose, reset_dynamics=True):
                self.assert_no_reset = not reset_dynamics
                self.pose = list(pose)
        shape = Shape()
        class Arm:
            fail = False
            def check_arm_collision(self):
                if self.fail: raise RuntimeError('backend failure')
                return shape.pose[0] > .22
        arm = Arm()
        scene = ForecastCollisionScene(arm, {'target':shape})
        forecast = ConstantVelocityDynamics().predict([object_state(velocity=(.1,0,0))],1.)
        scene.bind(forecast, 0.)
        self.assertTrue(scene.check(.5))
        self.assertEqual(shape.get_pose()[0], 9.)
        self.assertTrue(shape.assert_no_reset)
        arm.fail = True
        with self.assertRaisesRegex(RuntimeError, 'backend failure'):
            scene.check(.5)
        self.assertEqual(shape.get_pose()[0], 9.)
        self.assertTrue(scene.check(1.5))  # Unforecast time is conservatively rejected.


class EvaluationTests(unittest.TestCase):
    def row(self, method, episode, success, contract="same"):
        return EpisodeResult(method, "unit_test_only", episode, 0, success, "closed_loop_simulator", "test_checkpoint", contract)

    def test_paired_identity_and_known_discordance(self):
        a = [self.row("a", str(i), s) for i, s in enumerate([True, True, False, True])]
        b = [self.row("b", str(i), s) for i, s in enumerate([False, True, False, False])]
        result = compare(a, b, bootstrap_samples=500)
        self.assertEqual(result["paired_wins"], 2)
        self.assertEqual(result["paired_losses"], 0)
        self.assertEqual(result["paired_delta"], .5)
        with self.assertRaises(ValueError):
            compare(a, b[:-1])
        with self.assertRaises(ValueError):
            compare(a+[a[0]], b+[b[0]])

    def test_observation_contract_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            compare([self.row("a", "0", True)], [self.row("b", "0", True, "different")])

    def test_repeated_seeds_remain_in_raw_episode_bootstrap_clusters(self):
        from dataclasses import replace
        a, b = [], []
        for episode, win in [('0', True), ('1', False)]:
            for seed in range(4):
                a.append(replace(self.row('a', episode, win), seed=seed))
                b.append(replace(self.row('b', episode, not win), seed=seed))
        report = compare(a, b, bootstrap_samples=2000)
        self.assertEqual(report['independent_raw_episodes'], 2)
        self.assertEqual(report['paired_bootstrap_95'], [-1., 1.])
        self.assertIsNone(report['mcnemar_exact_p'])
        self.assertFalse(report['wilson_intervals_assume_independent_trials'])

    def test_replay_proxy_cannot_enter_success_table(self):
        with self.assertRaises(ValueError):
            EpisodeResult("a", "t", "0", 0, True, "offline_replay", "checkpoint", "contract")


if __name__ == "__main__":
    unittest.main(verbosity=2)
