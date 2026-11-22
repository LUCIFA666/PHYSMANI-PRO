import unittest
from types import SimpleNamespace
from collections import deque
from our_mani.execution_protocol_20261113_v2 import ExecutionRequest,ProtocolExecutor

POSE=[.3,.1,.9,0,0,0,1,0]

class Driver:
    def __init__(self):
        self.backend=SimpleNamespace(clock=lambda:1.)
        self.executor=SimpleNamespace(config=SimpleNamespace(clock_tolerance=1e-5),gripper_done=False,gripper_target=0.,reference_marker=object())
        self.static_completion_segments=frozenset();self.continuity=None;self._goal_history=deque()
        self.active_segment_id=None;self.active_packet=None;self.executed=[];self.continued=0
    def execute(self,action,**kwargs):
        self.executed.append((action,kwargs));self.active_segment_id=kwargs.get('segment_id');self.active_packet=action
        return dict(status='chunk_complete',executed_ticks=2,execution=[])
    def _continue_active(self,stop_requested):
        self.continued+=1
        return dict(status='chunk_complete',executed_ticks=2,execution=[])

class ProtocolTest(unittest.TestCase):
    def test_legacy_pose8_has_no_invented_timing_and_routes_to_existing_driver(self):
        request=ExecutionRequest.legacy(POSE,1.)
        self.assertEqual(request.pose8,tuple(POSE));self.assertIsNone(request.target_time_offset_seconds)
        self.assertEqual(ExecutionRequest.decode(request.to_wire()),request)
        timed=Driver();legacy=Driver();legacy.backend=timed.backend;legacy.executor=timed.executor
        bridge=ProtocolExecutor(timed,legacy_driver=legacy)
        bridge.execute(POSE,observation_time=1.)
        self.assertEqual(len(timed.executed),0);self.assertEqual(len(legacy.executed),1)
        action=legacy.executed[0][0]
        self.assertEqual((*action.position,*action.quaternion_xyzw,action.opening),tuple(POSE))

    def test_timed_sample_keeps_observation_and_sample_times_distinct(self):
        request=ExecutionRequest(POSE,1.,'timed_sample',.5,'approach')
        self.assertEqual(request.timed_target().pose.timestamp,1.)
        driver=Driver();bridge=ProtocolExecutor(driver);r=bridge.execute(request)
        self.assertEqual(r['execution_protocol']['target_sample_time'],1.5)
        self.assertEqual(driver.active_segment_id,'approach')

    def test_gripper_event_must_finish_before_lift_stage_is_accepted(self):
        driver=Driver();bridge=ProtocolExecutor(driver)
        close=ExecutionRequest(POSE,1.,'timed_sample',0.,'grasp','gripper')
        lift=ExecutionRequest(POSE,1.,'timed_sample',0.,'lift','static_pose')
        bridge.execute(close);r=bridge.execute(lift)
        self.assertFalse(r['gripper_completion_ack']['transition_accepted'])
        self.assertEqual(driver.active_segment_id,'grasp');self.assertEqual(driver.continued,1)
        driver.executor.gripper_done=True
        r=bridge.execute(lift);self.assertEqual(driver.active_segment_id,'lift')
        self.assertTrue(r['gripper_completion_ack']['transition_accepted'])

    def test_unsegmented_legacy_timed_profile_preserves_core_reference(self):
        driver=Driver();marker=driver.executor.reference_marker;bridge=ProtocolExecutor(driver)
        r=bridge.execute(POSE,observation_time=1.,legacy_lookahead_seconds=.5)
        self.assertIs(driver.executor.reference_marker,marker)
        self.assertTrue(r['execution_protocol']['phase_features_disabled_if_metadata_absent'])

    def test_waiting_for_fingers_does_not_freeze_a_moving_grasp_in_world_space(self):
        driver=Driver();bridge=ProtocolExecutor(driver)
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',.5,'grasp','gripper'))
        driver.backend.clock=lambda:1.1;driver.capture_stamp=1.;driver.grasp_delta=object()
        delta=driver.grasp_delta
        r=bridge.execute(ExecutionRequest(POSE,1.1,'timed_sample',.5,'lift'))
        self.assertIsNone(driver.capture_stamp);self.assertIs(driver.grasp_delta,delta)
        self.assertTrue(r['gripper_completion_ack']['transient_capture_hold_released_after_fresh_observation'])

    def test_legacy_and_new_routes_cannot_have_independent_reference_states(self):
        with self.assertRaises(ValueError):ProtocolExecutor(Driver(),legacy_driver=Driver())

    def test_return_from_legacy_route_clears_old_goal_history_without_reseeding_core(self):
        timed=Driver();legacy=Driver();legacy.backend=timed.backend;legacy.executor=timed.executor
        bridge=ProtocolExecutor(timed,legacy_driver=legacy);marker=timed.executor.reference_marker
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',.5,'approach'))
        timed._goal_history.append('stale_timed_goal')
        bridge.execute(POSE,observation_time=1.)
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',.5,'approach'))
        self.assertEqual(len(timed._goal_history),0)
        self.assertIs(marker,timed.executor.reference_marker)

    def test_same_stage_goal_refresh_cannot_erase_pending_gripper_completion(self):
        driver=Driver();bridge=ProtocolExecutor(driver)
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',0.,'grasp','gripper'))
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',0.,'grasp','continuous'))
        r=bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',0.,'lift'))
        self.assertFalse(r['gripper_completion_ack']['transition_accepted'])
        self.assertEqual(driver.active_segment_id,'grasp')

    def test_static_waypoint_clears_motion_extrapolation_but_preserves_reference(self):
        driver=Driver();bridge=ProtocolExecutor(driver);marker=driver.executor.reference_marker
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',0.,'lift'))
        driver._goal_history.append('moving_approach_history')
        bridge.execute(ExecutionRequest(POSE,1.,'timed_sample',0.,'lift','static_pose'))
        self.assertEqual(len(driver._goal_history),0);self.assertIs(marker,driver.executor.reference_marker)

    def test_release_policy_is_explicit_and_requires_driver_support(self):
        request=ExecutionRequest(POSE,1.,'timed_sample',.5,'release',release_policy='caller_intent')
        driver=Driver();bridge=ProtocolExecutor(driver)
        with self.assertRaises(ValueError):bridge.execute(request)
        driver.release_policy='terminal_pose';bridge.execute(request)
        self.assertEqual(driver.release_policy,'caller_intent')

    def test_ambiguous_future_static_and_nonbinary_commands_are_rejected(self):
        with self.assertRaises(ValueError):ExecutionRequest(POSE,1.,'timed_sample',.5,'lift','static_pose')
        with self.assertRaises(ValueError):ExecutionRequest(POSE,1.,'timed_sample',None)
        with self.assertRaises(ValueError):ExecutionRequest([*POSE[:7],.5],1.)
        with self.assertRaises(ValueError):ExecutionRequest.decode(POSE)
        with self.assertRaises(ValueError):ExecutionRequest(POSE,1.,stage_id='close',completion='gripper')

if __name__=='__main__':unittest.main()
