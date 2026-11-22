from .request import ExecutionRequest

class ProtocolExecutor:
    """Route legacy destinations unchanged; add timing, stage and physical ACKs."""
    def __init__(self,timed_driver,*,legacy_driver=None):
        if legacy_driver is not None and(
                legacy_driver.backend is not timed_driver.backend or legacy_driver.executor is not timed_driver.executor):
            raise ValueError('Legacy and timed drivers must share the same backend and persistent actuator core')
        self.timed_driver=timed_driver;self.legacy_driver=legacy_driver
        self._current_driver=timed_driver;self.last_accepted_request=None;self._stage_mode=None
        self._declared_static=set(timed_driver.static_completion_segments)
        self._pending_gripper_completion=None

    def __getattr__(self,name):return getattr(self._current_driver,name)

    def execute(self,value,*,observation_time=None,legacy_lookahead_seconds=None,stop_requested=lambda:False):
        request=ExecutionRequest.decode(value,observation_time=observation_time,
            legacy_lookahead_seconds=legacy_lookahead_seconds)
        if request.target_semantics=='destination':
            if self.legacy_driver is None:
                raise ValueError('Configure the existing destination driver for unchanged legacy pose8 routing')
            self._current_driver=self.legacy_driver
            result=self.legacy_driver.execute(request.world_goal(),stop_requested=stop_requested)
            result['execution_protocol']=dict(schema_version=2,route='unchanged_legacy_destination_driver',
                legacy_pose8_preserved=True,prediction_time_invented=False)
            self.last_accepted_request=request
            self._stage_mode=None
            self._pending_gripper_completion=None
            return result
        driver=self.timed_driver;self._current_driver=driver
        if request.release_policy!='driver_default'and not hasattr(driver,'release_policy'):
            raise ValueError('Configured driver does not implement explicit release policies')
        if request.release_policy=='aligned_drop'and not getattr(driver,'supports_aligned_drop',False):
            raise ValueError('Configured driver does not implement aligned drop release')
        if abs(request.observation_time-driver.backend.clock())>driver.executor.config.clock_tolerance:
            raise ValueError('Stale/future observation timestamp')
        previous=self._pending_gripper_completion
        gripper_ack=None
        completed=(previous is not None and driver.executor.gripper_done is True
            and driver.executor.gripper_target==previous.pose8[7])
        if completed:
            gripper_ack=dict(transition_accepted=True,active_stage_id=previous.stage_id,
                proposed_stage_id=request.stage_id,gripper_done=True,requested_opening=previous.pose8[7],
                mechanical_gripper_completion_is_not_grasp_or_task_success=True)
            self._pending_gripper_completion=None
        if (previous is not None and not completed and(previous.stage_id!=request.stage_id
                or previous.pose8[7]!=request.pose8[7])):
            capture_stamp=getattr(driver,'capture_stamp',None)
            released_capture_hold=(capture_stamp is not None and
                request.observation_time>capture_stamp+driver.executor.config.clock_tolerance)
            if released_capture_hold:
                # A fresh observation has arrived even though its next phase
                # is waiting for the fingers. Resume the retained moving
                # grasp trajectory instead of freezing its old world pose.
                driver.capture_stamp=None
            result=driver._continue_active(stop_requested)
            result['gripper_completion_ack']=dict(transition_accepted=False,active_stage_id=previous.stage_id,
                proposed_stage_id=request.stage_id,gripper_done=driver.executor.gripper_done,
                requested_opening=previous.pose8[7],physical_opening_target=driver.executor.gripper_target,
                transient_capture_hold_released_after_fresh_observation=released_capture_hold,
                mechanical_gripper_completion_is_not_grasp_or_task_success=True)
            result['execution_protocol']=dict(schema_version=2,route='wait_for_physical_gripper_completion',
                request_accepted=False,reference_state_reseeded=False)
            return result
        segmented=request.stage_id is not None
        if segmented!=self._stage_mode:
            if driver.continuity is not None:
                driver.continuity.history.clear();driver.continuity.segment=None;driver.continuity.lookahead=None
            driver.active_segment_id=None;driver.active_packet=None
            # Only goal observations are cleared. Actuator reference q/v/a
            # remains in the same persistent core and is never reseeded.
            driver._goal_history.clear();self._stage_mode=segmented
        if request.completion=='static_pose':self._declared_static.add(request.stage_id)
        if request.release_policy!='driver_default':driver.release_policy=request.release_policy
        if request.completion=='static_pose':
            # A fixed waypoint is not an extrapolation of the preceding
            # approach stream. Preserve the actuator's q/v/a history.
            driver._goal_history.clear()
        continuity=driver.continuity;static=driver.static_completion_segments
        try:
            if not segmented:driver.continuity=None;driver.static_completion_segments=frozenset()
            else:driver.static_completion_segments=frozenset(self._declared_static)
            result=driver.execute(request.timed_target(),segment_id=request.stage_id,stop_requested=stop_requested)
        finally:
            driver.continuity=continuity;driver.static_completion_segments=static
        accepted=(result.get('static_segment_completion',{}).get('transition_accepted')is not False
                  and result.get('goal_continuity',{}).get('accepted')is not False)
        if accepted:
            self.last_accepted_request=request
            if request.completion=='gripper':self._pending_gripper_completion=request
        if gripper_ack is not None:result['gripper_completion_ack']=gripper_ack
        result['execution_protocol']=dict(schema_version=2,route='timed_world_target',
            request_accepted=accepted,target_sample_time=request.observation_time+request.target_time_offset_seconds,
            stage_metadata_available=segmented,phase_features_disabled_if_metadata_absent=not segmented,
            metadata_origin=request.metadata_origin,legacy_pose8_preserved=True,reference_state_reseeded=False)
        result['execution_protocol'].update(release_policy=request.release_policy,
            static_waypoint_not_extrapolated=request.completion=='static_pose')
        return result
