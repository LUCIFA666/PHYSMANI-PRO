"""An unfinished same-stage gripper event must not freeze its pose stream."""
from dataclasses import replace
from ..execution_protocol_20261113_v2 import ExecutionRequest
from ..execution_protocol_20261113_v2.bridge import ProtocolExecutor as PreviousProtocol


class ProtocolExecutor(PreviousProtocol):
    def execute(self,value,*,observation_time=None,legacy_lookahead_seconds=None,stop_requested=lambda:False):
        request=ExecutionRequest.decode(value,observation_time=observation_time,
            legacy_lookahead_seconds=legacy_lookahead_seconds)
        pending=self._pending_gripper_completion
        driver=self.timed_driver
        completed=(pending is not None and driver.executor.gripper_done is True
            and driver.executor.gripper_target==pending.pose8[7])
        refresh=(request.target_semantics=='timed_sample'and pending is not None and not completed
            and request.stage_id==pending.stage_id and request.pose8[7]!=pending.pose8[7])
        effective=request
        if refresh:
            # Preserve the already accepted opening/closing event, while using
            # the newest pose, timestamp and explicit forecast offset. The next
            # stage still waits for the unchanged physical completion ACK.
            effective=replace(request,pose8=(*request.pose8[:7],pending.pose8[7]),completion='gripper')
        result=super().execute(effective,stop_requested=stop_requested)
        if refresh:
            result['pending_gripper_pose_refresh']=dict(
                stage_id=request.stage_id,caller_opening=request.pose8[7],
                committed_opening=pending.pose8[7],
                latest_observation_time=request.observation_time,
                latest_pose_accepted=result.get('execution_protocol',{}).get('request_accepted',False),
                opening_pulse_does_not_reverse_committed_event=True,
                next_stage_completion_barrier_preserved=True,
                actuator_reference_reseeded=False)
        return result
