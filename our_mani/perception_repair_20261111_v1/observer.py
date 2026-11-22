"""One current RGBD sample per actual50ms; membership never becomes a visual point."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from ..contracts import Pose
from ..local_mpc_control_20261103.public_shadow import sanitized_robot
from ..local_mpc_control_20261103.public_motion_bridge import PUBLIC_OBSERVATION_KEYS
from .tracker import ReacquiringPublicCubeTracker as CurrentDepthPixelCentreTrackerV70
from ..cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
from ..cube_corrected_live_20261104_v1.candidate_runtime import pack_public_row
from ..local_mpc_control_20261103.public_state import PublicStateAssembler
from ..local_mpc_control_20261103.goal_aware_pixel_centre_binding_v70 import calibration_identity,digest
from ..local_mpc_20261102.records import append,write_new
from ..cube_native_contact_feedback_live_20261104_v1.candidate_runtime import NativeContactPublicCacheV1,NativeContactPilotInferenceV1,NativeContactMatchedInferenceV1
from ..cube_native_contact_feedback_live_20261104_v1.carry import NativeContactCarryEstimate


class ObservedRepairedCubeBackend:
    def __init__(self,backend,output,*,checkpoint,checkpoint_sha256,forecast_checkpoint,forecast_checkpoint_sha256,
                 refiner_checkpoint,refiner_checkpoint_sha256,recorder=None,save_frames=True,
                 candidate_mode='legacy_pilot_with_corrected_forecast_development',
                 goal_sphere_filter=True,postconfirmed_maintenance=False,active_config=None,
                 grasp_config=None,require_public_goal_completion=False):
        if candidate_mode not in('legacy_pilot_with_corrected_forecast_development','matched_candidate_with_corrected_forecast'):raise ValueError('Explicit legacy or genuinely fitted matched family required')
        if goal_sphere_filter is not True or require_public_goal_completion is not False:
            raise ValueError('Original Sphere perception retained; visual completion cannot gate Native task success')
        torch.set_num_threads(2)
        self.backend=backend;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.tracker=CurrentDepthPixelCentreTrackerV70(refiner_checkpoint,refiner_checkpoint_sha256)
        self.forecaster=CubeResidualMotionDynamics(forecast_checkpoint,expected_sha256=forecast_checkpoint_sha256)
        self.candidate_cache=NativeContactPublicCacheV1(expected_forecast_sha256=forecast_checkpoint_sha256)
        inference=NativeContactPilotInferenceV1 if candidate_mode=='legacy_pilot_with_corrected_forecast_development'else NativeContactMatchedInferenceV1
        self.pilot_inference=inference(checkpoint,expected_sha256=checkpoint_sha256,forecast_sha256=forecast_checkpoint_sha256)
        self.candidate_mode=candidate_mode
        self.forecast_sha256=forecast_checkpoint_sha256;self.robot_history=PublicStateAssembler()
        self.carry=NativeContactCarryEstimate();self.latest_carry=None
        self.latest=self.latest_robot=self.latest_contact=None;self.last_free_forecast=None
        self.recorder=recorder;self.save_frames=save_frames;self.samples=0;self.last_time=None;self.times=[]
        self.goal_pose=None;self._last_held=False;self._last_source=None;self._calibration=None;self._calibration_digest=None
        write_new(self.output/'scope.json',dict(Native_contact_feedback_used=True,Native_target_pose_used_by_policy=False,
            current_measurement_route_changed=True,new_composite_position_route_qualified=False,
            NN_input_schema_or_weights_changed=False,NN_measurement_values_may_change=True,
            original_task_success_only=True,visual_probe_or_hold_enabled=False,
            close_intent_does_not_clear_unheld_GRU_history=True,controlled_carry_never_enters_FREE_history=True,
            obsolete_visual_arguments_ignored=['postconfirmed_maintenance','active_config','grasp_config']))

    def __getattr__(self,name):return getattr(self.backend,name)

    def sample(self,state=None):
        before=float(self.backend.clock());started=time.perf_counter()
        if self.last_time is not None and (before<=self.last_time or abs(before-self.last_time-.05)>1e-6):
            raise ValueError('Each actual50ms frame exactly once required')
        raw_state=self.backend.read_state()if state is None else state
        robot=sanitized_robot(raw_state)
        raw=self.backend.observe_rgbd();observation={k:raw[k]for k in PUBLIC_OBSERVATION_KEYS}
        if abs(float(observation['timestamp'])-before)>1e-8 or abs(robot.timestamp-before)>1e-8:raise ValueError('Same actual RGBD/robot time required')
        calibration=digest(calibration_identity(observation))
        if self._calibration_digest is None:self._calibration_digest=calibration
        elif calibration!=self._calibration_digest:raise ValueError('Original episode public camera calibration changed')
        contact=self.backend.read_contact_feedback()
        if abs(contact['timestamp']-before)>1e-9 or type(contact['held'])is not bool:raise ValueError('Actual current membership feedback required')
        target=None;error=None
        try:target,_=self.tracker.observe(observation)
        except(ValueError,RuntimeError)as exception:error=str(exception)
        if self.goal_pose is None and self.tracker.sphere is not None:
            self.goal_pose=Pose(self.tracker.sphere.center+self.tracker.sphere.marker_to_goal_offset)
        self.robot_history.observe_robot(robot)
        held=contact['held'];forecast=None;ready=False
        if held:
            if not self._last_held:self.forecaster.reset()
        elif target is None:
            self.forecaster.reset();self._last_source=None
        else:
            source=self.tracker.last_measurement_metadata.get('accepted_source')
            if self._last_held or(self._last_source is not None and source!=self._last_source):self.forecaster.reset()
            self.forecaster.observe([target],modes={'target':'external'},orientation_observability={'target':False})
            self._last_source=source
            if len(self.forecaster.history.get('target',()))>=5:
                forecast=self.forecaster.predict([target],.8,.05);ready=True;self.last_free_forecast=forecast
        result=dict(timestamp=before,mode='FREE'if not held else'UNRESOLVED',available=ready,
            free_forecast_available=ready,controlled_mode_available=False,object_observation=target,forecast=forecast,
            rotation_observable=False,goal_pose=self.goal_pose,observed_target_available=target is not None,
            frozen_checkpoint_sha256=self.forecast_sha256,Native_inputs_used=False,
            future_observations_used=False,observations_fabricated=False,forecaster_is_injected_test_double=False,
            reason=error or('Native held: FREE forecast disabled; controlled estimate separate'if held else'current_public_history'),
            tracker_dependency_injected=True,goal_marker_dependency_injected=False)
        self.latest_carry=self.carry.observe(robot,held=held,current_target=target,last_public_forecast=self.last_free_forecast)
        index=self.samples;close=float(self.backend.requested_opening)<.5
        self.candidate_cache.observe(pack_public_row(result,robot,close,index,f'rgbd/{index:04d}.npz',expected_forecast_sha256=self.forecast_sha256))
        if self.save_frames:
            folder=self.output/'rgbd';folder.mkdir(exist_ok=True);path=folder/f'{index:04d}.npz'
            arrays={'timestamp':np.asarray(before)}
            for group in('images','depths'):arrays.update({group+'_'+name:np.asarray(value)for name,value in observation['observation/'+group].items()})
            if self.recorder is None:
                with path.open('xb')as stream:np.savez_compressed(stream,**arrays)
            else:self.recorder.submit(path,arrays)
        if self._calibration is None:
            self._calibration={name:dict(intrinsics=c.intrinsics,camera_to_world=c.camera_to_world.xyzw(),depth_scale=c.depth_scale,timestamp_offset=c.timestamp_offset)for name,c in observation['observation/calibrations'].items()}
            write_new(self.output/'camera_calibration.json',self._calibration)
        append(self.output/'public_outputs.jsonl',dict(index=index,result=result,public_robot=robot,close_requested=close,rgbd=f'rgbd/{index:04d}.npz'))
        append(self.output/'native_contact_feedback.jsonl',dict(index=index,contact=contact,carry=self.latest_carry,
            task_judge=self.backend.native_task_judge_receipt(),Native_contact_feedback_used=True,
            Native_target_pose_used_by_policy=False,visual_success_gate=False))
        append(self.output/'position_diagnostics_extra.jsonl',dict(index=index,timestamp=before,tracker_diagnostics=deepcopy(self.tracker.last_measurement_metadata)))
        self.latest=result;self.latest_robot=robot;self.latest_contact=contact
        self.samples+=1;self.last_time=before;self._last_held=held;self.times.append(time.perf_counter()-started)
        if self.backend.clock()!=before:raise RuntimeError('Observer advanced physics')
        return result

    def tick(self,phase):
        state=self.backend.tick(phase);self.sample(state);return state

    def receipt(self):
        return dict(protocol='Native_membership_execution_public_numeric_NN_inputs_v1',samples=self.samples,
            Native_contact_feedback_used=True,Native_object_pose_used_by_policy=False,visual_success_or_reopen_gate=False,
            carry=self.latest_carry,original_task_judge=self.backend.native_task_judge_receipt(),
            candidate_transformer=self.pilot_inference.receipt(),candidate_cache=self.candidate_cache.receipt())
