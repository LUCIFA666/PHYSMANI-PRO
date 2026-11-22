"""Same real50ms observation stream, with actual requested-close boundaries."""
from pathlib import Path
import numpy as np
from ..cube_motion_capture_recovery_20261104_v2.collector import current_public_target,ARRAY_KEYS
from ..local_mpc_control_20261103.public_shadow import PublicShadowBackend
from ..local_mpc_20261102.records import write_new,append,sha


class TeacherObservationRecorder(PublicShadowBackend):
    def __init__(self,backend,output,*,bridge,parent,episode_id):
        super().__init__(backend,output,bridge=bridge,save_frames=False,
            controller_uses_public_forecast=False,observer_role='shadow',attachment_estimator_used=True)
        self.parent=parent;self.episode_id=episode_id;self.public_rows=0
        self._last_close=False;self._attempt=0
        (self.output/'public_frames').mkdir()
    def sample(self,state=None):
        before=self.backend.clock();snapshot=self.backend.read_snapshot()
        requested=float(self.backend.requested_opening)
        close=requested<.5
        if close and not self._last_close:self._attempt+=1
        self._last_close=close
        attempt=self.episode_id+'_attempt'+str(self._attempt)if close else None
        result=super().sample(state);target=current_public_target(result);index=self.samples-1
        relative=f'public_frames/{index:04}.npz';path=self.output/relative
        with path.open('xb')as f:np.savez_compressed(f,**{key:snapshot[key]for key in ARRAY_KEYS})
        meta={key:value for key,value in snapshot.items()if key not in ARRAY_KEYS}
        meta.update(file=relative,sha256=sha(path),physical_tick=self.backend._ticks,
            capture_additional_physics_steps=0,RGBD_shared_with_public_perception=True)
        write_new(self.output/'public_frames'/f'{index:04}.json',meta)
        append(self.output/'public_stream.jsonl',dict(schema='Cube_trueclock_motion_public_frame_v1',
            episode_id=self.episode_id,parent_id=self.parent['parent_id'],index=index,
            timestamp=snapshot['timestamp'],physics_timestamp=snapshot['physics_timestamp'],
            task_timestamp=snapshot['task_timestamp'],physics_time_origin=snapshot['physics_time_origin'],
            task_time_origin=snapshot['task_time_origin'],physical_tick=self.backend._ticks,
            target=target,robot={key:snapshot[key]for key in ('q','dq','tcp_pose','finger_joint_positions',
                'finger_joint_velocities','finger_open_amounts')},raw_RGBD=relative,
            raw_RGBD_sha256=meta['sha256'],full_public_snapshot=f'public_frames/{index:04}.json',
            close_requested=close,requested_opening=requested,attempt_id=attempt,
            forecast_available=result.get('free_forecast_available')is True,
            stale=not target['available'],Native_labels_in_input=False,
            collecting_policy_is_privileged_TRAIN_teacher=True))
        self.public_rows+=1
        if self.backend.clock()!=before:raise RuntimeError('Teacher public recorder added physics')
        return result
