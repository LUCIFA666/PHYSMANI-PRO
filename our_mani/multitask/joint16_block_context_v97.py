"""Public pre-contact block observations through the shared joint16 SE3 model.

The old block executor expects singular context/history metadata. This adapter
only translates that interface; it neither creates a second model nor learns
contact dynamics. Its zero orientation channel matches joint16 block_arrays.
Formal evaluation is authorized separately by the evaluator's frozen protocol.
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from ..contracts import ObjectState
from .context_se3 import MultiBodyContextDynamics
from .joint16_tracks_v31 import BLOCKS
from .specs import TASK_SPECS
from .evaluation import file_sha

SCHEMA='r10_joint16_single_block_shared_context_interface_v97'


class Joint16BlockContext:
    name='joint16_shared_SE3_block_context_v97'

    def __init__(self,checkpoint,*,task,root):
        if task not in BLOCKS:raise ValueError('only the two original moving-block tasks')
        self.root=Path(root).resolve();self.task=task
        self.engine=MultiBodyContextDynamics(checkpoint)
        self.path=self.engine.path;self.metadata=self.engine.metadata
        self.checkpoint_sha256=self.engine.checkpoint_sha256
        self.encoder_sha256=self.engine.encoder_sha256
        self.maximum_validated_horizon=self.engine.maximum_validated_horizon
        expected=set(TASK_SPECS)|set(BLOCKS)
        if (set(self.metadata.get('jointly_trained_tasks',()))!=expected
                or self.metadata['training_source_manifest']['source_episode_count']!=1600):
            raise ValueError('requires the completed shared joint16 checkpoint, not an old per-task model')
        sources=self.path.parent/self.metadata['training_source_manifest']['path_relative_to_checkpoint']
        entries=json.loads(sources.read_text())['entries']
        original=self.root/'artifacts/framework/protocol/train_manifest.json'
        roster=json.loads(original.read_text())['episodes']
        wanted={(row['task'],row['raw_episode_id']) for row in roster}
        actual={(row['task'],row.get('parent_raw_episode_id',row['raw_episode_id'])) for row in entries}
        if actual!=wanted or len(actual)!=1600 or Counter(task for task,_ in actual)!=Counter({t:100 for t in expected}):
            raise ValueError('shared checkpoint must retain the exact complete original TRAIN roster')
        self.binding=dict(schema=SCHEMA,task=task,checkpoint_sha256=self.checkpoint_sha256,
            encoder_sha256=self.encoder_sha256,original_TRAIN_manifest_sha256=file_sha(original),
            body_channel='public_RGBD_target_translation_and_past_velocity',
            rotation_input='identity',rotation_observable=False,angular_velocity_input='zero',
            external_forecasts_only_before_first_observed_close_or_contact=True,
            attached_dynamics_claimed=False,second_context_model_created=False,formal_release_granted=False)
        self.reset()

    def reset(self):
        """Episode-boundary reset; native cases construct a fresh adapter."""
        self.engine.reset();self.public_frame=None;self.blocked=False;self.events=[]

    def set_public_motion_state(self,*,timestamp,gripper_open,contact):
        # Accept only the three controller-observable fields; task_success and
        # native attachment/teacher state cannot enter the model through here.
        if (not isinstance(gripper_open,(bool,np.bool_)) or not isinstance(contact,(bool,np.bool_))
                or isinstance(timestamp,bool) or not np.isfinite(timestamp) or timestamp<0):
            raise ValueError('current public timestamp/opening/contact required')
        now=float(timestamp)
        if self.public_frame is not None and now<self.public_frame:
            raise ValueError('reset block context at episode boundaries')
        newly_blocked=not self.blocked and (not gripper_open or contact)
        self.blocked |= bool(not gripper_open or contact)
        self.public_frame=now
        if newly_blocked:
            self.engine.reset()
            self.events.append(dict(time=now,event='end_of_TRAIN_supported_exogenous_channel',
                public_gripper_open=bool(gripper_open),public_contact=bool(contact),
                native_task_success_used=False,contact_dynamics_not_extrapolated=True))

    def _current(self,objects,*,axis_aligned=True):
        if (len(objects)!=1 or not isinstance(objects[0],ObjectState)
                or objects[0].object_id!='target' or not objects[0].visible
                or self.public_frame is None or abs(objects[0].timestamp-self.public_frame)>1e-8):
            raise ValueError('context needs the single current public RGBD target and motion-state frame')
        obj=objects[0]
        if axis_aligned and (not np.allclose(obj.pose.rotation,np.eye(3),rtol=0,atol=1e-8)
                or not np.allclose(obj.angular_velocity,np.zeros(3),rtol=0,atol=1e-8)):
            raise ValueError('block context must match the original axis-aligned RGBD channel')
        return obj

    def observe(self,objects):
        obj=self._current(objects,axis_aligned=not self.blocked)
        if self.blocked:return False
        self.engine.observe([obj],modes={'target':'external'},orientation_observability={'target':False})
        return True

    def context_snapshot(self,object_id='target'):
        if object_id!='target' or self.blocked:
            raise ValueError('context is unavailable outside the TRAIN-supported pre-contact block channel')
        return self.engine.context_snapshot(object_id)

    def predict(self,objects,horizon,dt):
        if self.blocked:
            raise ValueError('context prediction is unavailable after first observed close/contact')
        self._current(objects)
        result=self.engine.predict(objects,horizon,dt)
        # Axis-aligned block tracking and its fitted cache declare no measured
        # SO3; preserve the old executor's exact external-forecast contract.
        count=len(result.times)
        metadata=dict(result.metadata,method=self.name,adapter_schema=SCHEMA,
            context=result.metadata['contexts']['target'],
            history_frames=result.metadata['history_frames']['target'],
            shared_joint16_checkpoint=True,block_rotation_channel_observed=False,
            learned_contact_or_attachment_dynamics=False)
        return replace(result,rotations=np.broadcast_to(np.eye(3),(count,1,3,3)).copy(),
            angular_velocities=np.zeros((count,1,3)),metadata=metadata)

    def receipt(self):
        return dict(self.binding,public_frame=self.public_frame,channel_blocked=self.blocked,
            channel_events=list(self.events),formal_release_granted=False)
