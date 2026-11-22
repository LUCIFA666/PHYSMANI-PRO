"""TRAIN development bridge to the unchanged R09 geometric block executor.

Public motion-state forwarding and read-only counted joint-window monitoring
add no render or physics call. Each actual case must still run in a fresh
simulator process. This helper does not authorize formal VAL or TEST.
"""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np

from .joint16_block_context_v97 import Joint16BlockContext
from .joint16_tracks_v31 import BLOCKS
from .evaluation import digest,file_sha

SCHEMA='r10_joint16_block_TRAIN_geometric_execution_bridge_v98'


class PublicCountedBlockEnvironment:
    def __init__(self,env,context=None):
        self.env=env;self.context=context;self.events=[];self.last_window=None
        self.observation_calls_forwarded=0
    def __getattr__(self,name):return getattr(self.env,name)
    def observe(self):
        result=self.env.observe();self.observation_calls_forwarded+=1
        if self.context is not None:
            robot=result[1]
            self.context.set_public_motion_state(timestamp=robot.timestamp,
                gripper_open=robot.gripper_open,contact=robot.contact)
        return result
    def _action(self,method,*args,**kwargs):
        env=self.env;dt=float(env.ledger.dt);before=int(env.ledger.ticks)
        q0=np.asarray(env.arm.get_joint_positions(),float)
        result=getattr(env,method)(*args,**kwargs)
        q1=np.asarray(env.arm.get_joint_positions(),float)
        if env.ledger.ticks!=before+1 or not np.isclose(dt,.05):
            raise ValueError('block commands must consume exactly one original counted 50ms tick')
        velocity=(q1-q0)/dt
        acceleration=None if self.last_window is None else (velocity-self.last_window)/dt
        vl=np.asarray(env.robot_model.velocity_limits,float);al=np.asarray(env.robot_model.acceleration_limits,float)
        if (q0.shape!=q1.shape or not np.isfinite(np.r_[q0,q1]).all()
                or not np.allclose(al,np.full_like(al,6.))):
            raise ValueError('nonfinite block joint state or changed original six-unit motion limit')
        self.events.append(dict(tick_before=before,tick_after=int(env.ledger.ticks),
            time_after=float(env.time),method=method,measured_q_before=q0.tolist(),measured_q_after=q1.tolist(),
            backward_window_velocity=velocity.tolist(),backward_window_acceleration=None if acceleration is None else acceleration.tolist(),
            velocity_limits=vl.tolist(),acceleration_limits=al.tolist(),
            position_passed=bool(np.all(q1>=env.robot_model.lower-1e-7)
                and np.all(q1<=env.robot_model.upper+1e-7)),
            speed_passed=bool(np.all(np.abs(velocity)<=vl+1e-6)),
            acceleration_passed=None if acceleration is None else bool(np.all(np.abs(acceleration)<=al+1e-6)),
            native_motor_parameters_changed=False,extra_physics_ticks=0,extra_camera_queries=0,
            metric_only=True))
        self.last_window=velocity.copy();return result
    def command_joints(self,*args,**kwargs):return self._action('command_joints',*args,**kwargs)
    def advance(self,*args,**kwargs):return self._action('advance',*args,**kwargs)
    def hold_tick(self,*args,**kwargs):return self._action('hold_tick',*args,**kwargs)
    def receipt(self):
        failed=[r for r in self.events if not r['position_passed'] or not r['speed_passed'] or r['acceleration_passed'] is False]
        return dict(schema=SCHEMA,all_stage_actual_window_checks_passed=not failed,
            events=deepcopy(self.events),violations=[r['tick_after'] for r in failed],
            maximum_actual_window_acceleration=max([abs(x) for r in self.events
                for x in (r['backward_window_acceleration'] or [])],default=0.),
            observation_calls_forwarded=self.observation_calls_forwarded,additional_camera_queries=0,
            additional_physics_ticks=0,all_stage_native_acceleration_guaranteed=False)


@contextmanager
def shared_context_factory_scope(evaluator,context):
    """Process-local library binding; original source files remain untouched."""
    old=evaluator.ContextLatentDynamics;old_value=evaluator.CandidateSuccessValue
    def refuse_value(*args,**kwargs):raise ValueError('Value is prohibited in the three-method joint16 experiment')
    def same_shared_context(path):
        if context is None or Path(path).resolve()!=context.path.resolve():
            raise ValueError('old separate Context checkpoint cannot enter the joint16 block executor')
        return context
    evaluator.ContextLatentDynamics=same_shared_context;evaluator.CandidateSuccessValue=refuse_value
    try:yield
    finally:evaluator.ContextLatentDynamics=old;evaluator.CandidateSuccessValue=old_value


def run_TRAIN_development_episode(env,row,config,*,method,root):
    root=Path(root).resolve()
    if method not in ('cv_geometry','context_geometry') or row.get('task') not in BLOCKS:
        raise ValueError('only the two moving-block tasks and authorized geometry methods')
    roster=json.loads((root/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    first=[r for r in roster if r['task']==row['task']][:4]
    if row not in first:raise ValueError('fixed first four original TRAIN rows; no TEST or successful-source replacement')
    if (config.get('dt')!=.05 or config.get('simulation_seconds')!=10.
            or config.get('global_joint_acceleration')!=6. or config.get('warmup_ticks')!=5):
        raise ValueError('preserve the original 50ms/10s/six-unit/five-frame block protocol')
    from scripts import evaluate_aligned_r09 as evaluator
    from ..context_dynamics import ContextLatentDynamics as original_context_class
    if evaluator.ContextLatentDynamics is not original_context_class:
        raise ValueError('fresh isolated process required; nested block model scopes are forbidden')
    c=deepcopy(config);predictor=method.split('_')[0];context=None
    if predictor=='context':
        pointer=c.get('joint16_context_checkpoint')
        if not isinstance(pointer,dict) or set(pointer)!={'path','sha256'}:
            raise ValueError('explicit frozen shared joint16 checkpoint pointer required')
        checkpoint=(root/pointer['path']).resolve()
        if not checkpoint.is_relative_to(root) or file_sha(checkpoint)!=pointer['sha256']:
            raise ValueError('shared joint16 checkpoint path/hash differs')
        context=Joint16BlockContext(checkpoint,task=row['task'],root=root)
        c['dynamics_checkpoint']=str(checkpoint.relative_to(root))
    # The old geometry kernel reads these strings before choosing its branch;
    # no Value file is admitted, opened or instantiated.
    c['cv_local_value']=c['context_local_value']='';c['value_input_context']=False
    boundary=PublicCountedBlockEnvironment(env,context)
    with shared_context_factory_scope(evaluator,context):
        record,frames=evaluator.run_episode(boundary,row,c,predictor=predictor,
            ranking='geometry',official=False,video=False)
    source_paths=('our_mani/multitask/joint16_block_episode_v98.py',
        'our_mani/multitask/joint16_block_context_v97.py','scripts/evaluate_aligned_r09.py',
        'our_mani/multitask/context_se3.py')
    binding=dict(schema=SCHEMA,method=method,task=row['task'],fixed_TRAIN_row_sha256=digest(row),
        original_geometric_component_sha256=record['controller_sha256'],
        sources_sha256={name:file_sha(root/name) for name in source_paths},
        shared_context=None if context is None else context.binding,Value_used=False,
        NGFF_used=False,FASTER_used=False,formal_VAL_TEST=False)
    record.update(legacy_geometric_component_controller_sha256=record['controller_sha256'],
        controller_sha256=digest(binding),joint16_block_execution=dict(binding,sha256=digest(binding)),
        all_stage_counted_joint_motion=boundary.receipt(),
        shared_context_receipt=None if context is None else context.receipt(),
        original_early_policy_termination_retained=True,
        task_budget_fully_consumed_or_native_success=bool(record['official_task_success']
            or np.isclose(env.time,10.)),
        demo_only=True,development_demo_only=True,formal_validation_metric=False,
        formal_TEST_metric=False,old_R09_release_not_reused=True)
    return record,frames
