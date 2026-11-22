"""Attach-only one-attempt prototype. This module never launches Native."""
from pathlib import Path
import json
import hashlib
from copy import deepcopy


def plain(value):
    from ..generic_reference_mpc_20261104_v1.train_labels import _plain
    return _plain(value)


class SeparateFileSink:
    def __init__(self,output):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'public').mkdir();(self.output/'private_labels').mkdir()

    def __call__(self,index,public,label):
        import numpy as np
        from ..generic_reference_mpc_20261104_v1.public_observation import validate_policy_packet
        validate_policy_packet(public)
        path=self.output/'public'/f'frame{index:06d}.npz'
        arrays={key:value for key,value in public.items()if isinstance(value,np.ndarray)}
        np.savez_compressed(path,**arrays)
        public_meta=dict(timestamp=public['timestamp'],camera_names=public['camera_names'],
            input_semantics=public['input_semantics'],arrays=dict(path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (path.with_suffix('.json')).write_text(json.dumps(plain(public_meta),allow_nan=False)+'\n')
        (self.output/'private_labels'/f'frame{index:06d}.json').write_text(json.dumps(plain(label),allow_nan=False)+'\n')


def one_attempt(step_command,ledger,observer,output,*,maximum_command_calls=40):
    """No reset/retry/quality filtering. Exception/failure/STOP partials remain."""
    if type(maximum_command_calls)is not int or maximum_command_calls<1:raise ValueError('Fixed attempt budget required')
    output=Path(output).resolve()
    # Caller/sink owns one already-fresh directory. Never erase/reuse a case.
    if not output.is_dir()or(output/'attempt.json').exists():raise ValueError('Fresh dedicated attempt directory required')
    result=dict(schema='official_dense_TRAIN_single_attempt_20261104_v1',attempts=1,
        task_success=False,complete=False,command_calls=0,command_calls_started=0,command_calls_completed=0,retried=False,Native_replay_of_HDF=False,
        role='TRAIN_generated_new_seed',qualification=False,partial_records_retained=True,
        observations_are_not_CF_zero_tick=True,task_clock_equals_physics_assumed=False)
    try:
        with observer:
            ledger.initial()
            for index in range(maximum_command_calls):
                if (output/'STOP').exists()or(output.parent/'STOP').exists():raise RuntimeError('Administrative STOP; no retry')
                result['command_calls_started']+=1
                value=step_command(index)
                if (type(value)is not dict or type(value.get('terminal'))is not bool
                        or type(value.get('success'))is not bool):raise TypeError('Actual official command result required')
                result['command_calls']+=1;result['command_calls_completed']+=1
                ledger.flush('official_command_return')
                if value['terminal']or value['success']:
                    result['task_success']=value['success'];result['terminal']=value['terminal'];break
            result.update(complete=True,status='success'if result['task_success']else'failure_or_budget_retained')
    except Exception as error:
        result.update(status='infrastructure_or_unobserved_clock_partial',error_type=type(error).__name__,error=str(error))
    finally:
        result.update(public_private_pairs=ledger.rows,observed_physics_steps=ledger.physics_steps,
            observed_task_step_calls=ledger.task_step_calls,clock_or_recording_fault=ledger.fault,
            all_role_labels_private=True,physical_motion_training_ready=False)
        (output/'attempt.json').write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    return result


class NativeReaders:
    """Read actual public sensor/proprio + separate privileged all-role labels.

    Caller must supply an already reset TRAIN environment and original sensors.
    No HDF camera/scene reset, pose setter, task initialization or physics occurs.
    """
    def __init__(self,task_environment,cameras,physics_clock,task_name,contract):
        import inspect
        from pyrep.backend import sim
        if physics_clock is not sim.simGetSimulationTime:
            raise TypeError('Original sim.simGetSimulationTime required, never tick*dt/fake clock')
        from ..generic_reference_mpc_20261104_v1.public_observation import CAMERAS
        task=task_environment._task;cls=type(task)
        if task_name!=contract['task']or cls.__name__!=contract['native_class']or cls.__module__!=contract['native_module']:
            raise TypeError('Exact real registered task instance required')
        source=Path(inspect.getfile(cls)).resolve();expected=contract['native_source']
        if str(source)!=expected['path']or hashlib.sha256(source.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Real official task source differs')
        if tuple(cameras)!=CAMERAS or not callable(physics_clock):raise ValueError('Original current four public sensors/actual clock required')
        self.env=task_environment;self.task=task;self.cameras=cameras;self.clock=physics_clock
        self.arm=task_environment._scene.robot.arm;self.gripper=task_environment._scene.robot.gripper
        self.origin=float(physics_clock());self.roles=self._roles(task_name)

    def _roles(self,task_name):
        from dataclasses import replace
        from ..multitask.specs import get_task_spec
        from ..multitask.native_tasks import resolve_native_roles
        from pyrep.objects.object import Object
        if task_name.removesuffix('_high_speed')=='pick_moving_target_on_the_table':
            return {'Cube':self.task.target_block}
        spec=get_task_spec(task_name)
        # All body roles, not only manipulated/dynamic roots: includes both
        # ball+hoop, ring+frame+all pegs, item+bin and every cup+spoke.
        return resolve_native_roles(self.task,replace(spec,roles=tuple(r for r in spec.roles if r.kind in ('body','joint'))),
                                    scene_object_resolver=Object.get_object)

    def clocks(self):return dict(physics=float(self.clock()),task=float(self.task.t))

    def public(self):
        import numpy as np
        from ..generic_reference_mpc_20261104_v1.public_observation import pack_public_observation
        rgb=[];depth=[];K=[];T=[];clips=[]
        for camera in self.cameras.values():
            camera.handle_explicitly()
            rgb.append((np.clip(camera.capture_rgb(),0,1)*255).astype(np.uint8))
            depth.append(np.asarray(camera.capture_depth(in_meters=True),dtype=np.float32))
            K.append(camera.get_intrinsic_matrix());matrix=np.eye(4);matrix[:3]=camera.get_matrix();T.append(matrix)
            clips.append([camera.get_near_clipping_plane(),camera.get_far_clipping_plane()])
        return pack_public_observation(timestamp=float(self.clock())-self.origin,rgb=np.stack(rgb),depth_m=np.stack(depth),
            K=np.asarray(K),T=np.asarray(T),camera_clips=np.asarray(clips),q=self.arm.get_joint_positions(),tcp_pose=self.arm.get_tip().get_pose())

    def private(self):
        bodies={};joints={}
        for role,obj in self.roles.items():
            if hasattr(obj,'get_joint_position'):joints[role]=dict(native_handle=obj.get_handle(),actual_position=obj.get_joint_position())
            else:bodies[role]=dict(native_handle=obj.get_handle(),pose_world_xyzw=obj.get_pose())
        return plain(dict(task_name_label_only=self.task.name,stage_label_only=getattr(self.task,'stage',None),
            bodies=bodies,joints=joints,q=self.arm.get_joint_positions(),dq=self.arm.get_joint_velocities(),
            actual_q_target_positions=self.arm.get_joint_target_positions(),tcp_pose_world_xyzw=self.arm.get_tip().get_pose(),
            physical_finger_positions=self.gripper.get_joint_positions(),physical_finger_velocities=self.gripper.get_joint_velocities(),
            finger_open_amounts=self.gripper.get_open_amount(),
            native_grasped_handles_label_only=[obj.get_handle()for obj in self.gripper.get_grasped_objects()],
            selected_cup_or_peg_not_policy_input=True,force=None,force_observed=False))


class OfficialExpertCommandStep:
    """Original expert prediction and Mover arm-then-gripper command semantics.

    No live-demo retry API. Training labels may be privileged; public readers
    remain independent. Original logger/setup is supplied by the real caller.
    """
    def __init__(self,task_environment,ledger):
        from physmani_bench.collect import ExpertActioner
        self.task_environment=task_environment;self.ledger=ledger
        self.actioner=ExpertActioner(policy=None,device='cpu')

    def __call__(self,index):
        import numpy as np
        output=self.actioner.predict(self.task_environment,step_id=index)
        # Same final trajectory and binary opening rounding as official _evaluate.
        trajectory=output['trajectory'][-1].cpu().numpy().copy()
        if trajectory.ndim!=2 or trajectory.shape[1]!=8 or len(trajectory)<1 or not np.isfinite(trajectory).all():
            raise ValueError('Actual finite official Nx8 expert command required')
        trajectory[:,-1]=trajectory[:,-1].round()
        success=terminal=False
        for row,action in enumerate(trajectory):
            self.ledger.set_command(dict(command_call=index,command_row=row,
                world_TCP_command_xyzw_opening=action.tolist(),expert_stage_label_only=output['stage'],
                desired_command_not_actual_TCP=True,handoff_tau=None))
            original_action=np.ones(9);original_action[:8]=action
            _,reward,terminal=self.task_environment.step(original_action)
            # reward/terminate are returned by the original native TaskEnvironment
            # step. Do not call success() again and mutate native history predicates.
            success=bool(reward==1)
            if success or terminal:break
        return dict(success=success,terminal=bool(terminal))


def observe_reset_TRAIN_attempt(task_environment,cameras,physics_clock,task_name,output,*,execute_native=False):
    """Root future attach entry. Never launch, reset, retry or own GPU/Native locks.

    Required external lifecycle: one new TRAIN reset with the protocol's fixed
    seed/variation, an original official logger, real four sensors, and correct
    exclusive global resource owner. The reset itself is not observed by this
    preliminary attach-only entry; it cannot certify a whole reset+demo run.
    """
    if execute_native is not True:raise ValueError('Source-only prototype; Root explicit real-environment attach required')
    from .protocol import prepare
    from .recording import DenseTrainingLedger
    from .native_hooks import ScopedNativeStepObserver
    plan=prepare();matches=[row for row in plan['attempts']if row['task']==task_name]
    if len(matches)!=1:raise ValueError('One of the fixed sixteen TRAIN versions required')
    row=matches[0]
    readers=NativeReaders(task_environment,cameras,physics_clock,task_name,row)
    sink=SeparateFileSink(output)
    ledger=DenseTrainingLedger(readers.clocks,readers.public,readers.private,sink,
                               maximum_physics_steps=row['maximum_physics_steps'])
    observer=ScopedNativeStepObserver(task_environment._scene.pyrep,task_environment._task,ledger)
    command=OfficialExpertCommandStep(task_environment,ledger)
    (Path(output)/'preparation_reference.json').write_text(json.dumps(dict(
        protocol_version=plan['version'],fixed_metadata_row=row,source_sha256=plan['source_sha256'],
        reset_was_external_not_observed=True,actual_seed_reset_verified=False,
        resource_owner_external=True,physical_motion_training_ready=False),indent=2,allow_nan=False)+'\n')
    return one_attempt(command,ledger,observer,output,maximum_command_calls=row['maximum_command_calls'])
