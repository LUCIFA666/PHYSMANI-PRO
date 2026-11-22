"""Full-source Native bridges; HDF development generation and author TEST reset."""
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
from ..cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
from ..highspeed_control_20261111_v1.extended_backend_60s import ExtendedTimeContactCubeBackend
from ..local_mpc_20261102.native import TruthNativeBackend,MovingBlockScenario,DT
from ..task_time_limits_20261111_v1 import TaskBudget
from .common import ROOT,OUT,sha,read


def verify_hdf_source(case):
    import h5py
    split=case['source_split'];path=Path(case['source_path']).resolve(strict=True)
    package={'train':'train_package_compressed','validation':'val_package_compressed'}[split]
    if path.name!='data.h5' or package not in path.parts or sha(path)!=case['source_sha256']:
        raise ValueError('Exact original split/source bytes required')
    with h5py.File(path,'r')as f:
        if str(f.attrs['task'])!=case['task'] or int(f.attrs['variation'])!=case['variation'] or 'episode'+str(int(f.attrs['episode']))!=case['raw_episode_id']:
            raise ValueError('Source task/variation/raw identity differs')
    return path


class SourceContactCubeBackend(ExtendedTimeContactCubeBackend):
    def __init__(self,case,*,execute_native=False):
        self._closed=True
        if execute_native is not True or case['source_split']not in('train','validation'):
            raise ValueError('Explicit scoped development source required')
        path=verify_hdf_source(case);self.source_case=deepcopy(case)
        from ..aligned.simulation import AlignedLocalSimulation,physics_time
        from pyrep.errors import ConfigurationError,IKError
        random.seed(case['seed']);np.random.seed(case['seed'])
        budget=TaskBudget.from_config(case['config']);raw=AlignedLocalSimulation.__new__(AlignedLocalSimulation)
        try:
            AlignedLocalSimulation.__init__(raw,case['task'],str(path),
                protocol=dict(dt=DT,simulation_seconds=budget.seconds,gripper_velocity=case['config']['gripper_velocity'],current_instance_masks=False),
                seed=case['seed'],servo_lead=1.)
        except BaseException:
            if hasattr(raw,'environment'):raw.environment.shutdown()
            raise
        super().__init__(case['task'],str(path),seed=case['seed'],variation=case['variation'],budget_ticks=budget.ticks,
            gripper_velocity=case['config']['gripper_velocity'],execute_native=True,_raw=raw,_physics_clock=physics_time,_ik_errors=(ConfigurationError,IKError))
        self.provenance.update(source_split=case['source_split'],original_source_path=str(path),original_source_sha256=case['source_sha256'],
            source_raw_episode_id=case['raw_episode_id'],source_variation=case['variation'],
            reset='source-calibrated original native generation; HDF is not a full archived scene replay',
            original_HDF_motion_or_expert_actions_replayed=False,privileged_policy_state=False,TEST_read=False,
            calibration_scope='same-source public first-frame robot home and camera metadata')
        self.initial_snapshot['source_provenance']=deepcopy(self.provenance)


def verify_archive_source(case):
    """Called only after full VAL release; TEST payload bytes never reach NN."""
    release=read(OUT/'validation_release.json')
    if release.get('complete')is not True or release.get('sources')!=40:
        raise ValueError('Complete full40 validation required before TEST source access')
    frozen=read(OUT/'frozen_models_before_validation.json')
    for key in('motion','candidate'):
        if sha(frozen[key]['path'])!=frozen[key]['sha256']:raise ValueError('Frozen models changed after VAL')
    manifest=read(OUT/'test_input_manifest.json')
    matches=[r for r in manifest['episodes']if r['task']==case['task']and r['raw_episode_id']==case['raw_episode_id']]
    if len(matches)!=1:raise ValueError('Exact original TEST source required')
    row=matches[0]
    if row['variation']!=case['variation']or row['official_episode_index']!=case['official_episode_index']or Path(row['path']).resolve()!=Path(case['source_path']).resolve():
        raise ValueError('Frozen TEST mapping changed')
    for name,item in row['source_required_files'].items():
        path=Path(row['path'])/name
        if path.stat().st_size!=item['bytes']or sha(path)!=item['sha256']:raise ValueError('TEST reset/calibration bytes changed: '+name)
    alias=Path(row['official_alias_path'])
    if not alias.is_symlink()or alias.resolve()!=Path(row['path']).resolve():raise ValueError('Verified author variation alias required')
    for name,digest in manifest['native_inputs_sha256'].items():
        if sha(name)!=digest:raise ValueError('Original native task/scene changed')
    return row


class ArchiveContactCubeBackend(NativeContactCubeBackendV1):
    """Retain archived robot/target/goal/distractors; no home staging or motion edits."""
    def __init__(self,case,*,execute_native=False):
        self._closed=True
        if execute_native is not True or case['source_split']!='test':raise ValueError('Explicit official TEST required')
        row=verify_archive_source(case);self.source_case=deepcopy(case);self.archive_row=row
        self.task_name,self.variation=case['task'],case['variation']
        self.native_task_success_latched=False;self.native_success_first_time=None
        from ..aligned.simulation import AlignedOfficialSimulation,physics_time
        from pyrep.errors import ConfigurationError,IKError
        budget=TaskBudget.from_config(case['config']);raw=AlignedOfficialSimulation.__new__(AlignedOfficialSimulation)
        random.seed(case['seed']);np.random.seed(case['seed'])
        try:
            AlignedOfficialSimulation.__init__(raw,case['task'],row['path'],protocol=dict(dt=DT,simulation_seconds=budget.seconds,
                gripper_velocity=case['config']['gripper_velocity'],current_instance_masks=False),seed=case['seed'],servo_lead=1.)
        except BaseException:
            if hasattr(raw,'environment'):raw.environment.shutdown()
            raise
        TruthNativeBackend.__init__(self,row['path'],MovingBlockScenario('native_generated',(0.,0.,0.),(0.,0.,0.)),
            budget_ticks=budget.ticks,seed=case['seed'],gripper_velocity=case['config']['gripper_velocity'],
            _raw=raw,_physics_clock=physics_time,_ik_errors=(ConfigurationError,IKError))
        self.provenance.update(task=case['task'],source_split='test',TEST_read=True,privileged_policy_state=False,
            official_public_input_evaluation=False,information_setting='Native binary membership control; RGBD numeric NN inputs',
            official_TEST_reset=True,reset='author reset_to_demo -> archived target_state -> init_episode -> pyrep.step -> disable expert',
            additional_home_staging=False,Native_task_conditions_unchanged=True,task_budget=budget.receipt(),
            current_and_past_public_inputs_only=True,TEST_used_for_fitting=False,paper_10s_protocol_comparable=False)
        self.initial_snapshot['source_provenance']=deepcopy(self.provenance)
        self._latch_original_task_success()

    def reset_scenario(self,scenario=None):
        if self._closed:raise RuntimeError('Cannot reset closed backend')
        if scenario is not None and scenario.name!='native_generated':raise ValueError('No TEST motion override')
        self._staged,self._fault=False,None;self.tick_records.clear();self.motion_events.clear()
        self._switch_handled,self._last_pose_sample=False,None
        self.raw.reset_official(self.archive_row)
        self.requested_opening=float(self.raw._requested_opening)
        self._physics_origin,self._task_origin=float(self._physics_clock()),float(self.task.t)
        self._staged=True;self._sample_native_result()
        self.staging_receipt=dict(excluded_from_control_budget=True,official_reset=deepcopy(self.raw.reset_receipt),
            reset_home_joints=None,extra_raw_settle_steps=0,native_motion_schedule_or_object_pose_assigned=False,
            original_archived_motion_restored=True,controlled_ticks=0,additional_robot_repositioning=False)
        self.initial_snapshot=dict(state=self.read_state(),staging=deepcopy(self.staging_receipt),native_bounds=self._geometry_snapshot(),
            joint_limits=self.joint_limits(),native_task=self.task_name,source_raw_episode_id=self.source_case['raw_episode_id'],
            official_TEST_reset=True,task_budget=TaskBudget.from_config(self.source_case['config']).receipt())
        return deepcopy(self.initial_snapshot)
