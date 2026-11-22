"""Two actual Native Cube tasks, original V72 mechanical methods unchanged.

Generated task resets retain their own motion distributions and Native success
conditions. Reset staging restores the TRAIN home, not a target-relative pose.
No motion segment or object pose is assigned by this new adapter.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np

from ..local_mpc_20261102.native import TruthNativeBackend, MovingBlockScenario, DT, _vector

TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')


class CubeTaskNativeBackendV1(TruthNativeBackend):
    def __init__(self, task_name, calibration_episode, *, seed, variation=0,
                 budget_ticks=160, gripper_velocity=.2, execute_native=False,
                 _raw=None, _physics_clock=None, _ik_errors=None):
        if execute_native is not True:
            raise ValueError('Explicit Root Native execution required before task/path access')
        if task_name not in TASKS or type(seed) is not int or type(variation) is not int or variation < 0:
            raise ValueError('Actual ordinary/high-speed Cube task, seed and variation required')
        if budget_ticks != 160 or type(budget_ticks) is not int:
            raise ValueError('Predeclared eight seconds /160 native ticks required')
        self.task_name, self.variation = task_name, variation
        if _raw is None:
            calibration = Path(calibration_episode).resolve(strict=True)
            if 'train_package_compressed' not in calibration.parts or calibration.name != 'data.h5':
                raise ValueError('TRAIN camera/home calibration only')
            random.seed(seed); np.random.seed(seed)
            from ..aligned.simulation import AlignedLocalSimulation, physics_time
            from pyrep.errors import ConfigurationError, IKError
            protocol = dict(dt=DT, simulation_seconds=budget_ticks*DT,
                            gripper_velocity=gripper_velocity, current_instance_masks=False)
            raw = AlignedLocalSimulation.__new__(AlignedLocalSimulation)
            try:
                AlignedLocalSimulation.__init__(raw, task_name, str(calibration),
                    protocol=protocol, seed=seed, servo_lead=1.)
            except BaseException:
                if hasattr(raw, 'environment'): raw.environment.shutdown()
                raise
            _raw, _physics_clock, _ik_errors = raw, physics_time, (ConfigurationError, IKError)
        scenario = MovingBlockScenario('native_generated', (0.,0.,0.), (0.,0.,0.))
        super().__init__(calibration_episode, scenario, budget_ticks=budget_ticks,
            seed=seed, gripper_velocity=gripper_velocity, _raw=_raw,
            _physics_clock=_physics_clock, _ik_errors=_ik_errors)
        self.provenance.update(task=task_name, truth_diagnostic_only=True,
            official_public_input_evaluation=False, Native_task_conditions_unchanged=True,
            motion_source='actual_task_init_episode_distribution_and_original_task_step',
            custom_uniform_or_speed_multiplier_applied=False,
            reset='generated_TRAIN_development; not HDF replay or official VAL/TEST',
            native_generated_motion_retained=True)

    def reset_scenario(self, scenario=None):
        if self._closed: raise RuntimeError('Cannot reset a closed backend')
        if scenario is not None and scenario.name != 'native_generated':
            raise ValueError('No custom trajectory may replace the actual Native task distribution')
        self._staged, self._fault = False, None
        self.tick_records.clear(); self.motion_events.clear()
        self._switch_handled, self._last_pose_sample = False, None
        self._staging_before = self._clock_tuple()
        self.raw.task_environment.set_variation(self.variation)
        descriptions, _ = self.raw.task_environment.reset()
        self.reset_descriptions = list(descriptions)
        self.native_default_target_pose = list(self.task.target_block.get_pose())
        self.task.disable_expert_plan()
        self.finish_home_reset()
        return deepcopy(self.initial_snapshot)

    def finish_home_reset(self):
        if self._staged:
            if self.raw.ledger.ticks: raise RuntimeError('Home reset forbidden after first tick')
            return deepcopy(self.staging_receipt)
        # Original V72 home/open/settle protocol. The task's sampled x/v/a/t0,
        # object pose, success sensor and task clock are deliberately retained.
        self.gripper.release()
        self.gripper.set_joint_positions([.04,.04], disable_dynamics=True)
        self.gripper.set_joint_target_positions([.04,.04])
        home = _vector(self.raw.home_joints, 7, 'TRAIN home joints')
        self.arm.set_joint_positions(home.tolist(), disable_dynamics=True)
        self.arm.set_joint_target_positions(home.tolist())
        for _ in range(3): self.raw.pyrep.step()
        self.raw._start_alignment()
        self.raw.ledger.budget = self.budget_ticks*DT
        self.raw._requested_opening = self.requested_opening = 1.
        self._physics_origin, self._task_origin = float(self._physics_clock()), float(self.task.t)
        self._staged = True
        self._sample_native_result()
        self.staging_receipt = dict(excluded_from_control_budget=True,
            clock_before=self._staging_before, clock_after=self._clock_tuple(),
            raw_physics_seconds=float(self._physics_clock())-self._staging_before[0],
            reset_home_joints=home.tolist(), controlled_ticks=0,
            disable_dynamics_used_only_during_staging=True, extra_raw_settle_steps=3,
            native_task=self.task_name, variation=self.variation,
            native_motion_schedule_or_object_pose_assigned=False,
            internal_reset_placement_attempts=None, descriptions_cached_from_reset=True)
        self.initial_snapshot = dict(state=self.read_state(), scenario=asdict(self.scenario),
            native_default_target_pose_before_override=self.native_default_target_pose,
            native_height_preserved=True, native_generated_motion_retained=True,
            staging=deepcopy(self.staging_receipt), native_bounds=self._geometry_snapshot(),
            joint_limits=self.joint_limits(), truth_diagnostic_only=True,
            official_public_input_evaluation=False, native_task=self.task_name)
        return deepcopy(self.staging_receipt)
