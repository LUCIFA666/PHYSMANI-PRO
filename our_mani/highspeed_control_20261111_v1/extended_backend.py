"""Original generated Cube mechanics with the newly authorized task budget.

The old V72 constructor intentionally fixes160 ticks. This new constructor
passes the declared budget to the original simulation/ledger at construction;
all reset, servo, tick, contact and task-success methods remain inherited.
"""
from pathlib import Path
import random
import numpy as np
from ..cube_native_contact_feedback_live_20261104_v1.backend import NativeContactCubeBackendV1
from ..cube_v72_live_20261104_v1.backend import TASKS
from ..local_mpc_20261102.native import TruthNativeBackend,MovingBlockScenario,DT
from .task_budget import TaskBudget


class ExtendedBudgetContactCubeBackend(NativeContactCubeBackendV1):
    def __init__(self,task_name,calibration_episode,*,seed,variation=0,budget_ticks=None,
                 gripper_velocity=.2,execute_native=False,_raw=None,_physics_clock=None,_ik_errors=None):
        self._closed=True
        self.native_task_success_latched=False;self.native_success_first_time=None
        budget=TaskBudget.from_config({} if budget_ticks is None else dict(budget_ticks=budget_ticks))
        if execute_native is not True:raise ValueError('Explicit Native execution required')
        if task_name not in TASKS or type(seed) is not int or type(variation) is not int or variation<0:
            raise ValueError('Actual Cube task, seed and variation required')
        self.task_name,self.variation=task_name,variation
        if _raw is None:
            calibration=Path(calibration_episode).resolve(strict=True)
            if 'train_package_compressed' not in calibration.parts or calibration.name!='data.h5':raise ValueError('TRAIN camera/home calibration only')
            random.seed(seed);np.random.seed(seed)
            from ..aligned.simulation import AlignedLocalSimulation,physics_time
            from pyrep.errors import ConfigurationError,IKError
            raw=AlignedLocalSimulation.__new__(AlignedLocalSimulation)
            try:
                AlignedLocalSimulation.__init__(raw,task_name,str(calibration),
                    protocol=dict(dt=DT,simulation_seconds=budget.seconds,gripper_velocity=gripper_velocity,current_instance_masks=False),
                    seed=seed,servo_lead=1.)
            except BaseException:
                if hasattr(raw,'environment'):raw.environment.shutdown()
                raise
            _raw,_physics_clock,_ik_errors=raw,physics_time,(ConfigurationError,IKError)
        scenario=MovingBlockScenario('native_generated',(0.,0.,0.),(0.,0.,0.))
        TruthNativeBackend.__init__(self,calibration_episode,scenario,budget_ticks=budget.ticks,
            seed=seed,gripper_velocity=gripper_velocity,_raw=_raw,_physics_clock=_physics_clock,_ik_errors=_ik_errors)
        self.provenance.update(task=task_name,truth_diagnostic_only=True,official_public_input_evaluation=False,
            Native_task_conditions_unchanged=True,motion_source='actual_task_init_episode_distribution_and_original_task_step',
            custom_uniform_or_speed_multiplier_applied=False,reset='generated_TRAIN_development; not HDF replay or official VAL/TEST',
            native_generated_motion_retained=True,task_budget=budget.receipt())
        self.initial_snapshot['task_budget']=budget.receipt()
        self._latch_original_task_success()
