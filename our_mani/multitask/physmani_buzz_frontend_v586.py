"""Supply original public initialization flags before conditional buzz tracking."""
from copy import deepcopy
from pathlib import Path
from .physmani_geometry_policy_v518 import PublicModelFrontend as OriginalFrontend
from .buzz_branch_v63 import PriorConditionedRegistration
from .buzz_runtime_v64 import binding as original_binding
from .evaluation import file_sha


def binding(root,config,task):
    return dict(schema='r10_explicit_public_context_before_original_buzz_registration_v586',task=task,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_buzz_frontend_v586.py'),
        original_conditional_registration=original_binding(root,config,task),
        inputs=['current_public_timestamp','current_public_gripper_open','current_public_contact'],
        old_actor_act_or_controller_executed=False,native_body_poses_or_outcomes_used=False,
        initial_only_TRAIN_prior_and_original_residual_gates_unchanged=True)


class PublicBuzzFrontend(OriginalFrontend):
    def __init__(self,actor):
        super().__init__(actor)
        adapter=getattr(actor.tracker._registration,'__self__',None)
        if not isinstance(adapter,PriorConditionedRegistration):
            raise ValueError('original source-bound conditional buzz registration must already be installed')
        self.conditional_registration=adapter

    def observe(self,observation,robot):
        self.conditional_registration.set_public_context(timestamp=robot.timestamp,
            gripper_open=robot.gripper_open,contact=robot.contact)
        return super().observe(observation,robot)

    def receipt(self):
        return dict(schema='r10_explicit_public_context_before_original_buzz_registration_v586',
            events=deepcopy(self.conditional_registration.events),old_actor_act_or_controller_executed=False,
            conditional_prior_or_quality_gates_changed=False,native_body_pose_or_outcome_inputs=False,
            extra_camera_or_physics_calls=0)
