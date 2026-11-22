"""Forward public TCP into the existing certified ring gauge before tracking.

The old adapter supplied TCP through actor.act(). The RML frontend deliberately
bypasses that old controller, so this equivalent public input must be explicit.
The same proved D4 orbit, selector and evidence_scope remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
from types import MethodType
import importlib
from .perception import MultiBodySE3Tracker
from .physmani_geometry_policy_v518 import PublicModelFrontend as OriginalFrontend
from .evaluation import file_sha

MODULES={'insert_onto_rotating_peg':'ring_operational_gauge_v60',
         'insert_onto_rotating_peg_high_speed':'high_peg_operational_gauge_v244'}


def binding(root,config,task):
    module=importlib.import_module('our_mani.multitask.'+MODULES[task])
    return dict(schema='r10_explicit_public_TCP_before_existing_ring_gauge_v570',task=task,
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_ring_frontend_v570.py'),
        original_gauge_binding=module.binding(root,config,task),
        original_D4_hypotheses_selector_and_typed_evidence_retained=True,
        full_orientation_promoted=False,old_actor_act_or_controller_executed=False,
        input='current_public_RobotState.pose_before_tracker.observe')


class PublicRingFrontend(OriginalFrontend):
    def __init__(self,actor):
        super().__init__(actor);task=actor.controller.library.task_spec.task_name
        module_name=MODULES[task];module=importlib.import_module('our_mani.multitask.'+module_name)
        if (not isinstance(actor.tracker,MultiBodySE3Tracker)
            or actor.tracker._symmetry_pose.__func__.__module__!=module.__name__):
            raise ValueError('exact existing source-bound ring operational gauge must be installed first')
        symmetry=actor.tracker.templates['ring'].finite_rotation_symmetry
        group=symmetry.certificate['group_rotations_body'];center=symmetry.certificate['center_body']
        local=actor.controller.library.for_phase('approach')[0].at(1.)[0].rotation
        self.current_public_tcp=None;self.gauge_events=[]
        def symmetry_pose(tracker,role,pose,points,previous):
            original=MultiBodySE3Tracker._symmetry_pose(tracker,role,pose,points,previous)
            if role!='ring':return original
            selected,proof=module.choose(original,group,center,previous=previous,
                tcp=self.current_public_tcp,local_grasp_rotation=local)
            self.gauge_events.append(dict(proof,time=float(tracker._current_timestamp),
                selected_xyzw=selected.xyzw().tolist(),public_TCP_supplied_before_tracking=True))
            return selected
        actor.tracker._symmetry_pose=MethodType(symmetry_pose,actor.tracker)

    def observe(self,observation,robot):
        self.current_public_tcp=robot.pose
        return super().observe(observation,robot)

    def receipt(self):
        return dict(schema='r10_explicit_public_TCP_before_existing_ring_gauge_v570',
            events=deepcopy(self.gauge_events),old_actor_act_or_controller_executed=False,
            original_D4_typed_evidence_scope_retained=True,full_orientation_promoted=False,
            extra_camera_or_physics_calls=0,native_body_pose_or_outcome_inputs=False)
