"""Current RGB-D translation fallback when a cup's handle pose is unavailable.

Reuse the existing multi-start CAD registration and its current-depth checks.
The previous pose is only an initialization. The fallback never certifies full
rotation, contact, or a grasp; collision therefore uses the original rotation
uncertainty enclosure. No past pose is relabelled as a current measurement.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import numpy as np
from .evaluation import file_sha


def binding(root):
    return dict(schema='r10_current_RGBD_cup_translation_hypotheses_enlarged_collision_v2',
        sources_sha256={p:file_sha(Path(root)/p) for p in ('our_mani/multitask/cup_translation_uncertainty.py',
            'our_mani/multitask/perception.py','our_mani/multitask/forecasting.py','our_mani/multitask/translation_uncertainty_mesh.py')},
        original_current_registration_RMS_inlier_depth_and_minimum_points_retained=True,
        maximum_near_optimal_translation_spread='twice original registration RMS, covered by enlarged all-rotation collision cube',
        full_rotation_never_promoted=True,grasp_full_R_and_native_success_unchanged=True,
        previous_pose_is_only_initialization=True,additional_physics_or_camera_calls=False,formal_VAL_TEST=False)


def install(actor,*,install_geometry=True):
    if actor.controller.feedback_step_calls:raise ValueError('install before first execution')
    tracker=actor.tracker;previous=tracker._registration;events=[]
    geometry=[]
    if install_geometry:
        from .translation_uncertainty_mesh import install as install_meshes
        geometry=install_meshes(actor,2*tracker.config.maximum_registration_rms_m)
    def registration(instance,members,clouds,previous_pose=None):
        result=previous(members,clouds,previous_pose)
        if result is not None or len(members)!=1:return result
        role=next(iter(members));template=instance.templates[role]
        if role not in ('cup0','cup1','cup2') or template.cup_feature_model is None:return result
        # Only this actor's synchronous registration call takes the generic
        # existing CAD path. Restore the exact typed template before returning.
        instance.templates[role]=replace(template,cup_feature_model=None)
        try:fit=previous(members,clouds,previous_pose)
        finally:instance.templates[role]=template
        row=dict(time=float(instance._current_timestamp),body=role,accepted=False,
            current_frame_only=True,full_rotation_promoted=False)
        if fit is not None and fit.get('valid'):
            poses=fit['retained_root_candidate_poses'];rms=fit['retained_candidate_rms_m']
            best=min(rms);near=[p.position for p,r in zip(poses,rms)
                if r<=best+instance.config.ambiguity_rms_slack_m]
            spread=max((float(np.linalg.norm(a-b)) for a in near for b in near),default=float('inf'))
            depth=fit.get('current_depth_pose_cost')
            depth_ok=depth is None and instance.config.free_space_weight==0.
            if isinstance(depth,dict):
                depth_ok=bool(depth.get('evidence_available') and depth['cost_m2']<=depth['maximum_cost_m2'])
            accepted=bool(spread<=2*instance.config.maximum_registration_rms_m and depth_ok)
            row.update(accepted=accepted,near_optimal_translation_spread_m=spread,
                current_surface_rms_m=float(fit['rms_m']),original_current_depth_gate_passed=depth_ok)
            if accepted:
                fit=dict(fit,rotation_ambiguous=True,cup_feature_registration=dict(
                    status='current_RGBD_translation_only_generic_CAD_fallback',full_rotation_ready=False,
                    current_point_count=sum(len(p) for p in clouds[role].values()),
                    previous_pose_is_only_initialization=True,full_rotation_promoted=False))
                result=fit
        events.append(row)
        return result
    tracker._registration=MethodType(registration,tracker)
    old=actor.metrics_report
    def metrics(instance):return dict(old(),cup_translation_uncertainty=dict(binding=binding(Path(__file__).resolve().parents[2]),
        events=deepcopy(events),geometry=deepcopy(geometry),original_source_templates_unchanged=True))
    actor.metrics_report=MethodType(metrics,actor)
    return events
