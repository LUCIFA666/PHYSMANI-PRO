"""Record existing public registration diagnostics without additional sensing."""
from copy import deepcopy
from types import MethodType


def install(actor):
    tracker=actor.tracker;previous=tracker.observe;events=[]
    if actor.controller.feedback_step_calls:raise ValueError('attach before counted execution')
    def observe(instance,*args,**kwargs):
        states,metadata=previous(*args,**kwargs)
        rows={}
        for name,m in metadata.get('bodies',{}).items():
            rows[name]={k:deepcopy(m.get(k)) for k in ('mask_pixels_by_camera','observed_point_count',
                'directly_visible','registration_valid','orientation_observable','pose_source',
                'pose_observed_until','pose_evidence_until','occlusion','current_depth_pose_cost',
                'cup_feature_registration','registration_selection_cost_m2')}
        events.append(dict(time=float(states[0].timestamp) if states else None,bodies=rows))
        return states,metadata
    tracker.observe=MethodType(observe,tracker)
    old=actor.metrics_report
    def metrics(instance):return dict(old(),public_registration_diagnostic=dict(events=deepcopy(events),
        extra_sensor_or_physics_queries=0,policy_commands_changed=False,native_truth_read=False))
    actor.metrics_report=MethodType(metrics,actor)
    return events
