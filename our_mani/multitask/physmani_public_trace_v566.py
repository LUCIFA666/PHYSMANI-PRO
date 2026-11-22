"""Read-only public frontend diagnostics; identical observation/admission calls."""
from copy import deepcopy
from .physmani_geometry_policy_v518 import PublicModelFrontend as OriginalFrontend


class TracedPublicFrontend(OriginalFrontend):
    def __init__(self,actor):
        super().__init__(actor);self.frames=[];self.readiness=[];self.references=[]
        original=actor.controller.generator.observed_reference
        def reference(state,identity,template,origin):
            event=dict(time=float(state.robot.timestamp),identity=identity,phase=template.phase,
                kind=template.reference_kind,expected_mode=template.reference_mode,
                observed_mode=state.perception_metadata.get('bodies',{}).get(identity,{}).get('motion_mode'))
            try:
                result=original(state,identity,template,origin);event['admitted']=True;return result
            except (ValueError,KeyError) as error:
                event.update(admitted=False,error=str(error),error_type=type(error).__name__);raise
            finally:self.references.append(event)
        actor.controller.generator.observed_reference=reference

    def observe(self,observation,robot):
        result=super().observe(observation,robot);a=self.actor;s=self.state
        self.frames.append(dict(time=float(robot.timestamp),role_ready=self.role_ready,
            mode_estimates=deepcopy(a.last_diagnostics['mode_estimates']),
            invalid_mode_ids=deepcopy(a.last_diagnostics['invalid_mode_ids']),role_bindings=deepcopy(s.role_bindings),
            forecast_ids=list(self.forecast.object_ids),forecast_start=float(self.forecast.times[0]),
            forecast_end=float(self.forecast.times[-1]),bodies={k:dict(visible=bool(v.visible),
                confidence=float(v.confidence),timestamp=float(v.timestamp),
                metadata={field:s.perception_metadata.get('bodies',{}).get(k,{}).get(field) for field in (
                    'registration_valid','directly_visible','orientation_observable','motion_mode',
                    'rotation_gauge','pose_evidence_until')}) for k,v in s.bodies.items()}))
        return result

    def ready(self,phase):
        # This is the same single readiness query as518, with its public
        # report retained. It adds no collision probe or camera/physics call.
        report=dict(ready=False,reason='current_public_roles_or_forecast_unavailable')
        if self.state is not None and self.forecast is not None and self.role_ready:
            invalid=self.actor.last_diagnostics['invalid_mode_ids'];modes=self.actor.last_diagnostics['mode_estimates']
            if not invalid and all(modes.get(k) in ('external','attached','released') for k in self.actor.body_ids):
                report=self.actor.collision_port.readiness(self.state,self.forecast,phase)
        self.readiness.append(dict(time=None if self.state is None else float(self.state.robot.timestamp),
            phase=phase,report=deepcopy(report)))
        return bool(report['ready'])

    def receipt(self):
        return dict(schema='r10_public_RML_frontend_readonly_trace_v566',frames=deepcopy(self.frames),
            readiness=deepcopy(self.readiness),references=deepcopy(self.references),
            policy_decisions_or_parameters_modified=False,extra_camera_or_physics_calls=0)
