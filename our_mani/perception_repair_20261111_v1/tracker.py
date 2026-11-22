"""Original qualified near measurement plus a new causal far reacquisition.

The added route is explicitly development-only and has no inherited NN
qualification claim. Every returned centre is fitted to this frame's actual
multi-camera depth. Source switches preserve the same physical measurement
history; missing frames are never interpolated.
"""
from collections import deque,Counter
from copy import deepcopy
import hashlib
import numpy as np
from ..contracts import ObjectState,Pose
from ..benchmark_perception import box_surface
from ..local_mpc_control_20261103.depth_pixel_centre_goal_perception_v70 import CurrentDepthPixelCentreTrackerV70
from .replay import public_red_components
from .box_fit import fit_current_cube

SOURCE='PUBLIC_RGBD_CURRENT_CUBE_POSITION_REPAIRED_V1'
RECOVERABLE={'NN_unavailable_and_no_current_far_component','NN_unavailable_without_far_consistency_no_near_shell_fallback'}


class ReacquiringPublicCubeTracker:
    def __init__(self,*args,**kwargs):
        self.base=CurrentDepthPixelCentreTrackerV70(*args,**kwargs)
        self.history=deque(maxlen=5);self.last_velocity=np.zeros(3)
        self.last_measurement_metadata=None;self.last_measured_cloud=None;self.counts=Counter()

    def __getattr__(self,name):return getattr(self.base,name)

    def observe(self,observation):
        now=float(observation['timestamp']);route=None;fit_evidence=[]
        before=list(self.history)
        self.base.history.clear();self.base.history.extend(before)
        self.base.last_velocity=self.last_velocity.copy()
        try:
            target,metadata=self.base.observe(observation)
            route=self.base.last_measurement_metadata['accepted_source']
            center=target.pose.position;confidence=target.confidence
            cloud=self.base.last_measured_cloud
        except ValueError as error:
            if str(error) not in RECOVERABLE:raise
            self.last_measurement_metadata=deepcopy(self.base.last_measurement_metadata)
            self.last_measured_cloud=None
            # Reacquire the previously measured identity, never initialize a
            # new target from a shape fit after a long unobserved gap.
            if not before or now-before[-1][0]>.25+1e-8 or self.base.sphere is None:raise
            components,_=public_red_components(observation,self.base.sphere)
            possible=[];dt=now-before[-1][0]
            predicted=before[-1][1]+self.last_velocity*dt
            for component in components:
                if component['n']<60 or component['visible_views']<3 or not component['initial_height_gate']:continue
                fit,evidence=fit_current_cube(component,self.base.sphere.center)
                if fit is not None:
                    distance=float(np.linalg.norm(fit['center']-predicted))
                    evidence['causal_association_distance_m']=distance
                    evidence['original_association_limit_m']=.045+1.*dt
                    if distance<=.045+1.*dt:
                        possible.append((fit,evidence))
                    else:evidence.update(available=False,reason='original_causal_association_gate')
                fit_evidence.append(evidence)
            self.last_measurement_metadata['reacquisition_candidates']=fit_evidence
            if len(possible)!=1:
                self.counts['reacquisition_unavailable']+=1
                raise
            fit,evidence=possible[0];center=fit['center'];cloud=fit['points'];confidence=evidence['confidence']
            route=evidence['source'];metadata=dict(backend=route,simulator_pose_used=False,future_observations_used=False,
                measured_points=len(cloud),visible_views=len(set(fit['views'])),geometry_completion=True)
        # Every sample is from its real current frame. A gap starts a new
        # finite-difference window; a producer route change alone does not.
        if before and abs(now-before[-1][0]-.05)>1e-6:self.history.clear()
        if self.history and now<=self.history[-1][0]:raise ValueError('unique increasing public measurement time required')
        self.history.append((now,np.asarray(center).copy()))
        velocity=np.zeros(3)
        if len(self.history)>1:
            times=np.array([r[0] for r in self.history]);times-=times.mean()
            xyz=np.array([r[1] for r in self.history]);velocity=(times[:,None]*(xyz-xyz.mean(0))).sum(0)/(times@times)
        self.last_velocity=velocity
        # Same original numerical pose/template ABI. The fit rotation is
        # internal evidence; no new observed orientation is asserted to the NN.
        pose=Pose(center);target=ObjectState('target',now,pose,box_surface(pose,self.base.prior.dimensions,256),velocity,np.zeros(3),float(confidence))
        self.last_measured_cloud=np.asarray(cloud).copy();self.counts[route]+=1
        legacy=deepcopy(self.base.last_measurement_metadata)
        self.last_measurement_metadata=dict(timestamp=now,measurement_available=True,accepted_source=SOURCE,
            current_measurement_route=route,position_measurement_qualified=False,
            original_NN_measurement_route_qualification=legacy.get('position_measurement_qualified'),
            new_composite_route_qualified=False,Native_inputs_used=False,future_observations_used=False,
            observations_fabricated=False,source_switch_does_not_fabricate_history=True,
            current_measured_points=len(cloud),current_support_sha256=hashlib.sha256(self.last_measured_cloud.tobytes()).hexdigest(),
            velocity_history_real_timestamps=[r[0] for r in self.history],legacy_diagnostics=legacy,reacquisition_candidates=fit_evidence,
            orientation_observed=False,NN_template_is_current_measured_cloud=False)
        return target,dict(metadata,measurement_route=route,composite_development_route=True,
            position_measurement_qualified=False,Native_inputs_used=False,future_observations_used=False)
