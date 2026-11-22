"""Current multi-view50mm cube fit with observed orientation, not axis alignment.

This is a new development measurement route, not the old NN certificate.
Only observed XYZ/color/view samples and the public initial sphere are used.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def surface_residual(points,center,orientation,half=.025):
    vector=[0.,0.,orientation] if np.asarray(orientation).ndim==0 else orientation
    R=Rotation.from_rotvec(vector).as_matrix()
    local=(np.asarray(points)-center)@R
    d=np.abs(local)-half
    return np.linalg.norm(np.maximum(d,0.),axis=1)+np.minimum(np.max(d,axis=1),0.)


def fit_current_cube(component,sphere_center):
    points=np.asarray(component['points'],float);views=np.asarray(component['views']);colors=np.asarray(component['colors'],float)
    evidence=dict(available=False,source='current_public_multiview_oriented_cube_development',
        Native_inputs_used=False,future_observations_used=False,geometry_completed_as_observation=False,
        old_NN_certificate_reused=False,measurement_route_qualified=False,points=len(points),visible_views=len(set(views)))
    def reject(reason):return None,dict(evidence,reason=reason)
    if len(points)<60 or len(set(views))<3:return reject('insufficient_current_multiview_support')
    # The original R100/ratio3.5 pixel selection is retained by the caller.
    # Lambertian shading makes genuine target faces fall below R170.
    if colors.shape!=(len(points),3) or np.any(colors[:,0]<100):return reject('original_R100_current_support_required')
    lo,hi=np.quantile(points,[.01,.99],axis=0)
    if np.any(hi-lo<.025) or np.any(hi-lo>.09):return reject('current_component_shape_not_cube_sized')
    initial=(lo+hi)/2;initial[2]=np.quantile(points[:,2],.98)-.025
    if abs(initial[2]-.775)>.08:return reject('original_table_initialization_height_gate')
    # All starts see this same current cloud; no motion/label/Native prior.
    candidates=[]
    for yaw in np.arange(0.,np.pi/2,np.pi/12):
        solve=least_squares(lambda z:surface_residual(points,z[:3],z[3:]),np.r_[initial,0.,0.,yaw],
            bounds=(np.r_[initial-.025,[-np.pi]*3],np.r_[initial+.025,[np.pi]*3]),
            loss='soft_l1',f_scale=.0015,max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        candidates.append((np.mean(np.minimum(np.abs(solve.fun),.01)**2),solve.x))
    _,best=min(candidates,key=lambda c:c[0]);center=best[:3]
    R=Rotation.from_rotvec(best[3:]).as_matrix();local=(points-center)@R
    residual=np.abs(surface_residual(points,center,best[3:]));q95,q99=np.quantile(residual,[.95,.99])
    evidence.update(center=center.copy(),rotation=R.copy(),surface_error_p95_m=float(q95),surface_error_p99_m=float(q99))
    if q95>.003 or q99>.005:return reject('current_oriented_surface_residual_exceeds_3mm_5mm')
    support=residual<=.005
    for axis in (0,1):
        if np.quantile(local[support,axis],.01)>-.020 or np.quantile(local[support,axis],.99)<.020:
            return reject('opposing_horizontal_face_support_missing')
    axis=np.argmax(np.abs(local),axis=1);sign=(local[np.arange(len(local)),axis]>=0).astype(int)
    faces=2*axis+sign;covered=[int(face) for face in range(6) if np.sum(support&(faces==face))>=8]
    normals=np.array([sign*R[:,axis] for axis in range(3) for sign in (-1,1)])
    top=int(np.argmax(normals[:,2]))
    if len(covered)<3 or top not in covered:return reject('three_current_faces_including_top_required')
    # Preserve far-only use. Compute the nearest compatible centre interval
    # in the fitted yaw frame with the original6mm point-axis allowance.
    accepted=points[support];rotated=accepted@R
    lower=np.max(rotated-.031,axis=0);upper=np.min(rotated+.031,axis=0)
    if np.any(lower>upper):return reject('oriented_compatible_center_interval_empty')
    sphere=np.asarray(sphere_center)@R
    minimum_distance=float(np.linalg.norm(np.maximum(np.maximum(lower-sphere,sphere-upper),0.)))
    evidence.update(far_minimum_compatible_center_distance_m=minimum_distance,far_only_scope_m=.12,observed_faces=covered)
    if minimum_distance<=.12:return reject('near_goal_requires_original_NN_measurement')
    evidence.update(available=True,reason=None,accepted_current_points=int(support.sum()),
        current_point_support_sha_required=True,confidence=min(1.,len(set(views[support]))/3)*min(1.,support.sum()/60))
    return dict(center=center,points=accepted,views=views[support],rotation=R),evidence
