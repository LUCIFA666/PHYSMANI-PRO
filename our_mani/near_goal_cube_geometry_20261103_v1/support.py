"""E research aggregation of current D points; no hidden face fabrication.

Compatible intervals are conditional on the declared near-axis 50 mm Cube
and point error allowance. They are not statistical confidence intervals or
certified physical bounds. Original point/view/crop/scope gates remain the
caller's responsibility and cannot be overridden by this aggregate.
"""
import numpy as np

PROTOCOL_SHA256='841bcc030b0ffa9dc7caa9a64c4e04ff45455e792ee28878f68a2980f3ffda53'
HALF_SIDE_M=.025
POINT_COMPATIBILITY_PADDING_M=.00025
MAXIMUM_DEVELOPMENT_RADIUS_M=.005
MINIMUM_FACE_COORDINATE_SAMPLES=3


def aggregate_known_cube_support(points,axis,coordinate_proposals):
    """Use face medians where supported, otherwise bounded current support.

    Inputs must be ONLY the accepted current D rows. No past centre, robot,
    forecast, GT, object pose or configurable threshold argument is accepted.
    Subthreshold face proposals are ignored entirely when bounding that axis.
    ``support_available`` is a geometric aggregation result, not the complete
    measurement availability or a model/control qualification.
    """
    points,axis,coordinate_proposals=map(np.asarray,(points,axis,coordinate_proposals))
    if (points.ndim!=2 or points.shape[1:]!=(3,) or points.dtype.kind not in 'fiu'
        or axis.shape!=(len(points),) or coordinate_proposals.shape!=(len(points),)
        or coordinate_proposals.dtype.kind not in 'fiu'):
        raise ValueError('Current accepted D points[N,3], axis[N] and proposals[N] required')
    n=len(points)
    out=dict(centre=None,diagnostic_centre_only=None,support_available=False,
        intervals=[[None,None] for _ in range(3)],radius_each=[None]*3,joint_radius=None,
        axis_counts=[int((axis==j).sum()) for j in range(3)],observed_axis=[False]*3,
        coordinate_sources=['bounded_current_support']*3,reasons=[],current_D_accepted_points=n,
        minimum_face_coordinate_samples=MINIMUM_FACE_COORDINATE_SAMPLES,
        point_compatibility_padding_m=POINT_COMPATIBILITY_PADDING_M,
        maximum_development_radius_m=MAXIMUM_DEVELOPMENT_RADIUS_M,
        known_5cm_axis_aligned_cube_prior=True,orientation_observed=False,
        face_invented_for_missing_axis=False,statistical_confidence_interval=False,
        certified_physical_bound=False,model_or_control_qualified=False,
        original_point_view_confidence_crop_scope_gates_still_required=True)
    if n==0:
        out['reasons'].append('no_current_D_accepted_points');return out
    if axis.dtype.kind not in 'iu' or not np.isin(axis,[0,1,2]).all():
        out['reasons'].append('invalid_current_D_accepted_axis');return out
    if not np.isfinite(points).all():
        out['reasons'].append('nonfinite_current_D_support');return out
    values=np.full(3,np.nan);radii=np.full(3,np.nan)
    observed=np.asarray(out['axis_counts'])>=MINIMUM_FACE_COORDINATE_SAMPLES
    out['observed_axis']=observed.tolist()
    for j in range(3):
        if observed[j]:
            out['coordinate_sources'][j]='visible_face_median'
            proposals=np.asarray(coordinate_proposals[axis==j],np.float64)
            if not np.isfinite(proposals).all():
                out['reasons'].append('nonfinite_supported_face_coordinate_'+str(j));continue
            with np.errstate(over='ignore',invalid='ignore'):
                value=float(np.median(proposals))
                lo=value-POINT_COMPATIBILITY_PADDING_M;hi=value+POINT_COMPATIBILITY_PADDING_M
            radius=POINT_COMPATIBILITY_PADDING_M
        else:
            # ALL accepted current D points, not just points assigned to j.
            with np.errstate(over='ignore',invalid='ignore'):
                lo=float(points[:,j].max())-HALF_SIDE_M-POINT_COMPATIBILITY_PADDING_M
                hi=float(points[:,j].min())+HALF_SIDE_M+POINT_COMPATIBILITY_PADDING_M
                value=lo+.5*(hi-lo);radius=.5*(hi-lo)
            if np.isfinite(lo) and np.isfinite(hi) and lo>hi:
                out['intervals'][j]=[lo,hi]
                out['reasons'].append('empty_current_support_interval_'+str(j));continue
        if not np.isfinite([lo,hi,value,radius]).all() or radius<0:
            out['reasons'].append('nonfinite_derived_coordinate_or_radius_'+str(j));continue
        values[j]=value;radii[j]=radius
        out['intervals'][j]=[lo,hi];out['radius_each'][j]=float(radius)
    if np.isfinite(values).all():out['diagnostic_centre_only']=values.tolist()
    if np.isfinite(radii).all():
        with np.errstate(over='ignore',invalid='ignore'):
            joint=float(np.linalg.norm(radii))
        if np.isfinite(joint):
            out['joint_radius']=joint
            if joint>MAXIMUM_DEVELOPMENT_RADIUS_M:
                out['reasons'].append('joint_current_support_radius_exceeds_fixed_5mm')
        else:out['reasons'].append('nonfinite_joint_current_support_radius')
    if not out['reasons'] and out['joint_radius'] is not None and out['diagnostic_centre_only'] is not None:
        out['support_available']=True;out['centre']=out['diagnostic_centre_only']
    return out
