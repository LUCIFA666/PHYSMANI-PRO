"""Axis/polarity agreement among current near-optimal registration hypotheses.

Agreement is a causal registration diagnostic, not an accuracy certificate or
proof of geometric symmetry. No native state, future frame or GT is accepted.
"""
import numpy as np


def near_optimal_axis_evidence(hypotheses, best_rotation, axis_local, *, directed,
                               best_rms_m, rms_slack_m, ambiguity_radians):
    axis=np.asarray(axis_local,float);best=np.asarray(best_rotation,float)
    if (axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-6
            or best.shape!=(3,3) or not np.isfinite(best).all() or type(directed) is not bool
            or not np.isfinite([best_rms_m,rms_slack_m,ambiguity_radians]).all()
            or best_rms_m<0 or rms_slack_m<0 or not 0<ambiguity_radians<np.pi):
        raise ValueError('explicit finite registration-axis hypotheses and thresholds required')
    if not np.allclose(best.T@best,np.eye(3),atol=1e-5) or np.linalg.det(best)<.99999:
        raise ValueError('axis evidence requires proper rotations')
    reference=best@axis;angles=[]
    for rms,rotation in hypotheses:
        rotation=np.asarray(rotation,float)
        if (not np.isfinite(rms) or rms<0 or rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
                or np.linalg.det(rotation)<.99999):
            raise ValueError('malformed axis rotation hypothesis')
        if rms<=best_rms_m+rms_slack_m:
            cosine=float(np.dot(reference,rotation@axis))
            if not directed:cosine=abs(cosine)
            angles.append(float(np.arccos(np.clip(cosine,-1.,1.))))
    maximum=max(angles,default=float(np.pi))
    return dict(axis_observable=bool(angles and maximum<=ambiguity_radians),
        axis_polarity='directed' if directed else 'undirected',near_optimal_hypotheses=len(angles),
        maximum_near_optimal_axis_deviation_rad=maximum,
        axis_world=reference.tolist(),accuracy_certified=False,
        evidence='agreement of current RGB-D/CAD registration hypotheses under declared residual slack')
