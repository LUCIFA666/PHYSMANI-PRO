import numpy as np

def outer_reach(snapshot,position):
    origins=np.asarray(snapshot['joint_origins_world'],float)
    tip=np.asarray(snapshot['tcp_reference_matrix'],float)[:3,3]
    # Triangle inequality for the existing revolute chain, with 5 cm reserve
    # for native joint compliance/calibration. This only proves impossibility;
    # membership in this sphere never certifies IK or collision feasibility.
    radius=float(np.linalg.norm(np.diff(np.vstack((origins,tip)),axis=0),axis=1).sum()+.05)
    distance=float(np.linalg.norm(np.asarray(position)-origins[0]))
    return dict(admissible=distance<=radius,outer_radius_m=radius,distance_from_first_joint_m=distance,
                robot_geometry_only=True,necessary_not_sufficient=True)
