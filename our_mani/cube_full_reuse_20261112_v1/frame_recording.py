"""Actual current public pixels hashed before optional sparse raw-frame storage."""
import hashlib
import numpy as np

def capture_evidence(observation,index,calibration):
    h=hashlib.sha256()
    for group in('images','depths'):
        for name,value in sorted(observation['observation/'+group].items()):
            array=np.ascontiguousarray(value)
            h.update((group+'/'+name+'/'+str(array.dtype)+'/'+str(array.shape)).encode())
            h.update(array.tobytes())
    return dict(index=index,timestamp=float(observation['timestamp']),rgbd_sha256=h.hexdigest(),
        camera_calibration_sha256=calibration,raw_rgbd_saved=index%20==0,raw_rgbd_stride=20,
        actual_RGBD_captured=True,Native_target_pose_or_future_used=False)
