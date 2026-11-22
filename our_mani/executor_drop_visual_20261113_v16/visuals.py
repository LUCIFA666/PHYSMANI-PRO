"""Read public rendered camera images without advancing the physical clock."""
import numpy as np

def capture_public_RGB(backend):
    before=backend._clock_tuple();frames={}
    for name,camera in backend.cameras.items():
        camera.handle_explicitly()
        pixels=np.asarray(camera.capture_rgb())
        if pixels.ndim!=3 or pixels.shape[-1]!=3 or not np.isfinite(pixels).all():
            raise ValueError('Finite public RGB camera image required')
        frames[name]=(np.clip(pixels,0,1)*255).astype(np.uint8)
    if before!=backend._clock_tuple():raise RuntimeError('Public image diagnostic advanced physics')
    return frames
