"""Original P4 matching loss. Training scales are not task acceptance gates."""
import torch
from ..local_mpc_p4_20261103.model import matching_loss

# The direct alias preserves the original per-example matching objective and
# parent-balanced aggregation contract for the FP32 training pilot.
cube_matching_loss=matching_loss


def fp32_matching_loss(output,target_position,target_rotation,target_time,**kwargs):
    """Same formula with explicit FP32 arithmetic, no new loss or supervision."""
    tensors={key:value.float()if isinstance(value,torch.Tensor)and value.is_floating_point()else value
             for key,value in output.items()}
    with torch.autocast(device_type=target_position.device.type,enabled=False):
        return matching_loss(tensors,target_position.float(),target_rotation.float(),target_time.float(),**kwargs)
