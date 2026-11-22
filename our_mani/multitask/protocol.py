"""R10 runtime errors and fixed shared physical-clock settings."""
import numpy as np


class MultitaskProtocolError(Exception):
    """State/clock corruption is infrastructure, never a policy failure."""


class MultitaskInfrastructureError(Exception):
    """Native runtime or cleanup failed."""


class EpisodeTerminatedError(MultitaskProtocolError):
    """A caller attempted another action after a native terminal result."""


def validate_protocol(protocol):
    for field, expected in (('dt', .05), ('simulation_seconds', 10.), ('gripper_velocity', .2)):
        value = protocol.get(field)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or abs(value - expected) > 1e-12:
            raise MultitaskProtocolError('R10 common physical setting differs: ' + field)
    if protocol.get('current_instance_masks') is not True:
        raise MultitaskProtocolError('R10 requires shared current instance masks')
    if not protocol.get('information_setting'):
        raise MultitaskProtocolError('declare the shared simulation-assisted information setting')

