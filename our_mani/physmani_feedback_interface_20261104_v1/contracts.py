"""Mechanical feedback contracts; no task, attachment or policy certificate."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
import numpy as np

VERSION = 'physmani_feedback_interface_20261104_v1'
RECEIPT_SCHEMA = 'physmani_two_tick_feedback_receipt_20261104_v1'


class GripperRequestV1(str, Enum):
    KEEP = 'KEEP'
    OPEN = 'OPEN'
    CLOSE = 'CLOSE'


def checked_gripper_request(value):
    if isinstance(value, GripperRequestV1):
        return value.value
    if type(value) is not str or value not in ('KEEP', 'OPEN', 'CLOSE'):
        raise ValueError('Explicit KEEP/OPEN/CLOSE request required; BC intent is not an actuator command')
    return value


def vector(value, size, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError('Finite ' + name + ' of shape ' + str((size,)) + ' required')
    return tuple(float(v) for v in out)


def plain(value):
    """JSON-safe complete control evidence, including nonfinite diagnostics."""
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Enum):
        return plain(value.value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'nonfinite_diagnostic': str(value)}
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise TypeError('Control evidence keys must be strings')
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    raise TypeError('Unsupported control evidence value: ' + type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def action_bundle(value):
    if isinstance(value, dict):
        if 'joint_goal' not in value:
            raise ValueError('Selected action bundle must contain joint_goal')
        result = deepcopy(value)
        result['joint_goal'] = vector(result['joint_goal'], 7, 'selected joint goal')
    else:
        result = {'joint_goal': vector(value, 7, 'selected joint goal')}
    # Validate serializability now, never after physics has moved.
    digest(result)
    return result
