"""Parent-side dispatch; every ours command consumes exactly one native tick."""
import numpy as np

from .protocol import MultitaskProtocolError


def dispatch_counted_command(env, command):
    if not isinstance(command, dict) or command.get('command') not in ('hold', 'joint_trajectory'):
        raise MultitaskProtocolError('unknown public actor command')
    before = env.ledger.ticks
    if command['command'] == 'hold':
        env.hold_tick()
    else:
        joints = np.asarray(command.get('joints'), dtype=float)
        if joints.shape != np.asarray(env.arm.get_joint_positions()).shape or not np.isfinite(joints).all():
            raise MultitaskProtocolError('actor emitted invalid joint reference')
        opening = command.get('opening')
        if not isinstance(opening, (int, float, np.number)) or not np.isfinite(opening) or not 0 <= opening <= 1:
            raise MultitaskProtocolError('actor emitted invalid gripper reference')
        env.command_joints(joints, opening=float(opening), phase='ours_' + str(command.get('phase', 'control')))
    if env.ledger.ticks != before + 1:
        raise MultitaskProtocolError('an ours command must consume exactly one common physical tick')
    return dict(ticks=1, phase=command.get('phase'), public_command_kind=command.get('public_command_kind'),
                counted_dt=env.ledger.dt, command=command['command'])
