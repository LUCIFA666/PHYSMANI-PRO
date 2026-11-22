"""Local CPU NGFF RPC keeps simulator and learned-model dependencies isolated."""
import json
from pathlib import Path
import socket
import struct
import time

import numpy as np

from .contracts import Forecast, ObjectState, Pose


def receive(connection):
    def exact(size):
        pieces = []
        while size:
            block = connection.recv(min(size, 1024*1024))
            if not block:
                raise ConnectionError('forecast RPC connection closed before complete message')
            pieces.append(block)
            size -= len(block)
        return b''.join(pieces)
    size = struct.unpack('!I', exact(4))[0]
    if size > 32*1024*1024:
        raise ValueError('forecast RPC message exceeds size limit')
    return json.loads(exact(size).decode('utf-8'))


def send(connection, message):
    body = json.dumps(message, separators=(',', ':'), allow_nan=False).encode('utf-8')
    connection.sendall(struct.pack('!I', len(body))+body)


def pack_state(state):
    return {'object_id':state.object_id, 'timestamp':state.timestamp, 'pose':state.pose.xyzw().tolist(),
            'points_world':state.points_world.tolist(), 'linear_velocity':state.linear_velocity.tolist(),
            'angular_velocity':state.angular_velocity.tolist(), 'confidence':state.confidence, 'visible':state.visible}


def unpack_state(data):
    return ObjectState(data['object_id'], data['timestamp'], Pose.from_xyzw(data['pose']), data['points_world'],
                       data['linear_velocity'], data['angular_velocity'], data['confidence'], data['visible'])


def pack_forecast(forecast):
    return {**{key:getattr(forecast, key).tolist() for key in ('times','positions','rotations','linear_velocities','angular_velocities')},
            'object_ids':list(forecast.object_ids), 'metadata':forecast.metadata,
            'points_world':forecast.points_world.tolist() if forecast.points_world is not None else None}


class RemoteNGFF:
    def __init__(self, path, *, timeout=15.):
        self.path, self.timeout = str(Path(path)), timeout

    def predict(self, objects, horizon, dt):
        start = time.perf_counter()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.path)
            send(connection, {'objects':[pack_state(o) for o in objects], 'horizon':horizon, 'dt':dt})
            response = receive(connection)
        if 'error' in response:
            raise RuntimeError('NGFF worker: '+response['error'])
        data = response['forecast']
        metadata = {**data['metadata'], 'local_rpc_wall_seconds':time.perf_counter()-start,
                    'execution_runtime':'isolated_project_CPU_worker', 'rpc_socket':self.path}
        return Forecast(tuple(data['object_ids']), data['times'], data['positions'], data['rotations'],
                        data['linear_velocities'], data['angular_velocities'],
                        np.asarray(data['points_world']) if data['points_world'] is not None else None, metadata)
