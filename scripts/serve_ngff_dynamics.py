"""Project-only local socket for frozen NGFF inference, never training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import torch
from evaluate_domain_dynamics import make_model
from our_mani.remote_dynamics import receive, send, unpack_state, pack_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default=str(ROOT/'runtime/framework/ngff_dynamics.sock'))
    args = parser.parse_args()
    path = Path(args.socket)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('socket must remain inside this project')
    if path.exists():
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                path.unlink()
            else:
                raise RuntimeError('NGFF socket already has a live server')
    torch.set_num_threads(4)
    model = make_model(1., 'cpu')
    print(json.dumps({'stage':'frozen_inference_only', 'checkpoint_sha256':model.checkpoint_sha256,
                      'created_utc':datetime.now(timezone.utc).isoformat(), 'device':'CPU'}), flush=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        try:
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(4)
            while True:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(15.)
                    try:
                        request = receive(connection)
                        if request.get('health'):
                            send(connection, {'checkpoint_sha256':model.checkpoint_sha256, 'training':False, 'device':'CPU'})
                            continue
                        objects = [unpack_state(o) for o in request['objects']]
                        horizon, dt = float(request['horizon']), float(request['dt'])
                        if len(objects)!=1 or not .05 <= horizon <= 1.1 or not .01 <= dt <= .1:
                            raise ValueError('CPU transfer worker supports one object, horizon <=1.1s and dt .01..0.1s')
                        send(connection, {'forecast':pack_forecast(model.predict(objects, horizon, dt))})
                    except Exception as error:
                        try:
                            send(connection, {'error':str(error)})
                        except (OSError, ValueError):
                            pass
        finally:
            path.unlink(missing_ok=True)


if __name__=='__main__':
    main()
