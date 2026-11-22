"""Exercise the OFFICIAL FASTER WebSocket client against a local protocol test server.

The server deliberately has no neural network. This verifies ndarray transport
and streamed callbacks only and cannot produce policy performance metrics.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/FASTER/packages/openpi-client/src')]

import numpy as np
from websockets.sync.server import serve
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main():
    expected = np.tile(np.array([.2, .1, .8, 0., 1., 0., 0., 1.], np.float32), (4, 1))
    received = []
    def handler(connection):
        packer = msgpack_numpy.Packer()
        connection.send(packer.pack({'kind': 'protocol_test_double_not_a_trained_policy'}))
        data = msgpack_numpy.unpackb(connection.recv())
        np.testing.assert_equal(data['state'], np.arange(8, dtype=np.float32))
        np.testing.assert_equal(data['image'], np.full((4, 4, 3), 128, np.uint8))
        for actions in (expected[:1], expected[1:3], expected[3:]):
            connection.send(packer.pack({'type': 'partial', 'actions': actions}))
        connection.send(packer.pack({'type': 'final', 'actions': expected}))
    with serve(handler, '127.0.0.1', 0) as server:
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = None
        try:
            client = WebsocketClientPolicy('127.0.0.1', port)
            result = client.infer_streaming({'state': np.arange(8, dtype=np.float32),
                                            'image': np.full((4, 4, 3), 128, np.uint8)},
                                            on_actions_ready=lambda a: received.append(a))
            np.testing.assert_equal(np.concatenate(received), expected)
            np.testing.assert_equal(result['actions'], expected)
        finally:
            if client is not None:
                client._ws.close()
            server.shutdown()
            thread.join(timeout=2.)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'kind': 'official_faster_client_local_protocol_smoke', 'passed': True,
              'partial_chunk_sizes': [len(v) for v in received], 'arrays_roundtrip_exact': True,
              'socket_scope': 'loopback_ephemeral_port', 'real_model_inference': False,
              'checkpoint_used': None}
    output = ROOT/'artifacts/framework/smoke/faster_transport_report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
