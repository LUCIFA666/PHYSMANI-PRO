"""Run real streaming inference through the simulator's Python 3.10 client."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from our_mani.faster import validate_metadata
from our_mani.physmani_data import RecordedEpisode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18471)
    args = parser.parse_args()
    row = next(r for r in json.loads((ROOT/'artifacts/framework/protocol/validation_manifest.json').read_text())['episodes']
               if r['task']=='pick_moving_target_on_the_table')
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    metadata = client.get_server_metadata()
    validate_metadata(metadata)
    with RecordedEpisode(row['path']) as episode:
        observation = episode.observation(5)
    observation = {k:v for k,v in observation.items() if k in ('timestamp', 'observation/images', 'observation/state')}
    observation.update(prompt='pick up the red block and lift it up to the target',
        handover_goal={'pose': np.array([.2, 0, .845, 0, 1, 0, 0], np.float32),
                       'linear_velocity': np.zeros(3, np.float32), 'angular_velocity': np.zeros(3, np.float32),
                       'radii': np.array([.04, .04, .04], np.float32), 'remaining_seconds': np.float32(.7)},
        action_prefix=np.empty((0, 8), np.float32), delay=np.int32(0))
    runs = []
    try:
        for run in range(2):
            chunks, first = [], []
            start = time.perf_counter()
            def ready(actions):
                if not first:
                    first.append(time.perf_counter()-start)
                chunks.append(np.asarray(actions).copy())
            result = client.infer_streaming(observation, on_actions_ready=ready)
            emitted = np.concatenate(chunks)
            np.testing.assert_allclose(emitted, result['actions'], atol=2e-4, rtol=2e-4)
            if emitted.shape != (16, 8) or not np.isfinite(emitted).all():
                raise RuntimeError('invalid neural streaming output')
            runs.append({'wall_seconds': time.perf_counter()-start, 'first_ready_seconds': first[0],
                         'chunk_sizes': [len(c) for c in chunks], 'partial_final_max_difference': float(np.max(np.abs(emitted-result['actions'])))})
    finally:
        client._ws.close()
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'kind': 'real_neural_faster_cross_python_streaming_transport',
              'metadata': metadata, 'client_python': sys.version, 'runs': runs, 'formal_test_used': False,
              'closed_loop_success_measured': False}
    (ROOT/'artifacts/framework/smoke/faster_neural_transport.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
