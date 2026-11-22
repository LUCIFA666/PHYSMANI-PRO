"""Serve an immutable real FASTER checkpoint on this project's loopback port."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/FASTER/src'), str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
for key, value in {'HF_HOME': ROOT/'runtime/framework/huggingface',
                   'OPENPI_DATA_HOME': ROOT/'artifacts/framework/openpi_cache',
                   'XDG_CACHE_HOME': ROOT/'runtime/framework/cache',
                   'JAX_COMPILATION_CACHE_DIR': ROOT/'runtime/framework/jax-cache',
                   'TMPDIR': ROOT/'runtime/framework/tmp'}.items():
    os.environ.setdefault(key, str(value))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.30')
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import StreamingWebsocketPolicyServer
from our_mani.checkpoint_snapshot import snapshot_checkpoint
from our_mani.faster_config import make_config


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--port', type=int, default=18471)
    parser.add_argument('--repo-id', default='hdf5/physmani_train_all16')
    args = parser.parse_args()
    source = Path(args.checkpoint).resolve()
    if (source/'our_mani_receipt.json').exists():
        receipt = json.loads((source/'our_mani_receipt.json').read_text())
        checkpoint = source
    else:
        checkpoint = ROOT/'artifacts/framework/faster_pinned'/('step_'+str(int(source.name)+1).zfill(8))
        receipt = snapshot_checkpoint(source, checkpoint)
    if not receipt.get('optimizer_updates', 0):
        raise ValueError('trained optimizer updates are required')
    config = make_config(args.repo_id, 'gs://openpi-assets/checkpoints/pi05_base/params')
    policy = create_trained_policy(config, checkpoint, sample_kwargs={'num_steps': 10, 'alpha': .6, 'u0': .9})
    metadata = {**config.policy_metadata, 'has_trained': True,
                'checkpoint_sha256': receipt['checkpoint_sha256'],
                'checkpoint': str(checkpoint), 'optimizer_updates': receipt['optimizer_updates'],
                'training_repo_id': args.repo_id, 'streaming_ready_rule': 'denoising_time_exactly_zero',
                'kind': 'real_neural_faster_development_checkpoint', 'formal_test_used': False,
                'created_utc': datetime.now(timezone.utc).isoformat()}
    files = [ROOT/'third_party/FASTER/src/openpi'/p for p in
             ('policies/our_mani_policy.py','policies/policy.py','models/pi0_faster.py')]
    metadata['inference_adaptation_code_sha256'] = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    status = ROOT/'artifacts/framework/faster_server_status.json'
    status.write_text(json.dumps({**metadata, 'host': '127.0.0.1', 'port': args.port, 'stage': 'serving'}, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    StreamingWebsocketPolicyServer(policy, host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
