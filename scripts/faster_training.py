"""Use official normalization/training code with our isolated FASTER config."""
import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'third_party/FASTER/src'),
               str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
os.environ.setdefault('HF_HOME', str(ROOT/'runtime/framework/huggingface'))
os.environ.setdefault('HF_LEROBOT_HOME', str(ROOT/'artifacts/framework/lerobot'))
os.environ.setdefault('OPENPI_DATA_HOME', str(ROOT/'artifacts/framework/openpi_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'runtime/framework/cache'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(ROOT/'runtime/framework/jax-cache'))
os.environ.setdefault('OUR_MANI_STOP_FILE', str(ROOT/'runtime/framework/state/STOP'))
from our_mani.faster_config import make_config


def official_module(name):
    # A real, importable module name is required by multiprocessing spawn:
    # normalization workers unpickle the upstream RemoveStrings transform.
    return importlib.import_module(f'third_party.FASTER.scripts.{name}')


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('config', 'norm-stats', 'train'))
    parser.add_argument('--repo-id', default='local/our_mani_physmani_grasp2')
    parser.add_argument('--initial-params', default='gs://openpi-assets/checkpoints/pi05_base/params')
    parser.add_argument('--exp-name', default='physmani_grasp2')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--updates', type=int, default=30000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = make_config(args.repo_id, args.initial_params, exp_name=args.exp_name,
                         batch_size=args.batch_size, updates=args.updates, resume=args.resume)
    if args.command == 'config':
        import jax
        print(json.dumps({'name': config.name, 'devices': [str(d) for d in jax.devices()],
                          'batch_size': config.batch_size, 'updates': config.num_train_steps,
                          'checkpoint_dir': str(config.checkpoint_dir), 'model': repr(config.model)}, indent=2))
        return
    if args.command == 'norm-stats':
        if args.repo_id.startswith('hdf5/physmani_train_'):
            from our_mani.faster_hdf5 import compute_norm_stats
            compute_norm_stats(config)
            return
        from openpi.training import config as registry
        module = official_module('compute_norm_stats')
        previous = registry.get_config
        try:
            registry.get_config = lambda name: config if name == config.name else previous(name)
            module.main(config.name)
        finally:
            registry.get_config = previous
    else:
        dataset = Path(os.environ['HF_LEROBOT_HOME'])/args.repo_id
        stats = config.assets_dirs/config.data.assets.asset_id/'norm_stats.json'
        if (not dataset.exists() and not args.repo_id.startswith('hdf5/physmani_train_')) or not stats.exists():
            raise RuntimeError('convert the TRAIN dataset and compute normalization statistics before training')
        if args.repo_id.startswith('hdf5/physmani_train_'):
            from our_mani.faster_hdf5 import register_dataset_factory
            register_dataset_factory()
            start_update = 0
            if args.resume:
                indices = [int(p.name) for p in config.checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit()]
                if not indices:
                    raise RuntimeError('no saved optimizer checkpoint available for exact sampling resume')
                start_update = max(indices)+1
            os.environ['OUR_MANI_SAMPLER_START_UPDATE'] = str(start_update)
        official_module('train').main(config)


if __name__ == '__main__':
    main()
