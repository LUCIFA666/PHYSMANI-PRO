"""Explicit full32 schedule160 commands; default only checks closed TRAIN data."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.near_goal_depth_refiner_20261103_v4.training import (
    load_full32_dataset, train_capture, EPOCHS, OPTIMIZER_STEPS,
)
from our_mani.near_goal_depth_refiner_20261103_v4.evaluation import oracle_diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--oracle-check', action='store_true')
    modes.add_argument('--train', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    if args.train:
        if args.output is None:
            parser.error('--train requires fresh --output')
        print(train_capture(args.manifest, args.output, device=args.device, workers=args.workers))
        return 0
    dataset = load_full32_dataset(args.manifest)
    if args.oracle_check:
        oracle = oracle_diagnostic(dataset)
        print(json.dumps(oracle, indent=2, allow_nan=False))
        return 0 if oracle['ready_for_segmentation_fit'] else 3
    for sample in dataset:
        if sample['features'].shape != (4, 9, 64, 64):
            raise ValueError('Original public crop schema changed')
    dataset.revalidate()
    print(json.dumps(dict(structural_check_complete=True, scope='full32', shots=256,
        epochs_when_explicitly_fit=EPOCHS, required_actual_optimizer_steps=OPTIMIZER_STEPS,
        training_started=False, Native_started=False, model_or_threshold_qualified=False, TEST_read=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
