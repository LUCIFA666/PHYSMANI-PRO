"""Explicit TRAIN binary-depth-refiner CLI; default only validates closed data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.near_goal_depth_refiner_20261103.data import DepthInfluenceDataset, LABEL_PROTOCOL
from our_mani.near_goal_depth_refiner_20261103.evaluation import oracle_diagnostic
from our_mani.near_goal_depth_refiner_20261103.training import train_capture


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
    parser.add_argument('--allow-pilot-smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.train:
        if args.output is None:
            parser.error('--train requires a fresh --output')
        print(train_capture(args.manifest, args.output, device=args.device,
            workers=args.workers, allow_pilot_smoke=args.allow_pilot_smoke))
        return 0
    dataset = DepthInfluenceDataset(args.manifest)
    if args.oracle_check:
        report = oracle_diagnostic(dataset)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['ready_for_segmentation_fit'] else 3
    for sample in dataset:
        if sample['features'].shape != (4, 9, 64, 64):
            raise ValueError('Same-runtime public crop schema mismatch')
    dataset.revalidate()
    print(json.dumps(dict(structural_check_complete=True, scope=dataset.scope, shots=len(dataset),
        label_protocol=LABEL_PROTOCOL, training_started=False, Native_started=False,
        model_or_threshold_qualified=False, TEST_read=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
