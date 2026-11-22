"""Explicit fit-only v5. Default verifies sources and prints config, no data."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.near_goal_depth_refiner_20261103_v5.training import (
    fixed_config, implementation_source_map, train_selected,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--oracle-evidence', type=Path)
    parser.add_argument('--oracle-sha256')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    if not args.train:
        print(json.dumps(dict(config=fixed_config(), source_count=len(implementation_source_map()),
            training_started=False, calibration_started=False, model_qualified=False)))
        return 0
    if any(v is None for v in (args.manifest, args.output, args.oracle_evidence, args.oracle_sha256)):
        parser.error('--train requires manifest,fresh output,prior oracle evidence and expectedSHA')
    print(train_selected(args.manifest, args.output, oracle_evidence=args.oracle_evidence,
        oracle_evidence_sha256=args.oracle_sha256, device=args.device, workers=args.workers))
    return 0


if __name__ == '__main__': raise SystemExit(main())
