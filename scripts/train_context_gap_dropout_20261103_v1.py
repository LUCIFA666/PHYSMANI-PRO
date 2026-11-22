"""Prepare/check an independent Cube gap student; no training by default."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--train', action='store_true')
    modes.add_argument('--evaluate', action='store_true')
    parser.add_argument('--registry', required=True)
    parser.add_argument('--output')
    parser.add_argument('--checkpoint')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    from our_mani.context_gap_dropout_20261103_v1.training import check, train, evaluate_checkpoint
    if args.train:
        if not args.output:
            parser.error('--train requires a fresh --output')
        report = train(args.registry, args.output, device=args.device)
    elif args.evaluate:
        if not all((args.output, args.checkpoint, args.expected_sha256)):
            parser.error('--evaluate requires --output, --checkpoint and --expected-sha256')
        report = evaluate_checkpoint(args.registry, args.checkpoint, args.expected_sha256, args.output)
    else:
        report = check(args.registry, args.output)
        report = {key: value for key, value in report.items() if key not in ('identity', 'actual_timestamp_parent_ledger', 'teacher_spec')}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
