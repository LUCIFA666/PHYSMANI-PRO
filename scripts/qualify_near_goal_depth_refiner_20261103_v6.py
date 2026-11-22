"""Default identity check; --qualify explicitly consumes the fresh4 cohort once."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.near_goal_depth_refiner_20261103_v6.qualification import (
    check_qualification_inputs, qualify_selected_checkpoint,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--fresh-manifest', required=True)
    parser.add_argument('--output')
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args(argv)
    if not args.qualify:
        print(json.dumps(check_qualification_inputs(args.selected, args.expected_sha256, args.fresh_manifest)))
        return 0
    if not args.output:
        parser.error('--qualify requires new --output')
    result = qualify_selected_checkpoint(args.selected, args.expected_sha256, args.fresh_manifest, args.output)
    print(json.dumps(dict(complete=True, selection=result['selection'], Native_control_qualified=False)))
    return 0 if result['selection']['calibration_passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
