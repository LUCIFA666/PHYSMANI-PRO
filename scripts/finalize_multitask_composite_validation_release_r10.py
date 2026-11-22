"""Read-only full VAL acceptance of final ours1100 plus verified baseline275."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.composite_validation_release import finalize_composite_validation_release
from our_mani.multitask.evaluation import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition', required=True)
    parser.add_argument('--coverage-certificate', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    def pointer(name):
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT): raise ValueError('proof inputs must stay inside desktop project')
        return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))
    result = finalize_composite_validation_release(ROOT, pointer(args.composition),
        pointer(args.coverage_certificate), args.out)
    print(json.dumps(dict(complete=result['complete'], formal_test_permitted=result['formal_test_permitted'],
        stage=result.get('stage'), errors=result.get('errors', []), output=args.out), ensure_ascii=False))
    if not result['complete']: raise SystemExit(2)


if __name__ == '__main__': main()
