"""Independent physical-clock TRAIN16 diagnostic; check is the default."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--predict', action='store_true')
    modes.add_argument('--score', action='store_true')
    parser.add_argument('--student')
    parser.add_argument('--student-sha256')
    parser.add_argument('--predictions')
    parser.add_argument('--output')
    parser.add_argument('--include-native-labels', action='store_true')
    args = parser.parse_args()
    from our_mani.context_physical_eval_20261103_v1.evaluation import check, predict, score
    if args.predict:
        if not all((args.student,args.student_sha256,args.output)) or args.include_native_labels:
            parser.error('--predict requires student/SHA/fresh output and cannot read labels')
        result = predict(args.student,args.student_sha256,args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','files')}))
    elif args.score:
        if not args.predictions or not args.output:
            parser.error('--score requires --predictions and fresh report --output')
        result = score(args.predictions,args.output,include_native_labels=args.include_native_labels)
        print(json.dumps(dict(complete=True,parents=result['parents'],anchors=result['anchors'],qualification=False)))
    else:
        if args.include_native_labels:
            parser.error('Default check cannot read labels')
        result = check(args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('identity','frame_ledger','source_sha256')}))


if __name__ == '__main__':
    main()
