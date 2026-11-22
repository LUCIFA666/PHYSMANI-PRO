"""CPU metadata/source-only one-attempt protocol. No Native execution mode."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional fresh metadata-only plan JSON')
    args=parser.parse_args(argv)
    from our_mani.official_dense_train16_20261104_v1.protocol import prepare
    plan=prepare()
    if args.output is not None:
        with args.output.open('x')as stream:json.dump(plan,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(complete=True,preparation_only=True,execution_authorized=False,
        tasks=len(plan['attempts']),source_count=len(plan['source_sha256']),Native=False,GPU=False,
        HDF_payload_reads=0,TEST_reads=0,physical_motion_training_ready=False,
        planning_gap_supported=False,output=None if args.output is None else str(args.output.resolve())),allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
