"""V10 scope12 only. Default preflight; --qualify explicitly consumes new80.

No fitting, threshold/radius search, Native capture or automatic deployment.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.cube_position_measurement_20261103_v10.qualification import (
    check_qualification_inputs,qualify_selected_checkpoint)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--position-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--qualify',action='store_true')
    args=p.parse_args(argv)
    if not args.qualify:
        report=check_qualification_inputs(args.selected,args.expected_sha256,args.position_manifest)
        print(json.dumps(report,allow_nan=False));return 0
    if args.output is None:p.error('--qualify requires a fresh --output')
    report=qualify_selected_checkpoint(args.selected,args.expected_sha256,args.position_manifest,args.output)
    print(json.dumps(dict(complete=True,scope12_EF_qualified=report['scope12_EF_qualified'],
        axis_general_qualified=False,active_certified_routes=report['active_certified_routes'],
        composite_global_coverage_qualified=report['composite_global_coverage_qualified']),allow_nan=False))
    return 0 if report['scope12_EF_qualified']else 3


if __name__=='__main__':raise SystemExit(main())
