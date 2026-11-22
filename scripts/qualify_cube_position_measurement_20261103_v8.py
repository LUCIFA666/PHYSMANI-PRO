"""Default source check; --qualify consumes new640..647 once for position only."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.cube_position_measurement_20261103_v8.qualification import check_qualification_inputs,qualify_selected_checkpoint

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--position-manifest',required=True);p.add_argument('--output')
    p.add_argument('--qualify',action='store_true');a=p.parse_args(argv)
    if not a.qualify:
        print(json.dumps(check_qualification_inputs(a.selected,a.expected_sha256,a.position_manifest)));return 0
    if not a.output:p.error('--qualify requires new --output')
    result=qualify_selected_checkpoint(a.selected,a.expected_sha256,a.position_manifest,a.output)
    print(json.dumps(dict(complete=True,decision=result['decision'],model_qualified=False,raw_mask_model_qualified=False)))
    return 0 if result['position_measurement_qualified'] else 3

if __name__=='__main__':raise SystemExit(main())
