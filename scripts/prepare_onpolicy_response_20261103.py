"""Default preflight only; --prepare explicitly joins eight closed TRAIN cases."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_onpolicy_response_20261103.data import preflight_plan,prepare_dataset

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/plan.json')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();protocol=args.protocol or args.plan.with_name('response_fit_protocol.json')
    if args.prepare:
        if args.output is None:parser.error('--prepare requires a fresh --output')
        result=prepare_dataset(args.plan,protocol,args.output)
    else:result=preflight_plan(args.plan,protocol)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
