"""CPU check by default; --capture explicitly starts one owned Native parent."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103.perception_capture import check_capture_plan,capture_parent

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--capture',action='store_true')
 p.add_argument('--plan',type=Path,required=True);p.add_argument('--parent-id');p.add_argument('--output',type=Path);a=p.parse_args(argv)
 if a.capture and(not a.parent_id or not a.output):p.error('--capture requires --parent-id and --output')
 result=capture_parent(ROOT,a.plan,a.parent_id,a.output)if a.capture else check_capture_plan(ROOT,a.plan,a.parent_id)
 print(json.dumps(result,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x),allow_nan=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
