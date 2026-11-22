"""Dual-crop CF40/520 static capture CPU check by default; explicit --capture starts one parent."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103 import current_cube_dual_crop_capture_v1 as producer
from our_mani.local_mpc_20261103.depth_owner_capture import plain


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--check',action='store_true');g.add_argument('--capture',action='store_true')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--parent-id');p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.capture and (not a.parent_id or not a.output):p.error('--capture requires --parent-id and --output')
    result=producer.capture_parent(ROOT,a.plan,a.parent_id,a.output) if a.capture else producer.check_capture_plan(ROOT,a.plan,a.parent_id)
    print(json.dumps(plain(result),allow_nan=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
