"""Check only by default; explicit frozen Native replay/query never executes C."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103.public_prefix_static_collision_replay_20261103 import check,execute,plain

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True,type=Path);p.add_argument('--anchor-id');p.add_argument('--output',type=Path);p.add_argument('--source-freeze',type=Path);p.add_argument('--source-freeze-sha256');p.add_argument('--execute',action='store_true');p.add_argument('--print-source-freeze',action='store_true');a=p.parse_args(argv)
 if a.print_source_freeze:
  if a.execute:p.error('Freeze inspection cannot execute')
  result=check(a.request,a.anchor_id,collect_sources=True)
 elif a.execute:
  if not a.anchor_id or a.output is None or not a.source_freeze or not a.source_freeze_sha256:p.error('Explicit anchor/output/external freeze required')
  result=execute(a.request,a.anchor_id,a.output,a.source_freeze,a.source_freeze_sha256)
 else:result=check(a.request,a.anchor_id,a.source_freeze,a.source_freeze_sha256)
 print(json.dumps(plain(result),allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
