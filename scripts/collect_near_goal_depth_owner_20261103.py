"""CPU check by default; --capture explicitly starts one owned Native parent."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103 import depth_owner_capture as smoke

def _producer(plan):
 schema=json.loads(Path(plan).read_text()).get('schema')
 if schema=='P3_near_goal_depth_owner_full32_plan_20261103_v1':
  from our_mani.local_mpc_20261103 import depth_owner_full32
  return depth_owner_full32
 if schema=='P3_near_goal_depth_owner_pilot64_plan_20261103_v2':
  from our_mani.local_mpc_20261103 import depth_owner_pilot_v2 as depth_owner_pilot
  return depth_owner_pilot
 return smoke

def check_capture_plan(root,plan,parent_id=None):
 return _producer(plan).check_capture_plan(root,plan,parent_id)

def capture_parent(root,plan,parent_id,output):
 return _producer(plan).capture_parent(root,plan,parent_id,output)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--check',action='store_true');g.add_argument('--capture',action='store_true')
 p.add_argument('--plan',type=Path,required=True);p.add_argument('--parent-id');p.add_argument('--output',type=Path);a=p.parse_args(argv)
 if a.capture and(not a.parent_id or not a.output):p.error('--capture requires --parent-id and --output')
 result=capture_parent(ROOT,a.plan,a.parent_id,a.output)if a.capture else check_capture_plan(ROOT,a.plan,a.parent_id)
 print(json.dumps(result,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x),allow_nan=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
