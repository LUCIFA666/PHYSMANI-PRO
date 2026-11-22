"""Default CPU contract check. Actual fitting requires explicit --fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.scene_task_latent_20261103.data import write_new
from our_mani.scene_task_latent_20261103.training import check_prepared,fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/scene_task_latent_20261103/pretrain_v1.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--fit',action='store_true')
    parser.add_argument('--device')
    args=parser.parse_args()
    if args.fit and args.check:parser.error('choose --check or --fit')
    config=json.loads(args.config.read_text())
    if args.device:config['device']=args.device
    if args.fit:
        if args.output is None:parser.error('--fit requires fresh --output')
        result=fit(args.dataset,args.output,config)
    else:
        result=check_prepared(args.dataset,cpu_threads=int(config['cpu_threads']))
        if args.output:write_new(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
