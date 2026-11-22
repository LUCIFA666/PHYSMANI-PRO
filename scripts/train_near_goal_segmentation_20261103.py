"""Explicit offline near-goal TRAIN commands; default is a read-only check."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.near_goal_segmentation_20261103.data import CaptureDataset
from our_mani.near_goal_segmentation_20261103.evaluation import oracle_diagnostic
from our_mani.near_goal_segmentation_20261103.training import train_capture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True,type=Path)
    group=parser.add_mutually_exclusive_group();group.add_argument('--check',action='store_true');group.add_argument('--oracle-check',action='store_true');group.add_argument('--train',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--device',default='cpu');parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--allow-pilot-smoke',action='store_true');args=parser.parse_args()
    if args.train:
        if args.output is None:parser.error('--train requires new --output')
        print(train_capture(args.manifest,args.output,device=args.device,workers=args.workers,allow_pilot_smoke=args.allow_pilot_smoke));return 0
    dataset=CaptureDataset(args.manifest)
    if args.oracle_check:
        result=oracle_diagnostic(dataset);print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['ready_for_segmentation_fit'] else 3
    for sample in dataset:
        if sample['features'].shape!=(4,9,64,64):raise ValueError('crop schema mismatch')
    print(json.dumps(dict(ready=True,scope=dataset.scope,independent_scenes=len(dataset),features=[4,9,64,64],
        labels_physically_separate=True,training_started=False,Native_started=False,TEST_read=False)));return 0

if __name__=='__main__':raise SystemExit(main())
