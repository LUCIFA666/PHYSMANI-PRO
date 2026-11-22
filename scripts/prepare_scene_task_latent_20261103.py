"""Explicit first-frame TRAIN sensor preparation; never opens official VAL/TEST."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.scene_task_latent_20261103.data import prepare_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261103)
    args=parser.parse_args();manifest=prepare_dataset(args.output,seed=args.seed)
    print(json.dumps(dict(output=str(args.output),complete=manifest['complete'],split=manifest['split']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
