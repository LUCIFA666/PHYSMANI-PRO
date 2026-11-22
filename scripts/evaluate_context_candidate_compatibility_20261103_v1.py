"""Prepare public P4 A/B bundles by default; explicit --run evaluates frozen TF."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    from our_mani.context_candidate_compatibility_20261103_v1 import prepare,run
    result=run(a.output)if a.run else prepare(a.output)
    print(json.dumps({key:result[key]for key in ('version','all_anchors')},ensure_ascii=False))


if __name__=='__main__':main()
