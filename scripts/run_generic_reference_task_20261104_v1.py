"""Inspect the actual16task generic reference-MPC port; never auto-launch Native."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.generic_reference_mpc_20261104_v1.runner import check


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task');p.add_argument('--check',action='store_true')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if args.execute:
        p.error('No trained world-command runtime hook supplied yet; Root must bind actual BC/e artifacts and fresh admission before Native')
    print(json.dumps(check(args.task),indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
