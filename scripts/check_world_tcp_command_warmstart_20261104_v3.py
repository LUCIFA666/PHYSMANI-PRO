"""Default CPU/source check for v3; no model checkpoint or pixels read."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.world_tcp_command_warmstart_20261104_v3.checks import check


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(check(args.subset),ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
