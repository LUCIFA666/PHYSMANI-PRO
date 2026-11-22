"""Prepare labels-only command-run index; default reads only metadata/stat."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.world_tcp_command_warmstart_20261104_v1.data import metadata_check,prepare_index
from our_mani.world_tcp_command_warmstart_20261104_v1.model import CandidateConfig,parameter_count


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-index',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare_index:
        if args.output is None:parser.error('--prepare-index requires fresh --output')
        result=prepare_index(args.manifest,args.output)
    else:
        result=metadata_check(args.manifest)
        result['candidate_exact_parameters']=parameter_count()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
