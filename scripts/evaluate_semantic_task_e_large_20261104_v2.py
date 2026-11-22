"""Development selection classification only; default has no model forward."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.semantic_task_e_large_20261104_v2.common import check_preparation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--root-authorization',type=Path)
    parser.add_argument('--root-authorization-sha256')
    parser.add_argument('--pipeline-owned-locks',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args(argv)
    if args.evaluate:
        if any(value is None for value in (args.checkpoint,args.checkpoint_sha256,args.output,
            args.root_authorization,args.root_authorization_sha256)):
            parser.error('Explicit selected checkpoint/output/Root authorization pointers required')
        from our_mani.semantic_task_e_large_20261104_v2.evaluation import evaluate_selected
        result=evaluate_selected(args.manifest,args.data_root,args.checkpoint,args.checkpoint_sha256,
            args.output,args.root_authorization,args.root_authorization_sha256,
            pipeline_owned=args.pipeline_owned_locks)
    else:
        result=check_preparation(args.manifest,args.data_root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
