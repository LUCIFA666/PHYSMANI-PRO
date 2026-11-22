"""Prepare or explicitly preflight/fit the fixed normalized207M eight-operation encoder."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.semantic_task_e_large_20261104_v2.common import check_preparation,source_map


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--root-authorization',type=Path)
    parser.add_argument('--root-authorization-sha256')
    parser.add_argument('--preflight-report',type=Path)
    parser.add_argument('--preflight-sha256')
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--oom-predecessor',type=Path)
    parser.add_argument('--oom-predecessor-sha256')
    parser.add_argument('--pipeline-owned-locks',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--gpu-preflight',action='store_true')
    mode.add_argument('--train',action='store_true')
    args=parser.parse_args(argv)
    if args.gpu_preflight or args.train:
        if args.output is None or args.root_authorization is None or args.root_authorization_sha256 is None:
            parser.error('Execution requires fresh --output and exact Root authorization path/SHA')
        from our_mani.semantic_task_e_large_20261104_v2.training import gpu_preflight,train_selected
        common=dict(manifest=args.manifest,data_root=args.data_root,output=args.output,
            authorization=args.root_authorization,authorization_sha=args.root_authorization_sha256,
            pipeline_owned=args.pipeline_owned_locks)
        if args.gpu_preflight:
            result=gpu_preflight(**common,microbatch=args.microbatch,
                oom_predecessor=args.oom_predecessor,oom_sha=args.oom_predecessor_sha256)
        else:
            if args.preflight_report is None or args.preflight_sha256 is None:
                parser.error('--train requires actual --preflight-report and --preflight-sha256')
            if args.microbatch!=4 or args.oom_predecessor is not None or args.oom_predecessor_sha256 is not None:
                parser.error('Fit microbatch is bound by successful preflight, not a new CLI choice')
            result=train_selected(**common,preflight_report=args.preflight_report,preflight_sha=args.preflight_sha256)
    else:
        result=check_preparation(args.manifest,args.data_root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
