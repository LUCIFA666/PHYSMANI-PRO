"""Root-only fine cache canary/build; labels join only after full public freeze."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset',type=Path)
    parser.add_argument('--encoder-checkpoint',type=Path)
    parser.add_argument('--encoder-sha256')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--root-authorization',type=Path)
    parser.add_argument('--root-authorization-sha256')
    parser.add_argument('--preflight-report',type=Path)
    parser.add_argument('--preflight-sha256')
    parser.add_argument('--public-cache-manifest',type=Path)
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--oom-predecessor',type=Path)
    parser.add_argument('--oom-predecessor-sha256')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--gpu-canary',action='store_true')
    modes.add_argument('--build-cache',action='store_true')
    modes.add_argument('--join-labels',action='store_true')
    args=parser.parse_args(argv)
    from our_mani.world_tcp_command_warmstart_20261104_v3 import cache
    if args.join_labels:
        if args.public_cache_manifest is None or args.output is None:parser.error('Join needs frozen --public-cache-manifest and fresh --output')
        result=cache.join_command_labels(args.public_cache_manifest,args.output)
    elif args.gpu_canary or args.build_cache:
        if any(v is None for v in (args.subset,args.encoder_checkpoint,args.encoder_sha256,args.output,args.root_authorization,args.root_authorization_sha256)):
            parser.error('Actual GPU stage requires subset, selected encoder wholeSHA, fresh output and exact Root authorization')
        shared=dict(subset=args.subset,encoder_checkpoint=args.encoder_checkpoint,encoder_sha=args.encoder_sha256,
            output=args.output,authorization=args.root_authorization,authorization_sha=args.root_authorization_sha256)
        if args.gpu_canary:
            result=cache.gpu_canary(**shared,microbatch=args.microbatch,oom_predecessor=args.oom_predecessor,oom_sha=args.oom_predecessor_sha256)
        else:
            if args.preflight_report is None or args.preflight_sha256 is None:parser.error('Build requires actual same-source canary path/SHA')
            result=cache.build_public_cache(**shared,preflight_report=args.preflight_report,preflight_sha=args.preflight_sha256)
    else:
        if args.subset is None:parser.error('Default check requires fixed --subset')
        from our_mani.world_tcp_command_warmstart_20261104_v3.checks import check
        from our_mani.world_tcp_command_warmstart_20261104_v3.common import fixed_config,source_map,verify_sources
        result=check(args.subset);sources=source_map();verify_sources(sources)
        result.update(pilot_config=fixed_config(),pipeline_source_sha256=sources,cache_or_GPU_or_fit_started=False)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
