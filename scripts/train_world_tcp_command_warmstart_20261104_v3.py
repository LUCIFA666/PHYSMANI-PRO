"""Default CPU source check; Root explicitly runs fixed-five-epoch GPU BC."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.world_tcp_command_warmstart_20261104_v3.training import check,fit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-manifest');p.add_argument('--output');p.add_argument('--device',default='cuda:0')
    p.add_argument('--preflight-report');p.add_argument('--preflight-sha256')
    p.add_argument('--root-authorization');p.add_argument('--root-authorization-sha256')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--check',action='store_true');modes.add_argument('--train',action='store_true')
    a=p.parse_args(argv)
    if a.train:
        if not all((a.cache_manifest,a.output,a.preflight_report,a.preflight_sha256,a.root_authorization,a.root_authorization_sha256)):
            p.error('Actual joined cache, Root authorization, successful canary/SHA and fresh output required')
        value=fit(a.cache_manifest,a.output,a.preflight_report,a.preflight_sha256,
            a.root_authorization,a.root_authorization_sha256,device=a.device)
    else:value=check(a.cache_manifest)
    print(json.dumps(value,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
