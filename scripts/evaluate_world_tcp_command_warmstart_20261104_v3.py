"""Selected-v3 fixed5168 command VAL: numeric freeze then diagnostic score."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.world_tcp_command_warmstart_20261104_v3.training import check
from our_mani.world_tcp_command_warmstart_20261104_v3.evaluation import predict,score


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-manifest');p.add_argument('--selected');p.add_argument('--expected-sha256')
    p.add_argument('--output');p.add_argument('--device',default='cpu')
    modes=p.add_mutually_exclusive_group()
    for mode in ('check','predict','score','evaluate'):modes.add_argument('--'+mode,action='store_true')
    a=p.parse_args(argv)
    if a.predict or a.evaluate:
        if not all((a.cache_manifest,a.selected,a.expected_sha256,a.output)):
            p.error('Exact selected/SHA, joined cache and fresh output required')
        value=predict(a.selected,a.expected_sha256,a.cache_manifest,a.output,device=a.device)
        if a.evaluate:value=score(a.output)
    elif a.score:
        if not a.output:p.error('Frozen prediction output required')
        value=score(a.output)
    else:value=check(a.cache_manifest)
    print(json.dumps(value,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
