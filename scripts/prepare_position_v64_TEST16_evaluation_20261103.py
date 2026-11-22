"""Reserved16 evaluation-only tools. Default checks metadata, never materializes.

Later --materialize requires a new explicit Root authorization after full
devVAL8 closure; sourcefreeze binding is another append-only metadata phase.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261103.position_v64_test16_protocol import (
    check_preparation,materialize,bind_source_freeze,
)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group();g.add_argument('--materialize',action='store_true');g.add_argument('--bind-source-freeze',type=Path)
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256');p.add_argument('--source-freeze-sha256')
    a=p.parse_args(argv)
    if a.materialize:
        if a.authorization is None or not a.authorization_sha256:p.error('later explicit Root authorization+SHA required')
        result=materialize(a.authorization,a.authorization_sha256)
    elif a.bind_source_freeze is not None:
        if not a.source_freeze_sha256:p.error('prospective sourcefreeze SHA required')
        result=bind_source_freeze(a.bind_source_freeze,a.source_freeze_sha256)
    else:result=check_preparation()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
