"""Large-model fixed208 selection or single-use104 reserved freeze then score."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.current_cube_rgbd_centre_large_20261104_v2.training import check
from our_mani.current_cube_rgbd_centre_large_20261104_v2.evaluation import predict, score


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest'); p.add_argument('--selected'); p.add_argument('--expected-sha256')
    p.add_argument('--encoder-checkpoint'); p.add_argument('--output')
    p.add_argument('--role', choices=('selection', 'reserved_fresh_centre_VAL'), default='reserved_fresh_centre_VAL')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--predict', action='store_true'); modes.add_argument('--score', action='store_true')
    a = p.parse_args(argv)
    if a.predict:
        if not all((a.manifest, a.selected, a.expected_sha256, a.encoder_checkpoint, a.output)):
            p.error('Exact selected/SHA/manifest/encoder/fresh output required')
        result = predict(a.selected, a.expected_sha256, a.manifest, a.output, a.encoder_checkpoint, role=a.role)
    elif a.score:
        if not a.output: p.error('Frozen prediction directory required')
        result = score(a.output)
    else: result = check(a.manifest)
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == '__main__': raise SystemExit(main())
