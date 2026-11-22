"""Large-centre default source check; Root explicitly runs preflight or train."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.current_cube_rgbd_centre_large_20261104_v1.training import check, gpu_preflight, fit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest'); p.add_argument('--encoder-checkpoint'); p.add_argument('--expected-encoder-sha256')
    p.add_argument('--output'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--preflight'); p.add_argument('--expected-preflight-sha256')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--gpu-preflight', action='store_true'); modes.add_argument('--train', action='store_true')
    a = p.parse_args(argv)
    if a.gpu_preflight or a.train:
        if not all((a.manifest, a.encoder_checkpoint, a.expected_encoder_sha256, a.output)):
            p.error('Explicit manifest/encoder/SHA/fresh output required')
        if a.train:
            if not a.preflight or not a.expected_preflight_sha256:
                p.error('Actual same-source GPU preflight and external SHA required before training')
            result = fit(a.manifest, a.encoder_checkpoint, a.expected_encoder_sha256, a.output,
                a.preflight, a.expected_preflight_sha256, device=a.device)
        else:
            result = gpu_preflight(a.manifest, a.encoder_checkpoint, a.expected_encoder_sha256, a.output, device=a.device)
    else: result = check(a.manifest)
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == '__main__': raise SystemExit(main())
