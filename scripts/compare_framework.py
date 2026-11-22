"""Strict paired comparison after BOTH methods finish the locked test manifest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.evaluation import EpisodeResult, compare


def load(path):
    return [EpisodeResult(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/framework/protocol/test_manifest_grasp2.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/framework/comparison.json')
    args = parser.parse_args()
    expected = [(r['task'], r['raw_episode_id'], r['seed']) for r in json.loads(args.manifest.read_text())['episodes']]
    result = compare(load(args.ours), load(args.baseline), expected_keys=expected)
    result['locked_manifest'] = str(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
