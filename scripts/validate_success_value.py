"""Independent VAL calibration for one explicitly identified local controller."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.local_learning import SuccessValue, probability_metrics
from validate_local_models import read_rows, cluster_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = list(read_rows(ROOT/args.data).values())
    value = SuccessValue.load(ROOT/args.model)
    if {r['controller_sha256'] for r in rows} != {value.metadata['controller_sha256']}:
        raise RuntimeError('Value and outcome controller must have the same fingerprint')
    labels = np.asarray([r['stable_lift'] for r in rows], dtype=int)
    predictions = value(np.asarray([r['initial_features'] for r in rows]))
    by_task = {task: probability_metrics(predictions[[r['task']==task for r in rows]],
                                        labels[[r['task']==task for r in rows]])
               for task in sorted({r['task'] for r in rows})}
    selected = []
    for threshold in (.5, .7, .8):
        keep = predictions >= threshold
        chosen = [r for r, k in zip(rows, keep) if k]
        selected.append({'threshold': threshold, 'n': int(keep.sum()),
                         'stable_lift_rate': float(labels[keep].mean()) if keep.any() else None,
                         'episode_cluster_bootstrap_95': cluster_interval(chosen, labels[keep]) if keep.any() else None})
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'formal_test_used': False,
              'kind': 'controller_specific_independent_local_VAL_not_official_benchmark',
              'controller_sha256': value.metadata['controller_sha256'],
              'model': args.model, 'metrics': probability_metrics(predictions, labels),
              'per_task': by_task, 'selected_regions': selected}
    out = ROOT/args.out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
