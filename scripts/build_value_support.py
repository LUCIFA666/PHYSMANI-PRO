"""Freeze a TRAIN-only input support guard for the controller's Value."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from our_mani.local_learning import SuccessValue
from train_local_models import records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    rows = records(ROOT/args.data)
    value = SuccessValue.load(ROOT/args.model)
    if {r['controller_sha256'] for _,r in rows} != {value.metadata['controller_sha256']}:
        raise ValueError('support data must use the Value controller')
    identity = hashlib.sha256(b''.join(p.read_bytes() for p,_ in rows)).hexdigest()
    if identity != value.metadata['dataset_sha256']:
        raise ValueError('support data differ from the model training receipt')
    groups = sorted({(r['task'],r['source_episode']) for _,r in rows})
    rng, held = np.random.default_rng(810), set()
    for task in sorted({g[0] for g in groups}):
        selected = [g for g in groups if g[0]==task]
        held.update(selected[i] for i in rng.permutation(len(selected))[:max(1,len(selected)//5)])
    fit = np.asarray([r['initial_features'] for _,r in rows if (r['task'],r['source_episode']) not in held])
    select = np.asarray([r['initial_features'] for _,r in rows if (r['task'],r['source_episode']) in held])
    # Fixed physical slack accommodates measurement noise without authorizing
    # large extrapolations in forecast speed or local relative state.
    slack = np.r_[[.02]*3,[.05]*6,[.1]*6,0.,0.,0.]
    scale = np.maximum(value.std, np.r_[[.02]*3,[.05]*6,[.05]*6,1.,1.,.25])
    normalized = (fit-value.mean)/scale
    distance = cKDTree(normalized).query((select-value.mean)/scale)[0]
    threshold = float(np.quantile(distance,.99))
    report = {'kind':'TRAIN_only_support_for_controller_specific_Value', 'formal_test_used':False,
              'source':args.data, 'dataset_sha256':identity, 'controller_sha256':value.metadata['controller_sha256'],
              'value_checkpoint_sha256':hashlib.sha256((ROOT/args.model).read_bytes()).hexdigest(),
              'lower':(fit.min(0)-slack).tolist(), 'upper':(fit.max(0)+slack).tolist(),
              'mean':value.mean.tolist(), 'scale':scale.tolist(), 'normalized_fit_features':normalized.tolist(),
              'maximum_nearest_fit_distance':threshold,
              'distance_threshold_rule':'99th percentile of same TRAIN group-heldout selection states',
              'fit_n':len(fit), 'train_group_heldout_n':len(select)}
    path = (ROOT/args.model).with_name('value_support.json')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('normalized_fit_features','lower','upper','mean','scale')},indent=2))


if __name__=='__main__':
    main()
