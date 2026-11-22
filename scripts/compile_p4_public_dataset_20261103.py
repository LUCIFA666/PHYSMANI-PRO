"""Compile an explicitly preassigned TRAIN or development VAL teacher plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from our_mani.local_mpc_p4_20261103.data import public_features, successful_contacts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_assignment(job, expected_split):
    """Bind the predeclared parent/split/config before reading an episode."""
    case = (ROOT/job['path']).resolve()
    config_path = (ROOT/job['config']).resolve()
    family = ROOT/'runtime/framework/local_mpc_20261102'
    if family not in case.parents or ROOT not in config_path.parents:
        raise ValueError('prepared project config and runtime-family case required')
    if job.get('split') != expected_split or sha(config_path) != job['config_sha256']:
        raise ValueError('declared split/config identity differs')
    cfg = json.loads(config_path.read_text()); meta = cfg.get('P4_collection_metadata',{})
    if (cfg.get('dataset_use') != expected_split or cfg.get('seed') != job.get('seed')
            or meta.get('parent_id') != job.get('parent_id') or not job.get('parent_id')
            or meta.get('split') != expected_split or meta.get('episode_id') != job.get('episode_id')
            or meta.get('variant_id') != job.get('variant_id')
            or meta.get('parent_assigned_before_episode_generation') is not True):
        raise ValueError('parent/episode/seed/variant/split binding differs')
    if ('/train_package_compressed/' not in cfg.get('calibration_train_episode','')
            or cfg.get('task') != 'pick_moving_target_on_the_table'):
        raise ValueError('original TRAIN calibration for the declared block task only')
    return case, config_path, cfg


def compile_dataset(plan_path, output, limit=None, expected_split='train'):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_text()); jobs = plan['jobs']
    # Inspect assignment before opening any episode, including when compiling
    # only the fixed first tranche. A source with a mixed split is rejected.
    if expected_split not in ('train','val'):
        raise ValueError('only TRAIN or explicitly preassigned development VAL is supported; no TEST')
    if not jobs or any(job.get('split') != expected_split for job in jobs):
        raise ValueError('this compiler requires one preassigned split matching explicit role')
    if limit is not None:
        if not 1 <= limit <= len(jobs): raise ValueError('valid explicit tranche limit required')
        jobs = jobs[:limit]
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = {str(plan_path): sha(plan_path)}; tensors = []; ledger = []; cases = []
    frame_cache = {}
    for job in jobs:
        case, config_path, declared_config = validate_assignment(job, expected_split)
        parent = job['parent_id']; sources[str(config_path)] = sha(config_path)
        paths = {key: case / name for key, name in dict(
            inputs='run_inputs.json', result='result.json', exit='worker_exit.json',
            initial='initial_snapshot.json', ticks='native_ticks.jsonl',
            public='public_shadow/public_outputs.jsonl', scope='public_shadow/scope.json',
            calibration='public_shadow/camera_calibration.json').items()}
        owner = Path(str(case)+'.owner.actual_exit.json')
        if json.loads(owner.read_text())['actual_process_returncode'] != 0:
            raise ValueError('actual successful child exit required')
        sources[str(owner)] = sha(owner)
        loaded = {}
        for key, path in paths.items():
            sources[str(path)] = sha(path)
            loaded[key] = lines(path) if path.suffix == '.jsonl' else json.loads(path.read_text())
        cfg = loaded['inputs']['config']
        if (cfg.get('dataset_use') != expected_split or cfg.get('public_motion_shadow') is not True
                or loaded['scope'].get('controller_uses_public_forecast') is not False
                or loaded['scope'].get('object_GT_held_success_allowed_in_public_bridge') is not False
                or loaded['result'].get('source_unchanged') is not True
                or loaded['result'].get('complete') is not True or loaded['inputs'].get('TEST_read') is not False
                or loaded['inputs'].get('scene') != job['scene'] or cfg != declared_config
                or loaded['exit'].get('exit_code') != 0):
            raise ValueError('unchanged preassigned truth teacher with passive public50ms recording required')
        snapshot = loaded['initial']
        if (snapshot.get('native_height_preserved') is not True or
                abs(snapshot['state']['object_position'][2]-snapshot['native_default_target_pose_before_override'][2])>1e-8):
            raise ValueError('original Native tabletop height required')
        for relative, expected in loaded['inputs']['source_sha256'].items():
            archive = case / 'source_snapshot' / relative
            if sha(archive) != expected: raise ValueError('source archive differs: '+relative)
            sources[str(archive)] = expected
        labels = successful_contacts(loaded['initial']['state'], loaded['ticks'])
        public = loaded['public']; reasons = Counter(); count = 0
        if len(public) != len(loaded['ticks'])+1:
            raise ValueError('one initial plus every actual50ms public frame required')
        for index, row in enumerate(public):
            now = float(row['public_robot']['timestamp'])
            if abs(now-index*.05)>1e-6: raise ValueError('public frame stream has missing/extra time')
            matching = [label for label in labels if 1e-8 < label['tau']-now <= .8+1e-8]
            if not matching: continue
            try:
                features, provenance = public_features(row, public[:index+1])
            except ValueError as error:
                reasons[str(error)] += 1; continue
            for relative, timestamp in zip(provenance['real_history_rgbd'], provenance['real_history_timestamps']):
                frame = (case / 'public_shadow' / relative).resolve()
                if (case/'public_shadow').resolve() not in frame.parents:
                    raise ValueError('public frame outside capture directory')
                if frame not in frame_cache:
                    with np.load(frame, allow_pickle=False) as arrays:
                        captured = float(arrays['timestamp'])
                        for camera in ('front', '8', '16', '36'):
                            if 'images_'+camera not in arrays or 'depths_'+camera not in arrays:
                                raise ValueError('all four real RGB-D views required')
                    frame_cache[frame] = captured; sources[str(frame)] = sha(frame)
                if abs(frame_cache[frame]-timestamp)>1e-8:
                    raise ValueError('actual saved RGBD timestamp differs from public history')
            for label in matching:
                tensors.append(dict(**features,
                    target_position=np.asarray(label['object_tcp_position'],np.float32),
                    target_rotation=np.asarray(label['object_tcp_rotation'],np.float32),
                    target_dt=np.float32(label['tau']-now)))
                ledger.append(dict(parent_id=parent, split=expected_split, case=str(case), public_index=index,
                    timestamp=now, tau=label['tau'], label_event=label['label_event'],
                    label_acceptance=label['acceptance'], public_feature_provenance=provenance))
                count += 1
        cases.append(dict(parent_id=parent, case=str(case), accepted_contact_labels=len(labels),
                          public_training_anchors=count, rejected_anchor_reasons=dict(reasons)))
    parents = sorted({row['parent_id'] for row in ledger})
    if not tensors:
        raise ValueError('no eligible real public causal training anchors; no dataset generated')
    values = {key: np.stack([row[key] for row in tensors]) for key in tensors[0]}
    ids = np.asarray([parents.index(row['parent_id']) for row in ledger], dtype=np.int64)
    counts = Counter(ids.tolist())
    values['parent_index'] = ids
    values['sample_weight'] = np.asarray([1./counts[index] for index in ids],np.float32)
    with (output/'dataset.npz').open('xb') as stream: np.savez_compressed(stream, **values)
    report = dict(schema='P4_public_causal_teacher_pilot_dataset_v1', split=expected_split, cases=cases,
        anchors=len(ledger), positive_independent_parents=len(parents), parents=parents,
        shapes={key:list(value.shape) for key,value in values.items()},
        task_classifier_trained=False, task_latent='fixed_uniform16_placeholder_distinct_from_frozen_motion_context16',
        success_probability_supervised=False, multimodal_grasp_coverage_qualified=False,
        labels='privileged future labels only; actual first attached state followed by same-segment goal+liftedhold',
        input_scope='current/past public observations, robot sensors and causal frozen GRU forecasts only',
        dataset_sha256=sha(output/'dataset.npz'), source_sha256=sources,
        compiler_sha256=sha(__file__), feature_source_sha256=sha(ROOT/'our_mani/local_mpc_p4_20261103/data.py'),
        formal_TEST_read=False, VAL_read=expected_split=='val', training_started=False)
    for name, value in [('report.json',report),('anchor_ledger.json',ledger)]:
        with (output/name).open('x') as stream: json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--split',choices=('train','val'),default='train')
    args=parser.parse_args()
    report=compile_dataset(args.plan,args.output,args.limit,args.split)
    print(json.dumps({key:report[key] for key in ('anchors','positive_independent_parents','shapes')},ensure_ascii=False))


if __name__=='__main__': main()
