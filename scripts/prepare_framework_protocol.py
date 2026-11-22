"""Lock matching episode identities and audit reusable read-only benchmark assets."""
import hashlib
import json
from pathlib import Path
import sys
import pickle
from datetime import datetime, timezone

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.physmani_data import RecordedEpisode, CAMERAS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    external = Path('/home/ubuntu/桌面/phsimani')
    benchmark = external/'PhysMani/data/rmt/physmani_bench'
    source = external/'physmani_reproduction/evidence/test_manifest.json'
    manifest = json.loads(source.read_text())
    output = ROOT/'artifacts/framework/protocol'
    tasks = ("pick_moving_target_on_the_table", "pick_moving_target_on_the_table_high_speed")
    episodes = []
    for task, ids in manifest["raw_episode_ids"].items():
        if len(ids) != 40 or len(set(ids)) != 40:
            raise RuntimeError(f"invalid test identities: {task}")
        for index, raw_id in enumerate(ids):
            episodes.append({"task": task, "raw_episode_id": raw_id, "official_index": index,
                             "seed": 0, "path": str(benchmark/'test'/task/'all_variations/episodes'/raw_id)})
    if len(episodes) != 640:
        raise RuntimeError("full benchmark must contain 640 episodes")
    contract = {"cameras": list(CAMERAS), "source_resolution": [256, 256], "wrist_camera": False,
                "simulation_dt_seconds": .05, "observation_access": "causal_rgbd_and_robot_state",
                "privileged_target_state_as_policy_input": False,
                "robot": "released_physmani_panda", "matched_reset": "task_raw_episode_id_and_seed",
                "comparison_tracks": {"benchmark_timing": "match baseline discrete observation/control schedule",
                                      "real_time": "both methods use measured wall latency and asynchronous execution"}}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    write(output/'observation_contract.json', {"sha256": contract_hash, "contract": contract})
    write(output/'test_manifest_all16.json', {"source": str(source), "source_sha256": sha(source), "episodes": episodes})
    write(output/'test_manifest_grasp2.json', {"scope": "2 grasp tasks, not the full 16-task paper benchmark",
                                             "source_manifest_sha256": sha(source),
                                             "episodes": [r for r in episodes if r["task"] in tasks]})
    splits, identities, counts = {}, {}, {}
    for split, directory in (("train", "train_package_compressed"), ("validation", "val_package_compressed")):
        rows = []
        for path in sorted((benchmark/directory).glob('*/all_variations/episodes/*/data.h5')):
            task, raw_id = path.parts[-5], path.parts[-2]
            rows.append({"task": task, "raw_episode_id": raw_id, "path": str(path)})
        splits[split] = rows
        identities[split] = {(r["task"], r["raw_episode_id"]) for r in rows}
        counts[split] = {task: sum(r["task"] == task for r in rows) for task in sorted({r["task"] for r in rows})}
        write(output/f'{split}_manifest.json', {"split": split, "episodes": rows})
    identities['test'] = {(r['task'], r['raw_episode_id']) for r in episodes}
    same_named = {}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlap = identities[a] & identities[b]
        same_named[f'{a}:{b}'] = len(overlap)
    # Raw episode basenames are scoped to a split in the released packages.
    # Recompute recording signatures; a shared name alone is not data leakage.
    fingerprints = {}
    for split in ('train', 'validation'):
        rows = []
        for row in splits[split]:
            with h5py.File(row['path'], 'r') as f:
                gripper = np.c_[f['low_dim_obs/gripper_pose'][:], f['low_dim_obs/gripper_open'][:]].astype('<f4')
                image = f['observations/front/rgb'][0]
            rows.append({**row, 'frames': len(gripper),
                         'gripper_sha256': hashlib.sha256(gripper.tobytes()).hexdigest(),
                         'first_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        fingerprints[split] = rows
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(external/'PhysMani/packages/physmani_bench'))
    test_rows = []
    for row in episodes:
        path = Path(row['path'])
        observations = pickle.loads((path/'low_dim_obs.pkl').read_bytes())
        gripper = np.asarray([np.r_[s.gripper_pose, float(s.gripper_open)] for s in observations], dtype='<f4')
        image = np.asarray(Image.open(path/'front_rgb/0.png').convert('RGB'))
        test_rows.append({**row, 'frames': len(gripper),
                          'gripper_sha256': hashlib.sha256(gripper.tobytes()).hexdigest(),
                          'first_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
    fingerprints['test'] = test_rows
    signature = lambda r: (r['task'], r['frames'], r['gripper_sha256'], r['first_rgb_sha256'])
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlap = {signature(r) for r in fingerprints[a]} & {signature(r) for r in fingerprints[b]}
        if overlap:
            raise RuntimeError(f"possible duplicate recordings between {a} and {b}")
    write(output/'recording_fingerprints.json', fingerprints)
    sample = next(r for r in splits['train'] if r['task'] in tasks)
    with RecordedEpisode(sample['path']) as reader:
        observation = reader.observation(0)
        reader_check = {"episode": sample, "frames": len(reader),
                        "rgb_shapes": {c: list(v.shape) for c, v in observation['observation/images'].items()},
                        "depth_m_ranges": {c: [float(v.min()), float(v.max())] for c, v in observation['observation/depths'].items()},
                        "observation_keys": list(observation), "privileged_labels_exposed": False}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "external_assets_access": "read_only",
              "test_episodes": len(episodes), "initial_grasp_task_episodes": sum(r['task'] in tasks for r in episodes),
              "train_episodes": len(splits['train']), "validation_episodes": len(splits['validation']),
              "split_counts_by_task": counts, "same_named_episodes_across_splits": same_named,
              "episode_basename_namespace": "local to each split; not a global identity",
              "exact_recording_signature_overlap": False,
              "duplicate_audit_scope": "task, frame count, full float32 gripper trajectory, decoded first front RGB; not semantic similarity",
              "sample_hdf5_reader": reader_check, "observation_contract_sha256": contract_hash,
              "validation_selection": "select on validation only; lock model before any test evaluation",
              "paper_top3_checkpoint_aggregation": "separate reproduction track, never use test SR to select ours",
              "camera_comparison_note": "FASTER mosaics the same four source views into one image; architecture/resolution adaptation is disclosed",
              "status": "protocol_prepared_no_new_closed_loop_success_results"}
    write(output/'audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
