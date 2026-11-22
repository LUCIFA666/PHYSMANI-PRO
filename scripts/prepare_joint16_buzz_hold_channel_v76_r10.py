"""Add a disclosed causal hold-only rotation channel before joint16 training.

Original caches and native label values are immutable. New cache files change
only observed rotation/twist/channel masks and the corresponding supervised
rotation-availability mask. No TEST, native launch or optimizer is involved.
"""
from pathlib import Path
import sys,json
from copy import deepcopy
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.joint16_tracks_v31 import read_joint16_tracks
from our_mani.multitask.track_dataset import read_tracks,validate_motion_label_schema
from our_mani.multitask.buzz_hold_observations_v76 import derive_hold_observations
from our_mani.multitask.two_center_pose import load_two_center_pose_prior
B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'joint16_buzz_hold_channel_v76'
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')


def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))


def main():
    if OUT.exists():raise RuntimeError('preserve existing derived data')
    source=B/'joint16_body_track_adapter_v31/manifest.json';joint=json.loads(source.read_text())
    native_shard=ROOT/joint['shards'][0]['path'];assert file_sha(native_shard)==joint['shards'][0]['sha256']
    native=json.loads(native_shard.read_text());entries=[];proofs=[];inputs={str(source):file_sha(source),str(native_shard):file_sha(native_shard)}
    models={}
    for task in TASKS:
        p=B/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/models'/task/'model.json';m=json.loads(p.read_text())
        models[task]=load_two_center_pose_prior(ROOT,pointer(p),canonical_asset_pointer=m['canonical_basis']['TRAIN_CAD'])
        inputs[str(p)]=file_sha(p)
        for pt in (*m['TRAIN_evidence'].values(),m['canonical_basis']['TRAIN_CAD'],m['helper_source']):inputs[str(ROOT/pt['path'])]=pt['sha256']
    OUT.mkdir();(OUT/'tracks').mkdir()
    for old in native['entries']:
        entry=deepcopy(old)
        if old['task'] not in TASKS:entries.append(entry);continue
        path=ROOT/old['cache'];assert file_sha(path)==old['cache_sha256'];inputs[str(path)]=old['cache_sha256']
        if old.get('teacher_used') is True:raise ValueError('no-action hold channel cannot derive teacher operation rotations')
        with np.load(path,allow_pickle=False) as stored:z={k:stored[k].copy() for k in stored.files}
        if not np.all(z['public_command_kind'].astype(str)=='hold') or z['robot_contact'].any() or not z['robot_gripper_open'].all():raise ValueError('exact original public no-action hold scope required')
        observed,proof=derive_hold_observations(models[old['task']],z);derived=dict(z,**observed)
        derived['target_rotation_observable']=(observed['rotation_observable']&z['native_pose_valid']&z['target_pose_valid_before_twist_mask'])
        changed=('rotations','angular_velocities','rotation_observable','target_rotation_observable')
        if any(not np.array_equal(derived[k],z[k],equal_nan=True) for k in z if k not in changed):raise ValueError('original position/native mask/label values changed')
        validate_motion_label_schema(derived,native['dynamics_target_label_schema'],old['dynamics_target_label_schema'])
        destination=OUT/'tracks'/(old['split']+'_'+old['task']+'_'+old['raw_episode_id']+'.npz');np.savez_compressed(destination,**derived)
        entry.update(cache=str(destination.relative_to(ROOT)),cache_sha256=file_sha(destination),
            derived_observed_rotation_channel=proof['schema'],original_source_cache=dict(path=old['cache'],sha256=old['cache_sha256']),
            static_orientation_prior_source='original100_TRAIN_hold_parents_only',original_native_label_values_unchanged=True,
            new_conditional_target_rotation_mask=True,current_RGBD_direct_SO3_registration_claimed=False,
            deployment_orientation_channel_difference='hold training uses public two-centre prior; online action uses initial prior then current RGBD continuity v63; must verify post-training native smoke')
        entries.append(entry);proofs.append(dict(split=old['split'],task=old['task'],raw_episode_id=old['raw_episode_id'],proof=proof,
            original=entry['original_source_cache'],derived=pointer(destination),changed_fields=list(changed)))
    part=deepcopy(native);part['entries']=entries;part['observed_rotation_extension']=dict(schema='r10_no_action_hold_public_prior_extension_v76',
        source=pointer(ROOT/'our_mani/multitask/buzz_hold_observations_v76.py'),inputs_sha256=inputs,
        original_native_target_values_and_twist_masks_unchanged=True,TEST_read=False,teacher_or_future_native_labels_in_history=False,
        deployment_input_channel_difference_disclosed=True)
    shard=OUT/'r10_manifest.json';atomic_json(shard,part);read_tracks(ROOT,shard)
    joint['shards'][0]=dict(**pointer(shard),dynamics_target_label_schema=native['dynamics_target_label_schema'])
    joint['observation_extension']=part['observed_rotation_extension'];joint['model_training_started']=False
    atomic_json(OUT/'manifest.json',joint);meta,episodes=read_joint16_tracks(ROOT,OUT/'manifest.json')
    verify_inputs(ROOT,inputs);atomic_json(OUT/'channel_proofs.json',proofs)
    report=dict(complete=True,manifest=pointer(OUT/'manifest.json'),derived_buzz_acquisitions=len(proofs),
        source_counts={':'.join(k):v for k,v in sorted(Counter((p['split'],p['task']) for p in proofs).items())},
        all_original_joint_parents=len({e.source_key for e in episodes}),native_launched=False,optimizer_steps=0,GPU_used=False,TEST_read=False,
        original_caches_unchanged=True,original_native_targets_and_twist_masks_unchanged=True,
        conditional_rotation_channel=True,deployment_channel_difference_requires_smoke=True,formal_release=False,
        proofs=pointer(OUT/'channel_proofs.json'),source=pointer(Path(__file__)))
    atomic_json(OUT/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
