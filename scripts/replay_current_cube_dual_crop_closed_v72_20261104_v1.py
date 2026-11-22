"""Replay selected current RGBD centre weights on the six closed V72 cases.

This is a development diagnostic, not a new task rollout or TEST. Public-only
predictions for all1046 frames are frozen before Native position scoring. The
default starts neither inference nor a simulator. --watch-pilot waits for the
already authorized visual pilot to finish, then predicts and scores once.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1.training import (
    VERSION, SELECTED_ENCODER_SHA, training_source_map, pointer, write_new)
from our_mani.current_cube_rgbd_centre_20261103_v1.features import public_model_arrays
from our_mani.near_goal_segmentation_20261103.data import CAMERAS, validate_public

BASE = ROOT/'artifacts/framework/local_mpc_20261102'
PLAN = ROOT/'config/framework/P4_position_dwell_v72_development_canaries_v1/plan.json'
PLAN_SHA = '489ca3197b50dd0495ba0c3ba90957141d6cda1c001fcd63172e5b50e0f835e6'
PILOT = BASE/'current_cube_dual_crop_CF_pilot40_training_pipeline_v1'
ENCODER = ROOT/'artifacts/framework/near_goal_segmentation_20261103/full32_geometryF_unknownKL_fit_v5/selected.pt'
FIT = BASE/'current_cube_dual_crop_CF_pilot40_fit_v1'
GATE = ROOT/'scripts/evaluate_public_position_closed_cases_20261103.py'
EXPECTED_CASES = ('00_val_static','01_val_static','03_val_uniform','04_val_uniform','uniform_002','static_no_proxy')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def public_input(npz, calibration, first_result):
    """Exact original seven current public inputs, without a Native channel."""
    from scipy.spatial.transform import Rotation
    goal, meta = first_result['goal_pose'], first_result['goal_metadata']
    if (meta.get('simulator_pose_used') is not False or
        meta.get('backend') != 'rgbd_red_goal_sphere_train_calibrated_frame'):
        raise ValueError('Original public RGBD goal-sphere fit required')
    anchor = np.asarray(goal['position'], float)-np.asarray(meta['marker_to_goal_m'], float)
    transforms = []
    for camera in CAMERAS:
        pose = np.asarray(calibration[camera]['camera_to_world'], float)
        if pose.shape != (7,) or not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-5):
            raise ValueError('Original public camera xyz+xyzw required')
        transform = np.eye(4); transform[:3,3] = pose[:3]
        transform[:3,:3] = Rotation.from_quat(pose[3:]).as_matrix(); transforms.append(transform)
    with np.load(npz, allow_pickle=False) as raw:
        expected = {'timestamp'}|{'images_'+c for c in CAMERAS}|{'depths_'+c for c in CAMERAS}
        if set(raw.files) != expected: raise ValueError('Exact current raw RGBD-only cache required')
        value = dict(rgb=np.stack([raw['images_'+c] for c in CAMERAS]),
            depth_m=np.stack([raw['depths_'+c] for c in CAMERAS]),
            K=np.asarray([calibration[c]['intrinsics'] for c in CAMERAS]), T=np.stack(transforms),
            public_sphere_center=anchor, camera_names=np.asarray(CAMERAS),timestamp=raw['timestamp'].copy())
    validate_public(value)
    return value


def public_digest(value):
    digest = hashlib.sha256()
    for name in sorted(value):
        a=np.asarray(value[name]);digest.update((name+str(a.dtype)+str(a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def closed_gate():
    if sha(PLAN) != PLAN_SHA: raise ValueError('Fixed V72 development plan changed')
    spec=importlib.util.spec_from_file_location('original_closed_V72_gate',GATE)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    _, cases, proof=gate.preflight(PLAN)
    if tuple(c['id'] for c in cases) != EXPECTED_CASES: raise ValueError('All fixed six cases required')
    return cases, proof


def predict(selected, expected_sha, output):
    if sha(selected) != expected_sha: raise ValueError('Whole selected checkpoint SHA required')
    cases, gate = closed_gate()  # Before any Native truth is opened.
    output=Path(output).resolve()
    if output.exists() or BASE not in output.parents:
        raise FileExistsError('Fresh artifact replay outside all original cases required')
    import torch
    from our_mani.current_cube_rgbd_centre_20261103_v1.model import DualCropCubeCentreHeadV1
    sources=training_source_map()
    for path in (Path(__file__).resolve(),GATE,PLAN,ENCODER,Path(selected).resolve()): sources[str(path)]=sha(path)
    for name,digest in cases[0]['run']['source_sha256'].items(): sources[str(ROOT/name)]=digest
    payload=torch.load(selected,map_location='cpu',weights_only=True);meta=payload.get('metadata',{})
    if (meta.get('version')!=VERSION or meta.get('training_complete')is not True or
        meta.get('actual_optimizer_steps')!=520 or meta.get('epochs_completed')!=40 or
        meta.get('source_sha256')!=training_source_map() or sha(ENCODER)!=SELECTED_ENCODER_SHA):
        raise ValueError('Completed same-source forty-epoch centre head required')
    status=read(Path(selected).parent/'fit_status.json')
    if status.get('training_complete')is not True or status.get('selected_checkpoint')!=pointer(selected):
        raise ValueError('Actual selected fit completion required')
    torch.set_num_threads(2);model=DualCropCubeCentreHeadV1(encoder_trainable=True)
    model.load_selected_v5_encoder(ENCODER,SELECTED_ENCODER_SHA)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    output.mkdir(parents=True,exist_ok=False)
    predictions=[];bound={};calls=0
    with torch.no_grad():
        for case in cases:
            shadow=case['case']/'public_shadow';path=shadow/'public_outputs.jsonl'
            calibration_path=shadow/'camera_calibration.json';calibration=read(calibration_path)
            bound[str(path)]=sha(path);bound[str(calibration_path)]=sha(calibration_path)
            packets=list(rows(path));first=packets[0]['result'];cache=[];metadata=[]
            for packet in packets:
                if (output/'STOP').exists(): raise InterruptedError('Administrative replay STOP')
                path=(shadow/packet['rgbd']).resolve(strict=True)
                if shadow/'rgbd' not in path.parents:raise ValueError('Original public image cache only')
                bound[str(path)]=sha(path);inputs=public_input(path,calibration,first)
                if abs(float(inputs['timestamp'])-float(packet['public_robot']['timestamp']))>1e-8:
                    raise ValueError('Same-time original observation required; no interpolation')
                cache.append(public_model_arrays(inputs))
                observed=packet['result'].get('object_observation')
                metadata.append(dict(case_id=case['id'],index=packet['index'],timestamp=float(inputs['timestamp']),
                    expected_negative=case['negative'],public_sphere_center=inputs['public_sphere_center'].tolist(),
                    original_public_target_available=observed is not None,
                    original_public_centre=None if observed is None else observed['pose']['position'],
                    packed_public_input_sha256=public_digest(inputs)))
                if len(cache)==16 or packet is packets[-1]:
                    batch={k:torch.as_tensor(np.stack([c[k]for c in cache]))for k in cache[0]}
                    values=model(**batch);calls+=1
                    arrays={k:values[k].cpu().numpy()for k in ('centre_world','measurement_availability_logit','uncertainty_score_m','uncertainty_score_valid')}
                    if any(not np.isfinite(a).all()for a in arrays.values()):raise ValueError('Finite output for every frame required')
                    for i,row in enumerate(metadata):
                        probability=float(np.exp(-np.logaddexp(0.,-float(arrays['measurement_availability_logit'][i]))))
                        row.update(centre_world=arrays['centre_world'][i].tolist(),availability_probability=probability,
                            candidate_available=probability>=.5 and bool(arrays['uncertainty_score_valid'][i]),
                            uncertainty_score_m=arrays['uncertainty_score_m'][i].tolist())
                        predictions.append(row)
                    cache=[];metadata=[]
    if len(predictions)!=1046:raise ValueError('All1046 V72 frames must be retained')
    for mapping in (sources,bound):
        if any(sha(p)!=digest for p,digest in mapping.items()):raise ValueError('Method/public data changed during prediction')
    write_new(output/'public_predictions.json',predictions)
    freeze=dict(complete=True,predictions=pointer(output/'public_predictions.json'),selected=pointer(selected),
        cases=[dict(id=c['id'],path=str(c['case']))for c in cases],closed6_gate=gate,source_sha256=sources,
        public_data_sha256=bound,frames=1046,model_forward_batch_calls=calls,Native_GT_reads_before_freeze=0,
        threshold=.5,device='cpu',threads=2,development_only=True,TEST_read=False,
        no_control_actions_executed=True,runtime_admitted=False)
    write_new(output/'prediction_freeze.json',freeze)
    return dict(prediction_freeze=pointer(output/'prediction_freeze.json'),frames=1046)


def statistics(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean_m=float(a.mean()),p95_m=float(np.quantile(a,.95)),max_m=float(a.max()),within3cm=int((a<=.03).sum()))if len(a)else dict(count=0)


def score(output):
    output=Path(output);freeze=read(output/'prediction_freeze.json')
    for mapping in (freeze['source_sha256'],freeze['public_data_sha256']):
        if any(sha(p)!=digest for p,digest in mapping.items()):raise ValueError('Frozen inputs/method changed')
    if sha(freeze['predictions']['path'])!=freeze['predictions']['sha256'] or freeze.get('Native_GT_reads_before_freeze')!=0:
        raise ValueError('Complete public freeze required before scoring')
    write_new(output/'score_consumption.json',dict(prediction_freeze=pointer(output/'prediction_freeze.json'),one_attempt=True))
    predictions=read(freeze['predictions']['path']);cases,_=closed_gate();truth={};bound_truth={}
    for case in cases:
        for name in ('initial_snapshot.json','native_ticks.jsonl'):
            path=case['case']/name;bound_truth[str(path)]=sha(path)
            samples=[read(path)['state']]if name.endswith('.json')else list(rows(path))
            for sample in samples:truth[(case['id'],round(float(sample['time']),8))]=sample
    scored=[]
    for prediction in predictions:
        actual=truth[(prediction['case_id'],round(prediction['timestamp'],8))]
        centre=np.asarray(actual['object_position'],float);anchor=np.asarray(prediction['public_sphere_center'])
        in_scope=bool(np.linalg.norm(centre-anchor)<=.20+1e-6)
        old=prediction['original_public_centre']
        scored.append(dict(prediction,actual_held=actual['held'],actual_in_measurement_scope=in_scope,
            centre_error_m=float(np.linalg.norm(np.asarray(prediction['centre_world'])-centre)),
            original_centre_error_m=None if old is None else float(np.linalg.norm(np.asarray(old)-centre))))
    def summarize(group):
        inside=[r for r in group if r['actual_in_measurement_scope'] and not r['expected_negative']]
        accepted=[r for r in inside if r['candidate_available']]
        return dict(frames=len(group),true_in_scope=len(inside),new_available_in_scope=len(accepted),
            error_all_in_scope=statistics([r['centre_error_m']for r in inside]),
            error_new_available_in_scope=statistics([r['centre_error_m']for r in accepted]),
            error_original_available_in_scope=statistics([r['original_centre_error_m']for r in inside if r['original_centre_error_m']is not None]),
            outside_scope_new_available=sum(r['candidate_available']and not r['actual_in_measurement_scope']for r in group),
            NoProxy_new_available=sum(r['expected_negative']and r['candidate_available']for r in group))
    report=dict(complete=True,aggregate=summarize(scored),
        cases={c:summarize([r for r in scored if r['case_id']==c])for c in EXPECTED_CASES},
        held=summarize([r for r in scored if r['actual_held']]),
        original_target_unavailable=summarize([r for r in scored if not r['original_public_target_available']]),
        rows=scored,truth_files=bound_truth,prediction_freeze=pointer(output/'prediction_freeze.json'),
        posthoc_development_only=True,unchanged_original_trajectory=True,
        actual_new_task_rollouts=0,task_success_gain_demonstrated=False,
        in_scope_is_not_visibility_ground_truth=True,uncertainty_is_not_CI=True,runtime_admitted=False,TEST_read=False)
    if any(sha(p)!=digest for p,digest in bound_truth.items()):raise ValueError('Native truth changed during scoring')
    write_new(output/'report.json',report)
    return dict(report=pointer(output/'report.json'),aggregate=report['aggregate'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selected');parser.add_argument('--expected-sha256');parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--predict',action='store_true');mode.add_argument('--score',action='store_true');mode.add_argument('--watch-pilot',action='store_true')
    args=parser.parse_args()
    if args.watch_pilot:
        if args.output is None:parser.error('Fresh output required')
        watcher_source=sha(__file__)
        watch_path=args.output.with_suffix('.watch.json')
        write_new(watch_path,dict(pid=os.getpid(),source=pointer(__file__),pilot_status=str(PILOT/'status.json'),
            output=str(args.output),model_forward_calls_before_pilot_complete=0))
        def guard():
            if (args.output.with_suffix('.STOP').exists() or (args.output/'STOP').exists()
                or (PILOT/'STOP').exists() or
                (ROOT/'runtime/framework/local_mpc_20261102/STOP').exists()):
                raise InterruptedError('Administrative STOP')
            if sha(__file__)!=watcher_source:raise ValueError('Watcher source changed while waiting')
        begin=time.monotonic()
        while True:
            guard()
            status=read(PILOT/'status.json')
            if status['stage']=='stopped':raise RuntimeError('Pilot stopped; no replay')
            if status.get('complete')is True:break
            if time.monotonic()-begin>7200:raise TimeoutError('Pilot wait limit')
            time.sleep(5)
        with (ROOT/'runtime/framework/fixed_TRAIN104_geometry_native.lock').open('a')as first, (ROOT/'runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock').open('a')as second:
            for handle in (first,second):
                while True:
                    guard()
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(2)
            guard()
            selected=FIT/'selected.pt';predict(selected,sha(selected),args.output);result=score(args.output)
    elif args.predict:
        if not all((args.selected,args.expected_sha256,args.output)):parser.error('Selected/SHA/fresh output required')
        result=predict(args.selected,args.expected_sha256,args.output)
    elif args.score:
        if args.output is None:parser.error('Frozen output required')
        result=score(args.output)
    else:
        result=dict(complete=True,expected_cases=list(EXPECTED_CASES),expected_frames=1046,
            Native_started=False,model_forward_calls=0,training_started=False,TEST_read=False)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
