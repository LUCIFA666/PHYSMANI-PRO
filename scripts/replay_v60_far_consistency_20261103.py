"""Offline fixed V59 development observer replay, never Native or a controller.

Prediction mode reads only closure metadata, public camera calibration and
actual RGBD.  All four case outputs must be frozen before a separate posthoc
scorer may open labels.  No NN weights exist in this experiment: near unknown
therefore stays unknown.  This cannot demonstrate live V60 performance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.benchmark_perception import RedBoxTracker, BoxPrior
from our_mani.contracts import Pose
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103.goal_aware_perception import GoalAwareRedBoxTracker
from our_mani.local_mpc_20261103.refiner_far_consistency_v60 import component_far_consistency

BASE = ROOT/'runtime/framework/local_mpc_20261102/public_candidate_darkred_v59_development_v1'
CASES = ('00_val_static', '03_val_uniform', 'uniform_002', 'static_no_proxy')
SOURCES = (
    'scripts/replay_v60_far_consistency_20261103.py',
    'our_mani/local_mpc_20261103/refiner_far_consistency_v60.py',
    'our_mani/benchmark_perception.py', 'our_mani/contracts.py', 'our_mani/perception.py',
    'our_mani/goal_perception.py',
    'our_mani/local_mpc_control_20261103/goal_aware_perception.py',
    'artifacts/framework/local_policy/goal_marker_calibration.json',
)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def roster():
    jobs = read(BASE/'plan.json')['jobs']
    if tuple(Path(j['path']).name for j in jobs) != CASES:
        raise ValueError('Fixed full four-case V59 development roster required')
    rows = {}
    for name in CASES:
        case = BASE/name
        owner = read(BASE/(name+'.owner.actual_exit.json'))
        if (owner.get('actual_process_returncode') != 0 or Path(owner['output']).resolve() != case
                or read(case/'worker_exit.json').get('exit_code') != 0
                or read(case/'worker_exit.json').get('native_closed') is not True
                or read(case/'result.json').get('complete') is not True):
            raise ValueError('Only actually closed complete development cases accepted')
        frames = sorted((case/'public_shadow/rgbd').glob('*.npz'))
        if not frames or [p.stem for p in frames] != [f'{i:04d}' for i in range(len(frames))]:
            raise ValueError('Complete original public frame order required')
        rows[name] = dict(case=str(case), camera_calibration=case/'public_shadow/camera_calibration.json',
                          frames=frames, actual_exit=str(BASE/(name+'.owner.actual_exit.json')))
    return rows


def tracker():
    return GoalAwareRedBoxTracker(tracker=RedBoxTracker(
        prior=BoxPrior(minimum_red=100), complete_geometry=False))


def public_frame(path, calibrations):
    with np.load(path, allow_pickle=False) as z:
        expected = {'timestamp'} | {f'{k}_{n}' for k in ('images', 'depths') for n in calibrations}
        if set(z.files) != expected: raise ValueError('Only original public RGBD arrays accepted')
        o = {'timestamp': float(z['timestamp']),
             'observation/images': {n: z['images_'+n].copy() for n in calibrations},
             'observation/depths': {n: z['depths_'+n].copy() for n in calibrations},
             'observation/calibrations': calibrations}
    return o


def observe_one(t, observation):
    started = time.perf_counter(); target = metadata = None; error = None
    try:
        target, metadata = t.observe(observation)
        if t.sphere is None: target = None; error = 'initial_public_sphere_unavailable_no_unfiltered_fallback'
    except (ValueError, RuntimeError) as exc: error = type(exc).__name__+': '+str(exc)
    result = dict(raw_available=target is not None, raw_error=error,
                  current_NN_available=False, NN_weights_loaded=False,
                  source_geometry_completion=False)
    cloud = np.empty((0,3), dtype=float)
    if target is not None:
        if metadata.get('geometry_completion') is not False:
            raise ValueError('Actual chosen component, never completed proxy points required')
        cloud = np.asarray(target.points_world, dtype=float)
        proof = component_far_consistency(cloud, t.sphere.center)
        center = target.pose.position
        result.update(raw_centre=center.tolist(), raw_velocity=target.linear_velocity.tolist(),
            raw_confidence=float(target.confidence), raw_measured_points=len(cloud),
            raw_visible_views=metadata['visible_views'], far_consistency=proof,
            raw_centre_distance_to_sphere_m=float(np.linalg.norm(center-t.sphere.center)),
            fixed_halfdiagonal_test_passed=bool(np.linalg.norm(center-t.sphere.center)-np.sqrt(3)*.025>.08),
            raw_component_sha256=hashlib.sha256(cloud.tobytes()).hexdigest())
    result['wall_seconds'] = time.perf_counter()-started
    return result, cloud


def predict(output):
    rows = roster()  # All four closed before first public input is processed.
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = {str(ROOT/p): sha(ROOT/p) for p in SOURCES}
    for filename, digest in sources.items():
        source = Path(filename); dest = out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(source.read_bytes())
        if sha(dest) != digest: raise ValueError('Source changed while freezing')
    settings = dict(schema='V60_far_consistency_fixed_V59_public_replay_v1', cases=list(CASES),
        original_case_plan=dict(path=str(BASE/'plan.json'), sha256=sha(BASE/'plan.json')),
        source_sha256=sources, minimum_red=100, red_ratio=3.5, complete_geometry=False,
        half_side_m=.025, point_axis_development_allowance_m=.006, near_scope_m=.08,
        tolerance_or_threshold_search=False, production_binding_approved=False,
        NN_weights_loaded=False, NN_unavailable_near_is_unknown=True,
        legacy_history='original_tracker_all_current_observations_only',
        conditional_history='commit_only_far_consistent_current_measurement_unknown_clears_history',
        note='Fixed old robot trajectories; no actions executed, no live NN/closed-loop success claim')
    write(out/'settings.json',settings)
    input_sources={}; counts={}; frame_rosters={}
    for name, row in rows.items():
        data=read(row['camera_calibration'])
        if set(data)!= {'front','8','16','36'}: raise ValueError('Original four cameras required')
        calibrations={n:CameraCalibration(np.asarray(data[n]['intrinsics']),Pose.from_xyzw(data[n]['camera_to_world']))
                      for n in ('front','8','16','36')}
        input_sources[str(row['camera_calibration'])]=sha(row['camera_calibration'])
        raw, conditional = tracker(), tracker()
        clouds={'legacy':[], 'conditional':[]}; offsets={k:[0] for k in clouds}; frame_rosters[name]=[]
        previous=None
        with (out/(name+'.predictions.jsonl')).open('x') as stream:
            for i,path in enumerate(row['frames']):
                observation=public_frame(path,calibrations); now=observation['timestamp']
                if not np.isfinite(now) or (previous is not None and abs(now-previous-.05)>1e-6):
                    raise ValueError('Actual public50ms causal order required; no synthetic frames')
                previous=now; digest=sha(path);input_sources[str(path)]=digest
                legacy,lc=observe_one(raw,observation); proposed,pc=observe_one(conditional,observation)
                far=bool(proposed.get('far_consistency',{}).get('far_consistent',False)
                         and proposed.get('raw_confidence',0.)>=.5)
                # Current NN is unavailable throughout this geometry-only experiment.
                proposed.update(route='FAR_RAW_CONSISTENT' if far else 'UNKNOWN',
                    accepted_current_centre=proposed.get('raw_centre') if far else None,
                    original_confidence_gate_passed=bool(proposed.get('raw_confidence',0.)>=.5),
                    no_shell_fallback_for_unknown=True)
                if not far: conditional.reset()
                proposed['accepted_history_length_after']=len(conditional.history)
                result=dict(case=name,index=i,timestamp=now,input_path=str(path),input_sha256=digest,
                    legacy=legacy,conditional=proposed,
                    initial_public_sphere=None if conditional.sphere is None else conditional.sphere.center.tolist())
                stream.write(json.dumps(result,allow_nan=False)+'\n')
                for k,c in [('legacy',lc),('conditional',pc)]:
                    clouds[k].append(c);offsets[k].append(offsets[k][-1]+len(c))
                frame_rosters[name].append(dict(index=i,timestamp=now,input_sha256=digest))
        np.savez_compressed(out/(name+'.current_components.npz'),
            **{k+'_points':np.concatenate(v) for k,v in clouds.items()},
            **{k+'_offsets':np.asarray(v,np.int64) for k,v in offsets.items()})
        counts[name]=len(row['frames'])
        print(json.dumps(dict(case=name,public_frames=counts[name],labels_opened=False)),flush=True)
    for p,digest in {**sources,**input_sources}.items():
        if sha(p)!=digest:raise ValueError('Public/source file changed during replay')
    expected={'settings.json'}|{n+suffix for n in CASES for suffix in ('.predictions.jsonl','.current_components.npz')}
    outputs={n:sha(out/n) for n in sorted(expected)}
    write(out/'public_prediction_freeze.json',dict(schema='V60_full_public_prediction_freeze_v1',
        complete=True,cases=list(CASES),case_frames=counts,frame_rosters=frame_rosters,
        prediction_and_config_files_sha256=outputs,source_sha256=sources,public_input_sources_sha256=input_sources,
        GT_labels_opened_during_prediction=False,Native_started=False,fit_started=False,TEST_read=False))
    return dict(output=str(out),case_frames=counts,prediction_freeze_sha256=sha(out/'public_prediction_freeze.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predict',action='store_true');p.add_argument('--output')
    a=p.parse_args()
    if a.predict:
        if not a.output:p.error('--output required for fresh public prediction freeze')
        print(json.dumps(predict(a.output)))
    else:print(json.dumps(dict(default_check_only=True,cases={k:len(v['frames']) for k,v in roster().items()},Native_started=False)))
