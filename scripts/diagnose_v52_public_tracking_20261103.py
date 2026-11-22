"""Fixed-trajectory public RGBD tracker replay; truth loaded only after outputs.

The diagnostic cube fit is new, unqualified geometry code confined here. It
fits current observed red points to the declared5cm cube; previous observations
only seed/associate fits. It never emits a forecast, TCP-relative controlled
point, Native pose or missing-frame substitute as a visual measurement.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
from dataclasses import asdict
from collections import Counter,deque
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose,ObjectState
from our_mani.perception import CameraCalibration
from our_mani.benchmark_perception import RedBoxTracker,BoxPrior
from our_mani.local_mpc_control_20261103.goal_aware_perception import (
    GoalAwareRedBoxTracker,filter_observed_sphere_surface)
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
from our_mani.local_mpc_control_20261103.public_grasp_v42 import GoalAwareActiveVisualPublicBridge
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from our_mani.local_mpc_control_20261103.goal_aware_binding_v42 import public_RGBD_digest

CASES=('00_val_static','02_val_uniform','06_val_speed_change')
BASE=ROOT/'runtime/framework/local_mpc_20261102/P4_student_closedloop_devVAL_v52_v1'
PROTOTYPE_CONFIG=dict(dimension_m=.05,red_cluster_distance_m=.028,maximum_points=5000,
    fit_points=320,soft_L1_scale_m=.002,median_surface_residual_max_m=.003,
    p90_surface_residual_max_m=.006,minimum_two_axis_span_m=.020,
    minimum_confidence=.5,minimum_points=12,minimum_views=2,max_nfev=35,
    initial_table_height_gate_m=.08,association_distance_base_m=.045,association_speed_allowance_mps=1.,
    parameters_predeclared_before_GT_metrics=True,qualified=False)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=None if not len(a) else float(a.mean()),p95=None if not len(a) else float(np.quantile(a,.95)),maximum=None if not len(a) else float(a.max()))
def serialize(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)
def observation(public,row,cameras):
    p=public/row['rgbd']
    with np.load(p,allow_pickle=False) as frame:
        obs={'timestamp':float(frame['timestamp']),
             'observation/images':{n:np.array(frame['images_'+n],copy=True) for n in cameras},
             'observation/depths':{n:np.array(frame['depths_'+n],copy=True) for n in cameras},
             'observation/calibrations':cameras}
    if abs(obs['timestamp']-row['public_robot']['timestamp'])>1e-8:raise ValueError('real sameframe RGBD required')
    return obs,p
def red_components(obs,prior):
    points=[];views=[];red=[]
    for name,rgb in obs['observation/images'].items():
        image=np.asarray(rgb,float);mask=(image[...,0]>=prior.minimum_red)&(image[...,0]>prior.red_ratio*image[...,1])&(image[...,0]>prior.red_ratio*image[...,2])
        depth=obs['observation/depths'][name];valid=mask&np.isfinite(depth)&(depth>0)
        cloud=obs['observation/calibrations'][name].backproject(depth,mask)
        inside=((cloud>=prior.workspace_min)&(cloud<=prior.workspace_max)).all(1)
        points.append(cloud[inside]);views.extend([name]*int(inside.sum()));red.extend(image[valid,0][inside].tolist())
    cloud=np.concatenate(points);views=np.asarray(views);red=np.asarray(red)
    if len(cloud)>5000:
        take=np.linspace(0,len(cloud)-1,5000,dtype=int);cloud,views,red=cloud[take],views[take],red[take]
    if not len(cloud):return []
    tree=cKDTree(cloud);graph=tree.sparse_distance_matrix(tree,.028,output_type='coo_matrix');count,labels=connected_components(graph,directed=False)
    result=[]
    for label in range(count):
        chosen=labels==label
        if chosen.sum()<12:continue
        p=cloud[chosen];center=(np.quantile(p,.01,axis=0)+np.quantile(p,.99,axis=0))/2
        center[2]=np.quantile(p[:,2],.98)-.025
        result.append(dict(points=p,views=len(set(views[chosen])),red=float(red[chosen].mean()),
                           box_AABB_center=center,extent=np.quantile(p,.99,axis=0)-np.quantile(p,.01,axis=0)))
    return result
def cube_surface_distance(points,parameters):
    local=(points-parameters[:3])@Rotation.from_rotvec(parameters[3:]).as_matrix()
    q=np.abs(local)-.025
    return np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.)

class CurrentObservedCubeFit:
    def __init__(self):self.marker=GoalAwareRedBoxTracker();self.history=deque(maxlen=5);self.rotation=np.zeros(3)
    def observe(self,obs):
        if not self.marker._initial_fit_attempted:self.marker.initialize_from_public(obs)
        if self.marker.sphere is None:return None,dict(reason='initial_public_sphere_fit_unknown')
        filtered,trace=filter_observed_sphere_surface(obs,self.marker.sphere,self.marker.prior)
        candidates=[];diagnostics=[];now=obs['timestamp']
        for component in red_components(filtered,self.marker.prior):
            points=component['points'];confidence=min(1.,component['views']/3)*min(1.,len(points)/60)
            if component['views']<2 or confidence<.5:continue
            sample=points[np.linspace(0,len(points)-1,min(len(points),320),dtype=int)]
            starts=[np.r_[component['box_AABB_center'],self.rotation]]
            if self.history:starts.append(np.r_[self.history[-1][1],self.rotation])
            fits=[]
            for start in starts:
                solved=least_squares(lambda p:cube_surface_distance(sample,p),start,
                    loss='soft_l1',f_scale=.002,max_nfev=35)
                residual=np.abs(cube_surface_distance(points,solved.x))
                fits.append((float(np.median(residual)),float(np.quantile(residual,.9)),solved.x,residual))
            median,p90,pose,residual=min(fits,key=lambda v:(v[0],v[1]));center=pose[:3]
            span=np.sort(np.ptp((points-center)@Rotation.from_rotvec(pose[3:]).as_matrix(),axis=0))
            allowed=median<=.003 and p90<=.006 and span[1]>=.020
            if not self.history:allowed=allowed and abs(center[2]-.775)<=.08
            else:allowed=allowed and np.linalg.norm(center-self.history[-1][1])<=.045+(now-self.history[-1][0])
            item=dict(observed_points=len(points),views=component['views'],confidence=confidence,
                AABB_center=component['box_AABB_center'].tolist(),surface_fit_center=center.tolist(),
                median_surface_residual_m=median,p90_surface_residual_m=p90,allowed=bool(allowed))
            diagnostics.append(item)
            if allowed:
                continuity=0. if not self.history else np.linalg.norm(center-self.history[-1][1])
                candidates.append((median+p90+.1*continuity,center,pose[3:],points,confidence,item))
        if not candidates:return None,dict(reason='current_cube_geometry_insufficient_or_ambiguous',components=diagnostics,sphere_filter=trace)
        _,center,rotation,points,confidence,item=min(candidates,key=lambda v:v[0])
        self.history.append((now,center.copy()));self.rotation=rotation.copy()
        return ObjectState('target',now,Pose(center),points,np.zeros(3),np.zeros(3),confidence),dict(
            reason='current_observed_cube_surface_fit',selected=item,components=diagnostics,sphere_filter=trace,
            output_center_from_current_points=True,history_seed_is_not_measurement=True,orientation_output_unobserved=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),ROOT/'our_mani/benchmark_perception.py',ROOT/'our_mani/local_mpc_control_20261103/goal_aware_perception.py',ROOT/'our_mani/local_mpc_control_20261103/active_visual_attachment.py',ROOT/'our_mani/local_mpc_control_20261103/public_grasp_v42.py']
    source_before={str(p):sha(p) for p in source_paths};summaries=[]
    for name in CASES:
        case=BASE/name;actual=BASE/(name+'.owner.actual_exit.json')
        if json.loads(actual.read_text())['actual_process_returncode']!=0:raise ValueError('completed case actualexit0 required before diagnosis')
        public=case/'public_shadow';rows=lines(public/'public_outputs.jsonl');decisions=lines(case/'public_grasp_candidates_v5_decisions.jsonl');dby={round(r['time'],7):r for r in decisions};triggers={}
        for row in decisions:
            sp=(row['decision'].get('active_visual_attachment') or {}).get('probe')
            if sp:triggers[round(sp['trigger_time'],7)]=dby[round(sp['trigger_time'],7)]['actuator_state_before']
        cameras={n:CameraCalibration(v['intrinsics'],Pose.from_xyzw(v['camera_to_world'])) for n,v in json.loads((public/'camera_calibration.json').read_text()).items()}
        active=ActiveVisualConfig(enforce_probe_anchor_orientation_after_confirmation=False)
        wrappers={'original':ActiveVisualPublicBridge(active_config=active),'v42_strict_shell':GoalAwareActiveVisualPublicBridge(active_config=active)}
        prototype=CurrentObservedCubeFit();out=args.output/name;out.mkdir();ledger=[];source_files=[]
        for index,row in enumerate(rows):
            obs,path=observation(public,row,cameras);robot=PublicRobotSensors.from_mapping(row['public_robot']);rawhash=public_RGBD_digest(obs)
            frame=dict(index=index,time=obs['timestamp'],variants={},raw_RGBD_sha256=rawhash)
            for key,wrapper in wrappers.items():
                tick=time.perf_counter();result=wrapper.observe(obs,robot,close_requested=row['close_requested']);target=wrapper.binding.latest_visual_target
                if round(robot.timestamp,7) in triggers and target is not None:
                    wrapper.active.begin_probe(target,robot,triggers[round(robot.timestamp,7)])
                frame['variants'][key]=dict(position=None if target is None else target.pose.position.tolist(),
                    valid=bool(target is not None and target.visible and target.confidence>=.5),confidence=None if target is None else target.confidence,
                    mode=result['mode'],active_reason=wrapper.latest_attachment['reason'],active_failed=wrapper.active.failed,
                    filter_trace=result.get('goal_aware_filter_trace'),tracker_error=result.get('goal_aware_tracker_error'),wall_seconds=time.perf_counter()-tick)
            tick=time.perf_counter();target,meta=prototype.observe(obs)
            frame['variants']['observed_cube_fit']=dict(position=None if target is None else target.pose.position.tolist(),valid=target is not None,metadata=meta,wall_seconds=time.perf_counter()-tick)
            if public_RGBD_digest(obs)!=rawhash:raise RuntimeError('raw public frame modified')
            old=row['result'].get('object_observation');frame['original_recorded_position']=None if old is None else old['pose']['position']
            ledger.append(frame);source_files.append(dict(path=str(path),sha256=sha(path)))
            if index%40==0:print(name,index,flush=True)
        # Freeze public-only proposals before any ground-truth file is opened.
        predictions=out/'public_only_predictions.jsonl'
        with predictions.open('x') as f:
            for frame in ledger:f.write(json.dumps(frame,ensure_ascii=False,allow_nan=False,default=serialize)+'\n')
        prediction_sha=sha(predictions)
        truth=lines(public/'evaluation_labels.jsonl');truthby={round(r['time'],7):r for r in truth};metrics={}
        recorded_delta=[]
        for key in ('original','v42_strict_shell','observed_cube_fit'):
            good=[r for r in ledger if r['variants'][key]['valid']];errors=[];after=[];bad=[]
            for r in good:
                error=float(np.linalg.norm(np.asarray(r['variants'][key]['position'])-truthby[round(r['time'],7)]['object_position']));errors.append(error)
                if r['time']>=2.6:after.append(error)
                if error>.03:bad.append(dict(time=r['time'],position=r['variants'][key]['position'],error_m=error))
            metrics[key]=dict(valid_frames=len(good),unknown_frames=len(ledger)-len(good),position_error_m=stats(errors),post2p6_position_error_m=stats(after),wrong_association_over3cm=bad,
                frame_wall_seconds=stats([r['variants'][key]['wall_seconds'] for r in ledger]))
        for r in ledger:
            if r['original_recorded_position'] is not None and r['variants']['original']['position'] is not None:
                recorded_delta.append(float(np.linalg.norm(np.asarray(r['original_recorded_position'])-r['variants']['original']['position'])))
        report=dict(case=name,real_frames=len(ledger),scope='fixed old robot trajectory; public predictions frozen before GT metrics; no new closedloop outcome',
            public_predictions_sha256=prediction_sha,GT_only_posthoc=True,prototype_config=PROTOTYPE_CONFIG,metrics=metrics,
            original_replay_to_recorded_center_delta_m=stats(recorded_delta),initial_visual_sphere=asdict(wrappers['v42_strict_shell'].tracker.sphere) if wrappers['v42_strict_shell'].tracker.sphere is not None else None,
            actual_exit_sha256=sha(actual),public_source_sha256=sha(public/'public_outputs.jsonl'),RGBD_files=source_files)
        write(out/'report.json',json.loads(json.dumps(report,default=serialize)));summaries.append(dict(case=name,metrics=metrics,report=str(out/'report.json')))
        print(name,'DONE',[(k,v['valid_frames'],v['position_error_m']['maximum']) for k,v in metrics.items()],flush=True)
    source_after={str(p):sha(p) for p in source_paths}
    if source_before!=source_after:raise RuntimeError('production/replay source changed')
    write(args.output/'manifest.json',dict(complete=True,cases=summaries,source_sha256=source_before,source_unchanged=True,
        prototype_only=True,production_modified=False,Native_started=False,network_fitting_started=False,formal_TEST_read=False,
        scope='three completed development VAL case perception diagnosis only; geometry prototype unqualified, not heldout performance',
        NN_weights_or_6mm_attachment_threshold_modified=False))
if __name__=='__main__':main()
