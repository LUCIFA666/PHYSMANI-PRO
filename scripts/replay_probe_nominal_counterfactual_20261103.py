#!/usr/bin/env python3
"""Frozen-B probe action counterfactuals at measured state+h, CPU only.

No control implementation, Native, fitting, or counterfactual actual labels.
Every trajectory starts from this row's actual feedback and fresh h. Within a
trajectory actual q/dq/h then evolve by the frozen model's two50ms transitions.
"""
from pathlib import Path
import argparse,hashlib,json,sys
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
from our_mani.local_mpc_control_20261103.joint_goal_cem import PARAM_SPACE,NOMINAL_BASIS_VERSION
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_20261102 import mpc
from our_mani.local_mpc_p2_20261103.execution_models import validate_feedback_state
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from our_mani.local_mpc_20261102.records import write_new,append,plain

CASES={name:ROOT/'runtime/framework/local_mpc_20261102/public_v41_probe12_canaries_v1'/name for name in ('uniform_012','uniform_002')}
SOURCES=['our_mani/local_mpc_control_20261103/public_grasp_probe.py','our_mani/local_mpc_control_20261103/joint_goal_cem.py',
 'our_mani/local_mpc_control_20261103/response_factory.py','our_mani/local_mpc_control_20261103/planner_factory.py',
 'our_mani/local_mpc_control_20261103/joint_goals.py','our_mani/local_mpc_control_20261103/coupled_response.py',
 'our_mani/local_mpc_p2_20261103/execution_models.py','our_mani/local_mpc_p2_20261103/reference.py',
 'our_mani/local_mpc_p2_20261103/action_adapter.py','our_mani/local_mpc_p2_20261103/kinematics.py',
 'our_mani/local_mpc_p2_20261103_v2/servo_adapter.py','our_mani/local_mpc_p2_20261103_v2/servo.py',
 'our_mani/local_mpc_p2_20261103_v2/torch_fk.py','our_mani/local_mpc_20261102/constraints.py','our_mani/local_mpc_20261102/mpc.py',
 'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt',
 'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json',
 'scripts/replay_probe_nominal_counterfactual_20261103.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):return np.linalg.norm(x,axis=-1)
def angles(q,anchor):return (Rotation.from_quat(np.asarray(q).reshape(-1,4))*Rotation.from_quat(anchor).inv()).magnitude()
def stats(values):
 a=np.asarray(values,float)
 return dict(count=len(a),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(count=0)
def numpy_spec(value):
 result=deepcopy(value)
 for key in ('anchor_position','anchor_quaternion','target_position','target_quaternion','object_anchor_position','relative_r0'):
  if key in result:result[key]=np.asarray(result[key],float)
 return result


class ModelRecorder:
 def __init__(self,model):self.model=model;self.calls=[]
 def __getattr__(self,name):return getattr(self.model,name)
 def initialize_batch(self,*args,**kwargs):self.calls=[];return self.model.initialize_batch(*args,**kwargs)
 def step_batch(self,x,u,h,dt):
  out=self.model.step_batch(x,u,h,dt);self.calls.append(dict(x=deepcopy(x),u=deepcopy(u),nxt=deepcopy(out[0]),info=deepcopy(out[2])));return out


class CEMRecorder(VerificationProbePlanner):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.search_trace=[]
 def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
  out=super()._rollout_batch(initial,candidates,modes,controls,old,committed,fixed_sequence)
  scores,_,_,actions=out;win=int(np.argmin(scores));per=len(modes)//8
  reserved=(fixed_sequence is not None and candidates[int(modes[win])]['id']==committed and win%per==0)
  self.search_trace.append(dict(iteration=len(self.search_trace),sequences=len(modes),minimum_cost=float(scores[win]),
   winning_mode=int(modes[win]),winning_local_sample=win%per,winning_candidate_id=candidates[int(modes[win])]['id'],
   winning_proposal_kind='reserved_shifted_absolute_warm' if reserved else 'mode_mean' if win%per==0 else 'random_knot_sample',
   winning_first_joint_goal=actions['joint_goal'][win,0].copy(),finite_sequences=int(np.isfinite(scores).sum()),
   finite_sequences_by_mode=[int(np.isfinite(scores[i*per:(i+1)*per]).sum()) for i in range(8)]))
  return out


def candidate_ledger(row):
 # Probe rollouts use only the recorded ID/deadline/qualification. The other
 # required candidate geometry fields are inert schema placeholders, not new
 # observations; fixed probe_spec is the ONLY nominal/control geometry.
 return [mpc.validate_candidate(dict(id=c['id'],deadline=c['deadline'],native_pose_qualified=c['qualified'],
  object_tcp_position=[0,0,0],object_tcp_quaternion=[0,0,0,1],approach_direction=[0,0,1],pregrasp_offset=.07,
  opening=0.,required_opening=1.,allowed_relative_speed=.14)) for c in row['decision']['qualification']['cases']]


def summarize_rollout(planner,initial,candidates,controls,*,saved_sequence=None):
 recorder=ModelRecorder(planner.execution_model);old_model=planner.execution_model;planner.execution_model=recorder
 try:score,parts,states,actions=planner._rollout_batch(initial,candidates,np.arange(8),controls,initial['phase_state'],None,None)
 finally:planner.execution_model=old_model
 # K8 physical probe targets coincide. Qualification remains the original fixed
 # per-candidate ledger, with no new Native collision query.
 index=0;spec=planner.probe_spec;anchor=np.asarray(spec['anchor_position']);goal=np.asarray(spec['target_position']);quat=np.asarray(spec['anchor_quaternion'])
 p=states['tcp_position'][index];q=states['tcp_quaternion'][index];cmdp=actions['tcp_position'][index];cmdq=actions['tcp_quaternion'][index]
 micro=[];micro_angles=[];fail=[];ref_fe_dq=[]
 for k,call in enumerate(recorder.calls):
  joint=np.asarray(call['info']['predicted_actual_q'][index]);fk=planner.goal_helper.kinematics(joint);m_p=fk[:,:3,3];m_q=matrix_quaternion(fk[:,:3,:3]);micro.extend(m_p.tolist())
  micro_angles.append(float(angles(m_q,quat).max()))
  FE=(np.diff(np.vstack([call['x']['q'][index],joint]),axis=0)/.05)
  # Macro endpoint dq is the finite difference of the same last predicted q pair.
  ref_fe_dq.append(float(np.max(np.abs(FE[-1]-call['nxt']['dq'][index]))))
  flags=dict(model_feasible=bool(call['info']['feasible'][index]),joint_goal_feasible=bool(call['u']['joint_goal_feasible'][index]),
   original_nominal_native_qualification=bool(candidates[index]['native_pose_qualified']),
   command_probe15mm=bool(norm(call['u']['tcp_position'][index]-anchor)<=.015+1e-10),
   command_probe04rad=bool(angles(call['u']['tcp_quaternion'][index][None],quat)[0]<=.04+1e-10),
   both50ms_probe15mm=bool(np.all(norm(m_p-anchor)<=.015+1e-10)),
   both50ms_probe04rad=bool(np.all(angles(m_q,quat)<=.04+1e-10)))
  constraint=planner.constraint_cost
  flags['command_and_macro_workspace']=bool(constraint.batch(call['x'],call['u'],call['nxt'],np.array([c['native_pose_qualified'] for c in candidates]),call['nxt']['time'])[index]==0.)
  fail.append(dict(macro=k+1,endpoint_time=float(call['nxt']['time'][index]),gates=flags,
   failed_gates=[name for name,ok in flags.items() if not ok],
   maximum_micro_anchor_distance_m=float(norm(m_p-anchor).max()),maximum_micro_angle_rad=float(angles(m_q,quat).max())))
 micro=np.asarray(micro)
 result=dict(cost=float(score[index]),feasible=bool(np.isfinite(score[index])),cost_terms={key:float(value[index]) for key,value in parts.items()},
  constraint_cost=float(parts['constraint'][index]),all_constraints_by_macro=fail,
  nominal_recorded_collision_qualification_only=True,new_Native_collision_queries=0,future_continuous_collision_certified=False,
  H1_position=p[1].tolist(),H1_vertical_from_probe_anchor_m=float(p[1,2]-anchor[2]),H1_target_error_m=float(norm(p[1]-goal)),
  H8_position=p[-1].tolist(),H8_vertical_from_probe_anchor_m=float(p[-1,2]-anchor[2]),H8_target_error_m=float(norm(p[-1]-goal)),
  H8_vertical_overshoot_of12mm_m=float(p[-1,2]-goal[2]),
  max_macro_anchor_distance_m=float(norm(p-anchor).max()),max_micro_anchor_distance_m=float(norm(micro-anchor).max()),
  micro15mm_margin_m=float(.015-norm(micro-anchor).max()),macro15mm_margin_m=float(.015-norm(p-anchor).max()),
  max_micro_angle_rad=max(micro_angles),
  command_target_error_m=norm(cmdp-goal).tolist(),command_target_error_xyz_m=(cmdp-goal).tolist(),
  command_angle_rad=angles(cmdq,quat).tolist(),command_joint_goals=actions['joint_goal'][index].tolist(),
  predicted_micro_positions=micro.tolist(),predicted_macro_positions=p.tolist(),predicted_actual_q=states['q'][index].tolist(),predicted_actual_dq=states['dq'][index].tolist(),
  reference_target_changed_per50ms=[np.asarray(call['info']['target_changed'][index]).tolist() for call in recorder.calls],
  first_joint_goal_minus_actual_previous_target_rad=(actions['joint_goal'][index,0]-np.asarray(initial['actuator_state']['target_joints'])).tolist(),
  finite_difference_dq_max_residual_rad_s=max(ref_fe_dq),all_predicted_held_false=bool(not states['held'].any()))
 if saved_sequence is not None:
  result['selected_saved_joint_goal_reconstruction_max_abs_rad']=float(np.max(np.abs(actions['joint_goal'][index]-saved_sequence)))
 return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 source_start={f:sha(ROOT/f) for f in SOURCES};case_reports={}
 for name,path in CASES.items():
  source=path/'public_grasp_v41_decisions.jsonl';rows=list(map(json.loads,source.read_text().splitlines()));first=next(r for r in rows if r['decision']['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE')
  spec=first['decision']['probe'];identity=spec['identity'];lr=first['executor']['limits']
  limits=JointLimits(lr['native_joint_position_lower'],lr['native_joint_position_upper'],lr['native_joint_velocity_limits'],lr['controller_reference_acceleration_limits'],lr['controller_reference_jerk_limits'],lr['controller_reference_velocity_cap'])
  core,helper,provenance=build_planner(ROOT,limits,controller='coupled',action_space='joint',joint_goal_config={'lower':(-.1,-.65,.755)},terminal_config={'enabled':True},reference_frequency=6.)
  output=args.output/name;output.mkdir();entries=[];previous=None
  for row in rows:
   d=row['decision'];plan=d.get('plan')
   if d['mode']!='ACTIVE_VISUAL_VERIFICATION_PROBE' or not d.get('probe') or d['probe']['identity']!=identity:continue
   if not plan or not plan.get('sequence'):entries.append(dict(time=row['time'],eligible=False,reason='no_selected_sequence'));previous=None;continue
   initial=validate_feedback_state(d['state']);candidates=candidate_ledger(row)
   if abs(initial['time']-row['actuator_state_before']['reference_clock'])>1e-8:raise ValueError('same actual timestamp state/h required')
   if json.dumps(plain(initial['actuator_state']),sort_keys=True)!=json.dumps(row['actuator_state_before'],sort_keys=True):raise ValueError('stored state and actual h differ')
   before=deepcopy(initial)
   planner=VerificationProbePlanner(core.config,constraint_cost=core.planner.constraint_cost,execution_model=core.execution_model,goal_helper=helper,terminal_config=core.terminal_config,probe_spec=numpy_spec(d['probe']))
   planner._prepare_nominals(initial,candidates,initial['phase_state']);nominal=planner._nominals['joint_goal'][1]
   if np.max(np.abs(nominal[0]-plan['action']['nominal_joint_goal']))>1e-10:raise ValueError('fresh DLS nominal does not match recorded frozen helper')
   selected_q=np.asarray([a['joint_goal'] for a in plan['sequence']]);nominal_controls=np.zeros((8,8,7))
   selected_controls=selected_q[:,None,:]-nominal[None,:,:]
   first_controls=np.broadcast_to(selected_q[0][None,None,:]-nominal[None,:,:],(8,8,7)).copy()
   variants=dict(nominal_zero_controls=summarize_rollout(planner,initial,candidates,nominal_controls),
    selected_saved_sequence=summarize_rollout(planner,initial,candidates,selected_controls,saved_sequence=selected_q),
    selected_first_action_held_H8=summarize_rollout(planner,initial,candidates,first_controls))
   selected=variants['selected_saved_sequence'];saved=np.asarray([s['tcp_position'] for s in plan['predicted_states']])
   selected['saved_plan_TCP_reconstruction_max_m']=float(norm(np.asarray(selected['predicted_macro_positions'])-saved).max())
   selected['saved_plan_cost_reconstruction_abs']=abs(selected['cost']-plan['cost'])
   if selected['saved_plan_TCP_reconstruction_max_m']>1e-10 or selected['saved_plan_cost_reconstruction_abs']>1e-8:raise ValueError('counterfactual model/cost failed exact saved selected reconstruction')
   warm=None
   if previous is not None and abs(row['time']-previous['time']-.1)<1e-7 and previous.get('action_rejected_reason') is None:
    warm=deepcopy(previous['decision']['plan']['warm_start'])
   warm_eval=dict(input_available=warm is not None,inference_scope='previous accepted same-probe consecutive row warm_out; no future/actual predicted state mixing')
   if warm is not None:
    shift=min(8,max(0,int(round((initial['time']-warm['time'])/.1))))
    shifted=warm['sequence'][shift:]+[deepcopy(warm['sequence'][-1]) for _ in range(shift)]
    warm_q=np.asarray([a['joint_goal'] for a in shifted]);controls=warm_q[:,None,:]-nominal[None,:,:]
    variants['previous_warm_shifted_sequence']=summarize_rollout(planner,initial,candidates,controls)
    warm_eval.update(shift=shift,selected_first_q_minus_shifted_warm_rad=(selected_q[0]-warm_q[0]).tolist(),
      selected_first_max_abs_difference_rad=float(np.max(np.abs(selected_q[0]-warm_q[0]))),
      committed_id=warm['selected_candidate_id'],selected_id=plan['selected_candidate_id'])
   # Re-run full fixed-budget optimizers with the same current state and ledger,
   # once with inferred original warm and once cold. These are CPU proposals,
   # never counterfactual physical outcomes or candidate selection by GT.
   replans={}
   for label,input_warm in [('same_warm',warm),('cold',None)]:
    rp=CEMRecorder(core.config,constraint_cost=core.planner.constraint_cost,execution_model=core.execution_model,goal_helper=helper,terminal_config=core.terminal_config,probe_spec=numpy_spec(d['probe']))
    value=rp.plan(initial,candidates,warm_start=input_warm)
    a=value.get('action');replans[label]=dict(cost=value.get('cost'),feasible=a is not None,budget=value['budget'],selected_id=value.get('selected_candidate_id'),
      first_joint_goal=None if a is None else a['joint_goal'].tolist(),first_target_error_m=None if a is None else float(norm(a['tcp_position']-np.asarray(spec['target_position']))),
      first_vs_recorded_max_abs_rad=None if a is None else float(np.max(np.abs(a['joint_goal']-selected_q[0]))),
      iteration_minima=value['diagnostics'].get('iterations'),warm_evaluations=value['diagnostics'].get('warm_start_evaluations_counted'),
      search_trace=rp.search_trace,winning_search_proposal=min(rp.search_trace,key=lambda item:item['minimum_cost']))
   if replans['same_warm']['first_vs_recorded_max_abs_rad'] is not None and replans['same_warm']['first_vs_recorded_max_abs_rad']>1e-10:raise ValueError('original-warm CPU replay fails exact original first action')
   if json.dumps(plain(initial),sort_keys=True)!=json.dumps(plain(before),sort_keys=True):raise ValueError('actual feedback mutated by counterfactuals')
   entry=dict(time=row['time'],eligible=True,probe_identity=identity,variants=variants,warm=warm_eval,replans=replans,
    original_iterations=plan['diagnostics']['iterations'],original_warm_evaluations=plan['diagnostics']['warm_start_evaluations_counted'],
    actual_matched_selected_command=bool(np.array_equal(selected_q[0],row['executed_action']['joint_goal'])),actual_action_rejected=row['action_rejected_reason'])
   append(output/'rows.jsonl',entry);entries.append(entry);previous=row
   print(name,row['time'],'nominal',variants['nominal_zero_controls']['cost'],variants['nominal_zero_controls']['micro15mm_margin_m'],'selected',selected['cost'],'cold',replans['cold']['cost'],flush=True)
  report=dict(case=name,first_probe=spec,input_decision_sha256=sha(source),rows=len(entries),rows_path=str(output/'rows.jsonl'),
   scope='fixed actual state+h; model-only action counterfactual and warm/cold optimizer replay; no Native trajectories for alternatives',
   inert_candidate_geometry_placeholder_declared=True,recorded_ID_deadline_and_qualification_used_only=True,
   new_Native_collision_queries=0,formal_qualification=False,model_provenance=provenance,
   summary=dict(nominal_infeasible_rows=sum(not e['variants']['nominal_zero_controls']['feasible'] for e in entries if e['eligible']),
    saved_selected_feasible_rows=sum(e['variants']['selected_saved_sequence']['feasible'] for e in entries if e['eligible']),
    cold_vs_warm_different_first_action_rows=sum(e['replans']['cold']['first_vs_recorded_max_abs_rad'] is not None and e['replans']['cold']['first_vs_recorded_max_abs_rad']>1e-9 for e in entries if e['eligible'])))
  write_new(output/'report.json',report);case_reports[name]=report
 source_end={f:sha(ROOT/f) for f in SOURCES}
 if source_start!=source_end:raise RuntimeError('frozen source changed during counterfactual diagnostic')
 write_new(args.output/'manifest.json',dict(version='probe_nominal_selected_counterfactual_20261103_v1',complete=True,source_sha256=source_start,
  source_unchanged=True,cases={name:dict(report_path=str(args.output/name/'report.json'),sha256=sha(args.output/name/'report.json')) for name in case_reports},
  Native_started=False,fit_started=False,TEST_VAL_read=False,counterfactual_actual_feedback_not_invented=True))
if __name__=='__main__':main()
