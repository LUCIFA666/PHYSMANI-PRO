#!/usr/bin/env python3
"""Offline cached-nominal IK branch candidate, never a deployed controller.

Only a fixed command destination is cached. Every evaluation starts at its own
measured q/dq/fingers/h. Current collision checks on the cached branch are NOT
available offline: model costs are explicitly conditional, not ready actions.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import replay_probe_nominal_counterfactual_20261103 as cf
from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_p2_20261103.execution_models import validate_feedback_state
from our_mani.local_mpc_20261102.records import write_new,append,plain

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


class DiagnosticNominalIKCache:
 """Static destination only; no stored feedback, predicted state, or h."""
 def __init__(self,source_identity):self.source_identity=source_identity;self.key=None;self.entry=None;self.solve_count=0;self.reuse_count=0
 def resolve(self,initial,candidates,spec,helper):
  position=np.asarray(spec['target_position'],np.float64);quat=np.asarray(spec['target_quaternion'],np.float64)
  # Bitwise exact pose key, including quaternion sign: no approximate matching.
  key=(spec['identity'],position.tobytes(),quat.tobytes(),tuple(c['id'] for c in candidates),
       tuple(bool(c.get('native_pose_qualified',False)) for c in candidates),self.source_identity)
  if key!=self.key:
   solved=helper.adapter.solve(np.broadcast_to(initial['q'],(8,7)).copy(),np.broadcast_to(position,(8,3)).copy(),np.broadcast_to(quat,(8,4)).copy())
   self.entry=dict(joint_goal=solved['joint_goal'].copy(),feasible=solved['joint_goal_feasible'].copy(),
    ik_seed_actual_q_sha256=hashlib.sha256(np.asarray(initial['q'],np.float64).tobytes()).hexdigest(),solved_at_actual_time=initial['time'],
    pose=np.c_[np.broadcast_to(position,(8,3)),np.broadcast_to(quat,(8,4))].copy())
   self.key=key;self.solve_count+=1;reused=False
  else:self.reuse_count+=1;reused=True
  return deepcopy(self.entry),dict(reused=reused,solve_count=self.solve_count,reuse_count=self.reuse_count,
   solved_at_actual_time=self.entry['solved_at_actual_time'],cached_state_reused_for_prediction=False,
   ik_seed_actual_q_sha256=self.entry['ik_seed_actual_q_sha256'],seed_hash_is_provenance_only=True,virtual_h_cached=False,
   exact_pose_probe_identity_mask_and_source_required=True,static_destination_only=True,
   current_cached_branch_Native_collision_recheck_available=False,current_selected_path_collision_recheck_available=False)


class CachedNominalDiagnosticPlanner(cf.CEMRecorder):
 def __init__(self,*args,nominal_cache,**kwargs):super().__init__(*args,**kwargs);self.nominal_cache=nominal_cache
 def _prepare_nominals(self,initial,candidates,old):
  start=time.perf_counter();entry,receipt=self.nominal_cache.resolve(initial,candidates,self.probe_spec,self.goal_helper)
  self._nominals=dict(joint_goal=np.broadcast_to(entry['joint_goal'],(3,8,7)).copy(),
    feasible=np.broadcast_to(entry['feasible'],(3,8)).copy(),pose=np.broadcast_to(entry['pose'],(3,8,7)).copy(),
    time=np.full((3,8),self.probe_spec['trigger_time']))
  self.nominal_receipt=dict(DLS_calls=0 if receipt['reused'] else 1,poses=8,basis_version='diagnostic_exact_probe_pose_IK_branch_cache',
    wall_seconds=time.perf_counter()-start,cache=receipt,source='CPU_counterfactual_not_Runtime_factory',
    current_actual_state_and_h_still_used=True)


def make_limits(row):
 lr=row['executor']['limits'];return JointLimits(lr['native_joint_position_lower'],lr['native_joint_position_upper'],lr['native_joint_velocity_limits'],
  lr['controller_reference_acceleration_limits'],lr['controller_reference_jerk_limits'],lr['controller_reference_velocity_cap'])


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 sources={f:sha(ROOT/f) for f in cf.SOURCES};sources['scripts/replay_probe_cached_nominal_counterfactual_20261103.py']=sha(__file__);reports={}
 for case,path in cf.CASES.items():
  input_path=path/'public_grasp_v41_decisions.jsonl';rows=list(map(json.loads,input_path.read_text().splitlines()));first=next(r for r in rows if r['decision']['mode']=='ACTIVE_VISUAL_VERIFICATION_PROBE');identity=first['decision']['probe']['identity']
  core,helper,provenance=build_planner(ROOT,make_limits(first),controller='coupled',action_space='joint',joint_goal_config={'lower':(-.1,-.65,.755)},terminal_config={'enabled':True},reference_frequency=6.)
  source_identity=json.dumps(dict(FK=provenance['FK'],checkpoint_sha256=provenance['checkpoint']['sha256'],
    coupled_parameter_sha256=provenance['coupled_parameters']['sha256'],source_sha256=provenance['source_sha256'],limits=first['executor']['limits']),sort_keys=True)
  cache=DiagnosticNominalIKCache(source_identity);directory=args.output/case;directory.mkdir();ledger=[]
  for row in rows:
   d=row['decision']
   if d['mode']!='ACTIVE_VISUAL_VERIFICATION_PROBE' or not d.get('probe') or d['probe']['identity']!=identity:continue
   state=validate_feedback_state(d['state']);spec=cf.numpy_spec(d['probe']);candidates=cf.candidate_ledger(row)
   if json.dumps(plain(state['actuator_state']),sort_keys=True)!=json.dumps(row['actuator_state_before'],sort_keys=True):raise ValueError('fresh actual state/h mismatch')
   before=json.dumps(plain(state),sort_keys=True);zero=np.zeros((8,8,7))
   fresh=VerificationProbePlanner(core.config,constraint_cost=core.planner.constraint_cost,execution_model=core.execution_model,goal_helper=helper,terminal_config=core.terminal_config,probe_spec=spec)
   fresh._prepare_nominals(state,candidates,state['phase_state']);fresh_nominal=cf.summarize_rollout(fresh,state,candidates,zero)
   cached=CachedNominalDiagnosticPlanner(core.config,constraint_cost=core.planner.constraint_cost,execution_model=core.execution_model,goal_helper=helper,terminal_config=core.terminal_config,probe_spec=spec,nominal_cache=cache)
   cached._prepare_nominals(state,candidates,state['phase_state']);cache_receipt=deepcopy(cached.nominal_receipt)
   cached_nominal=cf.summarize_rollout(cached,state,candidates,zero)
   cold_plans={}
   for name,planner in [('fresh_DLS_cold',cf.CEMRecorder(core.config,constraint_cost=core.planner.constraint_cost,execution_model=core.execution_model,goal_helper=helper,terminal_config=core.terminal_config,probe_spec=spec)),
     ('cached_DLS_cold',cached)]:
    result=planner.plan(state,candidates,warm_start=None);a=result.get('action')
    cold_plans[name]=dict(cost=result.get('cost'),model_feasible=a is not None,budget=result['budget'],
      first_joint_goal=None if a is None else a['joint_goal'],first_target_error_m=None if a is None else float(cf.norm(a['tcp_position']-np.asarray(spec['target_position']))),
      selected_winning_proposal=min(planner.search_trace,key=lambda v:v['minimum_cost']),
      physical_action_qualified=False,Native_cached_branch_and_actual_path_collision_unobserved=True)
   if json.dumps(plain(state),sort_keys=True)!=before:raise ValueError('actual feedback mutated')
   record=dict(time=row['time'],probe_identity=identity,actual_state_time=state['time'],actual_h_time=state['actuator_state']['reference_clock'],
    cache=cache_receipt,variants=dict(fresh_DLS_nominal=fresh_nominal,cached_DLS_nominal=cached_nominal),cold_CEM=cold_plans,
    physical_cached_branch_qualification='unobserved; recorded fresh-branch masks used only as conditional model gate',
    counterfactual_actual_feedback_or_outcome_claimed=False)
   append(directory/'rows.jsonl',record);ledger.append(record)
   print(case,row['time'],'cached_nominalcost',cached_nominal['cost'],'fresh',fresh_nominal['cost'],'retarget cached/fresh',cached_nominal['reference_target_changed_per50ms'][0],fresh_nominal['reference_target_changed_per50ms'][0],flush=True)
  # Isolated key invalidation guard checks do not alter the experiment cache.
  trial=DiagnosticNominalIKCache(source_identity);initial=validate_feedback_state(first['decision']['state']);spec=cf.numpy_spec(first['decision']['probe']);candidates=cf.candidate_ledger(first)
  trial.resolve(initial,candidates,spec,helper);trial.resolve(initial,candidates,spec,helper)
  masked=deepcopy(candidates);masked[0]['native_pose_qualified']=not masked[0]['native_pose_qualified'];trial.resolve(initial,masked,spec,helper)
  moved=deepcopy(spec);moved['target_position']=moved['target_position']+np.array([1e-6,0,0]);trial.resolve(initial,masked,moved,helper)
  new=deepcopy(moved);new['identity']='different_actual_attempt';trial.resolve(initial,masked,new,helper)
  trial.source_identity='different_frozen_FK_or_model_source';trial.resolve(initial,masked,new,helper)
  if trial.solve_count!=5 or trial.reuse_count!=1:raise ValueError('cache key invalidation regression')
  report=dict(case=case,rows=len(ledger),input_sha256=sha(input_path),model_provenance=provenance,
    primary_cache_solve_count=cache.solve_count,primary_cache_reuse_calls=cache.reuse_count,
    cache_key_guards=dict(first_solve=True,identical_reuse=True,mask_changed_resolves=True,pose_changed_resolves=True,new_attempt_resolves=True,source_changed_resolves=True),
    collision_scope='current cached joint branch must be checked by Native point/path gates at every real decision; not available in this offline experiment; no reuse of recorded fresh-branch certificate is claimed',
    summary=dict(cached_model_feasible_nominal_rows=sum(r['variants']['cached_DLS_nominal']['feasible'] for r in ledger),
     fresh_expected_first50ms_retargets=sum(r['variants']['fresh_DLS_nominal']['reference_target_changed_per50ms'][0][0] for r in ledger),
     cached_expected_first50ms_retargets=sum(r['variants']['cached_DLS_nominal']['reference_target_changed_per50ms'][0][0] for r in ledger)),
    measured_counterfactual_trajectory_claimed=False)
  write_new(directory/'report.json',report);reports[case]=report
 if any(sha(ROOT/f)!=h for f,h in sources.items()):raise RuntimeError('source changed during offline cache candidate replay')
 write_new(args.output/'manifest.json',dict(version='fixed_probe_exact_pose_IK_cache_CPU_candidate_20261103_v1',complete=True,source_sha256=sources,source_unchanged=True,
   cases={name:str(args.output/name/'report.json') for name in reports},Native_started=False,fit_started=False,core_controller_modified=False,P4_or_teacher_source_modified=False,
   replay_scope='19 measured states+h from original first warm probes; cache is a static destination only; no future measured counterfactual state substituted'))
if __name__=='__main__':main()
