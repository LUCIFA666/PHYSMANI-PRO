"""CPU conditional-model replay of v53 destinations on 19 recorded states+h.

No Native collision service is available offline. Recorded masks belong to the
old fresh-DLS branch, so they are conditional model gates only. This script
does not construct a production CachedProbeQualification, certify an action,
replace measured future states, or claim a counterfactual Native outcome.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import replay_probe_nominal_counterfactual_20261103 as cf
from our_mani.local_mpc_control_20261103.probe_nominal_cache_v53 import ProbeNominalQCache,_fresh_feedback
from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
from our_mani.local_mpc_control_20261103.response_factory import build_planner
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_20261102.records import write_new,append,plain


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):value.update(block)
    return value.hexdigest()


class ConditionalModelCachePlanner(VerificationProbePlanner):
    """Offline diagnostic only: no production collision qualification token."""
    def __init__(self,*args,nominal_cache,source_sha256,nominal_mask,**kwargs):
        super().__init__(*args,**kwargs)
        self.nominal_cache=nominal_cache;self.source_sha256=source_sha256;self.nominal_mask=nominal_mask
    def _prepare_nominals(self,initial,candidates,old):
        actual=_fresh_feedback(initial)
        entry,diagnostic=self.nominal_cache.resolve(actual["q"],candidates,self.probe_spec,self.goal_helper,
            source_sha256=self.source_sha256,nominal_mask=self.nominal_mask)
        self._nominals=dict(joint_goal=np.broadcast_to(entry["joint_goal"],(3,8,7)).copy(),
            feasible=np.broadcast_to(entry["algebraic_feasible"]&entry["identity"]["nominal_mask"],(3,8)).copy(),
            pose=np.broadcast_to(entry["pose"],(3,8,7)).copy(),time=np.full((3,8),self.probe_spec["trigger_time"]))
        self.nominal_receipt=dict(basis_version="v53_cache_conditional_offline_replay_only",cache=diagnostic,
            DLS_calls=diagnostic["DLS_calls"],current_cached_branch_collision_observed=False,
            physical_action_qualified=False,actual_state_and_h_cached=False)


def limits(row):
    value=row["executor"]["limits"]
    return JointLimits(value["native_joint_position_lower"],value["native_joint_position_upper"],
        value["native_joint_velocity_limits"],value["controller_reference_acceleration_limits"],
        value["controller_reference_jerk_limits"],value["controller_reference_velocity_cap"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source_files=list(cf.SOURCES)+["scripts/replay_probe_nominal_cache_v53_20261103.py",
        "our_mani/local_mpc_control_20261103/probe_nominal_cache_v53.py"]
    sources={name:sha(ROOT/name) for name in source_files};reports={};total=0
    for name,expected in sources.items():
        target=args.output/"source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as out,(ROOT/name).open("rb") as original:shutil.copyfileobj(original,out)
        if sha(target)!=expected:raise RuntimeError("source changed during replay archival")
    for name,case in cf.CASES.items():
        input_path=case/"public_grasp_v41_decisions.jsonl"
        rows=[json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
        first=next(row for row in rows if row["decision"]["mode"]=="ACTIVE_VISUAL_VERIFICATION_PROBE")
        identity=first["decision"]["probe"]["identity"]
        recorded=[row for row in rows if row["decision"]["mode"]=="ACTIVE_VISUAL_VERIFICATION_PROBE"
            and row["decision"].get("probe",{}).get("identity")==identity]
        core,helper,provenance=build_planner(ROOT,limits(first),controller="coupled",action_space="joint",
            joint_goal_config={"lower":(-.1,-.65,.755)},terminal_config={"enabled":True},reference_frequency=6.)
        source_identity=dict(sources,recorded_config_and_source_envelope=sha(case/"run_inputs.json"))
        cache=ProbeNominalQCache();directory=args.output/name;directory.mkdir();ledger=[]
        for row in recorded:
            state=_fresh_feedback(row["decision"]["state"]);spec=cf.numpy_spec(row["decision"]["probe"])
            candidates=cf.candidate_ledger(row)
            mask=np.asarray([r.get("candidate_representation_defined",True) for r in row["decision"]["qualification"]["cases"]])
            if mask.dtype!=np.bool_:raise ValueError("recorded public representation mask must be explicit boolean")
            if json.dumps(plain(state["actuator_state"]),sort_keys=True)!=json.dumps(row["actuator_state_before"],sort_keys=True):
                raise ValueError("recorded actual state and h do not share the same feedback frame")
            untouched=json.dumps(plain(state),sort_keys=True)
            fresh=VerificationProbePlanner(core.config,execution_model=core.execution_model,goal_helper=helper,
                constraint_cost=core.planner.constraint_cost,terminal_config=core.terminal_config,probe_spec=spec)
            fresh._prepare_nominals(state,candidates,state["phase_state"])
            zero=np.zeros((8,8,7));fresh_nominal=cf.summarize_rollout(fresh,state,candidates,zero)
            cached=ConditionalModelCachePlanner(core.config,execution_model=core.execution_model,goal_helper=helper,
                constraint_cost=core.planner.constraint_cost,terminal_config=core.terminal_config,probe_spec=spec,
                nominal_cache=cache,source_sha256=source_identity,nominal_mask=mask)
            cached._prepare_nominals(state,candidates,state["phase_state"])
            cache_diagnostic=deepcopy(cached.nominal_receipt)
            cached_nominal=cf.summarize_rollout(cached,state,candidates,zero)
            plan=cached.plan(state,candidates,warm_start=None)
            if plan["budget"]["sequences"]!=256 or plan["budget"]["horizon"]!=8:
                raise ValueError("CPU replay budget changed")
            if json.dumps(plain(state),sort_keys=True)!=untouched:raise ValueError("recorded actual state/h was mutated")
            action=plan.get("action");last=np.asarray(state["actuator_state"]["target_joints"],float)
            entry=cache.lookup(cache_diagnostic["cache"]["cache_key"])
            record=dict(time=row["time"],actual_state_time=state["time"],actual_h_time=state["actuator_state"]["reference_clock"],
                cache=cache_diagnostic,probe_identity=identity,fixed_probe_deadline=spec.get("deadline"),
                original_probe_budget_seconds=None if "deadline" not in spec else spec["deadline"]-spec["trigger_time"],
                variants=dict(fresh_DLS_nominal=fresh_nominal,cached_DLS_nominal=cached_nominal),
                nominal_cached_q_minus_recorded_previous_goal_max_abs_rad=float(np.max(np.abs(entry["joint_goal"][0]-last))),
                conditional_cold_CEM=dict(cost=plan.get("cost"),model_feasible=action is not None,budget=plan["budget"],
                    selected_joint_goal=None if action is None else action["joint_goal"],
                    selected_target_error_m=None if action is None else float(cf.norm(action["tcp_position"]-spec["target_position"])),
                    selected_q_minus_cached_nominal_max_abs_rad=None if action is None else float(np.max(np.abs(action["joint_goal"]-entry["joint_goal"][0]))),
                    physical_action_qualified=False,live_cached_branch_and_selected_path_collision_unobserved=True),
                mask_scope="old recorded fresh-DLS collision mask is conditional only; not transferred as a cached-branch certificate",
                counterfactual_actual_feedback_or_task_outcome_claimed=False)
            append(directory/"rows.jsonl",record);ledger.append(record)
            print(json.dumps(dict(case=name,time=row["time"],cache_reused=cache_diagnostic["cache"]["reused"],
                nominal_H8_error_m=cached_nominal["H8_target_error_m"],physical_qualification=False)),flush=True)
        report=dict(case=name,rows=len(ledger),input_sha256=sha(input_path),frozen_model_provenance=provenance,
            cache_solve_count=cache.solve_count,cache_reuse_calls=cache.reuse_count,
            cache_primary_calls_per_row=2,extra_lookup_does_not_run_DLS=True,
            nominal_cost_scope="complete H8 starting from each recorded actual q/dq/fingers and same-time actual h",
            cached_nominal_model_feasible_rows=sum(r["variants"]["cached_DLS_nominal"]["feasible"] for r in ledger),
            fresh_nominal_model_feasible_rows=sum(r["variants"]["fresh_DLS_nominal"]["feasible"] for r in ledger),
            fresh_nominal_expected_first50ms_retargets=sum(r["variants"]["fresh_DLS_nominal"]["reference_target_changed_per50ms"][0][0] for r in ledger),
            cached_nominal_expected_first50ms_retargets=sum(r["variants"]["cached_DLS_nominal"]["reference_target_changed_per50ms"][0][0] for r in ledger),
            all_CEM_budgets_256_H8=True,Native_started=False,fit_started=False,physical_action_qualified=False,
            recorded_future_actual_state_substituted_into_rollout=False,counterfactual_Native_outcome_claimed=False,
            old_recorded_probe_duration_preserved_not_current_protocol_qualification=True)
        write_new(directory/"report.json",report);reports[name]=report;total+=len(ledger)
    if total!=19:raise ValueError("expected exactly the predeclared 19 first-probe states")
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):raise RuntimeError("replay model/dependency source changed")
    write_new(args.output/"manifest.json",dict(schema="v53_nominal_cache_19actual_states_conditional_CPU_replay_20261103_v1",
        complete=True,rows=total,source_sha256=sources,source_unchanged=True,Native_started=False,fit_started=False,
        existing_core_or_NN_modified=False,physical_action_qualified=False,
        cached_branch_collision_checked=False,TEST_partition_changed=False,
        cases={name:str(args.output/name/"report.json") for name in reports}))


if __name__=="__main__":main()
