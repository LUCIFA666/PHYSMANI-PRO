"""CPU-only static-destination cache and fresh-qualification contracts."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.probe_nominal_cache_v53 import (
    ProbeNominalQCache,CachedNominalVerificationProbePlanner,
    qualify_cached_probe_candidates,qualify_cached_probe_action)
from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
from tests.test_local_mpc_public_grasp_probe_20261103 import planner_fixture


SOURCES={"runtime_config":"a"*64,"world_FK_snapshot":"b"*64,"joint_and_finger_models":"c"*64}


class SyntheticNoStepCollision:
    """A deterministic CPU test double, never Native qualification evidence."""
    def __init__(self,blocked=()):self.calls=[];self.blocked=set(blocked)
    def probe_joint_collision(self,q):
        self.calls.append(np.asarray(q).copy());return len(self.calls)-1 in self.blocked


def fixture():
    base,state,candidates,_=planner_fixture()
    state["actuator_state"]=dict(reference_clock=state["time"],gripper_target=0.,reference_valid=False,
        reference_initialized=False,source="fresh_actual_executor_snapshot",predicted=False)
    return base,state,candidates


def qualified(base,state,candidates,cache,*,sources=None,mask=None,oracle=None,spec=None):
    sources=SOURCES if sources is None else sources
    mask=np.ones(8,bool) if mask is None else mask
    oracle=SyntheticNoStepCollision() if oracle is None else oracle
    rows,receipt=qualify_cached_probe_candidates(oracle,state,candidates,base.goal_helper,spec or base.probe_spec,
        nominal_cache=cache,source_sha256=sources,nominal_mask=mask)
    planner=CachedNominalVerificationProbePlanner(base.config,execution_model=base.execution_model,
        constraint_cost=base.constraint_cost,goal_helper=base.goal_helper,terminal_config=base.terminal_config,
        probe_spec=spec or base.probe_spec,nominal_cache=cache,source_sha256=sources,
        nominal_mask=mask,qualification=receipt)
    return planner,rows,receipt,oracle


class CacheTests(unittest.TestCase):
    def test_cache_contains_only_destination_identity_and_no_feedback_or_certificate(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();mask=np.ones(8,bool)
        entry,first=cache.resolve(state["q"],candidates,base.probe_spec,base.goal_helper,source_sha256=SOURCES,nominal_mask=mask)
        changed=state["q"]+np.array([1e-5,0,0,0,0,0,0])
        again,second=cache.resolve(changed,candidates,base.probe_spec,base.goal_helper,source_sha256=SOURCES,nominal_mask=mask)
        self.assertFalse(first["reused"]);self.assertTrue(second["reused"])
        self.assertEqual(cache.solve_count,1);np.testing.assert_array_equal(entry["joint_goal"],again["joint_goal"])
        self.assertEqual(set(cache._entry),{"cache_key","identity","joint_goal","algebraic_feasible","pose"})
        self.assertFalse(any(key in cache._entry for key in ("q","dq","state","actuator_state","h","predicted_states","qualification")))
        entry["joint_goal"][:]=999;entry["identity"]["source_sha256"]["runtime_config"]="e"*64
        untouched=cache.lookup(second["cache_key"])
        self.assertLess(abs(untouched["joint_goal"]).max(),3.)
        self.assertEqual(untouched["identity"]["source_sha256"]["runtime_config"],"a"*64)

    def test_exact_pose_attempt_deadline_mask_source_and_limits_invalidate(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();mask=np.ones(8,bool);spec=deepcopy(base.probe_spec)
        def resolve(c=candidates,s=spec,source=SOURCES,m=mask):
            return cache.resolve(state["q"],c,s,base.goal_helper,source_sha256=source,nominal_mask=m)
        resolve();resolve();self.assertEqual(cache.solve_count,1)
        spec["target_position"][0]+=1e-6;spec["anchor_position"][0]+=1e-6;resolve();self.assertEqual(cache.solve_count,2)
        spec["identity"]="next-attempt";spec["attempt_id"]=2;resolve();self.assertEqual(cache.solve_count,3)
        spec["deadline"]=spec["trigger_time"]+.8;resolve();self.assertEqual(cache.solve_count,4)
        spec["deadline"]+=.1;resolve();self.assertEqual(cache.solve_count,5)
        mask[2]=False;resolve();self.assertEqual(cache.solve_count,6)
        source=dict(SOURCES,runtime_config="d"*64);resolve(source=source);self.assertEqual(cache.solve_count,7)
        base.goal_helper.adapter.config=replace(base.goal_helper.adapter.config,damping=.012)
        resolve(source=source);self.assertEqual(cache.solve_count,8)
        base.goal_helper.limits=replace(base.goal_helper.limits,jerk=900.)
        resolve(source=source);self.assertEqual(cache.solve_count,9)
        reordered=deepcopy(candidates);reordered[0],reordered[1]=reordered[1],reordered[0]
        resolve(c=reordered,source=source);self.assertEqual(cache.solve_count,10)
        cache.reset();resolve(c=reordered,source=source);self.assertEqual(cache.solve_count,11)

    def test_each_frame_checks_cached_branch_again_and_ignores_old_fresh_DLS_flags(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache()
        for c in candidates:c["native_pose_qualified"]=False
        planner,rows,first,oracle=qualified(base,state,candidates,cache)
        self.assertTrue(all(first.qualified));self.assertEqual(len(oracle.calls),8)
        # Collision outcomes are current evidence, never saved in the cache.
        another=SyntheticNoStepCollision(blocked=(3,))
        _,rows2,second,_=qualified(base,state,candidates,cache,oracle=another)
        self.assertEqual(cache.solve_count,1);self.assertEqual(len(another.calls),8)
        self.assertFalse(rows2[3]["native_pose_qualified"]);self.assertTrue(rows2[2]["native_pose_qualified"])
        for a,b in zip(oracle.calls,another.calls):np.testing.assert_array_equal(a,b)
        self.assertFalse(second.receipt()["old_fresh_DLS_certificate_reused"])

    def test_stale_predicted_or_nonfinite_h_rejected_before_qualification(self):
        base,state,candidates=fixture()
        for change in ({"reference_clock":-.1},{"reference_clock":np.nan},{"reference_clock":True},
                       {"predicted":True},{"predicted":np.bool_(True)},{"source":"virtual_joint_prediction"},
                       {"gripper_target":1.},{"gripper_target":False}):
            value=deepcopy(state);value["actuator_state"].update(change)
            oracle=SyntheticNoStepCollision();cache=ProbeNominalQCache()
            with self.assertRaises(ValueError):qualified(base,value,candidates,cache,oracle=oracle)
            self.assertEqual(oracle.calls,[]);self.assertEqual(cache.solve_count,0)

    def test_current_qualification_cannot_be_serialized_or_reused_for_new_feedback(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();planner,rows,receipt,_=qualified(base,state,candidates,cache)
        later=deepcopy(state);later["time"]+=.1;later["actuator_state"]["reference_clock"]+=.1
        later["q"][0]+=1e-5
        with self.assertRaisesRegex(ValueError,"fresh actual state"):
            planner.plan(later,rows)
        planner.qualification=receipt.receipt()
        with self.assertRaises(TypeError):planner.plan(state,rows)
        planner.qualification=receipt;rows[0]["native_pose_qualified"]=False
        with self.assertRaisesRegex(ValueError,"qualification must agree"):
            planner.plan(state,rows)

    def test_repeated_normalization_does_not_forge_a_different_feedback_frame(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache()
        # Equivalent physical quaternions may have non-unit last-bit rounding
        # in actual sensors. Validation is deterministic from the same input.
        state["tcp_quaternion"]*=1.+2e-7
        state["object_quaternion"]=np.array([.2,.3,.4,.5])
        planner,rows,_,_=qualified(base,state,candidates,cache)
        result=planner.plan(state,rows)
        self.assertEqual(result["budget"]["sequences"],256)
        self.assertIsNotNone(result["action"])

    def test_first_cold_plan_matches_fresh_nominal_base_full_budget_and_no_fake_held(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();before=deepcopy(state)
        solve=base.goal_helper.adapter.solve;calls=[]
        with patch.object(base.goal_helper.adapter,"solve",side_effect=lambda *args:(calls.append(1) or solve(*args))):
            planner,rows,qualification,_=qualified(base,state,candidates,cache)
            result=planner.plan(state,rows)
        self.assertEqual(len(calls),1)  # qualification resolves once; planner resolves zero
        self.assertEqual(result["budget"]["iteration_sequences"],[88,88,80])
        self.assertEqual(result["budget"]["sequences"],256);self.assertEqual(result["budget"]["horizon"],8)
        self.assertEqual(result["diagnostics"]["execution_model_calls"],24)
        self.assertFalse(result["diagnostics"]["held_asserted"])
        self.assertTrue(all(not s["held"] for s in result["predicted_states"]))
        self.assertIsNotNone(result["action"]);self.assertEqual(result["action"]["opening"],0.)
        fresh=VerificationProbePlanner(base.config,execution_model=base.execution_model,goal_helper=base.goal_helper,
            probe_spec=base.probe_spec,constraint_cost=base.constraint_cost,terminal_config=base.terminal_config)
        equivalent=fresh.plan(state,rows)
        self.assertAlmostEqual(result["cost"],equivalent["cost"],places=11)
        np.testing.assert_array_equal(result["action"]["joint_goal"],equivalent["action"]["joint_goal"])
        np.testing.assert_array_equal(state["q"],before["q"]);self.assertEqual(state["actuator_state"],before["actuator_state"])
        for action in result["sequence"]:
            pose=base.goal_helper.kinematics(action["joint_goal"])
            np.testing.assert_array_equal(action["tcp_position"],pose[:3,3])
            np.testing.assert_allclose(action["tcp_quaternion"],matrix_quaternion(pose[:3,:3]),atol=1e-14)

    def test_fresh_state_and_h_initialize_next_plan_while_destination_is_unchanged(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();planner,rows,_,_=qualified(base,state,candidates,cache)
        first=planner.plan(state,rows);goal=cache._entry["joint_goal"].copy()
        actual=deepcopy(state);actual["time"]+=.1;actual["q"][1]+=1e-5;actual["dq"][1]=.001
        actual["actuator_state"].update(reference_clock=.1,reference_valid=True,reference_initialized=True,
            target_joints=goal[0].copy(),reference_positions=actual["q"].copy(),reference_velocities=actual["dq"].copy(),
            reference_accelerations=np.zeros(7))
        pose=base.goal_helper.kinematics(actual["q"]);actual["tcp_position"]=pose[:3,3]
        actual["tcp_quaternion"]=matrix_quaternion(pose[:3,:3])
        next_planner,next_rows,_,_=qualified(base,actual,candidates,cache)
        second=next_planner.plan(actual,next_rows,warm_start=first["warm_start"])
        self.assertEqual(cache.solve_count,1);np.testing.assert_array_equal(cache._entry["joint_goal"],goal)
        np.testing.assert_array_equal(second["predicted_states"][0]["q"],actual["q"])
        self.assertEqual(second["predicted_states"][0]["time"],.1)
        self.assertEqual(second["budget"]["sequences"],256)
        self.assertFalse(second["diagnostics"]["actual_state_and_h_reused"])

    def test_selected_action_gets_fresh_endpoint_and_seven_path_checks_exact_q(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();planner,rows,qualification,_=qualified(base,state,candidates,cache)
        result=planner.plan(state,rows);original=deepcopy(result["action"]);oracle=SyntheticNoStepCollision()
        goal,reason,receipt=qualify_cached_probe_action(oracle,state,result["action"],base.goal_helper,
            candidates=rows,probe_spec=base.probe_spec,nominal_cache=cache,source_sha256=SOURCES,nominal_mask=np.ones(8,bool),
            qualification=qualification,selected_candidate_id=result["selected_candidate_id"])
        self.assertIsNone(reason);np.testing.assert_array_equal(goal,original["joint_goal"])
        self.assertEqual(len(oracle.calls),8);self.assertEqual(receipt["DLS_solves"],0)
        self.assertEqual([p["alpha"] for p in receipt["collision_probes"]],[1.,.125,.25,.375,.5,.625,.75,.875])
        np.testing.assert_array_equal(result["action"]["joint_goal"],original["joint_goal"])
        blocked=SyntheticNoStepCollision(blocked=(2,))
        goal,reason,_=qualify_cached_probe_action(blocked,state,result["action"],base.goal_helper,
            candidates=rows,probe_spec=base.probe_spec,nominal_cache=cache,source_sha256=SOURCES,nominal_mask=np.ones(8,bool),
            qualification=qualification,selected_candidate_id=result["selected_candidate_id"])
        self.assertIsNone(goal);self.assertEqual(reason,"sampled_joint_path_arm_static_collision")

    def test_all_invalid_public_mask_retains_full_256_budget(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();oracle=SyntheticNoStepCollision()
        planner,rows,_,_=qualified(base,state,candidates,cache,mask=np.zeros(8,bool),oracle=oracle)
        result=planner.plan(state,rows)
        self.assertIsNone(result["action"]);self.assertEqual(result["budget"]["sequences"],256)
        self.assertEqual(oracle.calls,[])

    def test_changed_cache_identity_discards_probe_warm_without_changing_other_policy(self):
        base,state,candidates=fixture();cache=ProbeNominalQCache();planner,rows,_,_=qualified(base,state,candidates,cache)
        first=planner.plan(state,rows)
        changed=dict(SOURCES,runtime_config="f"*64)
        planner2,rows2,_,_=qualified(base,state,candidates,cache,sources=changed)
        second=planner2.plan(state,rows2,warm_start=first["warm_start"])
        self.assertEqual(cache.solve_count,2)
        self.assertEqual(second["diagnostics"]["incompatible_warm_cleared"],"incompatible_probe_or_fixed_nominal_cache_identity")
        self.assertEqual(second["budget"],first["budget"])
        self.assertNotEqual(first["warm_start"]["probe_nominal_cache_identity"],second["warm_start"]["probe_nominal_cache_identity"])


if __name__=="__main__":unittest.main()
