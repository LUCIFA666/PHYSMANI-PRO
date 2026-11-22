from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from our_mani.multitask.evaluation import digest
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,VAL_COUNTS
from our_mani.multitask import common_rml_protocol_v530 as common
from our_mani.multitask import three_method_release_v531 as release
from our_mani.multitask import formal_suite_admission_v532 as admission

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT.parent/'phsimani/PhysMani'


class Tests(unittest.TestCase):
    def fixture(self):
        profile=common.binding(ROOT,NATIVE);sha=common.fingerprint(profile)
        pointer=dict(path='CPU_fixture',sha256='a'*64)
        routes=dict(common_author_RML_profile=profile,shared_context_checkpoint_sha256='b'*64,
            routes=[dict(task=t,method=m,GPU_required=m=='physmani',configuration=pointer,
                frozen_inputs=pointer,worker=pointer,consumer=pointer,common_RML_profile_sha256=sha)
                for t in TASKS for m in METHODS])
        policy=release.policy_fingerprint(routes)
        manifest=dict(split='validation',episodes=[dict(task=t,raw_episode_id='source'+str(i),
            path='CPU_fixture',split='validation') for t in TASKS for i in range(VAL_COUNTS[t])])
        env=NS(arm=NS(max_velocity=3.,max_acceleration=12.,max_jerk=1000.),ledger=NS(dt=.05,budget=10.))
        observed=common.attest(env,dict(baseline_decision_ticks=10,dt=.05),profile)
        cases=[dict(task=r['task'],source_raw_episode_id=r['raw_episode_id'],method=m,
            split='validation',complete=True,cleanup_completed=True,infrastructure_error=None,
            official_task_success=False,TEST_used_for_fitting=False,
            all_stage_counted_joint_motion=dict(all_stage_actual_window_checks_passed=False),
            common_author_RML_parameters=deepcopy(observed),
            full16_frozen_execution=dict(policy_sha256=policy,source_row_sha256=digest(r)))
            for r in manifest['episodes'] for m in METHODS]
        return routes,manifest,cases

    def test_all_three_motion_violations_are_reported_without_an_extra_ours_release_gate(self):
        routes,manifest,cases=self.fixture();r=release.validation_release(cases,manifest,routes)
        self.assertTrue(release.require_test_release(r,routes))
        self.assertEqual(r['measured_motion_violations_by_method'],{m:315 for m in METHODS})
        self.assertFalse(r['ours_measured_motion_passed'])
        self.assertEqual(sum(r['native_successes_by_method'].values()),0)

    def test_missing_motion_native_cleanup_or_common_parameters_still_block(self):
        routes,manifest,cases=self.fixture()
        for patch_value in (dict(all_stage_counted_joint_motion={}),dict(cleanup_completed=False),
                            dict(infrastructure_error='native crash'),dict(common_author_RML_parameters={})):
            bad=deepcopy(cases);bad[0].update(patch_value)
            with self.assertRaises(ValueError):release.validation_release(bad,manifest,routes)
        with self.assertRaises(ValueError):release.validation_release(cases[:-1],manifest,routes)
        bad=deepcopy(cases);bad[0]['common_author_RML_parameters']['actual_parameters']['max_acceleration']=4.
        with self.assertRaises(ValueError):release.validation_release(bad,manifest,routes)

    def test_changed_parameters_sources_and_old_schema_cannot_reuse_release(self):
        routes,manifest,cases=self.fixture();r=release.validation_release(cases,manifest,routes)
        for key,value in (('max_acceleration',4.),('decision_ticks',9)):
            bad=deepcopy(routes);bad['common_author_RML_profile']['parameters'][key]=value
            with self.assertRaises(ValueError):release.require_test_release(r,bad)
        bad=deepcopy(routes);bad['routes'][0]['common_RML_profile_sha256']='c'*64
        with self.assertRaises(ValueError):release.require_test_release(r,bad)
        bad=dict(r,schema='r10_full16_three_method_validation_release_v1')
        with self.assertRaises(ValueError):release.require_test_release(bad,routes)

    def test_native_attestation_rejects_local_parameter_override_without_mutating_arm(self):
        profile=common.binding(ROOT,NATIVE)
        env=NS(arm=NS(max_velocity=3.,max_acceleration=4.,max_jerk=1000.),ledger=NS(dt=.05,budget=10.))
        with self.assertRaises(ValueError):common.attest(env,dict(baseline_decision_ticks=10,dt=.05),profile)
        self.assertEqual(env.arm.max_acceleration,4.)

    def test_TEST_manifest_cannot_be_read_before_the_complete_new_VAL_release(self):
        routes,_,_=self.fixture()
        with patch.object(admission,'admit',return_value=dict(routes=routes)),patch.object(admission,'locked_json') as reader:
            with self.assertRaisesRegex(ValueError,'VAL release'):
                admission.read_formal_source(ROOT,{}, {},'test')
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
