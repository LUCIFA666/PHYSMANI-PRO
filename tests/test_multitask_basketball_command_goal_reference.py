"""CPU boundaries on two actual approved auxiliary TRAIN basketball sources."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,semantic_acquire
from our_mani.multitask import command_goal_reference as goal,basketball_command_goal_reference as basketball
from our_mani.multitask.reference_time import load_rule as load_reference

ROOT=Path(__file__).resolve().parents[1]
SETUP=ROOT/'artifacts/framework/multitask_r10/command_goal_reference_rule_v4_revision2_setup'


class BasketballCommandGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=a.read(SETUP/'artifact_fit_configuration.json')
        cls.rule=goal.load_rule(ROOT,cls.config['command_goal_reference_rule'],purpose='production_fitting')
        cls.semantic=semantic_acquire.load_rule(ROOT,cls.config['semantic_acquire_derivation_rule'])
        cls.reference=load_reference(ROOT,cls.config['reference_time_derivation_rule'])
        cls.permission=cls.rule.basketball_permission;cls.cases={}
        manifest=a.read(ROOT/'artifacts/framework/multitask_r10/semantic_acquire_original200_aux200_CPU_fit_v5/TRAIN_400_derivation_proof_before_fit.json')
        for task in basketball.TASKS:
            for row in manifest['case_proofs']:
                if row['kind']!='auxiliary_r1' or not row['source_id'].startswith(task+':') or not row['semantic_proof']['segment_ids']:continue
                command=a.read(a.local(ROOT,row['command_receipt']['path']))
                if not any(e['phase']=='deposit' for e in command['phase_completion_events']):continue
                path=a.local(ROOT,row['canonical_receipt']['path']);record=a.read(path)
                with np.load(a.local(ROOT,row['cache']['path']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                cls.cases[task]=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=command,
                    command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256']);break
        assert set(cls.cases)==set(basketball.TASKS)

    def combined(self,case):
        return adapter.interaction_source(case,reference_time_rule=self.reference,
            semantic_acquire_rule=self.semantic,command_goal_rule=self.rule)[0]

    def test_real_auxiliary_source_keeps_all_three_views_and_raw_metrics(self):
        for task,case in self.cases.items():
            original=adapter.interaction_source(case)[0];combined=self.combined(case)
            self.assertTrue(combined.semantic_acquire_evidence.segments)
            self.assertIsNotNone(combined.command_goal_reference_evidence)
            self.assertTrue(any(ref and ref.reference_time_basis=='command_issue' for ref in combined.references))
            self.assertEqual(len(original.completion_events),len(combined.completion_events))
            for before,after in zip(original.completion_events,combined.completion_events):
                np.testing.assert_array_equal(before.terminal_reference_xyzw,after.terminal_reference_xyzw)
                np.testing.assert_array_equal(before.terminal_tcp_xyzw,after.terminal_tcp_xyzw)
                self.assertEqual(before.eligible_command_ids,after.eligible_command_ids)
            np.testing.assert_array_equal(original.command_ids,combined.command_ids)
            self.assertEqual(list(original.command_ledger),case['command_receipt']['commands'])

    def test_closed_and_open_commands_both_use_exact_original_source_forecast(self):
        for case in self.cases.values():
            result=self.combined(case);e=result.command_goal_reference_evidence
            commands={c['command_id']:c for c in case['command_receipt']['commands'] if c['raw_phase']=='deposit'}
            self.assertEqual({r.command_id for r in e.references}|{r['command_id'] for r in e.pending_commands},set(commands))
            self.assertEqual({commands[r.command_id]['desired_opening'] for r in e.references},{0.,1.})
            for r in e.references:
                self.assertEqual(r.intent_kind,goal.BASKETBALL_FORECAST)
                self.assertTrue(r.provenance['closed_and_open_both_use_original_forecast'])
                self.assertFalse(r.provenance['future_boundary_reflection_applied'])
                self.assertFalse(r.provenance['intent_label_is_actual_future_dynamics_observation'])
                self.assertFalse(r.provenance['runtime_native_driver_or_teacher_parameter_access_allowed'])
                self.assertEqual(r.provenance['source_declared_forecast_seconds'],.5)
            self.assertEqual(e.derivation_proof['basketball_source_intention']['exact_source_replay_error_m'],0.)

    def test_original_strict_verification_rejects_an_already_derived_source(self):
        case=self.cases[basketball.TASKS[0]]
        with self.assertRaisesRegex(ValueError,'original source phases/references/events'):
            goal.verify_original_source(case,self.combined(case),self.rule)

    def test_auxiliary_is_unknown_without_its_explicit_approved_context(self):
        case=self.cases[basketball.TASKS[0]];ordinary=adapter.interaction_source(case)[0]
        with self.assertRaisesRegex(ValueError,'unknown actual execution plan'):
            goal.derive_source(case,ordinary,replace(self.rule,auxiliary_contexts={}))

    def test_mutated_array_command_or_driver_cannot_become_intention_labels(self):
        case=self.cases[basketball.TASKS[1]]
        for field in ('array','command','driver'):
            bad=deepcopy(case)
            if field=='array':bad['arrays']['measured_tcp_xyzw'][0,0]+=.01
            elif field=='command':next(c for c in bad['command_receipt']['commands'] if c['raw_phase']=='deposit')['raw_waypoint_xyzw'][0]+=.01
            else:bad['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule'][1]['v'][0]+=.01
            with self.subTest(field=field),self.assertRaises(ValueError):self.combined(bad)

    def test_permission_refuses_unknown_scope_tampered_source_or_retroactive_history(self):
        original=a.read(a.local(ROOT,self.permission.pointer['path']))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            directory=Path(temp)
            for field in ('unknown','scope','source','history'):
                value=deepcopy(original)
                if field=='unknown':value['implicit_auxiliary_or_runtime_access']=True
                elif field=='scope':value['TEST_used']=True
                elif field=='source':value['kernel_sources']['motion_helper']['sha256']='f'*64
                else:
                    review=a.read(a.local(ROOT,value['readonly_current_source_review']['path']))
                    review['historical_execution_dependency_claimed']=True
                    p=directory/'review.json';p.write_text(json.dumps(review));value['readonly_current_source_review']=a.pointer(ROOT,p)
                p=directory/(field+'.json');p.write_text(json.dumps(value))
                with self.subTest(field=field),self.assertRaises(ValueError):
                    basketball.load_permission(ROOT,a.pointer(ROOT,p),purpose='production_fitting')

    def test_typed_intention_cannot_gain_future_truth_or_runtime_teacher_access(self):
        r=self.combined(self.cases[basketball.TASKS[0]]).command_goal_reference_evidence.references[0]
        for change in ({'intent_label_is_actual_future_dynamics_observation':True},
                       {'runtime_native_driver_or_teacher_parameter_access_allowed':True},
                       {'future_boundary_reflection_applied':True}):
            with self.assertRaises(ValueError):replace(r,provenance=dict(r.provenance,**change))


if __name__=='__main__':unittest.main()
