"""CPU source/label boundaries using one actual archived TRAIN bin parent."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter
from our_mani.multitask import command_goal_reference as goal,bin_command_goal_reference as bin_goal

ROOT=Path(__file__).resolve().parents[1]
SETUP=ROOT/'artifacts/framework/multitask_r10/command_goal_reference_rule_v4_revision2_setup'


class BinCommandGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=goal.load_rule(ROOT,a.pointer(ROOT,SETUP/'command_goal_rule.json'),purpose='production_fitting')
        cls.permission=cls.rule.bin_permission
        manifest=a.read(ROOT/'artifacts/framework/multitask_r10/command_goal_actual700_production_CPU_refit_v2/put_rubbish_in_moving_bin/TRAIN_fit_manifest.json')
        for row in manifest['case_proofs']:
            command=a.read(a.local(ROOT,row['command_receipt']['path']))
            if not any(c['raw_phase']=='deposit' and c['desired_opening']>=.5 for c in command['commands']):continue
            path=a.local(ROOT,row['canonical_receipt']['path']);record=a.read(path)
            with np.load(a.local(ROOT,row['cache']['path']),allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
            cls.case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=command,
                command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
            break
        cls.source=adapter.interaction_source(cls.case)[0]

    def test_closed_uses_current_issue_and_open_uses_original_unreflected_forecast(self):
        source,audit=adapter.interaction_source(self.case,command_goal_rule=self.rule)
        evidence=source.command_goal_reference_evidence
        self.assertEqual(evidence.task_name,bin_goal.TASK)
        rows={r['command_id']:r for r in self.case['command_receipt']['commands']}
        for intent in evidence.references:
            raw=rows[intent.command_id]
            if raw['desired_opening']>=.5:
                self.assertEqual(intent.intent_kind,goal.BIN_FORECAST)
                np.testing.assert_allclose(intent.intent_reference_xyzw[:2],raw['raw_waypoint_xyzw'][:2],atol=1e-6,rtol=0)
                self.assertFalse(intent.provenance['future_boundary_reflection_applied'])
            else:
                self.assertEqual(intent.intent_kind,goal.ISSUE)
                np.testing.assert_array_equal(intent.intent_reference_xyzw,intent.original_issue_reference_xyzw)
            self.assertFalse(intent.provenance['intent_label_is_actual_future_dynamics_observation'])
            self.assertFalse(intent.provenance['runtime_native_driver_or_teacher_parameter_access_allowed'])
        self.assertEqual(len(self.source.completion_events),len(source.completion_events))
        for before,after in zip(self.source.completion_events,source.completion_events):
            np.testing.assert_array_equal(before.terminal_reference_xyzw,after.terminal_reference_xyzw)
            np.testing.assert_array_equal(before.terminal_tcp_xyzw,after.terminal_tcp_xyzw)
        self.assertEqual(evidence.derivation_proof['bin_source_intention']['replay']['exact_source_replay_error_m'],0.)

    def test_actual_replay_rejects_changed_initial_driver_and_raw_body_frame(self):
        initial=self.case['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']
        bad=deepcopy(initial);bad['v'][0]+=.01
        with self.assertRaisesRegex(ValueError,'exactly reproduce'):
            bin_goal.replay_states(self.case['arrays'],bad,self.permission)
        arrays=deepcopy(self.case['arrays']);j=list(arrays['body_ids']).index('bin');arrays['target_positions'][10,j,0]+=.001
        with self.assertRaisesRegex(ValueError,'exactly reproduce'):
            bin_goal.replay_states(arrays,initial,self.permission)

    def test_raw_ledger_or_array_changes_fail_before_intention_can_replace_them(self):
        for field in ('array','ledger'):
            bad=deepcopy(self.case)
            if field=='array':bad['arrays']['measured_tcp_xyzw'][0,0]+=.01
            else:bad['command_receipt']['commands'][-1]['raw_waypoint_xyzw'][0]+=.01
            with self.subTest(field=field),self.assertRaises(ValueError):
                bin_goal.derive_source(bad,self.source,self.rule,self.permission)

    def test_permission_rejects_unknown_fields_wrong_scope_or_historical_claim(self):
        original=a.read(a.local(ROOT,self.permission.pointer['path']))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            directory=Path(temp)
            for kind in ('unknown','scope','historical','sourceSHA'):
                value=deepcopy(original)
                if kind=='unknown':value['implicit_native_permission']=True
                elif kind=='scope':value['production_fitting_enabled']=False
                elif kind=='sourceSHA':value['kernel_sources']['motion_helper']['sha256']='f'*64
                else:
                    review=a.read(a.local(ROOT,value['readonly_current_source_review']['path']))
                    review['historical_execution_dependency_claimed']=True
                    path=directory/'review.json';path.write_text(json.dumps(review));value['readonly_current_source_review']=a.pointer(ROOT,path)
                path=directory/(kind+'.json');path.write_text(json.dumps(value))
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    bin_goal.load_permission(ROOT,a.pointer(ROOT,path),purpose='production_fitting')

    def test_typed_forecast_cannot_claim_actual_future_or_reflected_teacher_goal(self):
        proof=bin_goal.derive_source(self.case,self.source,self.rule,self.permission)
        value=next(r for r in proof.references if r.intent_kind==bin_goal.FORECAST)
        for change in ({'intent_label_is_actual_future_dynamics_observation':True},
                       {'future_boundary_reflection_applied':True}):
            with self.assertRaises(ValueError):replace(value,provenance=dict(value.provenance,**change))


if __name__=='__main__':unittest.main()
