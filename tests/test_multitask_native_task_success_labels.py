"""Actual TRAIN success/censor label boundaries; no model or native execution."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,native_task_success_labels as labels
ROOT=Path(__file__).resolve().parents[1]


class NativeSuccessLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=goal.load_rule(ROOT,a.pointer(ROOT,ROOT/'artifacts/framework/multitask_r10/command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'),purpose='production_fitting')
        cls.cases={};data=a.read(ROOT/'artifacts/framework/multitask_r10/button_native_success_independent_labels_actual200_v1/labels.json')
        for task in labels.TASKS:
            for positive in (True,False):
                row=next(r for r in data if r['task_name']==task and r['raw_task_success'] is positive);ptrs=row['raw_bindings']
                path=a.local(ROOT,ptrs['canonical_receipt']['path']);record=a.read(path)
                with np.load(a.local(ROOT,ptrs['cache']['path']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                cls.cases[(task,positive)]=dict(path=path,record=record,entry=record['entry'],arrays=arrays,
                    command_receipt=a.read(a.local(ROOT,ptrs['commands']['path'])),command_receipt_verified=True,command_receipt_sha256=ptrs['commands']['sha256'])

    def extract(self,case):return labels.extract(case,adapter.interaction_source(case)[0],self.rule)

    def test_actual_success_is_verified_without_fabricating_command_completion(self):
        for task in labels.TASKS:
            e=self.extract(self.cases[(task,True)]);self.assertTrue(e.raw_task_success)
            self.assertIsNotNone(e.success_completion);self.assertFalse(e.success_completion.raw_command_completed)
            self.assertFalse(e.success_completion.provenance['geometry_tolerance_override_allowed'])
            self.assertEqual(e.success_completion.terminal_frame,len(self.cases[(task,True)]['arrays']['times'])-1)
            self.assertFalse(e.derivation_proof['geometry_BC_or_runtime_gate_modified'])

    def test_original_unsuccessful_budget_censored_parents_stay_in_label_view(self):
        for task in labels.TASKS:
            e=self.extract(self.cases[(task,False)]);self.assertFalse(e.raw_task_success)
            self.assertTrue(e.budget_censored);self.assertIsNone(e.success_completion)
            self.assertEqual(e.status,'original_unsuccessful_budget_censored')
            self.assertEqual(e.original_masks_sha256,goal.fingerprint(self.cases[(task,False)]['arrays']))

    def test_native_outcome_or_terminal_tick_cannot_be_changed_in_memory(self):
        case=self.cases[(labels.TASKS[0],True)]
        for change in ('outcome','tick','completion'):
            bad=deepcopy(case)
            if change=='outcome':bad['record']['alignment']['native_outcome']['success']=False
            elif change=='tick':bad['record']['alignment']['native_terminal_tick']+=1
            else:bad['record']['entry']['teacher_used']=False
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'canonical native outcome/record'):
                labels.extract(bad,adapter.interaction_source(case)[0],self.rule)

    def test_raw_event_or_array_rewrite_and_validation_split_rejected(self):
        for change in ('event','array','split'):
            bad=deepcopy(self.cases[(labels.TASKS[1],True)])
            if change=='event':bad['command_receipt']['phase_completion_events'][-1]['attempt_end_time']-=.05
            elif change=='array':bad['arrays']['measured_tcp_xyzw'][-1,0]+=.01
            else:bad['entry']['split']='validation'
            with self.subTest(change=change),self.assertRaises(ValueError):self.extract(bad)

    def test_typed_success_cannot_override_geometry_or_be_runtime_observation(self):
        e=self.extract(self.cases[(labels.TASKS[0],True)]).success_completion
        for change in ({'geometry_tolerance_override_allowed':True},{'label_is_runtime_observation':True},{'native_success_is_waypoint_completion':True}):
            with self.subTest(change=change),self.assertRaises(ValueError):replace(e,provenance=dict(e.provenance,**change))
        with self.assertRaises(ValueError):goal.freeze_runtime_reference(candidate_forecast_pose=e,predictor='cv',observed_until=0,selected_at=1,forecast_time=2)

if __name__=='__main__':unittest.main()
