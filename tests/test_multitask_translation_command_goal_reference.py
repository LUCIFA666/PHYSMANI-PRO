"""CPU boundaries on three real TRAIN sources; no simulator or policy claims."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile,unittest
import numpy as np
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,translation_command_goal_reference as trans
ROOT=Path(__file__).resolve().parents[1];SETUP=ROOT/'artifacts/framework/multitask_r10/command_goal_reference_rule_v5_translation_revision3_setup'


class TranslationCommandGoalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=a.read(SETUP/'artifact_fit_configuration_original300_bound.json')
        cls.rule=goal.load_rule(ROOT,cls.config['command_goal_reference_rule'],purpose='production_fitting')
        cls.cases={}
        for task in trans.TASKS:
            manifest=a.read(ROOT/f'artifacts/framework/multitask_r10/full14_pending_reference_readonly_diagnostic_v6/{task}/cases.json')
            for row in manifest:
                commands=a.read(a.local(ROOT,row['command_receipt']['path']))
                if task==trans.HIGH_BIN and not any(c['raw_phase']=='deposit' and c['desired_opening']>=.5 for c in commands['commands']):continue
                path=a.local(ROOT,row['canonical_receipt']['path']);record=a.read(path)
                with np.load(a.local(ROOT,row['cache']['path']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                cls.cases[task]=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=commands,
                    command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256']);break
        assert set(cls.cases)==set(trans.TASKS)

    def derived(self,case):return adapter.interaction_source(case,command_goal_rule=self.rule)[0]

    def test_real_button_scope_is_only_approach_and_preserves_raw_metrics_masks(self):
        for task in trans.TASKS[:2]:
            case=self.cases[task];before=adapter.interaction_source(case)[0];after=self.derived(case);proof=after.command_goal_reference_evidence
            commands={c['command_id']:c for c in before.command_ledger}
            self.assertTrue(proof.references)
            self.assertEqual({r.command_id for r in proof.references}|{r['command_id'] for r in proof.pending_commands},
                {c['command_id'] for c in before.command_ledger if c['raw_phase']=='approach'})
            for r in proof.references:
                self.assertEqual(commands[r.command_id]['raw_phase'],'approach');self.assertEqual(r.intent_kind,goal.ISSUE)
                np.testing.assert_array_equal(r.intent_reference_xyzw,r.original_issue_reference_xyzw)
            self.assertEqual(proof.original_masks_sha256,goal.fingerprint(case['arrays']))
            self.assertEqual(before.references,after.references);np.testing.assert_array_equal(before.command_ids,after.command_ids)
            self.assertEqual(list(after.command_ledger),case['command_receipt']['commands'])
            for b,c in zip(before.completion_events,after.completion_events):
                np.testing.assert_array_equal(b.terminal_reference_xyzw,c.terminal_reference_xyzw)
                self.assertEqual(b.command_completed,c.command_completed)

    def test_real_high_bin_current_closed1_and_open_half_are_all_explicit(self):
        case=self.cases[trans.HIGH_BIN];source=self.derived(case);e=source.command_goal_reference_evidence
        self.assertEqual({r.provenance['source_declared_forecast_seconds'] for r in e.references},{0.,.5,1.})
        commands={c['command_id']:c for c in source.command_ledger}
        self.assertEqual({r.command_id for r in e.references}|{r['command_id'] for r in e.pending_commands},
            {cid for cid,c in commands.items() if c['raw_phase']=='deposit'})
        for r in e.references:
            delay=r.provenance['source_declared_forecast_seconds'];c=commands[r.command_id]
            self.assertEqual(r.intent_kind,trans.FORECAST if delay else goal.ISSUE)
            if delay:self.assertEqual(delay,.5 if c['desired_opening']>=.5 else 1.)
            else:self.assertEqual(commands[r.command_id-1]['raw_stage'],'wp2')
            self.assertFalse(r.provenance['future_boundary_reflection_applied'])
            self.assertFalse(r.provenance['intent_label_is_actual_future_dynamics_observation'])
        p=e.derivation_proof['translation_source_intention'];self.assertEqual(p['exact_forecast_goal_xy_maximum_error_m'],0.)
        self.assertEqual(p['source_arithmetic_replay']['maximum_body_replay_error_m'],0.)

    def test_legacy_v4_task_map_does_not_gain_new_scope_implicitly(self):
        case=self.cases[trans.TASKS[0]];original=adapter.interaction_source(case)[0]
        legacy=replace(self.rule,tasks=goal.TASK_PHASES,translation_permission=None)
        proof=goal.derive_source(case,original,legacy);self.assertEqual(proof.references,())
        self.assertNotIn(trans.TASKS[0],goal.TASK_PHASES)

    def test_raw_array_command_driver_or_unknown_actual_plan_rejected(self):
        for mutation in ('array','command','driver','actual_plan'):
            bad=deepcopy(self.cases[trans.HIGH_BIN])
            if mutation=='array':bad['arrays']['target_positions'][1,0,0]+=.01
            elif mutation=='command':next(c for c in bad['command_receipt']['commands'] if c['raw_phase']=='deposit')['raw_waypoint_xyzw'][0]+=.01
            elif mutation=='driver':bad['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']['v'][0]+=.01
            else:bad['record']['plan_sha256']='f'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.derived(bad)

    def test_already_derived_source_or_absent_permission_cannot_reenter_raw_verification(self):
        case=self.cases[trans.HIGH_BIN];derived=self.derived(case)
        # Reference/event arrays stay raw, but fabricated command bytes still fail.
        with self.assertRaises(ValueError):
            bad=replace(derived,command_ledger=tuple(dict(c,generated_at=c['generated_at']+.05) for c in derived.command_ledger))
            goal.verify_original_source(case,bad,self.rule)
        with self.assertRaisesRegex(ValueError,'explicit v5 permission'):
            adapter.interaction_source(case,command_goal_rule=replace(self.rule,translation_permission=None))

    def test_permission_unknown_fields_scope_source_and_fake_history_fail_closed(self):
        permission=self.rule.translation_permission;original=a.read(a.local(ROOT,permission.pointer['path']))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            directory=Path(temp)
            for mutation in ('unknown','scope','source','history'):
                value=deepcopy(original)
                if mutation=='unknown':value['allow_runtime_native_forecast']=True
                elif mutation=='scope':value['TEST_used']=True
                elif mutation=='source':value['kernel_sources']['motion_helper']['sha256']='e'*64
                else:
                    review=a.read(a.local(ROOT,value['readonly_current_source_review']['path']));review['historical_execution_dependency_claimed']=True
                    rp=directory/'review.json';rp.write_text(json.dumps(review));value['readonly_current_source_review']=a.pointer(ROOT,rp)
                p=directory/(mutation+'.json');p.write_text(json.dumps(value))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):trans.load_permission(ROOT,a.pointer(ROOT,p),purpose='production_fitting')

    def test_v5_cannot_drop_permission_or_mislabel_it_v4(self):
        original=a.read(a.local(ROOT,self.rule.pointer['path']))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            for mutation in ('permission','version','tasks'):
                value=deepcopy(original)
                if mutation=='permission':value.pop('translation_intention_derivation_permission')
                elif mutation=='version':value['schema']=goal.FIT_RULE_SCHEMA
                else:value['tasks']=goal.TASK_PHASES
                p=Path(temp)/(mutation+'.json');p.write_text(json.dumps(value))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):goal.load_rule(ROOT,a.pointer(ROOT,p),purpose='production_fitting')

    def test_high_bin_typed_forecast_refuses_future_truth_runtime_access_or_unknown_delay(self):
        r=next(r for r in self.derived(self.cases[trans.HIGH_BIN]).command_goal_reference_evidence.references if r.intent_kind==trans.FORECAST)
        for change in ({'source_declared_forecast_seconds':.25},{'intent_label_is_actual_future_dynamics_observation':True},
            {'runtime_native_driver_or_teacher_parameter_access_allowed':True},{'future_boundary_reflection_applied':True},
            {'source_branch_selected_from_original_prior_stage_not_outcome':False}):
            with self.subTest(change=change),self.assertRaises(ValueError):replace(r,provenance=dict(r.provenance,**change))

if __name__=='__main__':unittest.main()
