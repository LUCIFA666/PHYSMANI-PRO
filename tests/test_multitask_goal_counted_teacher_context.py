"""V6 CPU boundaries. Whole100 auditor mocked only for pure admission fixtures.

Actual registry/source metadata and an actual committed counted case are used.
The production positive whole100 gate is separately exercised without mocks
only after the real collection is complete. No fixture creates native data.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,requery_acquisition as counted
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/framework/multitask_r10'


class GoalCanonicalGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=goal.load_rule(ROOT,a.pointer(ROOT,BASE/'command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'),purpose='production_fitting')
        row=a.read(BASE/'full14_pending_reference_readonly_diagnostic_v6/push_moving_button/cases.json')[0]
        path=a.local(ROOT,row['canonical_receipt']['path']);record=a.read(path)
        with np.load(a.local(ROOT,row['cache']['path']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        cls.case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=a.read(a.local(ROOT,row['command_receipt']['path'])),command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
        cls.source=adapter.interaction_source(cls.case)[0]

    def test_valid_original_canonical_still_has_same_raw_arrays_and_events(self):
        proof=goal.verify_original_source(self.case,self.source,self.rule)
        self.assertTrue(proof.derivation_proof['actual_raw_source_verified']);self.assertEqual(proof.references,())
        self.assertEqual(proof.original_masks_sha256,goal.fingerprint(self.case['arrays']))

    def test_unused_record_outcome_driver_clock_or_parent_cannot_be_replaced(self):
        for change in ('outcome','driver','clock','parent'):
            case=deepcopy(self.case)
            if change=='outcome':case['record']['alignment']['native_outcome']['success']=not case['record']['alignment']['native_outcome']['success']
            elif change=='driver':case['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']['v'][0]+=.00000001
            elif change=='clock':case['record']['alignment']['native_terminal_tick']=123456
            else:case['record']['parent_source_group']='train:push_moving_button:episode99999'
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'canonical record or entry'):
                goal.verify_original_source(case,self.source,self.rule)

    def test_entry_must_be_exact_canonical_entry_even_when_record_is_unchanged(self):
        case=deepcopy(self.case);case['entry']=dict(case['entry'],diagnostic_extra_field='unbound')
        with self.assertRaisesRegex(ValueError,'canonical record or entry'):goal.verify_original_source(case,self.source,self.rule)


class CountedAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=GoalCanonicalGuardTests.rule if hasattr(GoalCanonicalGuardTests,'rule') else goal.load_rule(ROOT,a.pointer(ROOT,BASE/'command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'),purpose='production_fitting')
        cls.registry=a.pointer(ROOT,ROOT/goal.OWNED_COUNTED_REGISTRY_PATH)
        cls.actual=counted.verify_registration(ROOT,cls.registry,live=False,require_ready=True)
        cls.auditor=a.pointer(ROOT,Path(counted.__file__))

    def fixture(self,temp):
        context=deepcopy(self.actual);collection=Path(temp)/'synthetic_admission_collection';collection.mkdir()
        context['current']['collection']=collection
        rows=context['plan']['cases'];key=context['plan']['plan_sha256'];ids={c['case_id'] for c in rows}
        value=dict(version=counted.MANIFEST_VERSION,complete=True,expected=100,completed=100,missing_case_ids=[],
            execution_registration=self.registry,entries=[deepcopy(c) for c in rows])
        path=collection/'manifest.json';path.write_text(json.dumps(value))
        proof=dict(audit_passed=True,complete=True,base_plan_sha256=next(iter(self.rule.contexts.values()))['plan']['plan_sha256'],
            primary_plan=next(iter(self.rule.contexts.values()))['registration']['plan'],primary_Context_view_unmodified=True,primary_K1_roster_unmodified=True,
            acquisition_projection=value,per_case_actual_plan_sha256={cid:key for cid in ids},
            per_case_feedback_audit={cid:dict(actual_feedback_arrays_verified=True) for cid in ids},verified_files_sha256={})
        row=dict(registration=self.registry,manifest=a.pointer(ROOT,path),auditor=self.auditor)
        return context,proof,row

    def load(self,context,proof,rows):
        with patch.object(counted,'verify_registration',return_value=context) as vr,patch.object(counted,'audit_manifest',return_value=proof) as am:
            result=goal._load_counted_teacher_contexts(ROOT,rows,self.rule.contexts,{})
            vr.assert_called_once_with(ROOT,self.registry,live=False,require_ready=True)
            am.assert_called_once_with(ROOT,rows[0]['manifest']['path'],require_complete=True)
            return result

    def test_complete100_projection_has_separate_context_and_requires_complete_true_api(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            context,proof,row=self.fixture(temp);loaded=self.load(context,proof,[row])
            self.assertEqual(set(loaded),{context['plan']['plan_sha256']})
            self.assertNotIn(context['plan']['plan_sha256'],self.rule.contexts)
            self.assertEqual(loaded[context['plan']['plan_sha256']]['pointer'],self.registry)

    def test_missing_duplicate_unknown_registry_auditor_and_partial_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            context,proof,row=self.fixture(temp)
            for change in ('missing','duplicate','registry','auditor','extra','partial'):
                rows=[deepcopy(row)]
                if change=='missing':rows=[]
                elif change=='duplicate':rows*=2
                elif change=='registry':rows[0]['registration']['sha256']='f'*64
                elif change=='auditor':rows[0]['auditor']['sha256']='e'*64
                elif change=='extra':rows[0]['allow_partial']=True
                else:
                    p=a.local(ROOT,rows[0]['manifest']['path']);v=a.read(p);v['complete']=False;p.write_text(json.dumps(v));rows[0]['manifest']=a.pointer(ROOT,p)
                with self.subTest(change=change),patch.object(counted,'audit_manifest') as audit,self.assertRaises(ValueError):
                    goal._load_counted_teacher_contexts(ROOT,rows,self.rule.contexts,{})
                audit.assert_not_called()

    def test_wrong_base_context_feedback_parent_coverage_or_auditor_closure_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            context,proof,row=self.fixture(temp)
            for change in ('base','Context','plan_map','feedback_missing','feedback_false','parents','source','archive'):
                c=deepcopy(context);p=deepcopy(proof)
                if change=='base':p['base_plan_sha256']='a'*64
                elif change=='Context':p['primary_Context_view_unmodified']=False
                elif change=='plan_map':p['per_case_actual_plan_sha256'][next(iter(p['per_case_actual_plan_sha256']))]='b'*64
                elif change=='feedback_missing':p['per_case_feedback_audit'].pop(next(iter(p['per_case_feedback_audit'])))
                elif change=='feedback_false':p['per_case_feedback_audit'][next(iter(p['per_case_feedback_audit']))]['actual_feedback_arrays_verified']=False
                elif change=='parents':p['acquisition_projection']['entries'][0]['parent_source_group']='train:wrong:episode0'
                elif change=='source':c['current']['sources'][str(Path(counted.__file__).resolve())]['sha256']='c'*64
                else:c['registry']['execution']['source_snapshot']['sha256']='d'*64
                with self.subTest(change=change),patch.object(counted,'verify_registration',return_value=c),patch.object(counted,'audit_manifest',return_value=p),self.assertRaises(ValueError):
                    goal._load_counted_teacher_contexts(ROOT,[row],self.rule.contexts,{})

    def test_actual_committed_counted_case_uses_real_feedback_auditor_and_stays_unknown_without_context(self):
        context=self.actual;case=next(c for c in context['plan']['cases'] if (context['current']['collection']/'cases/train'/c['task']/c['case_id']/'case.json').exists())
        path=context['current']['collection']/'cases/train'/case['task']/case['case_id']/'case.json';record=a.read(path);entry=record['entry']
        with np.load(a.local(ROOT,entry['cache']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        own=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=a.read(a.local(ROOT,entry['teacher_command_receipt'])),command_receipt_verified=True,command_receipt_sha256=record['files_sha256'][entry['teacher_command_receipt']])
        source=adapter.interaction_source(own)[0]
        with self.assertRaisesRegex(ValueError,'unknown actual execution plan'):goal.verify_original_source(own,source,self.rule)
        # Deliberately isolated unit contract: full100 admission is not claimed;
        # per-case raw/source/parent/10feedback validation below is unmocked.
        ptr=dict(path=str(context['current']['collection']/'manifest.json'),sha256='f'*64)
        augmented=replace(self.rule,counted_teacher_contexts={context['plan']['plan_sha256']:dict(pointer=self.registry,manifest=ptr,auditor=self.auditor,context=context,whole100_proof={})})
        verified=goal.verify_original_source(own,source,augmented)
        self.assertTrue(verified.derivation_proof['counted_teacher_acquisition']['whole100_complete'])
        self.assertFalse(verified.derivation_proof['counted_teacher_acquisition']['is_infrastructure_recovery'])
        self.assertEqual(verified.derivation_proof['actual_source_snapshot'],context['current']['registration']['source_snapshot'])
        derived=goal.derive_source(own,source,augmented)
        self.assertEqual(derived.derivation_proof['actual_source_snapshot'],context['current']['registration']['source_snapshot'])
        self.assertIn('counted_teacher_acquisition',goal.evidence_proof(derived))

if __name__=='__main__':unittest.main()
