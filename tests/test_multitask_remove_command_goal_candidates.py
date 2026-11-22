"""Actual TRAIN candidates remain separate from raw and runtime references."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,remove_command_goal_candidates as candidate
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/framework/multitask_r10'


class RemoveReferenceCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=goal.load_rule(ROOT,a.pointer(ROOT,BASE/'command_goal_reference_rule_v5_translation_revision3_setup/command_goal_rule_original_only.json'),purpose='production_fitting')
        cls.permission=a.pointer(ROOT,BASE/'remove_reference_independent_actual200_CPU_candidate_v1/candidate_permission.json');cls.cases={}
        for task in candidate.TASKS:
            for row in a.read(BASE/f'full14_pending_reference_readonly_diagnostic_v6/{task}/cases.json'):
                commands=a.read(a.local(ROOT,row['command_receipt']['path']))
                if not any(e['phase']=='acquire' for e in commands['phase_completion_events']):continue
                path=a.local(ROOT,row['canonical_receipt']['path']);record=a.read(path)
                with np.load(a.local(ROOT,row['cache']['path']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                cls.cases[task]=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=commands,
                    command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256']);break

    def derive(self,case,rotating=False,permission=None):
        return candidate.derive_candidate(case,adapter.interaction_source(case)[0],self.rule,permission or self.permission,include_unqualified_rotating=rotating)

    def test_acquire_candidate_exposes_spoke_and_original_cup_bindings_separately(self):
        for case in self.cases.values():
            original=adapter.interaction_source(case)[0];before=goal.fingerprint(case['arrays']);e=self.derive(case)
            self.assertTrue(e.references);self.assertEqual({r.phase for r in e.references},{'acquire'})
            for r in e.references:
                self.assertEqual(r.binding.reference_role,'selected_spoke');self.assertEqual(r.original_binding.reference_role,'selected_cup')
                self.assertTrue(r.source_semantics_qualified);self.assertFalse(r.provenance['runtime_enabled'])
            self.assertEqual(before,goal.fingerprint(case['arrays']))
            self.assertEqual(list(original.command_ledger),case['command_receipt']['commands'])

    def test_all_real_acquire_raw_goals_share_source_spoke_local_geometry(self):
        for case in self.cases.values():
            e=self.derive(case);ledger={c['command_id']:c for c in case['command_receipt']['commands']};relative=[]
            for r in e.references:
                pose=Pose.from_xyzw(r.intent_reference_xyzw);raw=Pose.from_xyzw(ledger[r.command_id]['raw_waypoint_xyzw'])
                relative.append((pose.rotation.T@(raw.position-pose.position),pose.rotation.T@raw.rotation))
            self.assertTrue(relative)
            for p,r in relative:
                np.testing.assert_allclose(p,relative[0][0],atol=1e-6,rtol=0)
                self.assertLess(Rotation.from_matrix(r@relative[0][1].T).magnitude(),1e-6)

    def test_rotating_numerical_hypothesis_keeps_unrecorded_hierarchy_unqualified(self):
        for case in self.cases.values():
            e=self.derive(case,True);rr=[r for r in e.references if r.phase!='acquire'];self.assertTrue(rr)
            self.assertTrue(all(not r.source_semantics_qualified and r.provenance['waypoint_parent_hierarchy_proven'] is False for r in rr))
            with self.assertRaises(ValueError):replace(rr[0],source_semantics_qualified=True)

    def test_candidate_cannot_enter_production_source_or_runtime_reference(self):
        case=self.cases[candidate.TASKS[0]];e=self.derive(case);source=adapter.interaction_source(case)[0]
        with self.assertRaises(ValueError):replace(source,command_goal_reference_evidence=e)
        with self.assertRaises(ValueError):goal.freeze_runtime_reference(candidate_forecast_pose=e.references[0],predictor='context',observed_until=0,selected_at=1,forecast_time=2)
        with self.assertRaises(ValueError):replace(e.references[0],binding=e.references[0].original_binding)

    def test_fake_hierarchy_runtime_permission_or_unknown_field_rejected(self):
        original=a.read(a.local(ROOT,self.permission['path']))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            for change in ('hierarchy','runtime','unknown','source'):
                d=deepcopy(original)
                if change=='hierarchy':d['waypoint_parent_hierarchy_recorded']=True
                elif change=='runtime':d['runtime_enabled']=True
                elif change=='unknown':d['silent_spec_override']=True
                else:d['task_sources'][candidate.TASKS[0]]['sha256']='f'*64
                path=Path(temp)/(change+'.json');path.write_text(json.dumps(d))
                with self.subTest(change=change),self.assertRaises(ValueError):self.derive(self.cases[candidate.TASKS[0]],permission=a.pointer(ROOT,path))

    def test_changed_raw_arrays_commands_or_canonical_outcome_rejected(self):
        for change in ('array','command','outcome'):
            case=deepcopy(self.cases[candidate.TASKS[1]])
            if change=='array':case['arrays']['target_rotations'][1,0]=np.eye(3)
            elif change=='command':case['command_receipt']['commands'][0]['reference_body']='spoke0'
            else:case['record']['alignment']['native_outcome']['success']=not case['record']['alignment']['native_outcome']['success']
            with self.subTest(change=change),self.assertRaises(ValueError):self.derive(case)

if __name__=='__main__':unittest.main()
