"""CPU boundaries: missing priors and indirect poses cannot become observations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

script=Path(__file__).resolve().parents[1]/'scripts/prepare_rigid_orientation_pilot_r10.py'
spec=importlib.util.spec_from_file_location('rigid_pilot_tested',script)
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


class RigidPilotTests(unittest.TestCase):
    def test_missing_completed_train_evidence_never_creates_rigid_prior(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'artifacts';base.mkdir()
            with patch.object(prep,'ROOT',root),patch.object(prep.pilot,'BASE',base):
                report=prep.audit_priors(base/'no_completed_cases',base/'manifest.json')
            assert report['layout_map']=={} and len(report['task_status'])==14
            assert report['task_success_used_for_prior_selection'] is False

    def test_prior_digest_failure_blocks_plan_preparation_before_any_runtime_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'raw.npz';source.write_bytes(b'changed raw TRAIN labels')
            manifest=root/'prior.json';manifest.write_text(json.dumps(dict(TRAIN_only=True,TEST_used=False,
                inputs_sha256={str(source):'0'*64})))
            with patch.object(prep.pilot,'build_plan') as build:
                with self.assertRaisesRegex(ValueError,'TRAIN prior changed'):
                    prep.prepare_plan('future collection',manifest,root/'plan.json')
                build.assert_not_called()

    def test_unseen_child_group_pose_is_indirect_and_visibility_cannot_be_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'artifacts';base.mkdir();directory=base/'pilot';directory.mkdir()
            direct='candidate_full_except_ball_none__direct';group='candidate_full_except_ball_none__rigid_group'
            results=[]
            for task in prep.TASK_SPECS:
                report=dict(task=task,variants=[direct,group],frames=[dict(metric_only=dict(body_ids=['spoke2']),
                    tracker_diagnostics={direct:dict(bodies={'spoke2':dict(registration_valid=False,directly_visible=False,pose_source='past_evidence_only')}),
                        group:dict(bodies={'spoke2':dict(registration_valid=True,directly_visible=False,
                            pose_source='current_group_RGBD_and_declared_rigid_layout')})})])
                p=directory/(task+'.json');p.write_text(json.dumps(report))
                results.append(dict(task=task,completed=True,report=str(p.relative_to(root)),report_sha256=prep.pilot.sha(p)))
            queue=directory/'queue_report.json';queue.write_text(json.dumps(dict(complete=True,results=results)))
            with patch.object(prep,'ROOT',root),patch.object(prep.pilot,'BASE',base):
                report=prep.indirect_evidence(directory,base/'indirect.json')
                for task,variants in report['cases'].items():
                    row=variants[group]['spoke2']
                    assert row['group_indirect_not_visible_frames']==1 and row['group_visible_frames']==0
                p=root/results[0]['report'];data=json.loads(p.read_text())
                data['frames'][0]['tracker_diagnostics'][group]['bodies']['spoke2']['directly_visible']=True
                p.write_text(json.dumps(data));results[0]['report_sha256']=prep.pilot.sha(p)
                queue.write_text(json.dumps(dict(complete=True,results=results)))
                with self.assertRaisesRegex(ValueError,'direct visibility'):
                    prep.indirect_evidence(directory,base/'false_visibility.json')


if __name__=='__main__':unittest.main()
