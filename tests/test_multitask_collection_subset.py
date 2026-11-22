"""A fixed integration subset must retain the full acquisition denominator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import collect_multitask_training_r10 as collection


class CollectionSubset(unittest.TestCase):
    def test_verified_subset_does_not_claim_full_plan_complete(self):
        plan = dict(protocol={}, cases=[dict(case_id=key,split='train',task='push_moving_button')
            for key in ('first','second')])
        collection.BASE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=collection.BASE) as directory:
            directory = Path(directory)
            first = collection.case_directory(directory, plan['cases'][0])/'case.json'
            first.parent.mkdir(parents=True); first.write_text('{}')
            with patch.object(collection,'verify_plan',return_value=plan), \
                 patch.object(collection,'require_r09_release',return_value={}), \
                 patch.object(collection,'validate_case_result',return_value={'process_id':123}), \
                 patch.object(collection,'merge_completed',return_value={'complete':False,'completed':1,'expected':2}), \
                 patch.object(collection.subprocess,'Popen') as native:
                result = collection.run_queue('frozen-plan',directory,'guard',case_ids=['first'],workers=1)
            report = json.loads((directory/'queue_report.json').read_text())
            self.assertFalse(result['complete']); self.assertFalse(report['complete'])
            self.assertTrue(report['selected_complete']); self.assertEqual(report['selected_expected'],1)
            self.assertEqual(report['expected'],2); native.assert_not_called()
            self.assertEqual([c['case_id'] for c in plan['cases']], ['first','second'])

    def test_unknown_empty_or_duplicate_subset_cannot_start_a_worker(self):
        plan = dict(protocol={},cases=[dict(case_id='first')])
        for ids in ([],['missing'],['first','first']):
            with patch.object(collection,'verify_plan',return_value=plan), \
                 patch.object(collection.subprocess,'Popen') as native:
                with self.assertRaises(collection.CollectionError):
                    collection.run_queue('plan','unused','guard',case_ids=ids)
                native.assert_not_called()

    def test_explicit_stop_marker_cancels_pending_cases_without_a_failure_label(self):
        plan = dict(protocol={}, cases=[dict(case_id='pending',split='train',task='push_moving_button')])
        with tempfile.TemporaryDirectory(dir=collection.BASE) as directory:
            directory = Path(directory); stop = directory/'STOP'; stop.touch()
            with patch.object(collection,'verify_plan',return_value=plan), \
                 patch.object(collection,'require_r09_release',return_value={}), \
                 patch.object(collection,'COLLECTION_STOP',stop), \
                 patch.object(collection,'merge_completed',return_value={'complete':False,'completed':0,'expected':1}), \
                 patch.object(collection.subprocess,'Popen') as native:
                collection.run_queue('plan',directory,'guard',workers=1)
            report = json.loads((directory/'queue_report.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['receipts'][0]['status'],'cancelled_stop_marker')
            self.assertFalse(report['receipts'][0]['policy_failure_label_assigned'])
            native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
