from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import json
import unittest
from our_mani.multitask.evaluation_roster16_v283 import (
    TASKS,METHODS,validate_roster,case_grid,route,core14_selection_scope)
from our_mani.multitask.protocol import MultitaskProtocolError

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'artifacts/framework/protocol'
        cls.train=json.loads((base/'train_manifest.json').read_text())
        cls.val=json.loads((base/'validation_manifest.json').read_text())

    def test_actual1600_TRAIN_and315_VAL_cover_all16_and_three_methods(self):
        self.assertEqual(len(validate_roster(self.train,'train')),1600)
        grid=case_grid(self.val,'validation')
        self.assertEqual(len(grid),945)
        self.assertEqual(len({(r['task'],r['raw_episode_id'],r['method']) for r in grid}),945)
        self.assertEqual({r['task'] for r in grid},set(TASKS))
        self.assertEqual({r['method'] for r in grid},set(METHODS))

    def test_old14_task_subset_or_duplicate_cannot_masquerade_as_full16(self):
        subset=deepcopy(self.val)
        subset['episodes']=[r for r in subset['episodes'] if route(r['task'],'cv_geometry').component=='r10_multitask']
        with self.assertRaises(MultitaskProtocolError):validate_roster(subset,'validation')
        duplicate=deepcopy(self.val);duplicate['episodes'][-1]=deepcopy(duplicate['episodes'][0])
        with self.assertRaises(MultitaskProtocolError):validate_roster(duplicate,'validation')

    def test_value_NGFF_and_FASTER_are_not_routes(self):
        for method in ('cv_value','context_value','NGFF','FASTER'):
            with self.assertRaises(MultitaskProtocolError):route(TASKS[0],method)

    def test_core14_scope_keeps_full_source_rows_and_rejects_wrong_component(self):
        old=lambda *args:None;evaluator=NS(select_source=old)
        first=next(r for r in self.train['episodes'] if route(r['task'],'cv_geometry').component=='r10_multitask')
        with core14_selection_scope(evaluator):
            (row,original),values=evaluator.select_source(self.train,'train',first['task'],first['raw_episode_id'])
            self.assertEqual(row,first);self.assertIs(original,first);self.assertEqual(len(values),1600)
            block=next(r for r in self.train['episodes'] if route(r['task'],'cv_geometry').component=='aligned_block')
            with self.assertRaises(MultitaskProtocolError):evaluator.select_source(self.train,'train',block['task'],block['raw_episode_id'])
        self.assertIs(evaluator.select_source,old)

    def test_TEST_selection_is_closed_without_VAL_audit_without_reading_any_TEST_roster(self):
        with self.assertRaises(MultitaskProtocolError):validate_roster({'split':'test','episodes':[]},'test')


if __name__=='__main__':unittest.main()
