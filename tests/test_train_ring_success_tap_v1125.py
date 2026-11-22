"""Synthetic original-call profiling checks; never import or construct Native."""
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from our_mani.multitask.train_ring_success_tap_v1125 import (
    PassiveRingSuccessTap, TASKS, observer_context_factory, summarize,
)


def original_classes():
    namespace = {'__name__':'rlbench.backend.conditions'}
    code = '''
class DetectedCondition:
    def __init__(self, detector):
        self._detector=detector
        self._obj=object()
    def condition_met(self):
        met=self._detector.is_detected(self._obj)
        return met,False
class ConditionSet:
    def __init__(self, conditions, order_matters=False):
        self._conditions=conditions
        self._order_matters=order_matters
        self._simultaneously_met=True
        self._current_condition_index=0
    def condition_met(self):
        met=True
        if self._order_matters:
            if self._current_condition_index<len(self._conditions):
                for cond in self._conditions[self._current_condition_index:]:
                    ismet,term=cond.condition_met()
                    if not ismet:break
                    self._current_condition_index+=1
                met=self._current_condition_index>=len(self._conditions)
        else:
            for cond in self._conditions:
                ismet,term=cond.condition_met()
                met &= ismet
        return met,False
'''
    exec(compile(code,'/frozen/rlbench/backend/conditions.py','exec'),namespace)
    taskspace={'__name__':'rlbench.tasks.insert_onto_rotating_peg'}
    taskcode='''
class InsertOntoRotatingPeg:
    def __init__(self, condition):
        self._fail_conditions=[]
        self._success_conditions=[condition]
    def success(self):
        all_fails_met=all([cond.condition_met()[0] for cond in self._fail_conditions])
        if len(self._fail_conditions)>0 and all_fails_met:return False,True
        all_met=all([cond.condition_met()[0] for cond in self._success_conditions])
        return all_met,all_met
'''
    exec(compile(taskcode,'/frozen/rlbench/backend/task.py','exec'),taskspace)
    return namespace['DetectedCondition'],namespace['ConditionSet'],taskspace['InsertOntoRotatingPeg']


class Sensor:
    def __init__(self, result, *, error=None):
        self.result=result;self.error=error;self.calls=0
    def is_detected(self,obj):
        self.calls+=1
        if self.error is not None:raise self.error
        return self.result
    def get_name(self):raise AssertionError('extra Native name getter forbidden')
    def get_position(self):raise AssertionError('extra Native pose getter forbidden')
    def get_handle(self):raise AssertionError('extra Native identity getter forbidden')


def fixture(values=(True,False,True,True), *, order_matters=False, error_index=None):
    detected,group,task=original_classes()
    sensors=[Sensor(value,error=RuntimeError('original sensor exception') if i==error_index else None)
             for i,value in enumerate(values)]
    condition=group([detected(sensor) for sensor in sensors],order_matters=order_matters)
    return task(condition),sensors


class Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNone(sys.getprofile())
        self.addCleanup(sys.setprofile,None)

    def test_four_original_leaves_evaluated_once_even_after_false_and_labeled_by_registration(self):
        task,sensors=fixture()
        tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry')
        with tap:self.assertEqual(task.success(),(False,False))
        self.assertEqual([s.calls for s in sensors],[1,1,1,1])
        receipt=tap.receipt();self.assertTrue(receipt['restored']);self.assertEqual(receipt['observer_errors'],0)
        self.assertEqual((tap.success_calls,tap.success_returns,tap.condition_calls,tap.condition_returns),(1,1,5,5))
        calls=[e for e in receipt['events'] if e['event']=='original_condition_met_call']
        self.assertEqual([e['condition_path'] for e in calls],[['success',0],*([['success',0,i] for i in range(4)])])
        self.assertEqual([e['registered_leaf_ordinal'] for e in calls[1:]],[0,1,2,3])
        self.assertEqual([e['source_condition_label'] for e in calls[1:]],['success_detector'+str(i) for i in range(4)])
        returned=receipt['events'][-1]
        self.assertEqual(returned['leaf_returns'],[dict(ordinal=i,result=[v,False]) for i,v in enumerate((True,False,True,True))])
        self.assertTrue(returned['original_four_detector_registration'])
        self.assertEqual(summarize(receipt)['leaves'][1]['false_returns'],1)
        self.assertFalse(summarize(receipt)['absence_is_unknown'])
        json.dumps(receipt,allow_nan=False)  # No Native/Python object reference escapes the tap.

    def test_actual_ordered_short_circuit_is_observed_without_forcing_uninvoked_leaves(self):
        task,sensors=fixture((False,True,True,True),order_matters=True)
        with PassiveRingSuccessTap(TASKS[0],'context_geometry') as tap:
            self.assertEqual(task.success(),(False,False))
        self.assertEqual([s.calls for s in sensors],[1,0,0,0])
        row=tap.receipt()['events'][-1]
        self.assertEqual(row['condition_call_sequence'],[['success',0],['success',0,0]])
        self.assertFalse(row['original_four_detector_registration'])
        self.assertEqual(tap.condition_calls,2)
        self.assertTrue(summarize(tap.receipt())['absence_is_unknown'])

    def test_original_exception_propagates_unknown_return_and_profile_cleanup(self):
        task,sensors=fixture(error_index=1);tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry')
        with self.assertRaisesRegex(RuntimeError,'original sensor exception'):
            with tap:task.success()
        self.assertEqual([s.calls for s in sensors],[1,1,0,0])
        self.assertIsNone(sys.getprofile());self.assertTrue(tap.receipt()['restored'])
        self.assertEqual(tap.receipt()['scope_exception_type'],'RuntimeError')
        self.assertGreater(tap.nonstandard_returns,0)
        self.assertEqual(tap._success_frames,{});self.assertEqual(tap._condition_frames,{})
        self.assertTrue(summarize(tap.receipt())['absence_is_unknown'])

    def test_bounded_drops_keep_original_calls_and_unknown_absence(self):
        task,sensors=fixture();tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry',max_events=2)
        with tap:
            task.success();task.success()
        self.assertEqual([s.calls for s in sensors],[2,2,2,2])
        self.assertEqual(len(tap.events),2);self.assertEqual(tap.dropped_events,22)
        self.assertEqual((tap.success_returns,tap.condition_returns),(2,10))
        self.assertTrue(summarize(tap.receipt())['absence_is_unknown'])

    def test_existing_profiler_is_not_overwritten_and_scope_is_single_use(self):
        other=lambda *args:None
        sys.setprofile(other)
        tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry')
        with self.assertRaisesRegex(RuntimeError,'existing profiler'):tap.__enter__()
        self.assertIs(sys.getprofile(),other);sys.setprofile(None)
        with tap:pass
        self.assertTrue(summarize(tap.receipt())['absence_is_unknown'])
        with self.assertRaisesRegex(RuntimeError,'one ring telemetry scope'):tap.__enter__()

    def test_another_profiler_installed_inside_scope_is_preserved(self):
        other=lambda *args:None;tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry')
        with tap:sys.setprofile(other)
        self.assertIs(sys.getprofile(),other);self.assertFalse(tap.receipt()['restored'])

    def test_observer_errors_never_replace_original_success_or_predicates(self):
        task,sensors=fixture((True,True,True,True));tap=PassiveRingSuccessTap(TASKS[0],'cv_geometry')
        with patch.object(tap,'_observe',side_effect=ValueError('telemetry failure')):
            with tap:self.assertEqual(task.success(),(True,True))
        self.assertEqual([s.calls for s in sensors],[1,1,1,1]);self.assertGreater(tap.observer_errors,0)
        self.assertIsNone(sys.getprofile());self.assertTrue(tap.receipt()['restored'])

    def test_context_uses_existing_ledger_and_action_storage_only(self):
        task,_=fixture()
        class Env:
            def __init__(self):
                self.task=task;self.ledger=NS(ticks=11)
                self._native_action_sampling=NS(active=dict(action_index=3,label='geometry_execute',start_tick=10))
            def _tick(self,phase):return self.task.success()
        with PassiveRingSuccessTap(TASKS[0],'cv_geometry') as tap:Env()._tick('native_task_tick')
        row=tap.events[-1]
        self.assertEqual((row['counted_tick'],row['action_index'],row['action_label'],row['action_start_tick']),
                         (11,3,'geometry_execute',10))
        self.assertEqual(row['native_tick_phase'],'native_task_tick')

    def test_callback_requires_parent_permit_and_records_evaluation_only_receipt(self):
        selection=dict(task=TASKS[0],method='cv_geometry');events={}
        with self.assertRaisesRegex(ValueError,'parent-owned'):
            with observer_context_factory(None,None,selection,events):pass
        task,sensors=fixture()
        with observer_context_factory(None,None,selection,events,permit=object()):task.success()
        self.assertTrue(events['restored']);self.assertEqual(len(events['observer_receipts']),1)
        self.assertEqual([s.calls for s in sensors],[1,1,1,1])
        receipt=events['observer_receipts'][0]
        self.assertFalse(receipt['native_truth_passed_to_policy']);self.assertFalse(receipt['native_object_names_handles_or_poses_read'])


if __name__=='__main__':unittest.main()
