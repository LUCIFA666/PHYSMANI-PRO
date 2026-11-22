"""Synthetic receipt validation only; no simulator, files, VAL or TEST data."""
from copy import deepcopy
import unittest
from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case,VERSION,CORE_SCHEMA,BLOCK_SCHEMA
from our_mani.multitask.joint16_tracks_v31 import BLOCKS

PROTOCOL=dict(schema='r10_author_action_outcome_protocol_v741',implementation={'path':'fixture741','sha256':'a'*64},
    runtime_consumer={'path':'fixture744','sha256':'b'*64},original_runtime={'path':'fixture664','sha256':'c'*64},
    native_RML_parameters_unchanged=True)
NONE=dict(success=False,terminate=False,terminal_failure=False)
SUCCESS=dict(success=True,terminate=True,terminal_failure=False)
FATAL=dict(success=False,terminate=True,terminal_failure=True)


def fixture(*,block=False,method='context_geometry',fatal=False,success=False,hold=False,ticks=10):
    outcome=deepcopy(SUCCESS if success else FATAL if fatal else NONE)
    label='baseline_execute' if method=='physmani' else 'public_hold' if hold else 'geometry_execute'
    row=dict(label=label,start_tick=0,end_tick=ticks,ticks=ticks,max_ticks=1 if hold else 10,
        sampled_outcome=deepcopy(outcome),complete=True)
    receipt=dict(schema=BLOCK_SCHEMA if block else CORE_SCHEMA,completed_actions=[row] if ticks else [],
        last_tick=ticks,active_action=None,aborted=False,global_tick_modulus_used=False)
    if block:
        row['extra_task_success_calls']=0
        receipt.update(original_native_success_latch_preserved=True,native_failure_conditions=0,extra_task_success_calls=0)
    else:
        row.update(action_index=0,current_native_result=deepcopy(outcome))
        receipt.update(success_sampled_each_tick=True,failure_sampled_at_explicit_action_end=True,
            reset_failure_is_diagnostic_only=True,reset_success_semantics_preserved=True,aborted_actions=[],
            sampled_outcome=deepcopy(outcome),initial_native_result=deepcopy(SUCCESS if not ticks and success else NONE),
            intermediate_failure_samples=int(fatal))
    alignment=dict(native_success_latched=success,native_terminal_failure_latched=fatal,native_outcome_sampling=deepcopy(receipt))
    if not block:alignment.update(native_terminal_latched=success or fatal,native_outcome=deepcopy(outcome),
        native_terminal_tick=ticks if success or fatal else None)
    return dict(version=VERSION,complete=True,cleanup_completed=True,infrastructure_error=None,
        task=next(iter(BLOCKS)) if block else 'beat_the_rotating_buzz',method=method,
        counted_clock={'ticks':ticks},official_task_success=success,native_terminal_failure=fatal,
        native_outcome_sampling=receipt,author_outcome_protocol=deepcopy(PROTOCOL),alignment_receipt=alignment)


def sync(case):case['alignment_receipt']['native_outcome_sampling']=deepcopy(case['native_outcome_sampling'])


class ReceiptTests(unittest.TestCase):
    def check(self,case,block=False):return require_sampling_case(case,PROTOCOL,block=block)
    def reject(self,case,block=False):
        with self.assertRaises(ValueError):self.check(case,block)

    def test_native_success_failure_and_nonterminal_budget_are_all_valid(self):
        for flags in ({},{'fatal':True},{'success':True},{'method':'physmani','fatal':True},{'hold':True,'ticks':1}):
            with self.subTest(flags=flags):self.assertIs(self.check(case:=fixture(**flags)),case['native_outcome_sampling'])

    def test_block_schema_keeps_zero_extra_calls_and_no_fatal(self):
        for method in ('cv_geometry','context_geometry','physmani'):
            self.check(fixture(block=True,method=method,success=True),True)
        for key in ('native_failure_conditions','extra_task_success_calls'):
            for invalid in (True,1,None):
                case=fixture(block=True);case['native_outcome_sampling'][key]=invalid;sync(case);self.reject(case,True)
        self.reject(fixture(block=True,fatal=True),True)

    def test_protocol_requires_exact_SHA_and_boolean_types(self):
        case=fixture();case['author_outcome_protocol']['implementation']['sha256']='f'*64;self.reject(case)
        case=fixture();case['author_outcome_protocol']['native_RML_parameters_unchanged']=1;self.reject(case)

    def test_incomplete_aborted_or_safety_exception_receipts_cannot_be_clean(self):
        for mutate in (lambda c:c.update(complete=False),lambda c:c.update(cleanup_completed=False),
            lambda c:c.update(infrastructure_error={'type':'GeometryTargetRejected'}),
            lambda c:c['native_outcome_sampling'].update(aborted=True),
            lambda c:c['native_outcome_sampling'].update(active_action={'start_tick':0}),
            lambda c:c['native_outcome_sampling'].update(aborted_actions=[{'complete':False}]),
            lambda c:c['native_outcome_sampling']['completed_actions'][0].update(complete=False)):
            case=fixture();mutate(case);sync(case);self.reject(case)

    def test_action_interval_holes_overlap_counts_and_allowances_are_rejected(self):
        for update in ({'start_tick':1},{'end_tick':9},{'ticks':9},{'max_ticks':0},{'max_ticks':11},{'max_ticks':True},
                       {'label':'public_hold'},{'action_index':1},{'action_index':True}):
            case=fixture();case['native_outcome_sampling']['completed_actions'][0].update(update);sync(case);self.reject(case)
        case=fixture();case['native_outcome_sampling']['last_tick']=9;sync(case);self.reject(case)
        case=fixture();case['native_outcome_sampling']['completed_actions']=[];sync(case);self.reject(case)

    def test_hold_then_non_aligned_pose_and_partial_action_cover_exact_ticks(self):
        case=fixture(ticks=3);receipt=case['native_outcome_sampling'];first=receipt['completed_actions'][0]
        first.update(label='public_hold',end_tick=1,ticks=1,max_ticks=1)
        second=deepcopy(first);second.update(label='geometry_execute',action_index=1,start_tick=1,end_tick=3,ticks=2,max_ticks=10)
        receipt['completed_actions'].append(second);sync(case);self.check(case)
        second['start_tick']=0;sync(case);self.reject(case)

    def test_method_labels_and_terminal_action_cannot_be_followed(self):
        case=fixture(method='physmani');case['native_outcome_sampling']['completed_actions'][0]['label']='geometry_execute';sync(case);self.reject(case)
        case=fixture(fatal=True,ticks=2);row=case['native_outcome_sampling']['completed_actions'][0]
        row.update(end_tick=1,ticks=1);later=deepcopy(row);later.update(action_index=1,start_tick=1,end_tick=2)
        case['native_outcome_sampling']['completed_actions'].append(later);sync(case);self.reject(case)

    def test_current_end_failure_cannot_OR_old_failure_but_success_remains_latched(self):
        case=fixture(fatal=True);case['native_outcome_sampling']['completed_actions'][0]['current_native_result']=deepcopy(NONE)
        sync(case);self.reject(case)
        case=fixture(success=True);case['native_outcome_sampling']['completed_actions'][0]['current_native_result']=deepcopy(FATAL)
        sync(case);self.check(case)

    def test_top_native_fields_alignment_and_terminal_tick_must_agree(self):
        for mutate in (lambda c:c.update(official_task_success=True),lambda c:c.update(native_terminal_failure=True),
            lambda c:c['alignment_receipt'].update(native_success_latched=True),
            lambda c:c['alignment_receipt'].update(native_terminal_failure_latched=True),
            lambda c:c['alignment_receipt'].update(native_terminal_latched=True),
            lambda c:c['native_outcome_sampling'].update(sampled_outcome=deepcopy(FATAL)),
            lambda c:c['native_outcome_sampling'].update(global_tick_modulus_used=True)):
            case=fixture();mutate(case);sync(case);self.reject(case)
        case=fixture(fatal=True);case['alignment_receipt']['native_terminal_tick']=3;self.reject(case)

    def test_zero_action_reset_success_or_policy_negative_without_fatal(self):
        for block in (False,True):
            for success in (False,True):self.check(fixture(block=block,ticks=0,success=success),block)
        self.reject(fixture(ticks=0,fatal=True))
        case=fixture(ticks=0,success=True);case['native_outcome_sampling']['initial_native_result']=deepcopy(NONE);sync(case);self.reject(case)

    def test_original_runtime_version_and_registered_block_identity_are_required(self):
        case=fixture();case['version']='r10_full16_actual_split_common_author_RML_v664';self.reject(case)
        self.reject(fixture(),True);self.reject(fixture(block=True),False)


if __name__=='__main__':unittest.main()
