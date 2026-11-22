"""Invented six-tier grids only: no actual admission, cases or model input."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from our_mani.multitask import composite_TEST_reboot_resume_v1055 as core
from our_mani.multitask import composite_TEST_resume_v1040 as parent
spec=importlib.util.spec_from_file_location('fixture1040_CPU1055',Path(__file__).with_name('test_composite_TEST_resume_v1040.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def fixture(saved=1413,rejected=2,parent_completed=1409):
    prior,p1011,s1011,t1011,*rest=old.fixture(parent_completed,0,1266,584)
    rows=parent.resumed_jobs(p1011,prior,prior['execution_entry'])
    plan=dict(p1011,jobs=rows,imported_completed_cases=parent_completed)
    records={r['job_id']:dict(complete=True,returncode=0,native_success=bool(i%3),motion_pass=None if i%7==0 else False,
        timed_out=False,interrupted=False,automatic_retry=False) for i,r in enumerate(rows[:saved])}
    status=dict(s1011,policy_sha256=plan['policy_sha256'],completed=records,failed={},active=[rows[saved]['job_id']],checking=[rows[saved+1]['job_id']],
        native_slots=1,GPU_slots=1,drain_requested=False,stop_requested=False)
    ids=list(records); accepted=set(ids[:saved-rejected]); damaged=set(ids[saved-rejected:])
    evidence=[dict(job_id=k,recoverable_complete=k in accepted,native_success=records[k]['native_success'] if k in accepted else None,
        errors=[] if k in accepted else ['missing_invalid_evidence']) for k in ids]
    audit=dict(schema='r10_readonly_host_reboot_TEST_evidence_observation_v1054',audit_complete=True,formal_full_TEST_complete=False,
        inputs_unchanged_through_audit=True,old_outputs_modified=False,missing_evidence_reconstructed=False,
        unknown_counted_as_NativeFalse=False,success_based_case_selection=False,new_terminal_claimed_for_interrupted_owner=False,
        Native_started=False,scientific_inputs_read=False,raw_data_read=False,model_read=False,full_source_graph_read=False,
        old_owner_process=dict(same_process_alive=False),completed_evidence=evidence,saved_status_completed_count=saved,
        recoverable_complete_count=len(accepted),damaged_registered_count=len(damaged),unregistered_clean_native_terminals=[],
        current_boot_id='00000000-0000-0000-0000-000000000001',still_unclosed_count=1920-len(accepted))
    for name, values in dict(accepted_completed_job_ids=accepted,rejected_saved_completed_job_ids=damaged,
            saved_completed_job_ids=set(ids),remaining_job_ids={r['job_id'] for r in rows}-accepted).items():
        audit[name]=sorted(values);audit[name+'_sha256']=core.digest(sorted(values))
    control=dict(prior,schema=core.CONTROL_SCHEMA,explicit_host_reboot_recovery=True)
    for name,values in dict(completed=accepted,failed=set(),remaining={r['job_id'] for r in rows}-accepted,
            saved_completed=set(ids),rejected_saved_completed=damaged).items():
        control[name+'_job_ids']=sorted(values);control[name+'_key_sha256']=core.digest(sorted(values))
    return control,plan,status,audit,prior,p1011,s1011,t1011


class RebootContractTests(unittest.TestCase):
    def test_full1413_literal_status_adopts1411_and_recovers509_without_drain(self):
        c,p,s,a,*_=fixture();before=deepcopy((c,p,s,a))
        self.assertEqual(len(core.require_partition(c,p,s,a)),1920)
        self.assertEqual(len(c['completed_job_ids']),1411);self.assertEqual(len(c['remaining_job_ids']),509)
        self.assertEqual(len(s['completed']),1413);self.assertEqual((c,p,s,a),before)
        self.assertFalse(s['drain_requested']);self.assertTrue(s['active'])

    def test_six_tier_literal_lineage_and_only_pending_out_changes(self):
        c,p,s,a,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        for r,oldrow in zip(rows,p['jobs']):
            for field in ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','previous_row_sha256'):
                self.assertEqual(r[field],oldrow[field])
            self.assertEqual(r[core.LINEAGE_FIELD]['parent_row_sha256'],core.digest(oldrow))
            if r['imported_completed']:self.assertEqual(core.parent_from_imported(r),oldrow)
            else:
                inner=deepcopy(oldrow['physical_argv']);inner[inner.index('--out')+1]=core.option(r['physical_argv'],'--out')
                self.assertEqual(inner,r['physical_argv']);self.assertEqual(r['evidence_runtime'],core.RUNTIME)
                self.assertEqual(r['argv'].count(core.BATCH_ENTRY),1)
        new=dict(jobs=rows,expected_cases=1920,imported_completed_cases=1411,source_outcome_filtering=False)
        core.require_resume_plan(c,p,s,a,new)
        self.assertEqual(sum(r['evidence_runtime']==core.PREVIOUS for r in rows),2)

    def test_no_fake_native_false_success_filter_or_completion_loss(self):
        for mode in ('unknown_false','filter','skipdamaged','missing_saved','audit_duplicate','fake_full','live_old','source_changed'):
            c,p,s,a,*_=fixture()
            if mode=='unknown_false':a['completed_evidence'][-1]['native_success']=False
            elif mode=='filter':c['completed_job_ids'].pop();c['completed_key_sha256']=core.digest(c['completed_job_ids'])
            elif mode=='skipdamaged':c['remaining_job_ids'].remove(c['rejected_saved_completed_job_ids'][0]);c['remaining_key_sha256']=core.digest(c['remaining_job_ids'])
            elif mode=='missing_saved':s['completed'].pop(next(iter(s['completed'])))
            elif mode=='audit_duplicate':a['completed_evidence'].append(deepcopy(a['completed_evidence'][0]))
            elif mode=='fake_full':a['formal_full_TEST_complete']=True
            elif mode=='live_old':a['old_owner_process']['same_process_alive']=True
            else:p['jobs'][0]['raw']='different'
            with self.subTest(mode=mode),self.assertRaises(ValueError):core.require_partition(c,p,s,a)

    def test_completed_false_and_unknown_motion_remain_literal(self):
        c,p,s,a,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        self.assertTrue(any(s['completed'][r['job_id']]['native_success'] is False for r in rows if r['imported_completed']))
        self.assertTrue(any(s['completed'][r['job_id']]['motion_pass'] is None for r in rows if r['imported_completed']))
        row=next(r for r in rows if r['imported_completed']);row['resume_lineage_v1040']={}
        with self.assertRaises(ValueError):core.parent_from_imported(row)

    def test_pure_contract_performs_no_filesystem_reads(self):
        c,p,s,a,*_=fixture()
        with patch('builtins.open',side_effect=AssertionError('no file IO')),patch.object(Path,'read_bytes',side_effect=AssertionError('no file IO')):
            core.require_partition(c,p,s,a);core.resumed_jobs(p,c,c['execution_entry'])


if __name__=='__main__':unittest.main()
