"""Pure invented seven-tier grids. No actual admission, cases or model input."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from our_mani.multitask import composite_TEST_reboot_resume_v1070 as core
from our_mani.multitask import composite_TEST_reboot_resume_v1055 as parent
spec=importlib.util.spec_from_file_location('fixture1055_CPU1070',Path(__file__).with_name('test_composite_TEST_reboot_resume_v1055.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def fixture(saved=1419, parent_completed=1411):
    prior,p1041,s1041,a1054,*rest=old.fixture(saved=1413,rejected=2,parent_completed=1409)
    rows=parent.resumed_jobs(p1041,prior,prior['execution_entry'])
    plan=dict(p1041,jobs=rows,imported_completed_cases=parent_completed)
    records={k:deepcopy(s1041['completed'][k]) for k in prior['completed_job_ids']}
    fresh=[r for r in rows if r['job_id'] not in records][:saved-parent_completed]
    for i,row in enumerate(fresh):
        records[row['job_id']]=dict(complete=True,returncode=0,native_success=i not in (1,4,7),
            motion_pass=None if i%3==0 else False,timed_out=False,interrupted=False,automatic_retry=False)
    remaining=[r for r in rows if r['job_id'] not in records]
    status=dict(s1041,completed=records,failed={},active=[remaining[0]['job_id']],checking=[remaining[1]['job_id']],
        ready=[r['job_id'] for r in remaining[2:16]],drain_requested=False,stop_requested=False)
    accepted=set(records)
    evidence=[dict(job_id=k,recoverable_complete=True,native_success=v['native_success'],errors=[],
        evidence_inherited=k in prior['completed_job_ids'],literal_parent_row_matches=True)
        for k,v in records.items()]
    audit=dict(a1054,schema='r10_readonly_host_second_reboot_TEST_evidence_observation_v1067',
        completed_evidence=evidence,saved_status_completed_count=saved,recoverable_complete_count=saved,
        damaged_registered_count=0,still_unclosed_count=1920-saved,
        current_boot_id='00000000-0000-0000-0000-000000000002')
    for name,values in dict(accepted_completed_job_ids=accepted,rejected_saved_completed_job_ids=set(),
            saved_completed_job_ids=accepted,remaining_job_ids={r['job_id'] for r in rows}-accepted).items():
        audit[name]=sorted(values);audit[name+'_sha256']=core.digest(sorted(values))
    control=dict(prior,schema=core.CONTROL_SCHEMA)
    for name,values in dict(completed=accepted,failed=set(),remaining={r['job_id'] for r in rows}-accepted,
            saved_completed=accepted,rejected_saved_completed=set()).items():
        control[name+'_job_ids']=sorted(values);control[name+'_key_sha256']=core.digest(sorted(values))
    return control,plan,status,audit,prior,p1041,s1041,a1054,*rest


class SecondRebootContractTests(unittest.TestCase):
    def test_full1419_status_adopts_all_and_retries501_without_terminal(self):
        c,p,s,a,prior,*_=fixture();before=deepcopy((c,p,s,a))
        self.assertEqual(len(core.require_partition(c,p,s,a)),1920)
        self.assertEqual((len(c['completed_job_ids']),len(c['remaining_job_ids'])),(1419,501))
        self.assertEqual(len(prior['rejected_saved_completed_job_ids']),2)
        self.assertEqual(c['rejected_saved_completed_job_ids'],[])
        self.assertEqual((c,p,s,a),before)
        self.assertFalse(s['drain_requested']);self.assertTrue(s['active'])

    def test_seven_tier_literal_lineage_and_only_pending_out_changes(self):
        c,p,s,a,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        for r,oldrow in zip(rows,p['jobs']):
            for field in ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','resume_lineage_v1055','previous_row_sha256'):
                self.assertEqual(r[field],oldrow[field])
            self.assertEqual(r[core.LINEAGE_FIELD]['parent_row_sha256'],core.digest(oldrow))
            if r['imported_completed']:self.assertEqual(core.parent_from_imported(r),oldrow)
            else:
                inner=deepcopy(oldrow['physical_argv']);inner[inner.index('--out')+1]=core.option(r['physical_argv'],'--out')
                self.assertEqual(inner,r['physical_argv']);self.assertEqual(r['evidence_runtime'],core.RUNTIME)
                self.assertEqual(r['argv'].count(core.BATCH_ENTRY),1)
        new=dict(jobs=rows,expected_cases=1920,imported_completed_cases=1419,source_outcome_filtering=False)
        core.require_resume_plan(c,p,s,a,new)
        self.assertEqual(sum(r['evidence_runtime']==core.PREVIOUS for r in rows),8)
        self.assertEqual(sum(r['evidence_runtime']==core.FIFTH for r in rows),2)

    def test_no_fake_false_success_filter_completion_loss_or_old_boot_audit(self):
        for mode in ('unknown_false','filter','missing_saved','audit_duplicate','fake_full','live_old','source_changed','old_boot','lineage'):
            c,p,s,a,*_=fixture()
            if mode=='unknown_false':a['completed_evidence'][-1]['native_success']=None
            elif mode=='filter':c['completed_job_ids'].pop();c['completed_key_sha256']=core.digest(c['completed_job_ids'])
            elif mode=='missing_saved':s['completed'].pop(next(iter(s['completed'])))
            elif mode=='audit_duplicate':a['completed_evidence'].append(deepcopy(a['completed_evidence'][0]))
            elif mode=='fake_full':a['formal_full_TEST_complete']=True
            elif mode=='live_old':a['old_owner_process']['same_process_alive']=True
            elif mode=='old_boot':a['schema']='r10_readonly_host_reboot_TEST_evidence_observation_v1054'
            elif mode=='lineage':p['jobs'][0]['resume_lineage_v1055']={}
            else:p['jobs'][0]['raw']='different'
            with self.subTest(mode=mode),self.assertRaises(ValueError):core.require_partition(c,p,s,a)

    def test_false_and_unknown_motion_literal_and_inherited_fourth_lineage(self):
        c,p,s,a,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        self.assertTrue(any(s['completed'][r['job_id']]['native_success'] is False for r in rows if r['imported_completed']))
        self.assertTrue(any(s['completed'][r['job_id']]['motion_pass'] is None for r in rows if r['imported_completed']))
        row=next(r for r in rows if r['imported_completed']);row['resume_lineage_v1055']={}
        with self.assertRaises(ValueError):core.parent_from_imported(row)

    def test_pure_contract_performs_no_filesystem_reads(self):
        c,p,s,a,*_=fixture()
        with patch('builtins.open',side_effect=AssertionError('no file IO')),patch.object(Path,'read_bytes',side_effect=AssertionError('no file IO')):
            core.require_partition(c,p,s,a);core.resumed_jobs(p,c,c['execution_entry'])


if __name__=='__main__':unittest.main()
