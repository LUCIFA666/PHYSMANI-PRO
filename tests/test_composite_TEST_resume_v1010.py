"""Invented four-tier1920 grids only; no process, case, model or source inputs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from our_mani.multitask import composite_TEST_resume_v1010 as core
from our_mani.multitask import composite_TEST_resume_v955 as parent_core
spec=importlib.util.spec_from_file_location('fixture955_CPU1010',Path(__file__).with_name('test_composite_TEST_resume_v955.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def fixture(completed=1200,failures=0,parent_completed=584,original_completed=134):
    prior,p902,s902,t902,p753,c915=old.fixture(parent_completed,6,original_completed)
    rows=parent_core.resumed_jobs(p902,prior,prior['execution_entry'])
    p956=dict(p902,jobs=rows,imported_completed_cases=parent_completed)
    records={r['job_id']:dict(complete=True,returncode=0,native_success=False,motion_pass=None if i%7==0 else False,
        timed_out=False,interrupted=False,automatic_retry=False) for i,r in enumerate(rows[:completed])}
    failed={r['job_id']:dict(stage='CPU_preflight' if i%2 else 'native',complete=False,returncode=1,
        interrupted=False,timed_out=False,automatic_retry=False) for i,r in enumerate(rows[completed:completed+failures])}
    status=dict(s902,completed=records,failed=failed)
    terminal=dict(t902,completed=completed,failed=failures,native_successes=0)
    c=dict(prior,schema=core.CONTROL_SCHEMA)
    for name,values in (('completed',sorted(records)),('failed',sorted(failed)),
            ('remaining',sorted({r['job_id'] for r in rows}-set(records)))):
        c[name+'_job_ids']=values;c[name+'_key_sha256']=core.digest(values)
    return c,p956,status,terminal,p902,prior,p753,c915,s902,t902


class ContractTests(unittest.TestCase):
    def test_full_grid_dynamic_clean_drain_and_allFalse_None_preserved(self):
        for completed,failed in ((1200,0),(585,3),(1919,1)):
            c,p,s,t,*_=fixture(completed,failed);before=deepcopy((c,p,s,t))
            self.assertEqual(len(core.require_partition(c,p,s,t)),1920)
            self.assertEqual((c,p,s,t),before)
            self.assertTrue(all(r['native_success'] is False for r in s['completed'].values()))
            self.assertTrue(any(r['motion_pass'] is None for r in s['completed'].values()))

    def test_four_tiers_literal_parent_recovery_preserves_both_older_lineages(self):
        c,p,s,t,*_=fixture();before=deepcopy(p);rows=core.resumed_jobs(p,c,c['execution_entry'])
        counts={name:sum(r['evidence_runtime']==name for r in rows) for name in (core.ORIGINAL,core.INTERMEDIATE,core.PREVIOUS,core.RUNTIME)}
        self.assertEqual(counts,{core.ORIGINAL:134,core.INTERMEDIATE:450,core.PREVIOUS:616,core.RUNTIME:720})
        for row,parent in zip(rows,p['jobs']):
            self.assertEqual(row['resume_lineage'],parent['resume_lineage'])
            self.assertEqual(row['previous_row_sha256'],parent['previous_row_sha256'])
            self.assertEqual(row[core.LINEAGE_FIELD]['parent_row_sha256'],core.digest(parent))
            if row['imported_completed']:self.assertEqual(core.parent_from_imported(row),parent)
        self.assertEqual(p,before)
        core.require_resume_plan(c,p,s,t,dict(jobs=rows,expected_cases=1920,imported_completed_cases=1200,source_outcome_filtering=False))

    def test_only_pending_out_changes_and_exactly_one901_outer_wrap(self):
        c,p,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        for row,parent in zip(rows[1200:],p['jobs'][1200:]):
            inner=deepcopy(parent['physical_argv']);inner[inner.index('--out')+1]=core.option(row['physical_argv'],'--out')
            self.assertEqual(row['physical_argv'],inner);self.assertEqual(row['argv'].count(core.BATCH_ENTRY),1)
            self.assertNotIn(core.BATCH_ENTRY,row['physical_argv'])
            self.assertTrue(core.option(inner,'--out').startswith(core.OUTPUT+'/'))

    def test_reject_duplicate_missing_mismatched_source_and_active_or_STOP(self):
        changes=(lambda p,s,t:p['jobs'].pop(),lambda p,s,t:p['jobs'][-1].update(job_id=p['jobs'][0]['job_id']),
            lambda p,s,t:p['jobs'][0].update(raw='different_source'),lambda p,s,t:s.update(active=['live']),
            lambda p,s,t:s.update(checking=['live']),lambda p,s,t:t.update(stop_requested=True),lambda p,s,t:t.update(complete=True))
        for change in changes:
            c,p,s,t,*_=fixture();change(p,s,t)
            with self.subTest(change=change),self.assertRaises(ValueError):core.require_partition(c,p,s,t)

    def test_old_lineage_overwrite_or_tier_and_new_parent_digest_tamper_refused(self):
        for change in ('oldlineage','newfield','tier','outer','source'):
            c,p,s,t,*_=fixture();r=p['jobs'][600]
            if change=='oldlineage':r['resume_lineage']={}
            elif change=='newfield':r[core.LINEAGE_FIELD]={}
            elif change=='tier':r['evidence_runtime']=core.INTERMEDIATE
            elif change=='outer':r['argv']=r['physical_argv']
            else:r['worker_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):core.require_partition(c,p,s,t)
        c,p,*_=fixture();row=core.resumed_jobs(p,c,c['execution_entry'])[600];row['previous_row_sha256']='0'*64
        with self.assertRaises(ValueError):core.parent_from_imported(row)

    def test_no_filter_or_fake_native_int_motion_int_or_missing_motion(self):
        for mode in ('filter','skipfailed','nativeint','motionint','missingmotion','retry'):
            c,p,s,t,*_=fixture(1200,2);record=next(iter(s['completed'].values()))
            if mode=='filter':c['completed_job_ids'].pop()
            elif mode=='skipfailed':c['remaining_job_ids'].remove(c['failed_job_ids'][0]);c['remaining_key_sha256']=core.digest(c['remaining_job_ids'])
            elif mode=='nativeint':record['native_success']=0
            elif mode=='motionint':record['motion_pass']=0
            elif mode=='missingmotion':record.pop('motion_pass')
            else:record['automatic_retry']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):core.require_partition(c,p,s,t)

    def test_core_is_pure_and_task_registry_unchanged(self):
        self.assertEqual(core.TASKS,parent_core.TASKS);self.assertEqual(core.METHODS,parent_core.METHODS)
        c,p,s,t,*_=fixture()
        with patch('builtins.open',side_effect=AssertionError('no files')),patch.object(Path,'read_bytes',side_effect=AssertionError('no bytes')):
            core.require_partition(c,p,s,t);core.resumed_jobs(p,c,c['execution_entry'])

    def test_same_original954_environment_point_proof_not_resource_qualification(self):
        c,p,*_=fixture();proof=old.observation(c)
        self.assertIs(core.require_environment_recheck(proof,c),proof)
        proof['parsed_suite_lock_is_exact_hashed_payload']=False
        with self.assertRaises(ValueError):core.require_environment_recheck(proof,c)


if __name__=='__main__':unittest.main()
