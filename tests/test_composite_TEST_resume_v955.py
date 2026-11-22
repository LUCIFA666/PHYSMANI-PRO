"""Pure invented grids: no experiment files, processes or native APIs."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

from our_mani.multitask import composite_TEST_resume_v955 as core
from scripts import run_concurrent_resumed_formal_TEST_owner_v902_r10 as base


def fixture(completed=584, failures=6, original_completed=134):
    entry = dict(kind='batched901', pointer=dict(path=core.BATCH_ENTRY, sha256='b'*64),
        batch_control=dict(path='fixture/formal_batch_control.json', sha256='c'*64),
        batch_lock=dict(path='fixture/formal_batch_inputs.json', sha256='d'*64))
    jobs=[]
    for task in core.TASKS:
        for index in range(40):
            raw='invented_episode_'+str(index)
            for method in core.METHODS:
                key=task+'_'+raw+'_'+method
                argv=['bash','fixture/physical_python.sh',core.ORIGINAL_ENTRY,'--split','test',
                    '--task',task,'--raw-episode-id',raw,'--method',method,'--out',
                    'artifacts/framework/multitask_r10/fixture_original/'+task+'/'+raw+'/'+method]
                jobs.append(dict(job_id=key,task=task,raw=raw,method=method,argv=argv,
                    worker_sha256='a'*64,admission_driver=dict(path=core.ORIGINAL_ENTRY,sha256='a'*64),
                    preserved_metadata=dict(source_index=index,markers=['literal',False])))
    original_plan=dict(jobs=jobs,split='test',expected_cases=1920,policy_sha256='e'*64,
        source_outcome_filtering=False,automatic_retry=False)
    prior_control=dict(schema=base.CONTROL_SCHEMA,split='test',expected_cases=1920,source_count=640,
        completed_job_ids=[r['job_id'] for r in jobs[:original_completed]],
        execution_entry=deepcopy(entry))
    plan=dict(original_plan,jobs=base.resumed_jobs(original_plan,prior_control,entry),
        imported_completed_cases=original_completed)
    done={r['job_id']:dict(complete=True,returncode=0,native_success=False,motion_pass=None if i%7==0 else False,
        timed_out=False,interrupted=False,automatic_retry=False) for i,r in enumerate(plan['jobs'][:completed])}
    failed={r['job_id']:dict(stage='CPU_preflight' if i%2 else 'native',complete=False,returncode=1,
        interrupted=False,timed_out=False,automatic_retry=False) for i,r in enumerate(plan['jobs'][completed:completed+failures])}
    status=dict(split='test',expected=1920,completed=done,failed=failed,active=[],checking=[],native_slots=0,GPU_slots=0,
        automatic_retry=False,drain_requested=True,stop_requested=False)
    terminal=dict(split='test',policy_sha256='e'*64,complete=False,drained=True,drain_requested=True,stop_requested=False,
        automatic_retry=False,expected=1920,completed=completed,failed=failures,native_successes=0)
    control=dict(schema=core.CONTROL_SCHEMA,split='test',expected_cases=1920,source_count=640,
        policy_sha256='e'*64,user_authorized_resume=True,standing_authorization_reused=True,
        new_user_event_invented=False,physical_policy_changed=False,execution_entry=deepcopy(entry))
    for name,values in (('completed',sorted(done)),('failed',sorted(failed)),
            ('remaining',sorted({r['job_id'] for r in plan['jobs']}-set(done)))):
        control[name+'_job_ids']=values;control[name+'_key_sha256']=core.digest(values)
    return control,plan,status,terminal,original_plan,prior_control


def observation(control):
    for key in ('suite_lock','environment_manifest','environment_wrapper','environment_namespace_proof'):
        control[key]=dict(path='fixture/'+key+'.json',sha256='f'*64)
    return dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v954',complete=True,
        source=dict(path='scripts/check_frozen_TEST_environment_v954_r10.py',sha256='1'*64),
        Native_started=False,GPU_started=False,admission_or_TEST_started=False,
        declared_model_case_raw_bytes_read=False,scientific_inputs_or_old_locks_modified=False,
        original_v900_topology_constructor_passed=True,original_full_topology_final_check_passed=True,
        parsed_suite_lock_is_exact_hashed_payload=True,topology_is_point_observation_not_continuous=True,
        suite_lock=control['suite_lock'],declared_files=130292,unique_canonical_paths=130292,watched_directories=41468,
        actual_environment=dict(schema='r10_actual_frozen_OS_namespace_receipt_v820',complete=True,checked_OS_files=6418,
            all_current_OS_targets_full_byte_SHA_verified=True,manifest=control['environment_manifest'],
            wrapper=control['environment_wrapper'],namespace_qualification=control['environment_namespace_proof']))


class CompositeContractTests(unittest.TestCase):
    def test_registry_matches_the_real_frozen_roster(self):
        self.assertEqual(core.TASKS,base.TASKS);self.assertEqual(core.METHODS,base.METHODS)

    def test_all584_including_nativeFalse_unknown_motion_adopted_six_failures_pending(self):
        c,p,s,t,*_=fixture();before=deepcopy((c,p,s,t))
        self.assertEqual(len(core.require_partition(c,p,s,t)),1920)
        self.assertEqual((len(c['completed_job_ids']),len(c['failed_job_ids']),len(c['remaining_job_ids'])),(584,6,1336))
        self.assertTrue(set(c['failed_job_ids'])<=set(c['remaining_job_ids']))
        self.assertTrue(all(r['native_success'] is False for r in s['completed'].values()))
        self.assertEqual((c,p,s,t),before)

    def test_dynamic_C_and_failure_counts_are_not_hardcoded(self):
        for count,failed,old in ((7,1,2),(135,3,134),(1919,1,134)):
            c,p,s,t,*_=fixture(count,failed,old)
            self.assertEqual(len(core.require_partition(c,p,s,t)),1920)

    def test_imports_recover_exact_whole_parent_and_do_not_overwrite_old_lineage(self):
        c,p,s,t,*_=fixture();before=deepcopy(p)
        rows=core.resumed_jobs(p,c,c['execution_entry'])
        counts={name:sum(r['evidence_runtime']==name for r in rows) for name in (core.ORIGINAL,core.PREVIOUS,core.RUNTIME)}
        self.assertEqual(counts,{core.ORIGINAL:134,core.PREVIOUS:450,core.RUNTIME:1336})
        for row,parent in zip(rows,p['jobs']):
            self.assertEqual(row['previous_row_sha256'],parent['previous_row_sha256'])
            self.assertEqual(row['previous_worker_sha256'],parent['previous_worker_sha256'])
            if row['imported_completed']:
                self.assertEqual(core.parent_from_imported(row),parent)
        self.assertEqual(p,before)
        self.assertEqual(core.require_resume_plan(c,p,s,t,dict(jobs=rows,expected_cases=1920,
            imported_completed_cases=584,source_outcome_filtering=False)),{r['job_id']:r for r in p['jobs']})

    def test_new_attempts_change_only_out_and_outer_wrapper_once(self):
        c,p,*_=fixture();rows=core.resumed_jobs(p,c,c['execution_entry'])
        for row,parent in zip(rows[584:],p['jobs'][584:]):
            inner=deepcopy(parent['physical_argv']);inner[inner.index('--out')+1]=core.option(row['physical_argv'],'--out')
            self.assertEqual(row['physical_argv'],inner)
            self.assertEqual(row['argv'].count(core.BATCH_ENTRY),1)
            self.assertNotIn(core.BATCH_ENTRY,row['physical_argv'])
            self.assertTrue(core.option(inner,'--out').startswith(core.OUTPUT+'/'))
            self.assertNotEqual(core.option(inner,'--out'),core.option(parent['physical_argv'],'--out'))

    def test_wrong_grid_duplicate_coordinate_mismatched_sources_and_live_or_stopped_rejected(self):
        changes=(lambda p,s,t:p['jobs'].pop(),lambda p,s,t:p['jobs'][-1].update(job_id=p['jobs'][0]['job_id']),
            lambda p,s,t:p['jobs'][0].update(raw='invented_mismatch'),lambda p,s,t:p['jobs'][0].update(task='wrong'),
            lambda p,s,t:s.update(active=['live']),lambda p,s,t:s.update(native_slots=1),
            lambda p,s,t:t.update(stop_requested=True),lambda p,s,t:t.update(complete=True),
            lambda p,s,t:t.update(failed=0),lambda p,s,t:p.update(source_outcome_filtering=True))
        for change in changes:
            c,p,s,t,*_=fixture();change(p,s,t)
            with self.subTest(change=change),self.assertRaises(ValueError):core.require_partition(c,p,s,t)

    def test_completed_selection_failed_skipping_and_unclean_records_rejected(self):
        for mode in ('filter','skipfailed','overlap','nativeint','motionint','retry','missingmotion'):
            c,p,s,t,*_=fixture();record=next(iter(s['completed'].values()))
            if mode=='filter':c['completed_job_ids'].pop()
            elif mode=='skipfailed':c['remaining_job_ids'].remove(c['failed_job_ids'][0]);c['remaining_key_sha256']=core.digest(c['remaining_job_ids'])
            elif mode=='overlap':c['failed_job_ids'].append(c['completed_job_ids'][0]);c['failed_key_sha256']=core.digest(sorted(c['failed_job_ids']))
            elif mode=='nativeint':record['native_success']=0
            elif mode=='motionint':record['motion_pass']=0
            elif mode=='missingmotion':record.pop('motion_pass')
            else:record['automatic_retry']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):core.require_partition(c,p,s,t)

    def test_parent_alias_entry_tier_and_lineage_tampering_rejected(self):
        for mode in ('entry','tier','outer','source','lineage'):
            c,p,s,t,*_=fixture();row=p['jobs'][134]
            if mode=='entry':row['execution_entry']['kind']='original751'
            elif mode=='tier':row['evidence_runtime']=core.ORIGINAL
            elif mode=='outer':row['argv']=deepcopy(row['physical_argv'])
            elif mode=='source':row['worker_sha256']='0'*64
            else:row['resume_lineage']={}
            with self.subTest(mode=mode),self.assertRaises(ValueError):core.require_partition(c,p,s,t)
        c,p,*_=fixture();row=core.resumed_jobs(p,c,c['execution_entry'])[134]
        row['previous_row_sha256']='0'*64
        with self.assertRaises(ValueError):core.parent_from_imported(row)

    def test_contracts_have_no_file_or_process_IO(self):
        c,p,s,t,*_=fixture()
        with patch('builtins.open',side_effect=AssertionError('no files')),patch.object(Path,'read_bytes',side_effect=AssertionError('no bytes')):
            core.require_partition(c,p,s,t);core.resumed_jobs(p,c,c['execution_entry'])

    def test_954_point_observation_binding_is_strict_and_never_score_based(self):
        c,p,*_=fixture();proof=observation(c)
        self.assertIs(core.require_environment_recheck(proof,c),proof)
        for field in ('complete','original_v900_topology_constructor_passed','parsed_suite_lock_is_exact_hashed_payload'):
            changed=deepcopy(proof);changed[field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):core.require_environment_recheck(changed,c)
        changed=deepcopy(proof);changed['suite_lock']['sha256']='0'*64
        changed=deepcopy(proof);changed['actual_environment']['checked_OS_files']=0
        with self.assertRaises(ValueError):core.require_environment_recheck(changed,c)


if __name__=='__main__':unittest.main()
