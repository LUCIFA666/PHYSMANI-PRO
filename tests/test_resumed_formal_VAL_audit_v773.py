"""Pure resume provenance tests plus the actual unchanged full945 release gate."""
from copy import deepcopy
from pathlib import Path
import importlib.util,unittest
from scripts import audit_resumed_formal_VAL_v773_r10 as audit
from our_mani.multitask.evaluation import digest


def fixture():
    policy='e'*64;previous='runtime/framework/formal_VAL_v753'
    jobs=[]
    for index in range(945):
        method=('cv_geometry','context_geometry','physmani')[index%3]
        task='beat_the_rotating_buzz';raw='episode'+str(index)
        argv=['bash','scripts/run_local_sim.sh','scripts/run_formal_case_author_outcome_v751_r10.py',
            '--config','original_config.json','--manifest','validation.json','--frozen-inputs','core.json',
            '--suite','suite752.json','--suite-lock','suite752_lock.json','--source-inputs','VAL_source_lock.json',
            '--task',task,'--raw-episode-id',raw,'--method',method,'--split','validation',
            '--out',str(Path('artifacts/framework/multitask_r10/formal_suite16_v752/validation')/task/raw/method)]
        jobs.append(dict(job_id='job'+str(index),task=task,raw=raw,method=method,argv=argv,
            worker_sha256='original751',case_worker={'path':'original743','sha256':'a'*64},
            native_runtime={'path':'original744','sha256':'b'*64},source_row_sha256='source'+str(index)))
    old_plan=dict(split='validation',policy_sha256=policy,jobs=jobs,expected_cases=945)
    completed=[r['job_id'] for r in jobs[:95]];remaining=[r['job_id'] for r in jobs[95:]];interrupted=remaining[:2]
    old_status=dict(completed={name:{'complete':True} for name in completed},
        failed={name:dict(stage='native',complete=False,interrupted=True,timed_out=False) for name in interrupted},
        active=[],stop_requested=True)
    old_terminal=dict(complete=False,stop_requested=True,completed=95,failed=2,expected=945,policy_sha256=policy)
    control=dict(schema=audit.CONTROL_SCHEMA,split='validation',user_authorized_resume=True,expected_cases=945,
        previous_runtime=previous,completed_job_ids=completed,remaining_job_ids=remaining,interrupted_job_ids=interrupted,
        completed_key_sha256=digest(sorted(completed)),remaining_key_sha256=digest(sorted(remaining)),policy_sha256=policy)
    new_jobs=deepcopy(jobs)
    for index,row in enumerate(new_jobs):
        imported=index<95;row.update(imported_completed=imported,evidence_runtime=previous if imported else audit.RUNTIME)
        if not imported:row['argv'][row['argv'].index('--out')+1]=str(Path(audit.OUTPUT)/row['task']/row['raw']/row['method'])
    new_plan=dict(split='validation',policy_sha256=policy,jobs=new_jobs,expected_cases=945)
    return control,old_plan,old_status,old_terminal,new_plan


class ResumePlanTests(unittest.TestCase):
    def check(self,parts):
        return audit.require_resume_plan(*parts,runtime=audit.ROOT/audit.RUNTIME)

    def test_exact95_immutable_imports_and850_fresh_outputs_keep_one_full_grid(self):
        parts=fixture();old,new=self.check(parts)
        self.assertEqual(len(old),945);self.assertEqual(len(new),945)
        self.assertEqual(sum(r['imported_completed'] for r in new.values()),95)
        self.assertEqual(new['job94']['argv'],old['job94']['argv'])
        self.assertNotEqual(new['job95']['argv'],old['job95']['argv'])

    def test_imported_case_cannot_be_restarted_repathed_or_relabelled(self):
        for mutation in ('out','import_flag','evidence_runtime','extra_unrecorded_field'):
            parts=fixture();row=parts[-1]['jobs'][0]
            if mutation=='out':row['argv'][-1]='new_case_instead_of_import'
            elif mutation=='import_flag':row['imported_completed']=False
            elif mutation=='evidence_runtime':row['evidence_runtime']=audit.RUNTIME
            else:row['expected_identity']={'new':'not in the actual original plan'}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check(parts)

    def test_new_attempt_can_change_only_the_out_value_and_two_provenance_fields(self):
        for flag in ('--config','--suite','--suite-lock','--source-inputs','--manifest','--split','--method'):
            parts=fixture();row=parts[-1]['jobs'][95];row['argv'][row['argv'].index(flag)+1]='changed'
            with self.subTest(flag=flag),self.assertRaises(ValueError):self.check(parts)
        for mutation in ('wrong_output','old_evidence','source_sha','new_worker','duplicate_option'):
            parts=fixture();row=parts[-1]['jobs'][95]
            if mutation=='wrong_output':row['argv'][-1]=parts[1]['jobs'][95]['argv'][-1]
            elif mutation=='old_evidence':row['evidence_runtime']=parts[0]['previous_runtime']
            elif mutation=='source_sha':row['source_row_sha256']='changed'
            elif mutation=='new_worker':row['case_worker']['sha256']='c'*64
            else:row['argv']+=['--out','another']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check(parts)

    def test_missing_duplicate_and_selected_grid_cannot_drop_interrupted_cases(self):
        for mutation in ('missing','duplicate','completed_selection','remaining_selection','key_digest'):
            parts=fixture();control=parts[0]
            if mutation=='missing':parts[-1]['jobs'].pop()
            elif mutation=='duplicate':parts[-1]['jobs'][-1]=deepcopy(parts[-1]['jobs'][0])
            elif mutation=='completed_selection':control['completed_job_ids'][0]='job95'
            elif mutation=='remaining_selection':control['remaining_job_ids'][0]='not_an_original_case'
            else:control['completed_key_sha256']='0'*64
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check(parts)

    def test_only_frozen_explicit_user_STOP_can_authorize_the_two_replacements(self):
        for mutation in ('no_user_resume','old_complete','old_not_STOP','timeout','not_interrupted','extra_failure','active'):
            parts=fixture();control,_,status,terminal,_=parts;failed=status['failed'][control['interrupted_job_ids'][0]]
            if mutation=='no_user_resume':control['user_authorized_resume']=False
            elif mutation=='old_complete':terminal['complete']=True
            elif mutation=='old_not_STOP':terminal['stop_requested']=False
            elif mutation=='timeout':failed['timed_out']=True
            elif mutation=='not_interrupted':failed['interrupted']=False
            elif mutation=='extra_failure':status['failed']['other']={}
            else:status['active']=['still_running']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.check(parts)

    def test_policy_and_split_cannot_change_after_VAL_started(self):
        for location,key,value in ((0,'policy_sha256','f'*64),(1,'policy_sha256','f'*64),
            (4,'policy_sha256','f'*64),(0,'split','train'),(4,'split','test'),(4,'expected_cases',850)):
            parts=fixture();parts[location][key]=value
            with self.subTest(location=location,key=key),self.assertRaises(ValueError):self.check(parts)


class ActualOwnerAndReleaseTests(unittest.TestCase):
    def test_release_carries_project_resume_evidence_without_external_base_graph(self):
        base_name=str(audit.ROOT/'scripts/run_formal_suite16_owner_v753_r10.py')
        new_name=str(audit.ROOT/'runtime/framework/independent_resume_control_fixture.json')
        external_name=str(audit.ROOT.parent/'external_native_source.py')
        result=audit.project_resume_inputs({base_name:'a'*64,new_name:'b'*64,external_name:'c'*64},{base_name:'a'*64})
        self.assertEqual(result,{new_name:'b'*64})

    def launch_fixture(self):
        protocol=dict(schema='actual_protocol',runtime_consumer={'path':'744','sha256':'7'*64},original_runtime={'path':'664','sha256':'6'*64})
        control=dict(author_outcome_protocol=protocol,policy_sha256='p'*64,resume_owner=dict(path=audit.OWNER,sha256='2'*64))
        driver={'path':'751','sha256':'1'*64};guard={'path':'728','sha256':'8'*64}
        launch=dict(source_sha256='2'*64,split='validation',suite_lock_sha256='s'*64,split_source_lock_sha256='d'*64,
            policy_sha256='p'*64,admission_driver=driver,admission_guard=guard,execution_protocol_changed=True,
            native_runtime=protocol['runtime_consumer'],original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol)
        terminal=dict(complete=True,expected=945,completed=945,failed=0,split='validation',policy_sha256='p'*64,author_outcome_protocol=protocol)
        return launch,terminal,control,dict(suite_lock_sha='s'*64,source_lock_sha='d'*64,driver=driver,guard=guard)

    def test_actual772_owner_source_is_required_without_impersonating753(self):
        launch,terminal,control,kwargs=self.launch_fixture()
        audit.require_owner_launch(launch,terminal,control,**kwargs)
        for change in ('source','only850','partial','failed','policy','protocol'):
            bad_launch=deepcopy(launch);bad_terminal=deepcopy(terminal)
            if change=='source':bad_launch['source_sha256']='old753'
            elif change=='only850':bad_terminal['completed']=850
            elif change=='partial':bad_terminal['complete']=False
            elif change=='failed':bad_terminal['failed']=2
            elif change=='policy':bad_launch['policy_sha256']='other'
            else:bad_launch['author_outcome_protocol']={}
            with self.subTest(change=change),self.assertRaises(ValueError):audit.require_owner_launch(bad_launch,bad_terminal,control,**kwargs)

    def test_original945_gate_still_rejects_partial_or_duplicate_cohorts_without_reading_TEST(self):
        spec=importlib.util.spec_from_file_location('_common_release',Path(__file__).with_name('test_common_rml_release_v531.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        routes,manifest,cases=module.Tests().fixture()
        proof=audit.validation_release(cases,manifest,routes)
        self.assertEqual(proof['method_cases'],945);self.assertTrue(proof['formal_test_permitted'])
        for bad in (cases[:-1],cases[:-1]+[cases[0]]):
            with self.assertRaises(ValueError):audit.validation_release(bad,manifest,routes)


if __name__=='__main__':unittest.main()
