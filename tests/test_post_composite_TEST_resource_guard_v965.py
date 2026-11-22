"""Only source code and synthetic metadata/resources; no real gate or inventory."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import tempfile
import unittest
from unittest.mock import Mock,patch

from our_mani.multitask import post_composite_TEST_resource_guard_v965 as g
from our_mani.multitask import post_TEST_resource_guard_v924 as legacy

PROJECT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


old=module('CPU965_original924fixtures',PROJECT/'tests/test_post_TEST_resource_guard_v924.py')
motion=module('CPU965_contract963',PROJECT/g.MOTION_SOURCE)
post=module('CPU965_contract964',PROJECT/g.POST_SOURCE)
C=motion.C


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


class Fixture:
    """Actual writer shapes,134/450/1336, all nativeFalse and motionNone.

    Case/sidecar/manifest/source/model bytes do not exist in this fixture.
    All original small metadata hash relationships are constructed, not mocked.
    """
    def __init__(self):
        real_sha=legacy.sha
        self.codes={name:(PROJECT/name).read_bytes() for name in {*legacy.SOURCE_SHA,*g.SOURCE_SHA,*post.SOURCES}}
        prefix=b'CPU fake pinned source924: '
        def pinned(raw):return real_sha(self.codes[raw[len(prefix):].decode()]) if raw.startswith(prefix) else real_sha(raw)
        with patch.object(legacy,'sha',side_effect=pinned):base=old.Fixture(134)
        self.data=base.data;self.data.update(self.codes);self.reads=[]
        self.parent_control=deepcopy(self.data[legacy.CONTROL])
        parent_plan=self.data[g.PARENT+'/plan.json'];ids=[row['job_id'] for row in parent_plan['jobs']]
        all_records=deepcopy(self.data[g.PARENT+'/status.json']['completed'])
        for record in all_records.values():record['motion_pass']=None
        original_records=self.data[g.ORIGINAL+'/status.json']['completed']
        for record in original_records.values():record['motion_pass']=None
        # Keep the exact same imported record in the parent tier.
        all_records.update(deepcopy(original_records))
        self.data[g.PARENT+'/status.json'].update(completed={k:all_records[k] for k in ids[:584]},
            failed={k:dict(complete=False,automatic_retry=False,returncode=1) for k in ids[584:590]},
            active=[],checking=[],ready_jobs=[],native_slots=0,GPU_slots=0,CPU_ready=0,
            drain_requested=True,stop_requested=False)
        self.data[g.PARENT+'/terminal.json'].update(complete=False,completed=584,failed=6,
            drained=True,drain_requested=True,stop_requested=False)
        self.data[g.PARENT+'/DRAIN']=dict(reason='CPU synthetic environmental DRAIN')
        self.data[g.PARENT_CONTROLLER+'/status.json']=dict(stage='requires_concrete_repair',automatic_retry=False)
        self.data[g.PARENT_CONTROLLER+'/failure.json']=dict(complete=False,user_stop=False,automatic_retry=False)
        # Changed literal old records require their original915 pointers to be resealed.
        for key,n in (('previous_status','status.json'),('previous_terminal','terminal.json')):
            self.parent_control[key]=self.ptr(g.ORIGINAL+'/'+n)
        self.data[legacy.CONTROL]=self.parent_control
        self.data[legacy.CONTROL_LOCK]['control']=self.ptr(legacy.CONTROL)
        self.data[legacy.CONTROL_LOCK]['files'].update({legacy.CONTROL:self.ptr(legacy.CONTROL)['sha256'],
            **{p['path']:p['sha256'] for p in self.parent_control.values() if isinstance(p,dict) and set(p)=={'path','sha256'}}})
        ctl=deepcopy(self.parent_control)
        ctl.update(schema=C.CONTROL_SCHEMA,completed_job_ids=sorted(ids[:584]),failed_job_ids=sorted(ids[584:590]),remaining_job_ids=sorted(ids[584:]),
            completed_key_sha256=C.digest(sorted(ids[:584])),failed_key_sha256=C.digest(sorted(ids[584:590])),remaining_key_sha256=C.digest(sorted(ids[584:])),
            previous_resume_control=self.ptr(legacy.CONTROL),previous_resume_control_lock=self.ptr(legacy.CONTROL_LOCK),
            resume_controller=self.ptr(g.CONTROLLER_SOURCE),resume_owner=self.ptr(g.OWNER_SOURCE),
            resume_auditor=self.ptr(g.AUDITOR_SOURCE),result_auditor=self.ptr(g.AUDITOR_SOURCE))
        for n in ('launch','plan','status','terminal'):
            ctl['original_'+n]=self.ptr(g.ORIGINAL+'/'+n+'.json');ctl['previous_'+n]=self.ptr(g.PARENT+'/'+n+'.json')
        for n in ('launch','status','failure'):ctl['previous_controller_'+n]=self.ptr(g.PARENT_CONTROLLER+'/'+n+'.json')
        ctl['previous_DRAIN']=self.ptr(g.PARENT+'/DRAIN')
        check='runtime/framework/frozen_TEST_environment_recheck_v954.json'
        self.data[check]=dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v954',complete=True,
            source=base.unopened('scripts/check_frozen_TEST_environment_v954_r10.py'),Native_started=False,GPU_started=False,
            admission_or_TEST_started=False,declared_model_case_raw_bytes_read=False,scientific_inputs_or_old_locks_modified=False,
            original_v900_topology_constructor_passed=True,original_full_topology_final_check_passed=True,
            parsed_suite_lock_is_exact_hashed_payload=True,topology_is_point_observation_not_continuous=True,
            suite_lock=ctl['suite_lock'],declared_files=109,unique_canonical_paths=109,watched_directories=11,
            actual_environment=deepcopy(old.environment(711)))
        ctl['environment_recheck']=self.ptr(check);self.data[g.CONTROL]=ctl
        files=deepcopy(self.data[legacy.CONTROL_LOCK]['files'])
        files.update({name:self.ptr(name)['sha256'] for name in self.codes})
        for name in tuple(self.data):
            if g._allowed(name) and name not in (g.RESULT,g.CONTROL_LOCK):files[name]=self.ptr(name)['sha256']
        self.data[g.CONTROL_LOCK]=dict(schema=C.LOCK_SCHEMA,approved_for_execution=True,control=self.ptr(g.CONTROL),files=files)
        cp,lp=self.ptr(g.CONTROL),self.ptr(g.CONTROL_LOCK)
        common={k:deepcopy(self.data[g.PARENT+'/launch.json'][k]) for k in ('split','policy_sha256','author_outcome_protocol',
            'execution_protocol_changed','automatic_retry','admission_driver','admission_guard','native_runtime','original_native_runtime',
            'resource_profile','resource_qualification','execution_implementation')}
        common.update(resume_control=cp,resume_control_lock=lp,previous_resume_control=ctl['previous_resume_control'],
            previous_resume_control_lock=ctl['previous_resume_control_lock'],environment_recheck=ctl['environment_recheck'],frozen_environment=old.environment(1010))
        jobs=C.resumed_jobs(parent_plan,ctl,ctl['execution_entry'])
        owner=dict(common,pid=1010,start_ticks=1011,source_sha256=self.ptr(g.OWNER_SOURCE)['sha256'],
            execution_entry=ctl['execution_entry'],base_stage_owner=self.ptr(legacy.OLD_OWNER_SOURCE),parent_stage_owner=self.ptr(legacy.OWNER_SOURCE),
            imported_completed_cases=584,max_native_slots=4,max_GPU_slots=2,max_CPU_preflight_slots=2)
        controller=dict(pid=1110,start_ticks=1111,source_sha256=self.ptr(g.CONTROLLER_SOURCE)['sha256'],resume_control=cp,
            resume_control_lock=lp,automatic_retry=False,frozen_environment=old.environment(1110))
        self.data[g.CONTROLLER+'/launch.json']=controller
        self.data[g.CONTROLLER+'/status.json']=dict(stage='complete',automatic_retry=False,frozen_environment=controller['frozen_environment'])
        owner_args=[]
        for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
            owner_args.extend((flag,ctl[key]['path']))
        owner_args+=['--runtime',g.TEST_RUNTIME,'--out',g.ART+'/test','--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        audit_args=['--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        for stage,name,number,args in ((g.OWNER_STAGE,g.OWNER_SOURCE,1010,owner_args),(g.AUDIT_STAGE,g.AUDITOR_SOURCE,1210,audit_args)):
            self.data[stage+'/launch.json']=dict(pid=number,start_ticks=number+1,source_sha256=self.ptr(name)['sha256'],
                argv=[str(PROJECT/'.conda-env/bin/python'),'-B','-u',name,*args],automatic_retry=False,
                inherited_frozen_environment=controller['frozen_environment'],current_mount_namespace='CPU-private-mount')
            self.data[stage+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
        owner['argv']=self.data[g.OWNER_STAGE+'/launch.json']['argv'][3:];self.data[g.TEST_RUNTIME+'/launch.json']=owner
        self.data[g.TEST_RUNTIME+'/plan.json']=dict(common,expected_cases=1920,jobs=jobs,imported_completed_cases=584,source_outcome_filtering=False)
        records={};cases=[];drivers=[];batches=[];tiers=[]
        owners={g.ORIGINAL:self.data[g.ORIGINAL+'/launch.json'],g.PARENT:self.data[g.PARENT+'/launch.json'],g.TEST_RUNTIME:owner}
        for row in jobs:
            key=row['job_id'];runtime=row['evidence_runtime'];out=legacy.option(row['physical_argv'],'--out')
            record=deepcopy(all_records[key]) if row['imported_completed'] else base.completed(out)
            record['motion_pass']=None
            if runtime!=g.ORIGINAL:
                record['batch_driver_receipt']=base.unopened(out+'/batch_driver_receipt.json');batches.append(record['batch_driver_receipt'])
            records[key]=record;cases.append(record['case']);drivers.append(record['admission_driver_receipt']);actual=owners[runtime]
            tiers.append(dict(job_id=key,evidence_runtime=runtime,imported_completed=row['imported_completed'],actual_stage_owner_pid=actual['pid'],
                actual_stage_owner_start_ticks=actual['start_ticks'],resume_lineage=row['resume_lineage'],previous_row_sha256=row['previous_row_sha256'],
                execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],physical_argv=row['physical_argv'],case=record['case'],
                admission_driver_receipt=record['admission_driver_receipt'],batch_driver_receipt=record.get('batch_driver_receipt')))
        self.data[g.TEST_RUNTIME+'/status.json']=dict(common,expected=1920,completed=records,failed={},active=[],checking=[],ready_jobs=[],
            CPU_ready=0,native_slots=0,GPU_slots=0,CPU_launched=1336,imported_completed=584,owner_error=None,stop_requested=False,drain_requested=False)
        self.data[g.TEST_RUNTIME+'/terminal.json']=dict(common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,
            stop_requested=False,drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,
            formal_release_requires_independent_audit=True,imported_completed_cases=584,resumed_new_cases=1336,native_successes=0)
        result=deepcopy(self.data[legacy.RESULT]);result.update(schema=motion.RESULT_SCHEMA,source_sha256=self.ptr(g.AUDITOR_SOURCE)['sha256'],
            resume_control=cp,resume_control_lock=lp,environment_recheck=ctl['environment_recheck'],old584_results_not_rerun=True,
            motion_unknown_retained=True,zero_tick_cases_retained=True,old907_successful_terminal_claimed=False,
            original754_or906_single_result_claimed=False,cases=cases,admission_driver_receipts=drivers,batch_driver_receipts=batches,evidence_tiers=tiers,
            frozen_environment=old.environment(1210),auditor_invocation=dict(pid=1210,start_ticks=1211,source=self.ptr(g.AUDITOR_SOURCE),
                argv=self.data[g.AUDIT_STAGE+'/launch.json']['argv'][3:],inherited_physical_stage_identity_claimed=False),
            historical_infrastructure_attempts=dict(job_ids=ctl['failed_job_ids'],previous_status=ctl['previous_status'],previous_terminal=ctl['previous_terminal'],
                counted_as_native_task_failures=False,all_keys_reexecuted_once_in_new_namespace=True))
        result['actual_stage_owners']=[dict(runtime=runtime,launch=self.ptr(runtime+'/launch.json'),terminal=self.ptr(runtime+'/terminal.json'),
            pid=owners[runtime]['pid'],start_ticks=owners[runtime]['start_ticks'],completed=count,resource_limits={k:owners[runtime][k]
                for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}) for runtime,count in ((g.ORIGINAL,134),(g.PARENT,450),(g.TEST_RUNTIME,1336))]
        result['inputs_sha256']={str(PROJECT/name):self.ptr(name)['sha256'] for name in self.data if g._allowed(name)
            and not name.startswith(g.CONTROLLER+'/') and name not in (g.RESULT,legacy.RESULT)}
        result['inputs_sha256'][str(PROJECT/legacy.SOURCE_LOCK)]=ctl['source_inputs']['sha256']
        # Actual957 reads the904 qualification body, not its original audit
        # launch. The launch stays bound by the actual955 supplement instead.
        result['inputs_sha256'].pop(str(PROJECT/legacy.QUALIFICATION_LAUNCH),None)
        self.data[g.RESULT]=result;self.data[g.CONTROLLER+'/terminal.json']=dict(complete=True,automatic_retry=False,TEST_results=self.ptr(g.RESULT))
        self.data[g.AUDIT_STAGE+'/stdout.log']=dict(complete=True,schema=motion.RESULT_SCHEMA,split='test',cases=1920,out=g.RESULT,
            motion_is_diagnostic_only=True,native_successes_by_method=result['native_successes_by_method'])
        postctl=dict(schema=post.SCHEMA,execution_enabled=True,read_only_analysis=True,automatic_retry=False,policy_sha256=g.POLICY,
            runtime=g.POST_RUNTIME,controller_runtime=g.CONTROLLER,owner_runtime=g.TEST_RUNTIME,test_results=g.RESULT,motion_summary=g.MOTION,
            poll_seconds=1,sources={name:self.ptr(name) for name in post.SOURCES},resume_control=cp,resume_control_lock=lp,
            controller_launch=self.ptr(g.CONTROLLER+'/launch.json'))
        self.data[g.POST_CONTROL]=postctl
        names={*post.SOURCES,g.POST_CONTROL,g.CONTROL,g.CONTROL_LOCK,g.CONTROLLER+'/launch.json'}
        self.data[g.POST_LOCK]=dict(schema=post.LOCK_SCHEMA,execution_enabled=True,files={name:self.ptr(name)['sha256'] for name in names})
        pcp,plp=self.ptr(g.POST_CONTROL),self.ptr(g.POST_LOCK)
        host=dict(standard_host_analysis=True,frozen_environment_inherited_claimed=False,automatic_retry=False)
        self.data[g.POST_RUNTIME+'/launch.json']=dict(host,pid=1310,start_ticks=1311,source_sha256=self.ptr(g.POST_SOURCE)['sha256'],
            argv=[g.POST_SOURCE,'--execute','--lock-sha256',plp['sha256']],control=pcp,input_lock=plp,monitored_controller_launch=postctl['controller_launch'])
        summary=dict(schema=post.SUMMARY_SCHEMA,complete=True,read_only_analysis=True,split='test',task_count=16,source_count=640,case_count=1920,
            methods=list(post.METHODS),task_source_counts={task:40 for task in post.TASKS},policy_sha256=g.POLICY,source_sha256=self.ptr(g.MOTION_SOURCE)['sha256'],
            test_results=self.ptr(g.RESULT),resume_control=cp,resume_control_lock=lp,actual_test_auditor=self.ptr(g.AUDITOR_SOURCE),
            full1920_three_tier_TEST_results_and_complete958_observed=True,provenance_runtime_counts={g.ORIGINAL:134,g.PARENT:450,g.TEST_RUNTIME:1336},
            actual_stage_owners=deepcopy(result['actual_stage_owners']),new_release_granted_by_this_summary=False,native_false_retained=True,
            success_rate_or_acceleration_filter_used=False,native_GPU_or_jobs_started=False,TEST_used_for_fitting=False,TXT_or_handoff_written=False,
            inputs_sha256={name:self.ptr(name)['sha256'] for name in (g.RESULT,g.CONTROL,g.CONTROL_LOCK,g.MOTION_SOURCE)})
        self.data[g.MOTION]=summary;sp=self.ptr(g.MOTION)
        self.data[g.POST_RUNTIME+'/status.json']=dict(stage='complete',automatic_retry=False,motion_summary=sp)
        self.data[g.POST_RUNTIME+'/terminal.json']=dict(host,schema='r10_readonly_post_complete_composite_TEST_motion_terminal_v964',complete=True,
            motion_summary=sp,TEST_results=self.ptr(g.RESULT),control=pcp,input_lock=plp,resume_control=cp,resume_control_lock=lp,
            Native_GPU_TRAIN_started=False,TXT_modified=False)
        self.data[g.POST_STAGE+'/launch.json']=dict(host,pid=1410,start_ticks=1411,source=self.ptr(g.MOTION_SOURCE),source_sha256=self.ptr(g.MOTION_SOURCE)['sha256'],
            owner_pid=1310,owner_start_ticks=1311,control=pcp,input_lock=plp,TEST_results=self.ptr(g.RESULT),argv=[str(PROJECT/'.conda-env/bin/python'),'-B','-u',
                g.MOTION_SOURCE,'--test-results-sha256',self.ptr(g.RESULT)['sha256'],'--resume-lock-sha256',lp['sha256'],
                '--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK])
        self.data[g.POST_STAGE+'/terminal.json']=dict(complete=True,returncode=0,interrupted=False,automatic_retry=False)
        self.data[g.POST_STAGE+'/stdout.log']=dict(complete=True,output=g.MOTION,sha256=sp['sha256'],case_count=1920)

    def raw(self,name):
        value=self.data[name];return value if isinstance(value,bytes) else encode(value)
    def ptr(self,name):return dict(path=name,sha256=g.sha(self.raw(name)))
    def read(self,name):
        self.reads.append(name)
        if not g._allowed(name) or name not in self.data:raise AssertionError('non-metadata read: '+name)
        return self.raw(name)
    def pins(self):return dict(resume_lock_sha256=self.ptr(g.CONTROL_LOCK)['sha256'],motion_summary_sha256=self.ptr(g.MOTION)['sha256'],postprocessing_lock_sha256=self.ptr(g.POST_LOCK)['sha256'])
    def proof(self):return g.require_complete_TEST_metadata(PROJECT,self.ptr(g.RESULT)['sha256'],read_bytes=self.read,**self.pins())


class GuardTests(unittest.TestCase):
    def setUp(self):
        denied=Mock(side_effect=AssertionError('CPU fake reached production resource helper'))
        for name in ('_stop_guard','_reserve','_inventory','_is_alive'):
            scope=patch.object(g,name,denied);scope.start();self.addCleanup(scope.stop)
        scope=patch.object(g.os,'getpid',return_value=4242);scope.start();self.addCleanup(scope.stop)
    def fixture(self):return Fixture()
    def test_import_and_disabled_scope_are_inert(self):
        raw=(PROJECT/g.SOURCE).read_bytes();obj=ModuleType('CPU965_inert');obj.__file__=str(PROJECT/g.SOURCE)
        denied=Mock(side_effect=AssertionError('inert965 touched OS'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied),patch.object(g.importlib,'import_module',denied):
            exec(compile(raw,obj.__file__,'exec'),obj.__dict__)
            scope=obj.reserved_resources_after_TEST('unused','a'*64,resume_lock_sha256='b'*64,motion_summary_sha256='c'*64,postprocessing_lock_sha256='d'*64)
            with self.assertRaises(ValueError):scope.__enter__()

    def test_complete1920_metadata_False_None_and_historical6_retained(self):
        f=self.fixture();proof=f.proof()
        self.assertEqual(proof['provenance_runtime_counts'],{g.ORIGINAL:134,g.PARENT:450,g.TEST_RUNTIME:1336})
        self.assertEqual(len(proof['known_processes']),11);self.assertEqual(proof['task_count'],16)
        self.assertEqual(proof['postprocessing'],{key:f.ptr(name) for key,name in g.POST_PATHS.items()})
        self.assertFalse(proof['scores_used_for_gate']);self.assertTrue(proof['registration_only_compatibility'])
        self.assertEqual(len(proof['historical_infrastructure_attempts']['job_ids']),6)
        self.assertFalse(any(name.endswith('/case.json') or 'manifest' in name.lower() or 'CPU/' in name or '/native/' in name for name in f.reads))

    def test_all_external_pins_required_before_any_read(self):
        f=self.fixture()
        for key in ('resume_lock_sha256','motion_summary_sha256','postprocessing_lock_sha256'):
            pins=f.pins();pins[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):g.require_complete_TEST_metadata(PROJECT,f.ptr(g.RESULT)['sha256'],read_bytes=f.read,**pins)
        self.assertEqual(f.reads,[])

    def test_incomplete_actualTEST_does_not_open_result_or_postprocessing(self):
        for path,key,value in ((g.CONTROLLER+'/status.json','stage','formal_TEST_1920'),(g.TEST_RUNTIME+'/terminal.json','completed',1919),
                (g.AUDIT_STAGE+'/terminal.json','returncode',1)):
            f=self.fixture();f.data[path][key]=value
            with self.subTest(path=path),self.assertRaises(ValueError):f.proof()
            self.assertNotIn(g.RESULT,f.reads);self.assertFalse(set(g.POST_PATHS.values())&set(f.reads));self.assertNotIn(g.MOTION,f.reads)

    def test_wrong_source_or_external_result_SHA_refused(self):
        f=self.fixture();f.data[g.OWNER_SOURCE]+=b'changed'
        with self.assertRaises(ValueError):f.proof()
        f=self.fixture()
        with self.assertRaises(ValueError):g.require_complete_TEST_metadata(PROJECT,'f'*64,read_bytes=f.read,**f.pins())
        self.assertNotIn(g.MOTION,f.reads)

    def test_three_tiers_duplicate_or_historical_outcome_rewrite_refused(self):
        for mode in ('tier','duplicate','rewrite','sidecar'):
            f=self.fixture();result=f.data[g.RESULT]
            if mode=='tier':result['evidence_tiers'][134]['actual_stage_owner_pid']=1010
            elif mode=='duplicate':result['evidence_tiers'][1]=deepcopy(result['evidence_tiers'][0])
            elif mode=='sidecar':result['batch_driver_receipts'].pop()
            else:
                key=next(iter(f.data[g.ORIGINAL+'/status.json']['completed']))
                f.data[g.TEST_RUNTIME+'/status.json']['completed'][key]['native_success']=True
            if mode!='rewrite':f.data[g.CONTROLLER+'/terminal.json']['TEST_results']=f.ptr(g.RESULT)
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.proof()
            self.assertNotIn(g.MOTION,f.reads)

    def test_native_bool_strict_and_motion_unknown_not_coerced(self):
        for field,value in (('native_success',0),('native_success',None),('motion_pass',0),('motion_pass','false')):
            f=self.fixture();key=f.data[g.TEST_RUNTIME+'/plan.json']['jobs'][584]['job_id'];f.data[g.TEST_RUNTIME+'/status.json']['completed'][key][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):f.proof()

    def test_post_gate_requires_true_completed_parent_and_child_before_summary(self):
        for path,key,value in ((g.POST_RUNTIME+'/status.json','stage','saved_motion_v963'),(g.POST_RUNTIME+'/terminal.json','complete',False),
                (g.POST_STAGE+'/terminal.json','interrupted',True),(g.POST_STAGE+'/launch.json','owner_pid',999)):
            f=self.fixture();f.data[path][key]=value
            with self.subTest(path=path),self.assertRaises(ValueError):f.proof()
            self.assertNotIn(g.MOTION,f.reads)

    def test_exact_absolute964_source_entry_is_compatible(self):
        f=self.fixture();f.data[g.POST_RUNTIME+'/launch.json']['argv'][0]=str(PROJECT/g.POST_SOURCE)
        self.assertTrue(f.proof()['complete'])

    def test_post_argv_duplicate_flags_refused_and_missing_audit_membership_not_invented(self):
        f=self.fixture()
        self.assertNotIn(str(PROJECT/legacy.QUALIFICATION_LAUNCH),f.data[g.RESULT]['inputs_sha256'])
        f.data[g.POST_RUNTIME+'/launch.json']['argv'].append('--execute')
        with self.assertRaises(ValueError):f.proof()
        self.assertNotIn(g.MOTION,f.reads)

    def test_wrong_summary_or_post_smalllock_or_unknown_source_decl_refused(self):
        for mode in ('summary','lock','source'):
            f=self.fixture()
            if mode=='summary':f.data[g.MOTION]['actual_test_auditor']={'path':g.AUDITOR_SOURCE,'sha256':'f'*64}
            elif mode=='lock':f.data[g.POST_LOCK]['files']['extraneous'] = 'f'*64
            else:f.data[g.POST_CONTROL]['sources'][g.MOTION_SOURCE]['sha256']='f'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.proof()

    def test_metadata_reader_only_fixed_names_and_same_hash_payload(self):
        r=g._Metadata(PROJECT,lambda name:b'{"one":1}')
        with self.assertRaises(ValueError):r.raw('UNOPENED/case.json')
        with self.assertRaises(ValueError):r.json(g.RESULT,'f'*64)
        r.json(g.RESULT);r.callback=lambda name:b'{"one":2}'
        with self.assertRaises(ValueError):r.close()

    def test_temp_reader_rejects_symlink_and_stat_identity_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/g.RESULT;path.parent.mkdir(parents=True);path.write_bytes(b'{}')
            r=g._Metadata(root,None);r.json(g.RESULT)
            clone=path.with_name('replacement.json');clone.write_bytes(b'{}');clone.replace(path)
            with self.assertRaises(ValueError):r.close()
            path.unlink();path.symlink_to(root/'other.json');(root/'other.json').write_bytes(b'{}')
            with self.assertRaises(ValueError):g._Metadata(root,None).json(g.RESULT)

    def resources(self):return old.Resources()
    def lease(self,resources,proof=None):
        return g.reserved_resources_after_TEST(PROJECT,'a'*64,resume_lock_sha256='b'*64,motion_summary_sha256='c'*64,
            postprocessing_lock_sha256='d'*64,enabled=True,read_bytes=Mock(),stop_check=resources.stop,reserve=resources.reserve,
            inventory=resources.inventory,is_alive=resources.is_alive)
    def simple_proof(self):return dict(schema=g.VERSION,complete=True,known_processes=[dict(pid=i,start_ticks=i+1) for i in range(11)],gate_chain='actual958/956/957 then actual964/963')

    def test_shared_canonical924_registry_and_965_own_registry(self):
        resources=self.resources()
        with patch.object(g,'require_complete_TEST_metadata',return_value=self.simple_proof()):
            with self.lease(resources) as permit:
                self.assertIs(type(permit),legacy._Permit);self.assertIn(permit,legacy._LEASES)
                self.assertEqual(g.require_permit(permit,PROJECT,allow_test_callbacks=True)['gate_chain'],'actual958/956/957 then actual964/963')
                self.assertEqual(legacy.require_permit(permit,PROJECT,allow_test_callbacks=True)['schema'],g.VERSION)
                with self.assertRaises(ValueError):g.require_permit(permit,PROJECT)
                tap=importlib.import_module('our_mani.multitask.train_public_collision_tap_v934')
                with self.assertRaises(ValueError):tap._require_lease(permit,PROJECT)  # production934 cannot use CPU callbacks
                with patch.object(g.os,'getpid',return_value=999999),self.assertRaises(ValueError):g.require_permit(permit,PROJECT,allow_test_callbacks=True)
                with self.assertRaises(ValueError):g.require_permit(permit,PROJECT/'wrong',allow_test_callbacks=True)
            with self.assertRaises(ValueError):g.require_permit(permit,PROJECT,allow_test_callbacks=True)
            with self.assertRaises(ValueError):legacy.require_permit(permit,PROJECT,allow_test_callbacks=True)
        old_permit=legacy._Permit();legacy._LEASES[old_permit]=dict(pid=g.os.getpid(),root=PROJECT,proof={},stop=lambda _:None,test_only=False)
        try:
            with self.assertRaises(ValueError):g.require_permit(old_permit,PROJECT)
            with self.assertRaises((ValueError,TypeError)):g.require_permit(object(),PROJECT)
        finally:legacy._LEASES.pop(old_permit,None)

    def test_lease_cleanup_preserves_exception_and_releases_both_registries(self):
        resources=self.resources();original=RuntimeError('CPU caller failure');permit=None
        with patch.object(g,'require_complete_TEST_metadata',return_value=self.simple_proof()):
            try:
                with self.lease(resources) as permit:raise original
            except RuntimeError as caught:self.assertIs(caught,original)
        self.assertFalse(resources.held);self.assertNotIn(permit,g._LEASES);self.assertNotIn(permit,legacy._LEASES)

    def test_mocked_production_seams_share_unmodified934_validator(self):
        resources=self.resources();tap=importlib.import_module('our_mani.multitask.train_public_collision_tap_v934')
        with patch.object(g,'require_complete_TEST_metadata',return_value=self.simple_proof()),patch.object(g,'_stop_guard',resources.stop), \
                patch.object(g,'_reserve',resources.reserve),patch.object(g,'_inventory',resources.inventory),patch.object(g,'_is_alive',resources.is_alive):
            with g.reserved_resources_after_TEST(PROJECT,'a'*64,resume_lock_sha256='b'*64,motion_summary_sha256='c'*64,
                    postprocessing_lock_sha256='d'*64,enabled=True) as permit:
                self.assertEqual(tap._require_lease(permit,PROJECT)['schema'],g.VERSION)
                self.assertEqual(g.require_permit(permit,PROJECT)['schema'],g.VERSION)
                resources.stop_at=resources.stop_calls+1
                with self.assertRaises(InterruptedError):g.require_permit(permit,PROJECT)
        with self.assertRaises(ValueError):tap._require_lease(permit,PROJECT)

    def test_STOP_busy_live_and_inventory_failure_never_register(self):
        for mode in ('stop1','stop2','stop3','busy','live','postlive','host','ancestors','incomplete','native'):
            resources=self.resources()
            if mode.startswith('stop'):resources.stop_at=int(mode[-1])
            elif mode=='busy':resources.busy=True
            elif mode in ('live','postlive'):resources.live=True
            elif mode=='host':resources.visibility['host_visibility_available']=False
            elif mode=='ancestors':resources.visibility['ancestor_chain_complete']=False
            elif mode=='incomplete':resources.visibility['same_uid_inventory_incomplete']=True
            else:resources.visibility['live_project_native_processes']=[dict(pid=99)]
            with self.subTest(mode=mode),patch.object(g,'require_complete_TEST_metadata',return_value=self.simple_proof()),self.assertRaises((InterruptedError,RuntimeError,ValueError)):
                with self.lease(resources):self.fail('invalid resource lease yielded')
            self.assertFalse(resources.held);self.assertEqual(len(g._LEASES),0)

    def test_exact_lock_order_and_new_process_names_covered(self):
        self.assertEqual(g.LOCKS[:3],(g.CONTROLLER+'.lock',g.PARENT_CONTROLLER+'.lock','runtime/framework/ordered_resume_VAL_TEST_v775.lock'))
        self.assertIn(g.POST_RUNTIME+'.lock',g.LOCKS)
        for source in (g.OWNER_SOURCE,g.CONTROLLER_SOURCE,g.AUDITOR_SOURCE,g.POST_SOURCE,g.MOTION_SOURCE,
                'scripts/diagnose_TRAIN_public_collision_composite_v966_r10.py'):
            self.assertTrue(Path(source).name.startswith(legacy.NATIVE_PREFIXES+g.EXTRA_PREFIXES))


if __name__=='__main__':unittest.main()
