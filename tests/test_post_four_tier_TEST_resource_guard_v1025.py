"""Only source and synthetic metadata/resources; no real gate or inventory."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import tempfile
import unittest
from unittest.mock import Mock,patch

from our_mani.multitask import post_four_tier_TEST_resource_guard_v1025 as g
from our_mani.multitask import post_TEST_resource_guard_v924 as legacy

PROJECT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj);return obj


base=module('CPU1025_base965_fixture',PROJECT/'tests/test_post_composite_TEST_resource_guard_v965.py')
old=base.old
motion=module('CPU1025_motion1015_contract',PROJECT/g.MOTION_SOURCE)
post=module('CPU1025_post1016_contract',PROJECT/g.POST_SOURCE)
C=motion.C


def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


class Fixture:
    """Actual four-tier metadata,134/450/616/720, allFalse/motionNone.

    No case/sidecar/manifest/model payloads are constructed or opened. The
    original965 synthetic history supplies the literal old915/955 row layers.
    """
    def __init__(self):
        initial=base.Fixture();self.data=initial.data;self.reads=[]
        self.codes={name:(PROJECT/name).read_bytes() for name in {*legacy.SOURCE_SHA,*g.SOURCE_SHA,*post.SOURCES}}
        self.data.update(self.codes)
        previous_plan=self.data[g.PARENT+'/plan.json'];ids=[r['job_id'] for r in previous_plan['jobs']];adopted=1200
        all_records=deepcopy(self.data[g.PARENT+'/status.json']['completed'])
        old_result=deepcopy(self.data[base.g.RESULT]);parent=self.data[g.PARENT_CONTROL]
        self.data[g.PARENT+'/status.json'].update(completed={k:deepcopy(all_records[k]) for k in ids[:adopted]},failed={},
            active=[],checking=[],ready_jobs=[],CPU_ready=0,native_slots=0,GPU_slots=0,CPU_launched=adopted-584,
            drain_requested=True,stop_requested=False)
        self.data[g.PARENT+'/terminal.json'].update(complete=False,completed=adopted,failed=0,drained=True,
            drain_requested=True,stop_requested=False,resumed_new_cases=adopted-584,CPU_not_launched=1920-adopted)
        self.data[g.PARENT+'/DRAIN']={'reason':'CPU synthetic natural speed-change DRAIN'}
        self.data[g.PARENT_CONTROLLER+'/status.json']=dict(stage='requires_concrete_repair',automatic_retry=False)
        self.data[g.PARENT_CONTROLLER+'/failure.json']=dict(complete=False,user_stop=False,automatic_retry=False)
        # Never invent a completed old958/957/964/963 after this partial drain.
        for name in [g.PARENT_CONTROLLER+'/terminal.json',base.g.RESULT,base.g.MOTION,
            *base.g.POST_PATHS.values()]:self.data.pop(name,None)
        self.data[g.QUALIFICATION_LAUNCH]=dict(pid=1610,start_ticks=1611,source_sha256=self.ptr(g.QUALIFIER_SOURCE)['sha256'])
        self.data[g.QUALIFICATION]=dict(schema='r10_independent_concurrent_TRAIN_resource_qualification_v1005',complete=True,split='train',
            source_sha256=self.ptr(g.QUALIFIER_SOURCE)['sha256'],resource_profile=g.PROFILE,execution_entry=parent['execution_implementation'],
            policy_sha256=g.POLICY,author_outcome_protocol=legacy.PROTOCOL,case_count=12,infrastructure_failed=0,
            cleanup_completed=True,resource_bounds_verified=True,overlap_verified=True,independent_audit_complete=True,
            observed_native_peak=8,observed_GPU_peak=3,actual_CPU_preflight_peak=4,joint_supply_overlap={'complete':True},
            hardware_observation=dict(actual_three_compute_contexts_during_joint8_3_4_observed=True,unknown_samples=0,
                required_headroom_mib=2048,minimum_sampled_headroom_mib=12000),formal_score_contribution=False,TEST_read=False,
            actual_independent_audit=self.ptr(g.QUALIFICATION_LAUNCH))
        ctl=deepcopy(parent);ctl.update(schema=C.CONTROL_SCHEMA,completed_job_ids=sorted(ids[:adopted]),failed_job_ids=[],
            remaining_job_ids=sorted(ids[adopted:]),completed_key_sha256=C.digest(sorted(ids[:adopted])),
            failed_key_sha256=C.digest([]),remaining_key_sha256=C.digest(sorted(ids[adopted:])),
            previous_resume_control=self.ptr(g.PARENT_CONTROL),previous_resume_control_lock=self.ptr(g.PARENT_LOCK),
            resource_profile=g.PROFILE,resource_qualification=self.ptr(g.QUALIFICATION),resource_qualifier=self.ptr(g.QUALIFIER_SOURCE),
            resource_profile_changed=True,previous_resource_profile=parent['resource_profile'],previous_resource_qualification=parent['resource_qualification'],
            resume_controller=self.ptr(g.CONTROLLER_SOURCE),resume_owner=self.ptr(g.OWNER_SOURCE),resume_auditor=self.ptr(g.AUDITOR_SOURCE),
            result_auditor=self.ptr(g.AUDITOR_SOURCE))
        for name in ('launch','plan','status','terminal'):ctl['previous_'+name]=self.ptr(g.PARENT+'/'+name+'.json')
        for name in ('launch','status','failure'):ctl['previous_controller_'+name]=self.ptr(g.PARENT_CONTROLLER+'/'+name+'.json')
        ctl['previous_DRAIN']=self.ptr(g.PARENT+'/DRAIN');self.data[g.CONTROL]=ctl
        files=deepcopy(self.data[g.PARENT_LOCK]['files'])
        for name in tuple(self.data):
            if g._allowed(name) and name not in (g.RESULT,g.CONTROL_LOCK):files[name]=self.ptr(name)['sha256']
        self.data[g.CONTROL_LOCK]=dict(schema=C.LOCK_SCHEMA,approved_for_execution=True,control=self.ptr(g.CONTROL),files=files)
        cp,lp=self.ptr(g.CONTROL),self.ptr(g.CONTROL_LOCK);jobs=C.resumed_jobs(previous_plan,ctl,ctl['execution_entry'])
        common={k:deepcopy(self.data[g.PARENT+'/launch.json'][k]) for k in ('split','policy_sha256','author_outcome_protocol',
            'execution_protocol_changed','automatic_retry','admission_driver','admission_guard','native_runtime','original_native_runtime','execution_implementation')}
        common.update(resume_control=cp,resume_control_lock=lp,resource_profile=ctl['resource_profile'],resource_qualification=ctl['resource_qualification'],
            **{k:ctl[k] for k in ('previous_resume_control','previous_resume_control_lock','environment_recheck','resource_profile_changed',
                'previous_resource_profile','previous_resource_qualification')},frozen_environment=old.environment(5010))
        owner=dict(common,pid=5010,start_ticks=5011,source_sha256=self.ptr(g.OWNER_SOURCE)['sha256'],execution_entry=ctl['execution_entry'],
            base_stage_owner=self.ptr(legacy.OLD_OWNER_SOURCE),parent_stage_owner=self.ptr(g.PARENT_OWNER_SOURCE),
            imported_completed_cases=adopted,max_native_slots=8,max_GPU_slots=3,max_CPU_preflight_slots=4)
        controller=dict(pid=5110,start_ticks=5111,source_sha256=self.ptr(g.CONTROLLER_SOURCE)['sha256'],resume_control=cp,
            resume_control_lock=lp,automatic_retry=False,frozen_environment=old.environment(5110))
        self.data[g.CONTROLLER+'/launch.json']=controller
        self.data[g.CONTROLLER+'/status.json']=dict(stage='complete',automatic_retry=False,frozen_environment=controller['frozen_environment'])
        owner_args=[]
        for key,flag in (('suite','--suite'),('suite_lock','--suite-lock'),('source_inputs','--source-inputs'),('validation_release','--validation-release')):
            owner_args.extend((flag,ctl[key]['path']))
        owner_args+=['--runtime',g.TEST_RUNTIME,'--out',g.ART+'/test','--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        audit_args=['--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK,'--resume-lock-sha256',lp['sha256']]
        for stage,source,pid,args in ((g.OWNER_STAGE,g.OWNER_SOURCE,5010,owner_args),(g.AUDIT_STAGE,g.AUDITOR_SOURCE,5210,audit_args)):
            self.data[stage+'/launch.json']=dict(pid=pid,start_ticks=pid+1,source_sha256=self.ptr(source)['sha256'],
                argv=[str(PROJECT/'.conda-env/bin/python'),'-B','-u',source,*args],automatic_retry=False,
                inherited_frozen_environment=controller['frozen_environment'],current_mount_namespace='CPU-private-mount')
            self.data[stage+'/terminal.json']=dict(complete=True,returncode=0,automatic_retry=False)
        owner['argv']=self.data[g.OWNER_STAGE+'/launch.json']['argv'][3:];self.data[g.TEST_RUNTIME+'/launch.json']=owner
        self.data[g.TEST_RUNTIME+'/plan.json']=dict(common,expected_cases=1920,jobs=jobs,imported_completed_cases=adopted,source_outcome_filtering=False)
        records={};cases=[];drivers=[];batches=[];tiers=[]
        owners={runtime:self.data[runtime+'/launch.json'] for runtime in (g.ORIGINAL,g.MIDDLE,g.PARENT)};owners[g.TEST_RUNTIME]=owner
        for row in jobs:
            key=row['job_id'];runtime=row['evidence_runtime'];out=legacy.option(row['physical_argv'],'--out')
            record=deepcopy(all_records[key])
            if not row['imported_completed']:
                record.update(case=self.unopened(out+'/case.json'),admission_driver_receipt=self.unopened(out+'/admission_driver_receipt.json'),
                    batch_driver_receipt=self.unopened(out+'/batch_driver_receipt.json'))
            records[key]=record;cases.append(record['case']);drivers.append(record['admission_driver_receipt'])
            if 'batch_driver_receipt' in record:batches.append(record['batch_driver_receipt'])
            actual=owners[runtime]
            tiers.append(dict(job_id=key,evidence_runtime=runtime,imported_completed=row['imported_completed'],actual_stage_owner_pid=actual['pid'],
                actual_stage_owner_start_ticks=actual['start_ticks'],resume_lineage=row['resume_lineage'],resume_lineage_v1010=row['resume_lineage_v1010'],
                previous_row_sha256=row['previous_row_sha256'],execution_entry=row['execution_entry'],actual_outer_argv=row['argv'],
                physical_argv=row['physical_argv'],case=record['case'],admission_driver_receipt=record['admission_driver_receipt'],batch_driver_receipt=record.get('batch_driver_receipt')))
        self.data[g.TEST_RUNTIME+'/status.json']=dict(common,expected=1920,completed=records,failed={},active=[],checking=[],ready_jobs=[],CPU_ready=0,
            native_slots=0,GPU_slots=0,CPU_launched=1920-adopted,imported_completed=adopted,owner_error=None,stop_requested=False,drain_requested=False)
        self.data[g.TEST_RUNTIME+'/terminal.json']=dict(common,complete=True,expected=1920,completed=1920,failed=0,owner_error=None,
            stop_requested=False,drain_requested=False,drained=False,CPU_ready_unexecuted=[],CPU_not_launched=0,formal_release_requires_independent_audit=True,
            imported_completed_cases=adopted,resumed_new_cases=1920-adopted,native_successes=0)
        counts={g.ORIGINAL:134,g.MIDDLE:450,g.PARENT:adopted-584,g.TEST_RUNTIME:1920-adopted}
        result=old_result;result.update(schema=motion.RESULT_SCHEMA,source_sha256=self.ptr(g.AUDITOR_SOURCE)['sha256'],resume_control=cp,resume_control_lock=lp,
            resource_profile=ctl['resource_profile'],resource_qualification=ctl['resource_qualification'],resource_profile_changed=True,
            previous_resource_profile=ctl['previous_resource_profile'],previous_resource_qualification=ctl['previous_resource_qualification'],
            old_completed_results_not_rerun=True,old907_or958_successful_terminal_claimed=False,original754_906_or957_complete_result_claimed=False,
            cases=cases,admission_driver_receipts=drivers,batch_driver_receipts=batches,evidence_tiers=tiers,frozen_environment=old.environment(5210),
            auditor_invocation=dict(pid=5210,start_ticks=5211,source=self.ptr(g.AUDITOR_SOURCE),argv=self.data[g.AUDIT_STAGE+'/launch.json']['argv'][3:],
                inherited_physical_stage_identity_claimed=False),historical_infrastructure_attempts=[dict(runtime=runtime,job_ids=c['failed_job_ids'],
                previous_status=c['previous_status'],previous_terminal=c['previous_terminal'],counted_as_native_task_failures=False,
                all_keys_reexecuted_once_in_new_namespace=True) for runtime,c in ((g.MIDDLE,parent),(g.PARENT,ctl))])
        result['actual_stage_owners']=[dict(runtime=runtime,launch=self.ptr(runtime+'/launch.json'),terminal=self.ptr(runtime+'/terminal.json'),
            pid=owners[runtime]['pid'],start_ticks=owners[runtime]['start_ticks'],completed=counts[runtime],resource_limits={k:owners[runtime][k]
                for k in ('max_native_slots','max_GPU_slots','max_CPU_preflight_slots')}) for runtime in (g.ORIGINAL,g.MIDDLE,g.PARENT,g.TEST_RUNTIME)]
        result['inputs_sha256']={str(PROJECT/name):self.ptr(name)['sha256'] for name in self.data if g._allowed(name)
            and not name.startswith(g.CONTROLLER+'/') and name not in (g.RESULT,base.g.RESULT)}
        result['inputs_sha256'][str(PROJECT/legacy.SOURCE_LOCK)]=ctl['source_inputs']['sha256']
        for name in (legacy.QUALIFICATION_LAUNCH,g.QUALIFICATION_LAUNCH):result['inputs_sha256'].pop(str(PROJECT/name),None)
        self.data[g.RESULT]=result;self.data[g.CONTROLLER+'/terminal.json']=dict(complete=True,automatic_retry=False,TEST_results=self.ptr(g.RESULT))
        self.data[g.AUDIT_STAGE+'/stdout.log']=dict(complete=True,schema=motion.RESULT_SCHEMA,split='test',cases=1920,out=g.RESULT,
            motion_is_diagnostic_only=True,native_successes_by_method=result['native_successes_by_method'])
        self.post(cp,lp,counts,result)

    def post(self,cp,lp,counts,result):
        ctl=dict(schema=post.SCHEMA,execution_enabled=True,read_only_analysis=True,automatic_retry=False,policy_sha256=g.POLICY,
            runtime=g.POST_RUNTIME,controller_runtime=g.CONTROLLER,owner_runtime=g.TEST_RUNTIME,test_results=g.RESULT,motion_summary=g.MOTION,
            poll_seconds=1,sources={name:self.ptr(name) for name in post.SOURCES},resume_control=cp,resume_control_lock=lp,
            controller_launch=self.ptr(g.CONTROLLER+'/launch.json'))
        self.data[g.POST_CONTROL]=ctl
        names={*post.SOURCES,g.POST_CONTROL,g.CONTROL,g.CONTROL_LOCK,g.PARENT_CONTROL,g.CONTROLLER+'/launch.json'}
        self.data[g.POST_LOCK]=dict(schema=post.LOCK_SCHEMA,execution_enabled=True,files={n:self.ptr(n)['sha256'] for n in names})
        pcp,plp=self.ptr(g.POST_CONTROL),self.ptr(g.POST_LOCK);host=dict(standard_host_analysis=True,frozen_environment_inherited_claimed=False,automatic_retry=False)
        self.data[g.POST_RUNTIME+'/launch.json']=dict(host,pid=5310,start_ticks=5311,source_sha256=self.ptr(g.POST_SOURCE)['sha256'],
            argv=[g.POST_SOURCE,'--execute','--lock-sha256',plp['sha256']],control=pcp,input_lock=plp,monitored_controller_launch=ctl['controller_launch'])
        self.data[g.MOTION]=dict(schema=post.SUMMARY_SCHEMA,complete=True,read_only_analysis=True,split='test',task_count=16,source_count=640,case_count=1920,
            methods=list(post.METHODS),task_source_counts={t:40 for t in post.TASKS},policy_sha256=g.POLICY,source_sha256=self.ptr(g.MOTION_SOURCE)['sha256'],
            test_results=self.ptr(g.RESULT),resume_control=cp,resume_control_lock=lp,actual_test_auditor=self.ptr(g.AUDITOR_SOURCE),
            full1920_four_tier_TEST_results_and_complete1013_observed=True,provenance_runtime_counts=counts,actual_stage_owners=result['actual_stage_owners'],
            new_release_granted_by_this_summary=False,native_false_retained=True,success_rate_or_acceleration_filter_used=False,native_GPU_or_jobs_started=False,
            TEST_used_for_fitting=False,TXT_or_handoff_written=False,inputs_sha256={n:self.ptr(n)['sha256'] for n in (g.RESULT,g.CONTROL,g.CONTROL_LOCK,g.MOTION_SOURCE)})
        sp=self.ptr(g.MOTION);self.data[g.POST_RUNTIME+'/status.json']=dict(stage='complete',automatic_retry=False,motion_summary=sp)
        self.data[g.POST_RUNTIME+'/terminal.json']=dict(host,schema='r10_readonly_post_complete_composite_TEST_motion_terminal_v1016',complete=True,
            motion_summary=sp,TEST_results=self.ptr(g.RESULT),control=pcp,input_lock=plp,resume_control=cp,resume_control_lock=lp,Native_GPU_TRAIN_started=False,TXT_modified=False)
        self.data[g.POST_STAGE+'/launch.json']=dict(host,pid=5410,start_ticks=5411,source=self.ptr(g.MOTION_SOURCE),source_sha256=self.ptr(g.MOTION_SOURCE)['sha256'],
            owner_pid=5310,owner_start_ticks=5311,control=pcp,input_lock=plp,TEST_results=self.ptr(g.RESULT),argv=[str(PROJECT/'.conda-env/bin/python'),'-B','-u',g.MOTION_SOURCE,
                '--test-results-sha256',self.ptr(g.RESULT)['sha256'],'--resume-lock-sha256',lp['sha256'],'--resume-control',g.CONTROL,'--resume-lock',g.CONTROL_LOCK])
        self.data[g.POST_STAGE+'/terminal.json']=dict(complete=True,returncode=0,interrupted=False,automatic_retry=False)
        self.data[g.POST_STAGE+'/stdout.log']=dict(complete=True,output=g.MOTION,sha256=sp['sha256'],case_count=1920)

    def unopened(self,name):return dict(path=name,sha256=g.sha(('unopened:'+name).encode()))
    def raw(self,name):
        v=self.data[name];return v if isinstance(v,bytes) else encode(v)
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
        raw=(PROJECT/g.SOURCE).read_bytes();obj=ModuleType('CPU1025_inert');obj.__file__=str(PROJECT/g.SOURCE)
        denied=Mock(side_effect=AssertionError('inert1025 touched OS'))
        with patch.object(Path,'resolve',denied),patch.object(Path,'read_bytes',denied),patch.object(Path,'exists',denied),patch.object(g.importlib,'import_module',denied):
            exec(compile(raw,obj.__file__,'exec'),obj.__dict__)
            scope=obj.reserved_resources_after_TEST('unused','a'*64,resume_lock_sha256='b'*64,motion_summary_sha256='c'*64,postprocessing_lock_sha256='d'*64)
            with self.assertRaises(ValueError):scope.__enter__()

    def test_complete1920_metadata_False_None_and_historical6_retained(self):
        f=self.fixture();proof=f.proof()
        self.assertEqual(proof['provenance_runtime_counts'],{g.ORIGINAL:134,g.MIDDLE:450,g.PARENT:616,g.TEST_RUNTIME:720})
        self.assertEqual(len(proof['known_processes']),14);self.assertEqual(proof['task_count'],16)
        self.assertEqual(proof['postprocessing'],{key:f.ptr(name) for key,name in g.POST_PATHS.items()})
        self.assertFalse(proof['scores_used_for_gate']);self.assertTrue(proof['registration_only_compatibility'])
        self.assertEqual(len(proof['historical_infrastructure_attempts'][0]['job_ids']),6)
        self.assertFalse(any(name.endswith('/case.json') or 'manifest' in name.lower() or 'CPU/' in name or '/native/' in name for name in f.reads))

    def test_obsolete_three_tier_result_cannot_open_new_postprocessing(self):
        f=self.fixture();f.data[g.RESULT]['schema']='r10_three_tier_composite_formal_TEST_results_v957'
        f.data[g.CONTROLLER+'/terminal.json']['TEST_results']=f.ptr(g.RESULT)
        with self.assertRaises(ValueError):f.proof()
        self.assertNotIn(g.MOTION,f.reads);self.assertFalse(set(g.POST_PATHS.values())&set(f.reads))

    def test_duplicate_recorded_process_identity_never_grants_a_permit(self):
        f=self.fixture();f.data[g.POST_RUNTIME+'/launch.json']['pid']=f.data[g.CONTROLLER+'/launch.json']['pid']
        f.data[g.POST_RUNTIME+'/launch.json']['start_ticks']=f.data[g.CONTROLLER+'/launch.json']['start_ticks']
        f.data[g.POST_STAGE+'/launch.json']['owner_pid']=f.data[g.POST_RUNTIME+'/launch.json']['pid']
        f.data[g.POST_STAGE+'/launch.json']['owner_start_ticks']=f.data[g.POST_RUNTIME+'/launch.json']['start_ticks']
        with self.assertRaisesRegex(ValueError,'identities must be distinct'):f.proof()

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

    def test_four_tiers_duplicate_or_historical_outcome_rewrite_refused(self):
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
        for path,key,value in ((g.POST_RUNTIME+'/status.json','stage','saved_motion_v1015'),(g.POST_RUNTIME+'/terminal.json','complete',False),
                (g.POST_STAGE+'/terminal.json','interrupted',True),(g.POST_STAGE+'/launch.json','owner_pid',999)):
            f=self.fixture();f.data[path][key]=value
            with self.subTest(path=path),self.assertRaises(ValueError):f.proof()
            self.assertNotIn(g.MOTION,f.reads)

    def test_exact_absolute1016_source_entry_is_compatible(self):
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
    def simple_proof(self):return dict(schema=g.VERSION,complete=True,known_processes=[dict(pid=i,start_ticks=i+1) for i in range(1,15)],gate_chain='actual1013/1011/1012 then actual1016/1015')

    def test_shared_canonical924_registry_and_1025_own_registry(self):
        resources=self.resources()
        with patch.object(g,'require_complete_TEST_metadata',return_value=self.simple_proof()):
            with self.lease(resources) as permit:
                self.assertIs(type(permit),legacy._Permit);self.assertIn(permit,legacy._LEASES)
                self.assertEqual(g.require_permit(permit,PROJECT,allow_test_callbacks=True)['gate_chain'],'actual1013/1011/1012 then actual1016/1015')
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
        self.assertEqual(g.LOCKS[:3],(g.CONTROLLER+'.lock',g.PARENT_CONTROLLER+'.lock',g.MIDDLE_CONTROLLER+'.lock'))
        self.assertIn(g.POST_RUNTIME+'.lock',g.LOCKS)
        for source in (g.OWNER_SOURCE,g.CONTROLLER_SOURCE,g.AUDITOR_SOURCE,g.POST_SOURCE,g.MOTION_SOURCE,
                'scripts/diagnose_TRAIN_public_collision_composite_v966_r10.py'):
            self.assertTrue(Path(source).name.startswith(legacy.NATIVE_PREFIXES+g.EXTRA_PREFIXES))


if __name__=='__main__':unittest.main()
