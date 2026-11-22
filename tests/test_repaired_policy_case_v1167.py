"""CPU-only unified worker contracts; every environment/physical loop is a stub."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import repaired_policy_case_v1167 as subject
from our_mani.multitask import rml_case_preflight_v742 as substrate
from our_mani.multitask import rml_component_domain_v640 as domains
from scripts import run_repaired_policy_case_v1167_r10 as entry
from tests.test_repaired_policy_graph_v1143 import graph_spec,put
from tests.test_repaired_readiness_v1144 import protocol as sample_protocol, sampled_case

ROOT=Path(__file__).resolve().parents[1]


def write(root,name,value):
    return put(root,name,json.dumps(value,sort_keys=True).encode())


class Fixture:
    def __init__(self,root):
        self.root=Path(root);spec=graph_spec(root);self.calls=[]
        def source(name):return put(root,name,(ROOT/name).read_bytes())
        worker,preparer,consumer=(source(name) for name in (subject.WORKER,subject.SOURCE,subject.RUNTIME))
        spec['physical']['sources_map']['consumer']=consumer
        for branch,(name,callable_name) in subject.FACTORIES.items():
            ref=source(name);spec['physical']['sources_map'][branch]=ref
            spec['physical']['factories'][branch]=dict(source=ref,callable=callable_name)
        spec['physical']['factories'].pop('selected')
        spec['physical']['sources_map'].pop('factory')
        spec['control']['sources_map'].update(worker=worker,preparer1167=preparer)
        for component,name in (('resource_guard',subject.GUARD_SOURCE),('environment_wrapper',subject.ENV_SOURCE),
            ('component_purpose',subject.PURPOSE_SOURCE),('admission_guard',subject.ADMISSION_SOURCE),
            ('formal_source_reader',subject.ADMISSION_SOURCE),('split_source_lock_writer',subject.SOURCE_INPUTS_SOURCE)):
            ref=source(name);spec['control']['components'][component]=ref;spec['control']['sources_map'][component]=ref
        spec['control']['sources_map']['substrate']=source(subject.PREFLIGHT)
        self.profile={'opaque_static_profile':'CPU_fixture_only'}
        self.protocol=sample_protocol()
        spec['physical']['common_RML']['profile']=write(root,'metadata/common_profile.json',self.profile)
        spec['physical']['outcome_sampling']['protocol']=write(root,'metadata/outcome_protocol.json',self.protocol)
        namespace=write(root,'metadata/cpu_namespace_proof_v820.json',{'complete':True})
        spec['control']['static_inputs']['environment_namespace_qualification']=namespace
        rows=[dict(task=task,raw_episode_id='episode'+str(index),path='never_opened/'+task+'/'+str(index)+'.h5')
            for task in subject.graphs.TASKS for index in range(1,101)]
        spec['train_source_manifest']=write(root,subject.TRAIN_MANIFEST,dict(split='train',episodes=rows))
        core_name=subject.FACTORIES['core_geometry'][0]
        block_name=subject.FACTORIES['block_geometry'][0]
        files={name:spec['physical']['sources_map'][branch]['sha256']
            for branch,name in (('core_geometry',core_name),('block_geometry',block_name))}
        cpu=write(root,'metadata/physical_CPU_evidence.json',dict(complete=True,tests_exit_code=0,native_started=False,
            GPU_started=False,source_hashes=files))
        self.overlay=dict(integration_complete=True,unpinned_sources=[],files=files,required_sources=list(files),
            integration_cpu_evidence=cpu,original_745_execution_acceptance_granted=False,formal_VAL_TEST_qualification=False,
            core_factory=dict(module=core_name[:-3].replace('/','.'),source=core_name,
                function=subject.FACTORIES['core_geometry'][1],sha256=files[core_name]),
            block_factory=dict(module=block_name[:-3].replace('/','.'),source=block_name,
                function=subject.FACTORIES['block_geometry'][1],sha256=files[block_name]))
        spec['physical']['source_overlay_lock']=write(root,'metadata/source_overlay_lock.json',self.overlay)
        for row in spec['routes']:
            factory=subject._branch(row['task'],row['method']);row['physical_factory']=factory
            for scope in ('train','evaluation'):
                declaration=row[scope];declaration.update(worker=worker,preparer=preparer,consumer=consumer,physical_factory=factory,
                    case_schema=subject.CASE_SCHEMAS['train' if scope=='train' else 'validation'])
        self.spec=spec;self.ref=subject.graphs.write_policy_graph(root,spec,namespace='artifacts/framework/multitask_r10/graph1167')
        self.graph=subject.graphs.read_policy_graph(root,self.ref)

    def args(self,*,split='train',method='cv_geometry',task='insert_onto_rotating_peg',check=True,raw='episode1'):
        route=next(x for x in self.graph['routes'] if (x['task'],x['method'])==(task,method))
        value=NS(graph=self.ref['path'],config=route['configuration']['path'],split=split,task=task,raw_episode_id=raw,
            method=method,out=subject.ARTIFACT_PREFIX+split+'/'+('development_demo_only/' if split=='train' else '')+task+'/'+raw+'/'+method,
            runtime=subject.RUNTIME_PREFIX+split+'/'+task+'/'+raw+'/'+method,check=check,
            suite=None,suite_lock=None,source_inputs=None,admission=None)
        if split=='validation':
            value.suite='metadata/suite.json';value.suite_lock='metadata/suite_lock.json'
            value.source_inputs='metadata/validation_source_inputs.json';value.admission='metadata/admission.json'
            admission=write(self.root,value.admission,{'explicit':'CPU fixture'})
            stamp={key:self.graph[key] for key in subject.readiness.HASHES}
            source_inputs=write(self.root,value.source_inputs,dict(schema=self.graph['control']['contracts']['split_source_lock_schema'],
                producer_schema=subject.SOURCE_INPUTS_SCHEMA,producer=self.graph['control']['components']['split_source_lock_writer'],
                split='validation',source_count=315,source_manifest_sha256='a'*64,graph=self.ref,admission=admission,**stamp,
                TEST_used_for_fitting=False,source_read_after_admission=True,files={}))
            suite=write(self.root,value.suite,dict(schema='r10_repaired_validation_suite_v1147',graph=self.ref,admission=admission,
                source_inputs=source_inputs,source_manifests={'validation':{'path':'never_read_VAL_manifest.json','sha256':'a'*64}},**stamp))
            write(self.root,value.suite_lock,dict(schema=subject.SOURCE_INPUTS_SCHEMA+'_suite_evidence_lock',graph=self.ref,
                admission=admission,policy_sha256=self.graph['policy_sha256'],split='validation',formal_TEST_permitted=False,
                TEST_used_for_fitting=False,files={str(self.root/ref['path']):ref['sha256'] for ref in
                    (self.ref,admission,source_inputs,suite)}))
        return value

    def preflight(self,root,args,worker):
        self.calls.append((root,args,worker))
        if args.split=='train':
            domain=domains.for_train(root,{subject.TRAIN_MANIFEST:self.graph['train_source_manifest']['sha256']},
                args.task,args.raw_episode_id,args.method)
            row=dict(task=args.task,raw_episode_id=args.raw_episode_id,path='never_opened/data.h5')
            policy=None
        else:
            domain,row,proof=domains.for_evaluation(root,{}, {},args.split,args.task,args.raw_episode_id,args.method)
            policy=proof['policy_sha256']
        config=dict(context_checkpoint=self.graph['physical']['shared_context_checkpoint'],baseline_repo='/not-opened',
            baseline_checkpoint=str(root/'metadata/shared.pt'))
        configured=next(row for row in self.graph['routes'] if (row['task'],row['method'])==(args.task,args.method))
        model=next(iter(configured['models'].values()))
        config['learned_artifacts']={args.task:{name:model for name in ('geometry','interaction','rigid_layout')}}
        return dict(args=args,row=row,original_row=row,config=config,files={},out=root/args.out,formal=args.split=='validation',
            identity={'sha256':'original_identity','config_sha256':'oldconfig','inputs_sha256':'oldinputs'},
            component_domain=domain,model_preflight={'actual_shared16_block_adapter':{'CPU':'fixture'}},
            common_author_RML_profile=self.profile,full16_policy_sha256=policy,runtime_scope_restoration=None)

    @contextmanager
    def prepared(self,args):
        with patch.object(substrate,'prepare',side_effect=self.preflight),patch.object(substrate,'_checked'), \
             patch.object(substrate,'outcome_protocol_binding',return_value=self.protocol):
            yield subject.prepare(self.root,args)


class Source:
    def select(self):return None
    def _done(self,goal):return False
    def candidates(self,phase):return []
    def receipt(self):return {'original_source':True}


class Frontend:
    def observe(self,*args):return 'original observe'


class Dispatch:
    def __init__(self,source):self.source_policy=source


class CaseContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.f=Fixture(self.root)
        selected=patch.object(subject,'PHYSICAL_SOURCE_OVERLAY',self.f.graph['physical']['source_overlay_lock'])
        selected.start();self.addCleanup(selected.stop)

    def test_overlay_requires_both_registered_sources_and_factory_signatures(self):
        provenance=subject._source_overlay(self.root,self.f.graph)
        self.assertEqual(provenance['source_count'],2)
        self.assertTrue(provenance['qualified_as_source_provenance_only'])
        for field,branch in (('core_factory','core_geometry'),('block_factory','block_geometry')):
            declared=self.f.overlay[field];name,function=subject.FACTORIES[branch]
            self.assertEqual(declared['source'],name);self.assertEqual(declared['function'],function)
            self.assertEqual(self.f.overlay['files'][name],declared['sha256'])
            self.assertEqual(declared['sha256'],hashlib.sha256((self.root/name).read_bytes()).hexdigest())

    def test_overlay_rejects_missing_or_tampered_block_factory_and_missing_factory_sources(self):
        cases=('missing_block_factory','wrong_block_source','wrong_block_function','wrong_block_sha',
            'wrong_block_module','missing_block_source_in_files','missing_core_source_in_files')
        for index,kind in enumerate(cases):
            changed=deepcopy(self.f.overlay);graph=deepcopy(self.f.graph)
            if kind=='missing_block_factory':changed.pop('block_factory')
            elif kind=='wrong_block_source':changed['block_factory']['source']=subject.FACTORIES['core_geometry'][0]
            elif kind=='wrong_block_function':changed['block_factory']['function']='unregistered'
            elif kind=='wrong_block_sha':changed['block_factory']['sha256']='0'*64
            elif kind=='wrong_block_module':changed['block_factory']['module']='unregistered.module'
            else:
                branch='block_geometry' if kind=='missing_block_source_in_files' else 'core_geometry'
                changed['files'].pop(subject.FACTORIES[branch][0])
                changed['required_sources']=list(changed['files'])
            # Rebind valid CPU evidence to the actual altered file set, so the
            # negative assertion reaches the physical factory/source contract.
            changed['integration_cpu_evidence']=write(self.root,'metadata/overlay_negative_'+str(index)+'_cpu.json',
                dict(complete=True,tests_exit_code=0,native_started=False,GPU_started=False,
                    source_hashes=changed['files']))
            graph['physical']['source_overlay_lock']=write(self.root,
                'metadata/overlay_negative_'+str(index)+'.json',changed)
            with patch.object(subject,'PHYSICAL_SOURCE_OVERLAY',graph['physical']['source_overlay_lock']), \
                 self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'physical overlay must exactly bind'):
                subject._source_overlay(self.root,graph)

    def test_unselected_physical_overlay_is_rejected_before_json_read(self):
        changed=deepcopy(self.f.graph)
        changed['physical']['source_overlay_lock']={'root':'project','path':'metadata/other_overlay.json','sha256':'0'*64}
        with patch.object(subject.graphs,'read_project_json',side_effect=AssertionError('read before selected provenance')):
            with self.assertRaisesRegex(ValueError,'exact frozen1165 source provenance'):
                subject._source_overlay(self.root,changed)

    def test_old1146_worker_preparer_and_case_header_cannot_claim_new_graph_route(self):
        for field in ('worker','preparer','case_schema'):
            changed=deepcopy(self.f.graph)
            route=next(x for x in changed['routes'] if (x['task'],x['method'])==('insert_onto_rotating_peg','cv_geometry'))
            if field=='case_schema':route['train'][field]='r10_graph_bound_repaired_policy_case_v1146_TRAIN_case'
            else:
                name=(subject.WORKER if field=='worker' else subject.SOURCE).replace('1167','1146')
                route['train'][field]={'root':'project','path':name,'sha256':'0'*64}
            with self.subTest(field=field),patch.object(subject.graphs,'read_policy_graph',return_value=changed), \
                 patch.object(subject.importlib,'import_module',side_effect=AssertionError('no constructor')):
                with self.assertRaises(ValueError):subject.selection(self.root,self.f.args())

    def test_test_is_refused_before_graph_or_source_read(self):
        args=self.f.args();args.split='test'
        with patch.object(subject,'pointer',side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError,'TEST refused'):subject.selection(self.root,args)

    def test_real_graph_declares_actual_new_worker_preparer_consumer_and_three_branches(self):
        for method,task,branch in (('cv_geometry','insert_onto_rotating_peg','core_geometry'),
            ('context_geometry',subject.BLOCKS[0],'block_geometry'),('physmani','insert_onto_rotating_peg','physmani')):
            chosen=subject.selection(self.root,self.f.args(method=method,task=task))
            self.assertEqual(chosen['factory']['branch'],branch)
            self.assertEqual(chosen['declaration']['worker']['path'],subject.WORKER)
            self.assertEqual(chosen['declaration']['preparer']['path'],subject.SOURCE)
            self.assertEqual(chosen['declaration']['consumer']['path'],subject.RUNTIME)

    def test_unregistered_factory_and_legacy_case_schema_are_rejected_without_import(self):
        for kind in ('factory','schema'):
            changed=deepcopy(self.f.graph)
            route=next(x for x in changed['routes'] if (x['task'],x['method'])==('insert_onto_rotating_peg','cv_geometry'))
            if kind=='factory':changed['physical']['factories'][route['physical_factory']]['callable']='other_factory'
            else:route['train']['case_schema']='r10_full16_explicit_author_action_outcome_v744'
            with patch.object(subject.graphs,'read_policy_graph',return_value=changed):
                with self.assertRaises(ValueError):subject.selection(self.root,self.f.args())

    def test_wrong_config_worker_and_output_purpose_are_rejected(self):
        args=self.f.args();args.config='other.json'
        with self.assertRaises(ValueError):subject.selection(self.root,args)
        args=self.f.args();args.out=args.out.replace('development_demo_only/','')
        with self.assertRaises(ValueError):subject.selection(self.root,args)
        args=self.f.args();args.admission='not-a-TRAIN-argument.json'
        with self.assertRaises(ValueError):subject.selection(self.root,args)
        changed=deepcopy(self.f.graph)
        route=next(x for x in changed['routes'] if (x['task'],x['method'])==('insert_onto_rotating_peg','cv_geometry'))
        route['train']['worker']=route['train']['consumer']
        with patch.object(subject.graphs,'read_policy_graph',return_value=changed):
            with self.assertRaises(ValueError):subject.selection(self.root,self.f.args())

    def test_train_uses_real640_fixed_four_and_actual_new_worker_in742(self):
        args=self.f.args()
        with self.f.prepared(args) as prepared:
            actual=self.f.calls[-1]
            self.assertEqual(actual[2],self.root/subject.WORKER)
            self.assertEqual(actual[1].out,args.out)
            self.assertEqual(actual[1].manifest,subject.TRAIN_MANIFEST)
            self.assertTrue(actual[1].development_demo)
            self.assertEqual(prepared['component_domain'].require(args.task),'train_qualification')
            self.assertEqual(prepared['graph_witness']['purpose'],'train_qualification')
            self.assertEqual(prepared['graph_witness']['graph'],dict(root='project',**self.f.ref))
            self.assertEqual(prepared['identity']['graph_sha256'],self.f.ref['sha256'])
            self.assertEqual(prepared['identity']['policy_sha256'],self.f.graph['policy_sha256'])
            self.assertFalse(prepared['execution_protocol']['formal_score_contribution'])
            self.assertTrue(prepared['execution_protocol']['actual742_worker_is1167'])
            self.assertNotIn('actual742_worker_is1146',prepared['execution_protocol'])
        with self.assertRaises(ValueError):
            with self.f.prepared(self.f.args(raw='episode5')):pass

    def test_validation_prepare_cannot_precede_scoped_new_admission(self):
        args=self.f.args(split='validation')
        with patch.object(substrate,'prepare',side_effect=AssertionError('must not run')):
            with self.assertRaisesRegex(ValueError,'1147 scope'):subject.prepare(self.root,args)

    def test_validation_real640_evaluation_and_reader_restore_in_entire_scope(self):
        args=self.f.args(split='validation');chosen=subject.selection(self.root,args)
        old=importlib_original('our_mani.multitask.formal_suite_admission_v532');original=old.read_formal_source
        events=[];permit=object()
        @contextmanager
        def admitted(root,graph,admission,actual_permit):
            self.assertIs(actual_permit,permit);events.append('admit_enter')
            try:yield None
            finally:events.append('admit_exit')
        def reader(root,suite,files,split,*,validation_release=None):
            events.append('source_read')
            return {},[dict(task=args.task,raw_episode_id=args.raw_episode_id,path='never_opened_VAL/data.h5')],{'policy_sha256':self.f.graph['policy_sha256']}
        module=NS(__file__=str(self.root/subject.ADMISSION_SOURCE),admission_scope=admitted,read_formal_source=reader)
        def imported(name):return module if name==subject.ADMISSION_MODULE else importlib_original(name)
        error=RuntimeError('original Native stub failure')
        with patch.object(subject.importlib,'import_module',side_effect=imported):
            with self.assertRaises(RuntimeError) as caught:
                with subject.admission_scope(self.root,chosen,args,permit):
                    with self.f.prepared(args) as prepared:
                        self.assertEqual(prepared['component_domain'].require(args.task),'evaluation')
                        self.assertEqual(prepared['graph_witness']['actual_split'],'validation')
                        self.assertTrue(prepared['execution_protocol']['formal_score_contribution'])
                        self.assertIs(old.read_formal_source,reader);raise error
        self.assertIs(caught.exception,error);self.assertIs(old.read_formal_source,original)
        self.assertEqual(events,['admit_enter','source_read','admit_exit'])

    def test_formal_metadata_closed_bundle_never_opens_the_declared_VAL_manifest(self):
        args=self.f.args(split='validation');chosen=subject.selection(self.root,args)
        active={'graph':chosen['graph_ref'],'admission':subject.pointer(self.root,args.admission)}
        result=subject._formal_metadata(self.root,chosen,args,active)
        self.assertEqual(result['source_inputs'],subject.pointer(self.root,args.source_inputs))
        self.assertEqual(result['manifest']['path'],'never_read_VAL_manifest.json')
        self.assertFalse((self.root/result['manifest']['path']).exists())

    def test_formal_source_lock_cannot_be_swapped_or_relabelled_under_same_suite(self):
        cases=(('inputs','producer_schema'),('inputs','producer'),('inputs','split'),
            ('inputs','source_manifest_sha256'),('inputs','policy_sha256'),('inputs','graph'),
            ('suite','graph'),('suite','admission'),('suite','source_inputs'),
            ('lock','graph'),('lock','admission'),('lock','policy_sha256'),('lock','files'))
        for target,field in cases:
            args=self.f.args(split='validation');chosen=subject.selection(self.root,args)
            active={'graph':chosen['graph_ref'],'admission':subject.pointer(self.root,args.admission)}
            objects={key:json.loads((self.root/getattr(args,name)).read_text()) for key,name in
                (('inputs','source_inputs'),('suite','suite'),('lock','suite_lock'))}
            bad='wrong' if field not in ('producer','graph','admission','source_inputs','files') else (
                {} if field=='files' else {'path':'metadata/other.json','sha256':'b'*64})
            objects[target][field]=bad
            if target=='inputs':
                objects['suite']['source_inputs']=write(self.root,args.source_inputs,objects['inputs'])
            suite_ref=write(self.root,args.suite,objects['suite'])
            if target!='lock' or field!='files':
                objects['lock']['files'][str(self.root/args.suite)]=suite_ref['sha256']
                if target=='inputs':
                    objects['lock']['files'][str(self.root/args.source_inputs)]=objects['suite']['source_inputs']['sha256']
            write(self.root,args.suite_lock,objects['lock'])
            with self.subTest(target=target,field=field),self.assertRaises(ValueError):
                subject._formal_metadata(self.root,chosen,args,active)

    def test_post_admission_metadata_is_rechecked_after_execution(self):
        args=self.f.args(split='validation');chosen=subject.selection(self.root,args)
        active={'graph':chosen['graph_ref'],'admission':subject.pointer(self.root,args.admission)}
        metadata=subject._formal_metadata(self.root,chosen,args,active)
        prepared=dict(graph=self.f.graph,graph_ref=chosen['graph_ref'],formal_metadata=metadata,
            execution_protocol={'source':'CPU fixture'},identity={'execution_protocol_sha256':subject.digest({'source':'CPU fixture'})})
        with patch.object(substrate,'verify'),patch.object(subject.graphs,'read_policy_graph',return_value=self.f.graph), \
             patch.object(subject,'_source_overlay'):
            subject.verify(self.root,prepared)
            (self.root/args.source_inputs).write_text('{"changed_after_case":true}')
            with self.assertRaisesRegex(ValueError,'metadata changed'):subject.verify(self.root,prepared)

    def test_graph_frozen_model_profile_protocol_and_physical_CPU_evidence_must_match(self):
        args=self.f.args()
        with patch.object(substrate,'prepare',side_effect=self.f.preflight),patch.object(substrate,'_checked'), \
             patch.object(substrate,'outcome_protocol_binding',return_value={'changed':True}):
            with self.assertRaises(ValueError):subject.prepare(self.root,args)
        chosen=subject.selection(self.root,args);overlay=deepcopy(self.f.overlay);overlay['integration_complete']=False
        original=subject.graphs.read_project_json
        def read(root,ref):
            if ref==chosen['graph']['physical']['source_overlay_lock']:return overlay,ref
            return original(root,ref)
        with patch.object(subject.graphs,'read_project_json',side_effect=read):
            with self.assertRaises(ValueError):subject._source_overlay(self.root,chosen['graph'])

    def test_first_write_annotation_is_truthful_and_never_mutates_original_record(self):
        with self.f.prepared(self.f.args()) as prepared:
            old=dict(version='r10_full16_explicit_author_action_outcome_v744',identity=deepcopy(prepared['identity']),
                full16_frozen_execution={'old_consumer':True},official_task_success=False)
            original=deepcopy(old);value=subject.annotated_output(prepared,prepared['out']/'case.json',old)
            self.assertEqual(old,original);self.assertEqual(value['version'],subject.CASE_SCHEMAS['train'])
            self.assertEqual(value['inherited_consumer_record_version'],old['version'])
            self.assertFalse(value['official_task_success']);self.assertFalse(value['formal_score_contribution'])
            self.assertEqual(value['graph_witness'],prepared['graph_witness'])
            self.assertNotIn('full16_frozen_execution',value)
            self.assertEqual(value['inherited_consumer_source_execution_record'],{'old_consumer':True})

    def runtime(self,writes):
        return NS(__file__=str(self.root/subject.RUNTIME),_geometry=lambda *args:'old_geometry',
            _baseline=lambda *args:'old_baseline',atomic_json=lambda file,value:writes.append((file,value)))

    def test_core_scope_constructs_once_preserves_existing_observations_and_restores_on_error(self):
        with self.f.prepared(self.f.args()) as prepared:
            writes=[];runtime=self.runtime(writes);before=(runtime._geometry,runtime._baseline,runtime.atomic_json)
            resolver=NS(resolve=lambda *args:'original resolver');observer=object();actor=NS(mode_observer=observer,role_resolver=resolver)
            source=Source();frontend=Frontend();policy=Dispatch(source);executor=object();order=[]
            @contextmanager
            def integrated(root,config,task,current_actor,*,events):
                self.assertIs(current_actor,actor);order.append('enter')
                with patch.object(actor,'mode_observer',object()),patch.object(source,'_done',lambda goal:True), \
                     patch.object(frontend,'observe',lambda *args:'patched'):
                    try:yield frontend,policy,executor
                    finally:order.append('exit')
                events.update(restored=True)
            module=NS(integrated_components=integrated)
            creator=NS(create_actor=Mock(return_value=(actor,{'public':'proof'})))
            def imported(name):return creator if name=='our_mani.multitask.rml_component_extensions_v708' else importlib_original(name)
            error=RuntimeError('original loop error')
            with patch.object(subject,'verify'),patch.object(subject,'_import_factory',return_value=module), \
                 patch.object(subject.importlib,'import_module',side_effect=imported):
                with self.assertRaises(RuntimeError) as caught:
                    with subject.runtime_scope(self.root,prepared,runtime):
                        result=runtime._geometry(prepared,'same_env',['same_descriptions'])
                        self.assertEqual(result[2:],(frontend,policy,executor));self.assertIs(runtime._baseline,before[1])
                        runtime.atomic_json(prepared['out']/'inputs.json',{'original':True})
                        self.assertEqual(writes[0][1]['graph_witness'],prepared['graph_witness']);raise error
            self.assertIs(caught.exception,error);self.assertEqual((runtime._geometry,runtime._baseline,runtime.atomic_json),before)
            self.assertEqual(order,['enter','exit']);creator.create_actor.assert_called_once()
            self.assertTrue(prepared['runtime_scope_restoration']['restored'])
            self.assertIs(actor.mode_observer,observer);self.assertNotIn('_done',vars(source))

    def test_noncolour_core_actor_with_none_resolver_restores(self):
        with self.f.prepared(self.f.args(task='remove_cups_from_rotating_frame')) as prepared:
            actor=NS(mode_observer=object(),role_resolver=None);source=Source();frontend=Frontend()
            @contextmanager
            def integrated(*args,events):yield frontend,Dispatch(source),object()
            creator=NS(create_actor=Mock(return_value=(actor,{})))
            def imported(name):return creator if name=='our_mani.multitask.rml_component_extensions_v708' else importlib_original(name)
            with patch.object(subject,'verify'),patch.object(subject,'_import_factory',return_value=NS(integrated_components=integrated)), \
                 patch.object(subject.importlib,'import_module',side_effect=imported):
                runtime=self.runtime([])
                with subject.runtime_scope(self.root,prepared,runtime) as unused:
                    self.assertIsNone(unused)
                    runtime._geometry(prepared,'owned_env',[])
            self.assertTrue(prepared['runtime_scope_restoration']['restored'])

    def test_actual_configured_core_artifacts_cannot_be_replaced_by_unrelated_locked_model(self):
        args=self.f.args();original=self.f.preflight
        def preflight(*values):
            prepared=original(*values)
            prepared['config']['learned_artifacts'][args.task]['interaction']={'path':'metadata/other_model.json','sha256':'a'*64}
            return prepared
        with patch.object(substrate,'prepare',side_effect=preflight),patch.object(substrate,'_checked'), \
             patch.object(substrate,'outcome_protocol_binding',return_value=self.f.protocol):
            with self.assertRaisesRegex(ValueError,'core operation artifact missing'):subject.prepare(self.root,args)

    def test_block_scope_uses_same_env_and_original_block_factory_without_core_actor(self):
        with self.f.prepared(self.f.args(task=subject.BLOCKS[0])) as prepared:
            runtime=self.runtime([]);triple=(Frontend(),object(),object());events=[]
            @contextmanager
            def integrated(*args):
                self.assertEqual(args[-1],'owned_env');events.append('enter')
                try:yield triple
                finally:events.append('exit')
            module=NS(integrated_block_components=integrated,binding=Mock(return_value={'original1120':True}))
            with patch.object(subject,'verify'),patch.object(subject,'_import_factory',return_value=module):
                with subject.runtime_scope(self.root,prepared,runtime):
                    actual=runtime._geometry(prepared,'owned_env',[])
                    self.assertIsNone(actual[0]);self.assertEqual(actual[2:],triple)
            self.assertEqual(events,['enter','exit']);self.assertTrue(prepared['runtime_scope_restoration']['restored'])

    def test_physmani_original_baseline_and_geometry_functions_are_untouched(self):
        with self.f.prepared(self.f.args(method='physmani')) as prepared:
            runtime=self.runtime([]);baseline,geometry=runtime._baseline,runtime._geometry
            with patch.object(subject,'verify'),patch.object(subject,'_import_factory',side_effect=AssertionError('no factory import')):
                with subject.runtime_scope(self.root,prepared,runtime):
                    self.assertIs(runtime._baseline,baseline);self.assertIs(runtime._geometry,geometry)
            self.assertTrue(prepared['runtime_scope_restoration']['restored'])

    def test_writer_delegates_once_before_first_case_write_and_rejects_overwrite_or_escape(self):
        with self.f.prepared(self.f.args(method='physmani')) as prepared:
            writes=[];runtime=self.runtime(writes)
            with patch.object(subject,'verify'):
                with subject.runtime_scope(self.root,prepared,runtime):
                    runtime.atomic_json(prepared['out']/'case.json',{'version':subject.INHERITED_VERSION,'official_task_success':False})
                    self.assertEqual(writes[0][1]['version'],subject.CASE_SCHEMAS['train'])
                    with self.assertRaises(ValueError):runtime.atomic_json(prepared['out']/'case.json',{})
                    with self.assertRaises(ValueError):runtime.atomic_json(self.root/'escape.json',{})
            self.assertEqual(len(writes),1)

    def test_case_annotation_rejects_missing_or_different_inherited_consumer_version(self):
        with self.f.prepared(self.f.args()) as prepared:
            for value in ({},{'version':'another_runtime'}):
                with self.assertRaises(ValueError):subject.annotated_output(prepared,prepared['out']/'case.json',value)

    def test_factory_module_origin_cannot_differ_from_registered_file(self):
        chosen=subject.selection(self.root,self.f.args())
        with patch.object(subject.importlib,'import_module',return_value=NS(__file__='/different/factory.py')):
            with self.assertRaises(ValueError):subject._import_factory(self.root,chosen['factory'])

    def test_environment_requires_real_inner_process_and_graph_pinned820_fields(self):
        chosen=subject.selection(self.root,self.f.args());graph=chosen['graph']
        proof=dict(complete=True,observer_pid=os.getpid(),wrapper=graph['control']['components']['environment_wrapper'],
            manifest=graph['physical']['environment_manifest'],
            namespace_qualification=graph['control']['static_inputs']['environment_namespace_qualification'])
        module=NS(capture_namespace_evidence=Mock(return_value=proof))
        with patch.object(subject.importlib,'import_module',return_value=module):
            self.assertIs(subject.environment_proof(self.root,chosen),proof)
            proof['observer_pid']=-1
            with self.assertRaises(ValueError):subject.environment_proof(self.root,chosen)

    def test_cpu_entry_writes_witness_identity_sidecar_without_output_or_runtime_constructor(self):
        args=self.f.args();chosen=subject.selection(self.root,args)
        with self.f.prepared(args) as prepared:
            @contextmanager
            def permit_scope(*unused,**kwargs):yield 'owned_permit'
            @contextmanager
            def quiet(*unused,**kwargs):yield
            guard=NS(reserved_resources_after_TEST=Mock(side_effect=permit_scope),require_permit=Mock(return_value={'lease':'real-scope-stub'}))
            edges=NS(declared_edge_scope=quiet,evaluator_scope=quiet)
            def imported(name):
                if name=='our_mani.multitask.post_completed_TEST_resource_guard_v1106':return guard
                if name=='our_mani.multitask.frozen_input_edges_v463':return edges
                if name=='scripts.evaluate_multitask_case_r10':return NS()
                raise AssertionError('CPU must not import runtime/factory: '+name)
            with patch.object(entry,'ROOT',self.root),patch.object(entry.importlib,'import_module',side_effect=imported), \
                 patch.object(subject,'selection',return_value=chosen),patch.object(subject,'environment_proof',return_value={}), \
                 patch.object(subject,'prepare',return_value=prepared),patch.object(subject,'verify'):
                result=entry.run_case(args,root=self.root)
            self.assertFalse((self.root/args.out).exists());self.assertFalse(result['execution']['native_started'])
            self.assertFalse(result['execution']['GPU_started']);self.assertFalse(result['execution']['physical_factory_constructed'])
            self.assertEqual(result['graph_witness'],result['execution']['graph_witness'])
            self.assertEqual(result['identity'],result['execution']['identity'])
            self.assertTrue((chosen['runtime']/'CPU_check/launch.json').is_file())
            self.assertTrue((chosen['runtime']/'CPU_check/worker_sidecar.json').is_file())
            guard.reserved_resources_after_TEST.assert_called_once()

    def test_entry_failure_preserves_exception_and_explicit_infrastructure_receipt(self):
        args=self.f.args();chosen=subject.selection(self.root,args);error=RuntimeError('original preflight failure')
        @contextmanager
        def quiet(*args,**kwargs):yield 'permit'
        guard=NS(reserved_resources_after_TEST=quiet,require_permit=lambda *args:{})
        def imported(name):
            if name.endswith('v1106'):return guard
            if name.endswith('v463'):return NS(declared_edge_scope=quiet,evaluator_scope=quiet)
            return NS()
        with patch.object(entry,'ROOT',self.root),patch.object(entry.importlib,'import_module',side_effect=imported), \
             patch.object(subject,'selection',return_value=chosen),patch.object(subject,'environment_proof',return_value={}), \
             patch.object(subject,'prepare',side_effect=error):
            with self.assertRaises(RuntimeError) as caught:entry.run_case(args,root=self.root)
        self.assertIs(caught.exception,error)
        failure=json.loads((chosen['runtime']/'CPU_check/failure.json').read_text())
        self.assertFalse(failure['counted_as_policy_failure']);self.assertTrue(failure['original_exception_rethrown'])
        self.assertFalse((chosen['runtime']/'CPU_check/worker_sidecar.json').exists())

    def test_mock_loop_native_false_is_preserved_through_first_write_and_final_sidecar(self):
        # runtime.run is a pure JSON stub. No Native library/model is imported.
        args=self.f.args(method='physmani',check=False);chosen=subject.selection(self.root,args)
        with self.f.prepared(args) as prepared:
            @contextmanager
            def quiet(*args,**kwargs):yield 'permit'
            guard=NS(reserved_resources_after_TEST=quiet,require_permit=lambda *args:{})
            runtime=self.runtime([])
            runtime.atomic_json=lambda file,value:subject.write_json(file,value)
            def run(current):
                self.assertIs(current,prepared);current['out'].mkdir(parents=True,exist_ok=False)
                record=sampled_case(method='physmani',success=False,ticks=200)
                record.update(version='r10_full16_explicit_author_action_outcome_v744',identity=current['identity'],
                    task=args.task,process_id=os.getpid())
                runtime.atomic_json(current['out']/'case.json',record)
            runtime.run=Mock(side_effect=run)
            validate=Mock()
            def imported(name):
                if name.endswith('v1106'):return guard
                if name.endswith('v463'):return NS(declared_edge_scope=quiet,evaluator_scope=quiet)
                if name=='scripts.evaluate_multitask_case_r10':return NS()
                if name=='scripts.run_task_executor_smoke_author_suite16_v744_r10':return runtime
                if name=='our_mani.multitask.evaluation':return NS(validate_completed_case=validate)
                raise AssertionError('unexpected import in mock loop: '+name)
            with patch.object(entry,'ROOT',self.root),patch.object(entry.importlib,'import_module',side_effect=imported), \
                 patch.object(subject,'selection',return_value=chosen),patch.object(subject,'environment_proof',return_value={}), \
                 patch.object(subject,'prepare',return_value=prepared),patch.object(subject,'verify'):
                result=entry.run_case(args,root=self.root)
            runtime.run.assert_called_once_with(prepared);validate.assert_called_once()
            self.assertFalse(result['execution']['native_success']);self.assertFalse(result['execution']['native_failure'])
            self.assertTrue(result['execution']['native_false_preserved'])
            self.assertTrue(result['execution']['runtime_scope_restoration']['restored'])
            record=json.loads((prepared['out']/'case.json').read_text())
            self.assertEqual(record['version'],subject.CASE_SCHEMAS['train']);self.assertFalse(record['official_task_success'])
            self.assertEqual(record['counted_clock']['ticks'],200)
            self.assertEqual(record['graph_witness'],result['execution']['graph_witness'])
            self.assertEqual(record['identity'],result['execution']['identity'])


class StaticPhysicalAndControlContracts(unittest.TestCase):
    def test_actual1164_1153_744_and_full_physical_overlay_are_source_provenance_only(self):
        self.assertEqual(subject.FACTORIES['core_geometry'],('our_mani/multitask/author_native_release_core_v1164.py','integrated_components'))
        self.assertEqual(subject.FACTORIES['block_geometry'],('our_mani/multitask/retained_preclose_block_integration_v1153.py','integrated_block_components'))
        self.assertEqual(subject.FACTORIES['physmani'],(subject.RUNTIME,'_baseline'))
        provenance=subject._source_overlay(ROOT,{'physical':{'source_overlay_lock':subject.PHYSICAL_SOURCE_OVERLAY}})
        self.assertEqual(provenance['pointer'],subject.PHYSICAL_SOURCE_OVERLAY)
        self.assertEqual(provenance['source_count'],60)
        self.assertEqual(provenance['CPU_evidence']['sha256'],'755cdb1c70a1653e861ec73226e7532ccd49ea6b341a34bb7b6f233b0b4ee7ea')
        self.assertTrue(provenance['qualified_as_source_provenance_only'])
        self.assertNotIn('readiness',provenance)
        self.assertNotIn('admission',provenance)

    def test_native_runtime_sampling_admission_and_cli_match_frozen1146(self):
        # AST equality proves these copied branches retain their original call
        # order; only the new controller/schema/namespace identity is normalized.
        parent=ast.parse((ROOT/'our_mani/multitask/repaired_policy_case_v1146.py').read_text())
        current=ast.parse((ROOT/subject.SOURCE).read_text().replace('1167','1146'))
        parent_functions={n.name:n for n in parent.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        current_functions={n.name:n for n in current.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for name in ('admission_scope','_formal_metadata','_validate_graph_models','prepare','verify',
                     'annotated_output','_import_factory','runtime_scope','environment_proof'):
            self.assertEqual(ast.dump(current_functions[name],include_attributes=False),
                ast.dump(parent_functions[name],include_attributes=False),name)
        parent_entry=ast.parse((ROOT/'scripts/run_repaired_policy_case_v1146_r10.py').read_text())
        current_entry=ast.parse((ROOT/subject.WORKER).read_text().replace('1167','1146'))
        self.assertEqual(ast.dump(current_entry,include_attributes=False),ast.dump(parent_entry,include_attributes=False))


import importlib
importlib_original=importlib.import_module

if __name__=='__main__':unittest.main()
