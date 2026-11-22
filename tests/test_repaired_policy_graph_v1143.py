"""Pure CPU source-graph contracts using isolated, synthetic file pointers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.multitask import repaired_policy_graph_v1143 as subject


def put(root,name,data=b'fixture source'):
    path=Path(root)/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return dict(path=name,sha256=hashlib.sha256(data).hexdigest())


def graph_spec(root):
    """No candidate import, dataset reader, Native or real evaluation artifact."""
    root=Path(root)
    (root/'artifacts/framework/multitask_r10').mkdir(parents=True,exist_ok=True)
    compiler=put(root,subject.SOURCE,Path(subject.__file__).read_bytes())
    factory=put(root,'src/factory.py');consumer=put(root,'src/consumer.py')
    rml=put(root,subject.RML_SOURCE);outcome=put(root,subject.OUTCOME_SOURCE)
    physical_sources=dict(factory=factory,consumer=consumer,RML=rml,outcome=outcome)
    train=put(root,'src/train_worker.py');evaluation=put(root,'src/evaluation_worker.py')
    preparer=put(root,'src/preparer.py')
    components={name:put(root,'src/'+name+'.py') for name in subject.COMPONENTS}
    control_sources=dict(compiler=compiler,train_worker=train,evaluation_worker=evaluation,
        preparer=preparer,**components)
    checkpoint=put(root,'metadata/shared.pt',b'frozen checkpoint')
    model=put(root,'metadata/model.json',b'{"model":"frozen TRAIN model"}')
    configuration=put(root,'metadata/configuration.json',b'{"configuration":true}')
    inputs=put(root,'metadata/frozen_inputs.json',b'{"files":{}}')
    physical=dict(factories={'selected':dict(source=factory,callable='integrated_components')},
        source_overlay_lock=put(root,'metadata/source_overlay_lock.json',b'{"sources":{}}'),
        sources_map=physical_sources,complete_source_closure_declared=True,
        common_RML=dict(source=rml,profile=put(root,'metadata/common_profile.json',b'{"static_530_binding":true}')),
        outcome_sampling=dict(source=outcome,protocol=put(root,'metadata/outcome_protocol.json',b'{"static_741_binding":true}')),
        environment_manifest=put(root,'metadata/environment_manifest.json',b'{"environment":true}'),
        shared_context_checkpoint=checkpoint)
    control=dict(sources_map=control_sources,components=components,static_inputs={},
        contracts=dict(readiness_schema='r10_fresh_TRAIN192_readiness_v1144',
            admission_schema='new_frozen_graph_admission',split_source_lock_schema='new_post_admission_source_lock'),
        complete_source_closure_declared=True)
    routes=[]
    for task in subject.TASKS:
        for method in subject.METHODS:
            def declaration(purpose,worker):
                return dict(purpose=purpose,actual_split='train' if purpose=='train_qualification' else 'validation',
                    worker=worker,preparer=preparer,consumer=consumer,case_schema='new_actual_worker_case',
                    physical_factory='selected',factory_control='declared_frozen_factory_only',runtime_policy_override_allowed=False)
            routes.append(dict(task=task,method=method,configuration=configuration,frozen_inputs=inputs,
                models={'trained_model':model},checkpoints={} if method=='cv_geometry' else {'shared':checkpoint},
                physical_factory='selected',GPU_required=method=='physmani',
                train=declaration('train_qualification',train),evaluation=declaration('evaluation',evaluation)))
    return dict(schema=subject.SPEC_SCHEMA,tasks=list(subject.TASKS),methods=list(subject.METHODS),
        external_roots={},train_source_manifest=put(root,'metadata/original_TRAIN_manifest.json',
            b'opaque manifest: compiler must not parse or select episodes'),physical=physical,control=control,routes=routes)


class GraphContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.spec=graph_spec(self.root)

    def compile(self,spec=None):return subject.compile_policy_graph(self.root,self.spec if spec is None else spec)

    def test_complete_graph_is_deterministic_normalized_and_does_not_qualify(self):
        original=deepcopy(self.spec);first=self.compile();changed=deepcopy(self.spec)
        changed['routes'].reverse();changed['tasks'].reverse();changed['methods'].reverse()
        changed['physical']['sources_map']=dict(reversed(list(changed['physical']['sources_map'].items())))
        self.assertEqual(first,self.compile(changed));self.assertEqual(original,self.spec)
        self.assertEqual(len(first['routes']),48);self.assertEqual(first['roots']['project'],str(self.root))
        self.assertEqual(first['routes'][0]['train']['worker']['root'],'project')
        for key in ('physical_policy_sha256','control_graph_sha256','policy_sha256'):
            self.assertRegex(first[key],r'^[0-9a-f]{64}$')
        self.assertFalse(first['qualification_or_admission_granted'])
        self.assertFalse(first['formal_VAL_TEST_qualified']);self.assertFalse(first['readiness_generated'])
        self.assertFalse(first['future_evidence_pointers_included'])
        self.assertTrue(first['source_closure_and_worker_control_are_declarations_not_execution_proof'])

    def test_actual_eval_worker_change_changes_control_and_policy_but_not_physical(self):
        before=self.compile();new=put(self.root,'src/evaluation_worker2.py',b'new evaluation controller')
        self.spec['control']['sources_map']['evaluation_worker']=new
        for row in self.spec['routes']:row['evaluation']['worker']=new
        after=self.compile()
        self.assertEqual(before['physical_policy_sha256'],after['physical_policy_sha256'])
        self.assertNotEqual(before['control_graph_sha256'],after['control_graph_sha256'])
        self.assertNotEqual(before['policy_sha256'],after['policy_sha256'])

    def test_model_change_changes_physical_control_and_policy_hashes(self):
        before=self.compile();new=put(self.root,'metadata/model2.json',b'{"model":"new TRAIN fit"}')
        self.spec['routes'][0]['models']={'trained_model':new};after=self.compile()
        for key in ('physical_policy_sha256','control_graph_sha256','policy_sha256'):
            self.assertNotEqual(before[key],after[key])

    def test_future_evidence_is_separate_and_cannot_be_added_to_frozen_spec(self):
        before=self.compile();put(self.root,'runtime/readiness/proof.json',b'{"future":true}')
        self.assertEqual(before,self.compile())
        for key in ('readiness','admission','split_inputs','cohort','proof'):
            changed=deepcopy(self.spec);changed[key]={'path':'future.json','sha256':'a'*64}
            with self.subTest(key=key),self.assertRaises(subject.PolicyGraphError):self.compile(changed)
        self.spec['control']['static_inputs']['readiness']=put(self.root,'metadata/readiness.json',b'{}')
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_sources_map_cannot_smuggle_a_json_proof_as_source(self):
        self.spec['control']['sources_map']['hidden']=put(self.root,'metadata/proof.json',b'{}')
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_missing_or_pending_actual_eval_worker_prevents_compile_and_write(self):
        for bad in (None,{},dict(path='src/missing.py',sha256='a'*64)):
            changed=deepcopy(self.spec);changed['routes'][0]['evaluation']['worker']=bad
            with self.subTest(bad=bad),self.assertRaises(subject.PolicyGraphError):self.compile(changed)
            with self.assertRaises(subject.PolicyGraphError):
                subject.write_policy_graph(self.root,changed,namespace='artifacts/framework/multitask_r10/missing_worker')
        self.assertFalse((self.root/'artifacts/framework/multitask_r10/missing_worker').exists())
        for bad in ('pending',None):
            changed=deepcopy(self.spec);changed['routes'][0]['evaluation']['case_schema']=bad
            with self.assertRaises(subject.PolicyGraphError):self.compile(changed)

    def test_both_worker_purposes_must_declare_same_selected_physical_factory(self):
        for field,value in (('physical_factory','other'),('factory_control','runtime_override'),
                            ('runtime_policy_override_allowed',True),('actual_split','train'),('purpose','train_qualification')):
            changed=deepcopy(self.spec);changed['routes'][0]['evaluation'][field]=value
            with self.subTest(field=field),self.assertRaises(subject.PolicyGraphError):self.compile(changed)

    def test_grid_duplicates_missing_tasks_and_gpu_mislabel_are_rejected(self):
        changes=[]
        changed=deepcopy(self.spec);changed['routes'][-1]=deepcopy(changed['routes'][0]);changes.append(changed)
        changed=deepcopy(self.spec);changed['routes'].pop();changes.append(changed)
        changed=deepcopy(self.spec);changed['tasks'][0]='new_task';changes.append(changed)
        changed=deepcopy(self.spec);changed['methods'][0]='other';changes.append(changed)
        changed=deepcopy(self.spec);changed['routes'][0]['GPU_required']=True;changes.append(changed)
        for index,changed in enumerate(changes):
            with self.subTest(index=index),self.assertRaises(subject.PolicyGraphError):self.compile(changed)

    def test_control_roles_actual_preparer_and_factory_must_be_in_source_maps(self):
        for section,label in (('physical','factory'),('physical','consumer'),('physical','RML'),
                              ('control','train_owner'),('control','preparer'),('control','compiler')):
            changed=deepcopy(self.spec);changed[section]['sources_map'].pop(label)
            with self.subTest(section=section,label=label),self.assertRaises(subject.PolicyGraphError):self.compile(changed)
        self.spec['control']['components'].pop('formal_source_reader')
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_original530_741_sources_are_explicit_not_relabelled(self):
        self.spec['physical']['common_RML']['source']=self.spec['physical']['factories']['selected']['source']
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_legacy_readiness_schema_cannot_approve_repaired_policy(self):
        self.spec['control']['contracts']['readiness_schema']='r10_exact16_fixed_four_common_author_RML_readiness_v532'
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_context_checkpoint_is_explicit_shared_and_neural_routes_need_weights(self):
        for bad in ('empty','different'):
            changed=deepcopy(self.spec);row=next(x for x in changed['routes'] if x['method']=='context_geometry')
            row['checkpoints']={} if bad=='empty' else {'other':put(self.root,'metadata/other.pt',b'other weights')}
            with self.subTest(bad=bad),self.assertRaises(subject.PolicyGraphError):self.compile(changed)

    def test_sha_path_escape_symlink_and_dataset_sample_pointers_are_rejected(self):
        for name in ('../outside.json','/absolute.json','metadata/./model.json','metadata//model.json'):
            changed=deepcopy(self.spec);changed['routes'][0]['models']['trained_model']['path']=name
            with self.subTest(name=name),self.assertRaises(subject.PolicyGraphError):self.compile(changed)
        changed=deepcopy(self.spec);changed['routes'][0]['models']['trained_model']['sha256']='0'*64
        with self.assertRaises(subject.PolicyGraphError):self.compile(changed)
        target=self.root/'metadata/model.json';(self.root/'metadata/link.json').symlink_to(target)
        changed=deepcopy(self.spec);changed['routes'][0]['models']['trained_model']['path']='metadata/link.json'
        with self.assertRaises(subject.PolicyGraphError):self.compile(changed)
        changed=deepcopy(self.spec);changed['routes'][0]['models']={'sample':put(self.root,'data/episode.hdf5',b'not a model')}
        with self.assertRaises(subject.PolicyGraphError):self.compile(changed)

    def test_compiler_source_must_match_the_actually_running_tool(self):
        pointer=put(self.root,subject.SOURCE,b'forged compiler')
        self.spec['control']['sources_map']['compiler']=pointer
        with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_declared_external_native_sources_are_pinned_without_native_import(self):
        with TemporaryDirectory() as external:
            pointer=put(external,'third_party/arm.py',b'public frozen Native source')
            pointer['root']='native_repo';self.spec['external_roots']={'native_repo':external}
            self.spec['physical']['sources_map']['native_arm']=pointer
            graph=self.compile();self.assertEqual(graph['roots']['native_repo'],external)
            self.assertEqual(graph['physical']['sources_map']['native_arm']['root'],'native_repo')
            (Path(external)/'third_party/arm.py').write_text('changed')
            with self.assertRaises(subject.PolicyGraphError):self.compile()

    def test_write_is_explicit_first_write_and_roundtrip_rechecks_all_sources(self):
        namespace='artifacts/framework/multitask_r10/only_new_graph'
        pointer=subject.write_policy_graph(self.root,self.spec,namespace=namespace)
        self.assertEqual(subject.read_policy_graph(self.root,pointer),self.compile())
        with self.assertRaises(FileExistsError):subject.write_policy_graph(self.root,self.spec,namespace=namespace)
        (self.root/'src/factory.py').write_text('changed source')
        with self.assertRaises(subject.PolicyGraphError):subject.read_policy_graph(self.root,pointer)

    def test_no_existing_even_empty_namespace_is_reused_and_no_path_is_inferred(self):
        namespace='artifacts/framework/multitask_r10/existing';(self.root/namespace).mkdir()
        with self.assertRaises(FileExistsError):subject.write_policy_graph(self.root,self.spec,namespace=namespace)
        self.assertEqual(list((self.root/namespace).iterdir()),[])
        for invalid in ('runtime/new','../outside','artifacts/framework/multitask_r10'):
            with self.assertRaises(subject.PolicyGraphError):subject.write_policy_graph(self.root,self.spec,namespace=invalid)

    def test_stored_fingerprint_or_truth_claim_tampering_is_not_accepted(self):
        pointer=subject.write_policy_graph(self.root,self.spec,namespace='artifacts/framework/multitask_r10/tamper')
        graph=json.loads((self.root/pointer['path']).read_text());graph['qualification_or_admission_granted']=True
        changed=put(self.root,pointer['path'],json.dumps(graph).encode())
        with self.assertRaises(subject.PolicyGraphError):subject.read_policy_graph(self.root,changed)

    def test_json_reader_requires_explicit_project_pointer_and_matching_bytes(self):
        pointer=put(self.root,'metadata/receipt.json',b'{"complete":false}')
        value,normalized=subject.read_project_json(self.root,pointer)
        self.assertEqual(value,{'complete':False});self.assertEqual(normalized['root'],'project')
        pointer['root']='native_repo'
        with self.assertRaises(subject.PolicyGraphError):subject.read_project_json(self.root,pointer)

    def test_changed_file_during_compilation_cannot_get_frozen_hash(self):
        original=subject._Files.revalidate
        def race(files):
            (self.root/'src/factory.py').write_text('changed during compilation')
            return original(files)
        with patch.object(subject._Files,'revalidate',race):
            with self.assertRaises(subject.PolicyGraphError):self.compile()


if __name__=='__main__':unittest.main()
