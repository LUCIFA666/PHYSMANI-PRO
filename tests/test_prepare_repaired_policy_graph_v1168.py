"""Pure CPU temporary metadata; no real graph, source dataset or pilot runs."""
from copy import deepcopy
import ast
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from scripts import prepare_repaired_policy_graph_v1168_r10 as subject
from tests.test_repaired_policy_graph_v1143 import graph_spec,put
from tests.test_repaired_readiness_v1144 import write


class Fixture:
    def __init__(self,root):
        self.root=Path(root);spec=graph_spec(root);self.spec=spec
        cs=spec['control']['sources_map'];ps=spec['physical']['sources_map']
        for name in (subject.WORKER,subject.PREPARER,subject.SOURCE,subject.SIM,subject.GPU_SIM):
            cs[name]=put(root,name)
        ps['actual_consumer']=put(root,subject.CONSUMER)
        for name in subject.REVIEWED_SOURCE_PINS:
            reference=put(root,name,b'# synthetic reviewed transitive source; never imported\n')
            cs['reviewed:'+name]=dict(root='project',**reference)
            ps['reviewed:'+name]=dict(root='project',**reference)
        for pool in (cs,ps):
            for value in pool.values():value.setdefault('root','project')
        self.context=put(root,subject.CONTEXT,b'never-loaded synthetic shared checkpoint')
        for row in spec['routes']:
            if row['method']=='context_geometry':row['checkpoints']={'configured':self.context}
        rows=[dict(task=t,raw_episode_id='episode'+str(i),path='never-opened/'+t+'/data.h5')
            for t in subject.graphs.TASKS for i in range(100)]
        self.train=write(root,subject.TRAIN_MANIFEST,dict(split='train',episodes=rows))
        self.profile={'complete_static_profile':'synthetic fixture only'};self.protocol={'complete_static_protocol':'fixture only'}
        self.physical=write(root,'artifacts/framework/multitask_r10/synthetic_physical/source_overlay_lock.json',{'provenance_only':True})
        self.physical_cpu=write(root,'artifacts/framework/multitask_r10/synthetic_physical/CPU_evidence.json',{'fixture_only':True})
        self.env=write(root,'metadata/environment.json',{'environment_fixture':True})
        self.qual=write(root,'metadata/qualification.json',{'qualification_fixture':True})
        basefiles={str(self.root/p['path']):p['sha256'] for p in
            (*cs.values(),*ps.values(),self.context,self.train,spec['routes'][0]['configuration'])
            if p['path'] not in subject.REVIEWED_SOURCE_PINS}
        # Historical integrity keys survive verbatim. They need not be opened by
        # this helper (and are never parsed/selected as validation results).
        basefiles[str(root/'legacy'/'validation_manifest.json')]='f'*64
        basefiles[str(root/'legacy'/'case.json')]='e'*64
        self.basefiles=basefiles
        self.base_ref=write(root,subject.BASE_INPUTS,dict(files=basefiles,author_outcome_protocol=self.protocol))
        test=put(root,'tests/synthetic_control_test.py')
        tests={'tests/synthetic_control_test.py':dict(root='project',**test)}
        required={str(self.root/p['path']):p['sha256'] for p in (*cs.values(),*ps.values(),*tests.values())}
        self.inventory=dict(schema=subject.SCHEMA,roots=dict(project=str(self.root)),physical_sources_map=ps,control_sources_map=cs,
            control_components=spec['control']['components'],factories=spec['physical']['factories'],
            models_by_route=[{k:deepcopy(v) for k,v in row.items() if k not in ('train','evaluation','frozen_inputs')} for row in spec['routes']],
            test_sources=tests,physical_overlay=self.physical,physical_CPU_evidence=self.physical_cpu,
            old745_inputs=self.base_ref,old745_key_count=len(basefiles),original_TRAIN_manifest=self.train,
            common_RML_profile=self.profile,author_outcome_protocol=self.protocol,
            environment_manifest=self.env,environment_namespace_qualification=self.qual,
            required_control_CPU_source_hashes=required,worker_registry_matches_selected_physical_factories=True)
        self.inventory_ref=write(root,'artifacts/framework/multitask_r10/synthetic_inventory/source_inventory.json',self.inventory)
        self.cpu=write(root,'artifacts/framework/multitask_r10/synthetic_inventory/'+subject.CONTROL_CPU_NAME,dict(
            schema=subject.CONTROL_CPU_SCHEMA,complete=True,tests_exit_code=0,native_started=False,GPU_started=False,source_hashes=required))

    def prepare(self,namespace='artifacts/framework/multitask_r10/synthetic_preparation',**kwargs):
        with patch.object(subject,'collect_inventory',return_value=deepcopy(self.inventory)), \
             patch.object(subject,'BASE_COUNT',len(self.basefiles)),patch.object(subject,'CONTEXT_SHA',self.context['sha256']):
            return subject.prepare_only(self.root,self.inventory_ref,self.cpu,namespace=namespace,inventory_reviewed=True,**kwargs)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=Fixture(self.root)

    def test_default_inert_does_not_read_inventory_or_generate_files(self):
        with patch.object(subject,'collect_inventory',side_effect=AssertionError('no read')), \
             patch.object(subject,'prepare_only',side_effect=AssertionError('no prepare')),patch('sys.stdout',new_callable=io.StringIO):
            value=subject.main([])
        self.assertFalse(value['enabled']);self.assertFalse(value['files_written']);self.assertFalse(value['Native_started'])

    def test_source_metadata_cannot_be_VAL_TEST_manifest_before_any_hash_or_read(self):
        for name in ('artifacts/framework/protocol/validation_manifest.json','artifacts/framework/protocol/test_manifest.json'):
            with patch.object(subject,'pointer',side_effect=AssertionError('no hash')), \
                 patch.object(subject.graphs,'read_project_json',side_effect=AssertionError('no parse')):
                with self.assertRaises(ValueError):subject.metadata(self.root,name,filename='source_overlay_lock.json')
        for name in ('artifacts/framework/multitask_r10/old_validation/case.json',
                'artifacts/framework/multitask_r10/old_test/validation_manifest.json'):
            with patch.object(subject,'pointer',side_effect=AssertionError('no source/result hash')), \
                 patch.object(subject.graphs,'read_project_json',side_effect=AssertionError('no source/result parse')):
                with self.assertRaises(ValueError):subject._control_cpu(self.root,name,self.fixture.inventory)

    def test_explicit_inventory_only_firstwrite_has_no_graph_or_pilot_execution(self):
        namespace='artifacts/framework/multitask_r10/review_inventory'
        with patch.object(subject,'collect_inventory',return_value=self.fixture.inventory):
            reference=subject.write_inventory(self.root,self.fixture.physical,namespace=namespace)
            self.assertEqual(reference['path'],namespace+'/source_inventory.json')
            with self.assertRaises(FileExistsError):subject.write_inventory(self.root,self.fixture.physical,namespace=namespace)
        self.assertFalse((self.root/namespace/'graph').exists())

    def test_prepare_refuses_unreviewed_or_changed_inventory_and_old_registry_before_write(self):
        namespace='artifacts/framework/multitask_r10/not_prepared'
        with self.assertRaises(ValueError):subject.prepare_only(self.root,self.fixture.inventory_ref,self.fixture.cpu,namespace=namespace)
        for changed in ('registry','source'):
            actual=deepcopy(self.fixture.inventory)
            if changed=='registry':actual['worker_registry_matches_selected_physical_factories']=False
            else:actual['physical_sources_map']={}
            with patch.object(subject,'collect_inventory',return_value=actual),self.assertRaises(ValueError):
                subject.prepare_only(self.root,self.fixture.inventory_ref,self.fixture.cpu,namespace=namespace,inventory_reviewed=True)
        self.assertFalse((self.root/namespace).exists())

    def test_cpu_source_map_must_be_exact_and_exit_integer_zero(self):
        cpu,_=subject.metadata(self.root,self.fixture.cpu)
        for field,value in (('tests_exit_code',False),('GPU_started',True),('source_hashes',{})):
            changed=dict(cpu);changed[field]=value
            reference=write(self.root,'artifacts/framework/multitask_r10/bad_CPU/'+subject.CONTROL_CPU_NAME,changed)
            with self.assertRaises(ValueError):subject._control_cpu(self.root,reference,self.fixture.inventory)

    def test_collect_inventory_uses_selected_overlay_and_reports_registry_mismatch_without_writing(self):
        external=TemporaryDirectory();self.addCleanup(external.cleanup);native=Path(external.name)
        epoch=native/'epoch_99999.pth';epoch.write_text('synthetic released checkpoint')
        core=put(self.root,'src/selected_core.py');block=put(self.root,'src/selected_block.py')
        reviewed={name:put(self.root,name,b'# reviewed fixture, not imported') for name in subject.REVIEWED_SOURCE_PINS}
        put(self.root,subject.ORIGINAL_RUNTIME)
        for name in (subject.ROUTES,subject.ENV_MANIFEST,subject.ENV_QUALIFICATION):write(self.root,name,{'synthetic':True})
        codefiles={p['path']:p['sha256'] for p in (core,block)}
        cpu=write(self.root,'artifacts/framework/multitask_r10/inventory_fixture/CPU.json',dict(
            complete=True,tests_exit_code=0,native_started=False,GPU_started=False,source_hashes=codefiles))
        physical=dict(integration_complete=True,unpinned_sources=[],files=codefiles,required_sources=list(codefiles),
            original_745_execution_acceptance_granted=False,formal_VAL_TEST_qualification=False,integration_cpu_evidence=cpu,
            core_factory=dict(source=core['path'],sha256=core['sha256'],module='src.selected_core',function='integrated_components'),
            block_factory=dict(source=block['path'],sha256=block['sha256'],module='src.selected_block',function='integrated_block_components'))
        physical_ref=write(self.root,'artifacts/framework/multitask_r10/inventory_fixture/source_overlay_lock.json',physical)
        geometry=write(self.root,'metadata/actual_geometry.json',{})
        calibration=write(self.root,'metadata/actual_calibration.json',{})
        files=dict(self.fixture.basefiles,**{str(self.root/geometry['path']):geometry['sha256'],
            str(self.root/calibration['path']):calibration['sha256'],str(epoch):subject.sha(epoch)})
        configs={}
        for row in self.fixture.spec['routes']:
            configs[(row['task'],row['method'])]=dict(context_checkpoint=dict(path=subject.CONTEXT,sha256=self.fixture.context['sha256']),
                baseline_checkpoint=str(epoch),goal_calibration=calibration['path'],
                learned_artifacts={row['task']:dict(geometry=geometry,interaction=geometry,rigid_layout=geometry)})
        profile=dict(sources_sha256={},native_sources_sha256={})
        modules={core['path']:NS(SOURCES=(core['path'],)),block['path']:NS(SOURCES=(block['path'],)),
            subject.PREPARER:NS(FACTORIES={'core_geometry':('different.py','integrated_components'),
                'block_geometry':(block['path'],'integrated_block_components'),'physmani':(subject.CONSUMER,'_baseline')},
                PHYSICAL_SOURCE_OVERLAY=dict(physical_ref,root='project')),
            'our_mani/multitask/rml_case_preflight_v742.py':NS(SOURCES=())}
        components={label:p['path'] for label,p in self.fixture.spec['control']['components'].items()}
        base=dict(files=files,author_outcome_protocol=self.fixture.protocol)
        with patch.object(subject,'_substrate',return_value=(base,self.fixture.spec['routes'],configs,native,profile)), \
             patch.object(subject,'CONTROL_SOURCES',(subject.WORKER,subject.PREPARER,subject.SOURCE)), \
             patch.object(subject,'CONTROL_TESTS',tuple(self.fixture.inventory['test_sources'])), \
             patch.object(subject,'COMPONENTS',components),patch.object(subject,'CONTEXT_SHA',self.fixture.context['sha256']), \
             patch.object(subject,'COMPONENT_CALLABLES',{k:('main',) for k in components}), \
             patch.object(subject,'PHYSICAL_OVERLAY',dict(physical_ref,root='project')), \
             patch.object(subject,'REVIEWED_SOURCE_PINS',{name:ref['sha256'] for name,ref in reviewed.items()}), \
             patch.object(subject,'FACTORIES',dict(core_geometry=(core['path'],'integrated_components'),
                 block_geometry=(block['path'],'integrated_block_components'),physmani=(subject.CONSUMER,'_baseline'))), \
             patch.object(subject,'callable_metadata',return_value={'CPU_fixture':'declared_only'}), \
             patch.object(subject,'imported',side_effect=lambda root,name:modules[name]):
            result=subject.collect_inventory(self.root,physical_ref)
        self.assertEqual(len(result['models_by_route']),48)
        self.assertFalse(result['worker_registry_matches_selected_physical_factories'])
        self.assertEqual(result['factories']['core_geometry']['source']['path'],core['path'])
        self.assertTrue(result['legacy_immutable_integrity_lock_retained'])
        self.assertFalse(result['inventory_grants_readiness_or_admission'])
        for name,ref in reviewed.items():
            self.assertEqual(result['physical_sources_map']['project:'+name]['sha256'],ref['sha256'])
            self.assertEqual(result['control_sources_map']['project:'+name]['sha256'],ref['sha256'])
            self.assertEqual(result['required_control_CPU_source_hashes'][str(self.root/name)],ref['sha256'])

    def test_merge_preserves_every_legacy_key_and_never_adds_second_canonical_alias(self):
        file=self.root/'source.py';file.write_text('source');expected=subject.sha(file)
        base={str(file):expected,str(self.root/'legacy'/'case.json'):'a'*64}
        result=subject._merge(self.root,base,[(file,expected)])
        self.assertEqual(result,base)
        with self.assertRaises(ValueError):subject._merge(self.root,base,[(file,'b'*64)])
        with self.assertRaises(ValueError):subject._merge(self.root,base,[(self.root/'alias/../source.py',expected)])

    def test_model_mapping_uses_configured_artifacts_671_and_released_external_epoch(self):
        external=TemporaryDirectory();self.addCleanup(external.cleanup);native=Path(external.name)
        epoch=native/'train_logs/epoch_99999.pth';epoch.parent.mkdir();epoch.write_text('checkpoint')
        geometry=write(self.root,'metadata/geometry.json',{})
        calibration=write(self.root,'metadata/goal.json',{})
        files={str(self.root/geometry['path']):geometry['sha256'],str(self.root/calibration['path']):calibration['sha256'],
            str(self.root/subject.CONTEXT):self.fixture.context['sha256'],str(epoch):subject.sha(epoch)}
        with patch.object(subject,'CONTEXT_SHA',self.fixture.context['sha256']):
            cfg=dict(context_checkpoint=dict(path=subject.CONTEXT,sha256=self.fixture.context['sha256']),
                learned_artifacts={'insert_onto_rotating_peg':dict(geometry=geometry,interaction=geometry,rigid_layout=geometry,cv_value=None)})
            models,checkpoints=subject._models(self.root,native,files,dict(task='insert_onto_rotating_peg',method='context_geometry'),cfg)
            self.assertEqual(len(models),3);self.assertEqual(checkpoints['configured_shared_context']['path'],subject.CONTEXT)
            models,_=subject._models(self.root,native,files,dict(task=subject.BLOCKS[0],method='cv_geometry'),
                dict(context_checkpoint=cfg['context_checkpoint'],goal_calibration=calibration['path']))
            self.assertIn('public_goal_marker_calibration',models)
            models,checkpoints=subject._models(self.root,native,files,dict(task=subject.BLOCKS[0],method='physmani'),dict(baseline_checkpoint=str(epoch)))
            self.assertEqual(models['released_neural_policy']['root'],'native');self.assertEqual(models,checkpoints)

    def test_full_prepare_temp_graph_has48_routes_and_preserved_map_without_self_references(self):
        result=self.fixture.prepare();graph=subject.graphs.read_policy_graph(self.root,result['graph'])
        self.assertEqual(len(graph['routes']),48);self.assertFalse(result['qualification_granted'])
        merged=json.loads((self.root/result['merged_executable_inputs']['path']).read_text())
        for key,value in self.fixture.basefiles.items():self.assertEqual(merged['files'][key],value)
        self.assertNotIn(str(self.root/result['merged_executable_inputs']['path']),merged['files'])
        self.assertNotIn(str(self.root/result['graph']['path']),merged['files'])
        for name in subject.REVIEWED_SOURCE_PINS:
            self.assertNotIn(str(self.root/name),self.fixture.basefiles)
            self.assertEqual(merged['files'][str(self.root/name)],subject.sha(self.root/name))
        self.assertTrue(merged['legacy_immutable_integrity_lock_retained'])
        self.assertTrue(merged['no_VAL_TEST_sample_row_result_parsing_or_selection'])
        self.assertNotIn('approved_for_execution',merged)
        self.assertTrue(all(row['train']['worker']['path']==subject.WORKER and row['evaluation']['worker']['path']==subject.WORKER for row in graph['routes']))
        commands=json.loads((self.root/result['cpu_pilot_argv']['path']).read_text())['commands']
        self.assertEqual(len(commands),3)
        self.assertEqual({r['branch'] for r in commands},{'core_geometry','block_geometry','physmani'})
        for row in commands:
            self.assertEqual(row['argv'][-1],'--check');self.assertEqual(row['raw_episode_id'],'episode0')
            self.assertFalse(row['Native_started']);self.assertFalse(row['GPU_started'])
            self.assertIn(subject.GPU_SIM if row['method']=='physmani' else subject.SIM,row['argv'])
            self.assertIn(subject.WORKER,row['argv'])
            self.assertTrue(row['argv'][row['argv'].index('--out')+1].startswith('artifacts/framework/multitask_r10/repaired_policy_v1167/'))
            self.assertTrue(row['argv'][row['argv'].index('--runtime')+1].startswith('runtime/framework/repaired_policy_v1167/'))
        with self.assertRaises(ValueError):self.fixture.prepare()

    def test_actual_version_roles_and_physical_selection_are_explicit(self):
        self.assertEqual(subject.WORKER,'scripts/run_repaired_policy_case_v1167_r10.py')
        self.assertEqual(subject.PREPARER,'our_mani/multitask/repaired_policy_case_v1167.py')
        self.assertEqual(subject.COMPONENTS['train_auditor'],'our_mani/multitask/repaired_readiness_v1144.py')
        self.assertEqual(subject.COMPONENTS['train_owner'],'scripts/run_repaired_policy_owner_v1169_r10.py')
        self.assertEqual(subject.COMPONENTS['evaluation_owner'],subject.COMPONENTS['train_owner'])
        self.assertEqual(subject.COMPONENTS['evaluation_auditor'],'our_mani/multitask/repaired_validation_audit_v1170.py')
        self.assertEqual(subject.FACTORIES['core_geometry'],('our_mani/multitask/author_native_release_core_v1164.py','integrated_components'))
        self.assertEqual(subject.FACTORIES['block_geometry'],('our_mani/multitask/retained_preclose_block_integration_v1153.py','integrated_block_components'))
        self.assertEqual(subject.PHYSICAL_OVERLAY['sha256'],'7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929')
        self.assertEqual(subject.CONTROL_CPU_SCHEMA,'r10_repaired_control_CPU_evidence_v1168')
        self.assertEqual(set(subject.COMPONENT_CALLABLES),set(subject.COMPONENTS))
        self.assertIn(subject.PREDECESSOR_VERIFIER,subject.CONTROL_SOURCES)
        self.assertIn(subject.PREDECESSOR_VERIFIER_TEST,subject.CONTROL_TESTS)

    def test_wrong_or_data_physical_pointer_rejected_before_any_source_or_file_read(self):
        names=['artifacts/framework/protocol/validation_manifest.json',
            'artifacts/framework/multitask_r10/oldcase/case.json',
            dict(subject.PHYSICAL_OVERLAY,sha256='0'*64),dict(subject.PHYSICAL_OVERLAY,root='native')]
        for ref in names:
            with self.subTest(ref=ref),patch.object(subject,'pointer',side_effect=AssertionError('unexpected hash')), \
                 patch.object(subject,'_substrate',side_effect=AssertionError('unexpected substrate read')):
                with self.assertRaises(ValueError):subject.collect_inventory(self.root,ref)

    def test_control_callable_metadata_is_AST_only_even_with_dangerous_top_level_and_decorator(self):
        name='scripts/synthetic_future_owner.py';file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('raise AssertionError("must never import")\n@dangerous()\ndef main(argv=None, *, permit=None, **kwargs):\n    return None\n')
        with patch.object(subject.importlib,'import_module',side_effect=AssertionError('control import forbidden')):
            value=subject.callable_metadata(self.root,name,('main',))
        self.assertFalse(value['code_imported_or_executed'])
        self.assertEqual(value['source']['sha256'],subject.sha(file))
        f=value['callables']['main'];self.assertEqual(f['first_def_line'],3)
        self.assertEqual(f['positional'],['argv']);self.assertEqual(f['keyword_only'],['permit'])
        self.assertEqual(f['kwarg'],'kwargs');self.assertEqual(f['decorators'],['dangerous()'])
        with self.assertRaises(ValueError):subject.callable_metadata(self.root,name,('not_present',))
        with self.assertRaises(ValueError):subject.callable_metadata(self.root,'scripts/absent_future_auditor.py',('audit_validation',))

    def test_actual_model_bytes_are_verified_without_model_load_or_sample_read(self):
        file=self.root/'artifacts/weights.pt';file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'modelbytes')
        expected=subject.sha(file);ref=dict(path='artifacts/weights.pt',sha256=expected)
        files={str(file):expected}
        out=subject._model_pointer(self.root,self.root,files,ref)
        self.assertEqual(out['sha256'],expected)
        file.write_bytes(b'changed_modelbytes')
        with self.assertRaises(ValueError):subject._model_pointer(self.root,self.root,files,ref)

    def test_prepared_routes_schema_and_source_graph_exclude_postlaunch_evidence(self):
        result=self.fixture.prepare(namespace='artifacts/framework/multitask_r10/new_fixture')
        graph=subject.graphs.read_policy_graph(self.root,result['graph'])
        for row in graph['routes']:
            self.assertEqual(row['train']['case_schema'],'r10_graph_bound_repaired_policy_case_v1167_TRAIN_case')
            self.assertEqual(row['evaluation']['case_schema'],'r10_graph_bound_repaired_policy_case_v1167_validation_case')
        source_paths={p['path'] for side in ('physical','control') for p in graph[side]['sources_map'].values()}
        self.assertFalse(any(Path(p).name in ('readiness.json','cohort.json','admission.json','bundle.json') for p in source_paths))

    def test_real_control_sources_and_function_declarations_close_without_inventory_or_graph(self):
        project=Path(__file__).resolve().parents[1]
        missing=[name for name in (*subject.CONTROL_SOURCES,*subject.CONTROL_TESTS) if not (project/name).is_file()]
        if missing:self.skipTest('future control sources not yet complete: '+','.join(missing))
        with patch.object(subject,'collect_inventory',side_effect=AssertionError('no real inventory')), \
             patch.object(subject.graphs,'write_policy_graph',side_effect=AssertionError('no real graph')):
            rows={role:subject.callable_metadata(project,name,subject.COMPONENT_CALLABLES[role])
                  for role,name in subject.COMPONENTS.items()}
            actual=subject.imported(project,subject.PREPARER)
            self.assertEqual(actual.FACTORIES,subject.FACTORIES)
            self.assertEqual(actual.PHYSICAL_SOURCE_OVERLAY,subject.PHYSICAL_OVERLAY)
            self.assertEqual(actual.CASE_SCHEMAS['train'],'r10_graph_bound_repaired_policy_case_v1167_TRAIN_case')
            self.assertEqual(actual.CASE_SCHEMAS['validation'],'r10_graph_bound_repaired_policy_case_v1167_validation_case')
        self.assertEqual(set(rows),set(subject.COMPONENTS))
        for role,row in rows.items():
            self.assertFalse(row['code_imported_or_executed'])
            self.assertEqual(row['source']['sha256'],subject.sha(project/subject.COMPONENTS[role]))
            self.assertEqual(set(row['callables']),set(subject.COMPONENT_CALLABLES[role]))

    def test_actual_three_physical_factory_function_metadata_reads_source_only(self):
        project=Path(__file__).resolve().parents[1]
        with patch.object(subject.importlib,'import_module',side_effect=AssertionError('no factory construction')):
            result={branch:subject.callable_metadata(project,name,(function,))
                    for branch,(name,function) in subject.FACTORIES.items()}
        self.assertEqual(set(result),{'core_geometry','block_geometry','physmani'})
        for branch,(_,function) in subject.FACTORIES.items():
            self.assertEqual(set(result[branch]['callables']),{function})
            self.assertFalse(result[branch]['code_imported_or_executed'])

    def test_reviewed_source_hardpins_reject_missing_or_changed_bytes_before_substrate(self):
        name='src/reviewed.py';file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(b'# pinned source');expected=subject.sha(file)
        with patch.object(subject,'REVIEWED_SOURCE_PINS',{name:expected}):
            self.assertEqual(subject.reviewed_source_refs(self.root)[name]['sha256'],expected)
            file.write_bytes(b'# changed source')
            with self.assertRaisesRegex(ValueError,'reviewed transitive source bytes changed'):
                subject.reviewed_source_refs(self.root)
            file.unlink()
            with self.assertRaises(ValueError):subject.reviewed_source_refs(self.root)
        with patch.object(subject,'selected_physical_overlay',return_value={}), \
             patch.object(subject,'REVIEWED_SOURCE_PINS',{name:expected}), \
             patch.object(subject,'_substrate',side_effect=AssertionError('source drift must precede substrate read')):
            with self.assertRaises(ValueError):subject.collect_inventory(self.root,{})

    def test_actual_reviewed_AST_import_edges_and_local_dependencies_are_hash_bound(self):
        project=Path(__file__).resolve().parents[1]
        reviewed=subject.reviewed_source_refs(project)
        base=json.loads((project/subject.BASE_INPUTS).read_text())['files']
        for src,imported in (('our_mani/multitask/joint16_tracks_v31.py','track_dataset'),
                            ('our_mani/multitask/three_method_release_v531.py','three_method_release16')):
            tree=ast.parse((project/src).read_bytes())
            self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module==imported for n in ast.walk(tree)))
        observed=[]
        for source in reviewed:
            package=list(Path(source).parent.parts)
            for node in ast.walk(ast.parse((project/source).read_bytes())):
                if not isinstance(node,ast.ImportFrom) or not node.level:continue
                prefix=package[:len(package)-node.level+1]
                modules=[node.module] if node.module is not None else [v.name for v in node.names]
                for module in modules:
                    relative=str(Path(*prefix,*module.split('.')))+'.py';file=project/relative
                    if not file.is_file():continue
                    checksum=subject.sha(file);observed.append(relative)
                    self.assertTrue(reviewed.get(relative,{}).get('sha256')==checksum or base.get(str(file))==checksum,relative)
        self.assertTrue(observed)
        self.assertEqual(len(reviewed),2)
        # This is a bounded AST/declaration check, not a proof of all dynamic
        # imports. No import, track reader, model or qualification is invoked.


if __name__=='__main__':unittest.main()
