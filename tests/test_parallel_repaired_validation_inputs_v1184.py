"""CPU data-envelope contracts; all HDF5-named test files are temporary bytes."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import parallel_repaired_validation_inputs_v1184 as subject


class ValidationInputContracts(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        def write(name,value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            return subject.gate.point(self.root,path)
        self.write=write
        source=self.root/subject.SOURCE;source.parent.mkdir(parents=True);source.write_text('synthetic writer')
        self.graph_ref=write('metadata/graph.json',{})
        self.admission_ref=write('metadata/admission.json',{})
        self.routes_ref=write('metadata/preflight_routes.json',{})
        self.actual_routes_ref=write('metadata/actual_routes.json',{'actual_worker':'1180'})
        self.inherited_ref=write('metadata/inherited_preflight_entry.json',{'inherited_worker':'1167'})
        self.readiness_ref=write('metadata/readiness.json',{})
        self.manifest_ref=write('metadata/val_manifest.json',{})
        worker=write('metadata/worker.json',{})
        self.graph=dict(policy_sha256='policy',physical_policy_sha256='physical',control_graph_sha256='control',
            control=dict(components=dict(split_source_lock_writer=subject.gate.point(self.root,source))),
            routes=[dict(configuration=worker,frozen_inputs=worker,
                evaluation=dict(worker=worker,preparer=worker,consumer=worker))])
        self.admission=dict(routes=self.routes_ref,preflight_routes=self.routes_ref,actual_routes=self.actual_routes_ref,
            inherited742_preflight_entry=self.inherited_ref,source_manifests={'validation':self.manifest_ref},
            readiness=self.readiness_ref,tasks=['synthetic'],methods=['cv_geometry'])
        self.current=dict(graph=self.graph,admission=self.admission)
        self.rows=[]
        for index in range(315):
            source=self.root/f'val_package_compressed/task/episode{index}/data.h5';source.parent.mkdir(parents=True)
            source.write_bytes(b'frozen validation bytes')
            st=source.stat();self.rows.append(dict(path=str(source),task='synthetic',raw_episode_id=f'episode{index}',
                source_file_stat=dict(size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)))
        self.source=Path(self.rows[0]['path'])
        (self.root/subject.BUNDLE_PREFIX).mkdir(parents=True)

    @contextmanager
    def scope(self,*args):
        yield self.current

    def run_writer(self,namespace='new_inputs'):
        return subject.write_validation_inputs(self.root,self.graph_ref,self.admission_ref,
            namespace=subject.BUNDLE_PREFIX+namespace,permit=object())

    def patches(self):
        return (patch.object(subject.gate,'admission_scope',side_effect=self.scope),
            patch.object(subject.gate,'read_formal_source',return_value=({},self.rows,{})),
            patch.object(subject.gate,'_permit',return_value={}))

    def test_gate_precedes_any_data_hash_and_new_suite_binds_exact_graph(self):
        order=[]
        @contextmanager
        def admitted(*args):
            order.append('admitted');yield self.current
        original=subject._hash_file
        a,b,c=self.patches()
        with a,b,c,patch.object(subject.gate,'admission_scope',side_effect=admitted),\
             patch.object(subject,'_hash_file',side_effect=lambda path:(order.append('source_hash') or original(path))):
            result=self.run_writer()
        self.assertEqual(order[0],'admitted');self.assertEqual(order[1:],['source_hash']*315)
        lock=json.loads((self.root/result['source_inputs']['path']).read_text())
        suite=json.loads((self.root/result['suite']['path']).read_text())
        envelope=json.loads((self.root/result['suite_lock']['path']).read_text())
        self.assertEqual(lock['schema'],subject.gate.SPLIT_LOCK_SCHEMA)
        self.assertEqual(lock['producer_schema'],subject.SCHEMA)
        self.assertEqual(lock['files'][str(self.source)],original(self.source))
        self.assertEqual(suite['source_inputs'],result['source_inputs'])
        self.assertEqual(suite['graph'],self.graph_ref)
        self.assertEqual(suite['routes'],self.admission['preflight_routes'])
        self.assertEqual(suite['actual_routes'],self.admission['actual_routes'])
        self.assertNotEqual(suite['routes'],suite['actual_routes'])
        self.assertEqual(lock['inherited742_preflight_entry'],self.inherited_ref)
        self.assertTrue(lock['mixed_control_lineage'])
        self.assertFalse(suite['actual_execution_route_is_preflight_projection'])
        self.assertEqual(envelope['files'][str(self.root/result['suite']['path'])],result['suite']['sha256'])
        self.assertFalse(result['Native_started']);self.assertFalse(result['TEST_read'])
        bundle=json.loads((self.root/result['bundle']['path']).read_text())
        self.assertNotIn('bundle',bundle)
        self.assertEqual(bundle['source_inputs'],result['source_inputs'])

    def test_denied_admission_never_opens_source_or_claims_output_namespace(self):
        @contextmanager
        def denied(*args):
            raise ValueError('missing actual TRAIN192');yield
        with patch.object(subject.gate,'admission_scope',side_effect=denied),\
             patch.object(subject,'_hash_file',side_effect=AssertionError('unauthorized data read')):
            with self.assertRaises(ValueError):self.run_writer()
        self.assertFalse((self.root/(subject.BUNDLE_PREFIX+'new_inputs')).exists())

    def test_changed_stats_wrong_package_writer_or_evidence_and_repeat_reject(self):
        for bad in ('stats','train_package','writer','evidence','missing','duplicate'):
            with self.subTest(bad=bad):
                rows=deepcopy(self.rows);graph=deepcopy(self.graph);admission=deepcopy(self.admission)
                if bad=='stats':rows[0]['source_file_stat']['size_bytes']+=1
                if bad=='train_package':rows[0]['path']=str(self.source).replace('val_package_compressed','train_package_compressed')
                if bad=='writer':graph['control']['components']['split_source_lock_writer']['sha256']='wrong'
                if bad=='evidence':admission['readiness']=dict(self.readiness_ref,sha256='wrong')
                if bad=='missing':rows.pop()
                if bad=='duplicate':rows[1]=deepcopy(rows[0])
                self.current=dict(graph=graph,admission=admission)
                a,b,c=self.patches()
                with a,b,c,patch.object(subject.gate,'read_formal_source',return_value=({},rows,{})):
                    with self.assertRaises(ValueError):self.run_writer(bad)
        self.current=dict(graph=self.graph,admission=self.admission)
        a,b,c=self.patches()
        with a,b,c:
            self.run_writer('once')
            with self.assertRaises(FileExistsError):self.run_writer('once')


    def test_changed_actual_or_preflight_route_files_reject_before_bundle(self):
        for field in ('actual_routes','preflight_routes','inherited742_preflight_entry'):
            ref=self.admission[field];file=self.root/ref['path'];before=file.read_bytes()
            try:
                file.write_bytes(b'changed projection after admission')
                a,b,c=self.patches()
                with a,b,c,self.subTest(field=field),self.assertRaises(ValueError):self.run_writer('changed_'+field)
                self.assertFalse((self.root/(subject.BUNDLE_PREFIX+'changed_'+field)/'bundle.json').exists())
            finally:file.write_bytes(before)


if __name__=='__main__':
    unittest.main()
