"""Freeze existing VAL reset bytes only after the new TRAIN192 admission.

This is a post-admission data envelope, separate from the frozen policy graph.
No job is started and no TEST path is accepted or inspected.
"""
import hashlib
import json
from pathlib import Path

from . import parallel_repaired_admission_v1183 as gate

SOURCE = 'our_mani/multitask/parallel_repaired_validation_inputs_v1184.py'
SCHEMA = 'r10_post_admission_parallel_repaired_validation_inputs_v1184'
BUNDLE_PREFIX = 'artifacts/framework/multitask_r10/parallel_repaired_validation_inputs_v1184/'


def _stamp(st):
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)


def _hash_file(path):
    path=Path(path)
    gate.require(path.is_file() and path.resolve()==path,'canonical existing source file required')
    before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(data)
    gate.require(_stamp(before)==_stamp(path.stat()),'source changed while freezing its bytes')
    return digest.hexdigest()


def write_validation_inputs(root, graph_ref, admission_ref, *, namespace, permit):
    """Create one new source lock, suite and evidence lock; never reuse output."""
    root=Path(root).resolve()
    with gate.admission_scope(root,graph_ref,admission_ref,permit) as current:
        graph=current['graph'];admission=current['admission']
        gate.require(gate.plain(graph['control']['components']['split_source_lock_writer'])==
            gate.point(root,root/SOURCE),'actual split writer differs from frozen control graph')
        suite=dict(schema=gate.SUITE_SCHEMA,graph=gate.plain(graph_ref),
            admission=gate.plain(admission_ref),routes=admission['preflight_routes'],
            actual_routes=admission['actual_routes'],preflight_routes=admission['preflight_routes'],
            inherited742_preflight_entry=admission['inherited742_preflight_entry'],
            actual_execution_route_is_preflight_projection=False,mixed_control_lineage=True,
            source_manifests=admission['source_manifests'],readiness=admission['readiness'],
            policy_sha256=graph['policy_sha256'],physical_policy_sha256=graph['physical_policy_sha256'],
            control_graph_sha256=graph['control_graph_sha256'],tasks=admission['tasks'],methods=admission['methods'],
            formal_TEST_permitted=False,TEST_used_for_fitting=False,
            original_745_execution_acceptance_granted=False)
        initial_refs=[suite['graph'],suite['admission'],suite['actual_routes'],suite['preflight_routes'],
            suite['inherited742_preflight_entry'],
            suite['source_manifests']['validation']]
        envelope={str(root/ref['path']):ref['sha256'] for ref in initial_refs}
        _,rows,_=gate.read_formal_source(root,suite,envelope,'validation')
        gate.require(len(rows)==gate.VAL_COUNT and
            len({(row['task'],row['raw_episode_id']) for row in rows})==gate.VAL_COUNT,
            'complete unique admitted315-source roster required')
        name=gate.graphs._relative(namespace)
        gate.require(name.startswith(BUNDLE_PREFIX) and len(Path(name).parts)>4,
            'explicit fresh post-admission data namespace required')
        directory=root/name
        gate.require(directory.parent.is_dir() and directory.parent.resolve()==directory.parent,
            'canonical existing namespace parent required')
        directory.mkdir(exist_ok=False)
        files={str(root/suite['source_manifests']['validation']['path']):
            suite['source_manifests']['validation']['sha256']}
        for row in rows:
            source=Path(row['path'])
            gate.require(source.is_absolute() and source.name=='data.h5'
                and 'val_package_compressed' in source.parts,'only original VAL HDF5 sources allowed')
            before=source.stat()
            expected=row.get('source_file_stat')
            if expected is not None:
                actual=dict(size_bytes=before.st_size,mtime_ns=before.st_mtime_ns,
                    inode=before.st_ino,device=before.st_dev)
                gate.require(actual==expected,'original VAL source stat changed')
            checksum=_hash_file(source)
            gate.require(_stamp(before)==_stamp(source.stat()),'source changed after its roster check')
            gate.require(str(source) not in files or files[str(source)]==checksum,'conflicting VAL source declaration')
            files[str(source)]=checksum
        gate._permit(root,permit)
        source_lock=dict(schema=gate.SPLIT_LOCK_SCHEMA,producer_schema=SCHEMA,
            producer=gate.point(root,root/SOURCE),split='validation',
            source_manifest_sha256=suite['source_manifests']['validation']['sha256'],
            source_count=gate.VAL_COUNT,policy_sha256=graph['policy_sha256'],
            physical_policy_sha256=graph['physical_policy_sha256'],control_graph_sha256=graph['control_graph_sha256'],
            graph=gate.plain(graph_ref),admission=gate.plain(admission_ref),files=files,
            actual_routes=suite['actual_routes'],preflight_routes=suite['preflight_routes'],
            inherited742_preflight_entry=suite['inherited742_preflight_entry'],mixed_control_lineage=True,
            TEST_used_for_fitting=False,source_read_after_admission=True,
            original_VAL_bytes_reused_for_new_policy=True,new_independent_validation_dataset=False)
        path=directory/'source_inputs.json';gate._fresh(path,source_lock)
        suite['source_inputs']=gate.point(root,path)
        suite_path=directory/'suite.json';gate._fresh(suite_path,suite)
        refs=initial_refs+[admission['readiness'],suite['source_inputs'],gate.point(root,suite_path)]
        for route in graph['routes']:
            refs.extend((route['configuration'],route['frozen_inputs']))
            refs.extend(route['evaluation'][key] for key in ('worker','preparer','consumer'))
        for reference in refs:
            ref=gate.plain(reference)
            actual=gate.point(root,root/ref['path'])
            gate.require(actual==ref,'declared source/evidence changed before suite lock')
            envelope[str(root/ref['path'])]=ref['sha256']
        lock=dict(schema=SCHEMA+'_suite_evidence_lock',graph=gate.plain(graph_ref),
            admission=gate.plain(admission_ref),policy_sha256=graph['policy_sha256'],files=envelope,
            split='validation',formal_TEST_permitted=False,TEST_used_for_fitting=False)
        lock_path=directory/'suite_lock.json';gate._fresh(lock_path,lock)
        gate._permit(root,permit)
        result=dict(schema=SCHEMA,graph=gate.plain(graph_ref),admission=gate.plain(admission_ref),
            policy_sha256=graph['policy_sha256'],physical_policy_sha256=graph['physical_policy_sha256'],
            control_graph_sha256=graph['control_graph_sha256'],source_inputs=suite['source_inputs'],
            suite=gate.point(root,suite_path),suite_lock=gate.point(root,lock_path),
            actual_routes=suite['actual_routes'],preflight_routes=suite['preflight_routes'],
            inherited742_preflight_entry=suite['inherited742_preflight_entry'],mixed_control_lineage=True,
            source_count=gate.VAL_COUNT,Native_started=False,TEST_read=False)
        bundle_path=directory/'bundle.json';gate._fresh(bundle_path,result)
        return dict(result,bundle=gate.point(root,bundle_path))
