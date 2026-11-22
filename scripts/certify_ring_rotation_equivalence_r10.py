"""CPU-only finite TRAIN ring surface proof; installs no policy annotation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.rotation_equivalence import build_train_certificate,VerifiedFiniteSymmetry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='artifacts/framework/multitask_r10/TRAIN_smoke_collection_v7/manifest.json')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();manifest_path=(ROOT/args.manifest).resolve();out=(ROOT/args.out).resolve()
    if (not manifest_path.is_relative_to(ROOT) or not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10')
            or out.exists()):raise ValueError('explicit existing TRAIN manifest and fresh desktop R10 diagnostic directory required')
    manifest=json.loads(manifest_path.read_text());out.mkdir(parents=True)
    pointer=lambda p:dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
    spin=Rotation.from_rotvec(np.array([1.,0.,0.])[None]*np.arange(4)[:,None]*np.pi/2).as_matrix()
    group=np.concatenate((spin,spin@Rotation.from_rotvec([0.,np.pi,0.]).as_matrix()));entries=[]
    for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
        candidates=[e for e in manifest['entries'] if e['task']==task and e.get('teacher_used') is True]
        if not candidates:raise ValueError('fixed first TRAIN teacher acquisition missing: '+task)
        entry=candidates[0];mesh=(ROOT/entry['cache']).parent/'TRAIN_geometry.json'
        receipt=next(row for row in manifest['case_receipts'] if row['case_id']==mesh.parent.name)
        value=json.loads(mesh.read_text());vertices=np.asarray(value['bodies']['ring']['collision_vertices_body'])
        certificate=build_train_certificate(ROOT,pointer(mesh),'ring',group,(vertices.min(0)+vertices.max(0))/2,
            TRAIN_manifest_pointer=pointer(manifest_path),TRAIN_receipt_pointer={k:receipt[k] for k in ('path','sha256')})
        path=out/(task+'__ring_D4_certificate.json');atomic_json(path,certificate)
        VerifiedFiniteSymmetry.load(ROOT,pointer(path))
        entries.append(dict(task=task,TRAIN_parent_source=entry['parent_source_group'],certificate=pointer(path),
            group_order=certificate['group_validation']['order'],convex_parts=len(certificate['surface_coverage']),
            maximum_original_surface_Hausdorff_upper_bound_m=max(row['original_surface_Hausdorff_upper_bound_m']
                for row in certificate['group_surface_matches']),surface_tolerance_m=certificate['surface_tolerance_m'],
            reproducible_certificate_loaded=True,policy_skill_authorized=False,full_orientation_observable=False))
    report=dict(version='r10_TRAIN_ring_finite_surface_certificate_prototype_v1',complete=True,
        TRAIN_only=True,VAL_used=False,TEST_used=False,native_or_GPU_launched=False,
        source_manifest=pointer(manifest_path),source_script=pointer(Path(__file__)),entries=entries,
        tracker_or_actor_modified=False,actual_current_ICP_quotient_readiness_evaluated=False,
        full_orientation_promoted=False,automatic_affordance_permission=False,
        next_integration_requirements=[
            'Expose every retained current ICP hypothesis/RMS; certify consistency modulo D4 without claiming absolute accuracy.',
            'Freeze explicit TRAIN phase/body affordance permissions and transformed anchor/approach validation before acting.',
            'Preserve free-tip/base and approach-direction semantics of the selected peg; geometric ring symmetry does not exchange peg affordances.',
            'Use symmetry-aware supervision or validated action-orbit enumeration; native full-R labels cannot correct an online ambiguous gauge.',
            'A temporal gauge lift needs a causal motion/discontinuity bound; quotient agreement is not a full physical angular-velocity measurement.',
            'Keep current required phase support, model/source coverage and independent formal VAL/TEST gates.'])
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(complete=True,report=pointer(out/'report.json'),entries=entries),ensure_ascii=False))


if __name__=='__main__':main()
