"""Explicit TRAIN-only reference-kind overrides with original-source evidence.

Only the six audited translation-task phases are eligible. Roles, identities,
anchors and manipulated bindings remain unchanged; runtime actors consume the
reference kind in the fitted library and never consult native teacher evidence.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA='r10_TRAIN_reference_kind_overrides_v1'
EVIDENCE_RELATIVE='artifacts/framework/multitask_r10/phase_reference_affordance_config_DRAFT_v1.json'
ALLOWED={}
for _suffix in ('','_high_speed'):
    ALLOWED['moving_basketball_in_hoop'+_suffix]={'deposit':'hoop'}
    ALLOWED['push_moving_button'+_suffix]={'approach':'button','press':'button'}
    ALLOWED['put_rubbish_in_moving_bin'+_suffix]={'approach':'rubbish','acquire':'rubbish','deposit':'bin'}


class ReferenceOverrideError(ValueError):pass


def require(condition,message):
    if not condition:raise ReferenceOverrideError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def structural_overrides(protocol):
    if protocol is None or 'TRAIN_reference_kind_overrides' not in protocol:return {}
    mapping=protocol['TRAIN_reference_kind_overrides']
    require(isinstance(mapping,dict) and bool(mapping),'explicit nonempty audited override mapping required')
    for task,phases in mapping.items():
        require(task in ALLOWED and isinstance(phases,dict) and bool(phases),'unaudited task/empty reference override')
        for phase,row in phases.items():
            require(phase in ALLOWED[task] and isinstance(row,dict) and
                set(row)=={'reference_role','reference_kind'},'unsupported phase or reference-binding mutation')
            require(row['reference_kind']=='translation_only' and row['reference_role']==ALLOWED[task][phase],
                'only audited translation reference kind with unchanged semantic role is allowed')
    return mapping


def validate_overrides(protocol,*,root):
    """Verify evidence, all600 TRAIN cache hashes and author source fingerprints."""
    from .specs import get_task_spec
    root=Path(root).resolve();mapping=structural_overrides(protocol)
    if not mapping:
        require(protocol is None or 'TRAIN_reference_kind_override_evidence' not in protocol,
            'orphan override evidence without explicit mapping')
        return {},{}
    pointer=protocol.get('TRAIN_reference_kind_override_evidence')
    require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'override evidence requires explicit path/SHA')
    path=(root/pointer['path']).resolve()
    require(path==(root/EVIDENCE_RELATIVE).resolve() and path.is_file(),'only the audited original TRAIN affordance evidence is allowed')
    require(sha(path)==pointer['sha256'],'reference override evidence changed')
    evidence=json.loads(path.read_text())
    require(evidence.get('version')=='r10_phase_reference_and_symmetry_affordance_config_DRAFT_v1'
        and evidence.get('TRAIN_only_design_evidence') is True and evidence.get('TEST_used') is False
        and evidence.get('VAL_used') is False,'override evidence must be audited original TRAIN only')
    baseline=Path(protocol['baseline_repo']).resolve();inputs={str(path):pointer['sha256']}
    for name,expected in evidence['inputs_sha256'].items():
        item=(root/name).resolve()
        require(item.is_relative_to(root) or item.is_relative_to(baseline),'unsupported evidence source location')
        require(item.is_file() and sha(item)==expected,'affordance evidence source changed: '+str(item))
        inputs[str(item)]=expected
    diagnostic=(root/'artifacts/framework/multitask_r10/TRAIN_world_orientation_diagnostic_v1.json').resolve()
    require(str(diagnostic) in inputs,'original600 TRAIN audit is not bound by evidence')
    stats=json.loads(diagnostic.read_text())
    require(stats.get('version')=='r10_original_TRAIN_world_orientation_design_diagnostic_v1'
        and stats.get('TRAIN_only') is True and stats.get('sources_read')==600
        and stats.get('TEST_used') is False and stats.get('VAL_used') is False
        and stats.get('future_debug_target_XYZ_read') is False and set(stats['per_task'])==set(ALLOWED),
        '600 original TRAIN source audit is incomplete or includes forbidden supervision')
    counts=Counter();parent_keys=set();train=(root/'artifacts/framework/multitask_r10/demonstrations/train').resolve()
    for name,expected in stats['inputs_sha256'].items():
        item=(root/name).resolve()
        require(item.is_relative_to(root) and item.is_file() and sha(item)==expected,'original TRAIN audit input changed: '+str(item))
        inputs[str(item)]=expected
        if item.suffix=='.npz':
            require(item.is_relative_to(train),'original orientation evidence cache must be TRAIN')
            parts=item.relative_to(train).parts
            require(len(parts)==2 and parts[0] in ALLOWED,'foreign original TRAIN source in orientation audit')
            counts[parts[0]]+=1;parent_keys.add((parts[0],item.stem))
    require(counts==dict.fromkeys(ALLOWED,100) and len(parent_keys)==600,'600 distinct original TRAIN parents required')
    for task,phases in mapping.items():
        spec=get_task_spec(task);source=baseline/spec.native_source_relative_path
        require(inputs.get(str(source))==spec.native_source_sha256==sha(source),'author task source fingerprint differs')
        for phase,row in phases.items():
            audited=evidence['per_task'][task][phase]
            require(audited.get('reference_kind')==row['reference_kind'] and audited.get('reference_role')==row['reference_role']
                and audited.get('verified_original_TRAIN_parent_count')==100,'override lacks audited exact phase/role evidence')
            original=stats['per_task'][task][audited['world_orientation_prior_phase']]
            require(original['sources']==100 and np.isfinite(original['parent_mean_deviation_rad']['max'])
                and abs(original['parent_mean_deviation_rad']['max']-audited['max_parent_world_TCP_angle_variation_rad'])<=1e-12,
                'audited phase source count/rotation statistic differs')
            q=np.asarray(audited['descriptive_TRAIN_world_TCP_quaternion_xyzw'],float)
            other=np.asarray(original['descriptive_parent_equal_world_quaternion_xyzw'],float)
            require(q.shape==(4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1)<1e-6
                and Rotation.from_matrix(Rotation.from_quat(q).as_matrix()@Rotation.from_quat(other).as_matrix().T).magnitude()<1e-8,
                'audited TRAIN descriptive rotation differs')
    return mapping,inputs
