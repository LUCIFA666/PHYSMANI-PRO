"""Load ordinary, first-execution VAL or historical composite acquisitions.

The composite projection retains each raw receipt's actual plan SHA. It is
never converted to an invented single-plan acquisition.
"""
import json
from pathlib import Path

from .evaluation import file_sha

COMPOSITE_VERSION = 'r10_historical_composite_acquisition_lineage_v1'
BUNDLE_VERSION = 'r10_TRAIN_primary2800_plus_fixed_aux200_v1'
COUNTED_BUNDLE_VERSION = 'r10_TRAIN_primary3000_plus_counted_high_cup100_v1'


def load_acquisition_manifest(root, path):
    root = Path(root).resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('acquisition manifest must stay inside the desktop project')
    value = json.loads(path.read_text())
    composite = (value.get('schema') == COMPOSITE_VERSION
                 or value.get('version') == COMPOSITE_VERSION)
    first_val = 'shared_profile_execution_registration' in value
    bundle = value.get('version') == BUNDLE_VERSION or value.get('schema') == BUNDLE_VERSION
    counted_bundle = value.get('version') == COUNTED_BUNDLE_VERSION or value.get('schema') == COUNTED_BUNDLE_VERSION
    if sum((composite,first_val,bundle,counted_bundle))>1:
        raise ValueError('acquisition cannot combine historical composite, first VAL or auxiliary bundle markers')
    if not composite and not first_val and not bundle and not counted_bundle:
        return value, None
    if counted_bundle:
        from . import counted_teacher_bundle as auditor
        proof = auditor.audit_bundle(root,path,require_complete=True)
    elif bundle:
        from . import teacher_supplement as auditor
        proof = auditor.audit_bundle(root,path,require_complete=True)
    elif composite:
        from . import acquisition_lineage as auditor
        proof = auditor.audit_acquisition_lineage(root, path, require_complete=True)
    else:
        from . import validation_lineage as auditor
        proof = auditor.audit_validation_acquisition_lineage(root, path, require_complete=True)
    projected = proof['acquisition_projection']
    if (proof.get('audit_passed') is not True or proof.get('complete') is not True
            or projected.get('complete') is not True):
        raise ValueError('acquisition needs complete verified lineage')
    if composite or bundle or counted_bundle:
        expected_version = COUNTED_BUNDLE_VERSION if counted_bundle else BUNDLE_VERSION if bundle else COMPOSITE_VERSION
        if (projected.get('version') != expected_version
                or projected.get('plan_sha256') is not None
                or projected.get('single_plan_SHA_claimed') is not False
                or projected.get('projection_rewrites_original_receipts') is not False):
            raise ValueError('composite cannot forge a single plan or rewrite original receipts')
    elif (proof.get('execution_kind') != 'first_execution_VAL_shared_development_initialization'
            or proof.get('first_execution') is not True or projected != value
            or projected.get('version') != 'r10_causal_body_tracks_v1'
            or not isinstance(projected.get('plan_sha256'), str)
            or len(projected['plan_sha256']) != 64
            or proof.get('actual_plan_sha256_values') != [projected['plan_sha256']]):
        raise ValueError('first VAL execution must retain its original single-plan manifest')
    ids = [row['case_id'] for row in projected['case_receipts']]
    mapping = proof['per_case_actual_plan_sha256']
    if len(ids) != len(set(ids)) or set(mapping) != set(ids):
        raise ValueError('acquisition actual per-case plan identity is incomplete or inconsistent')
    if composite or bundle or counted_bundle:
        if (projected.get('per_case_actual_plan_sha256') != mapping
                or any(row.get('actual_plan_sha256') != mapping[row['case_id']]
                       for row in projected['case_receipts'])):
            raise ValueError('composite actual per-case plan identity is incomplete or inconsistent')
    elif any(digest != projected['plan_sha256'] for digest in mapping.values()):
        raise ValueError('first VAL actual per-case plan differs from its original single plan')
    proof = dict(proof)
    files = dict(proof['verified_files_sha256'])
    from . import acquisition_lineage
    for source in (Path(__file__).resolve(), Path(auditor.__file__).resolve(),
                   Path(acquisition_lineage.__file__).resolve()):
        expected = file_sha(source)
        if str(source) in files and files[str(source)] != expected:
            raise ValueError('acquisition audit source changed')
        files[str(source)] = expected
    proof['verified_files_sha256'] = files
    return projected, proof


def actual_case_plan_sha(manifest, proof, case_id):
    if proof is None:
        return manifest.get('plan_sha256')
    value = proof['per_case_actual_plan_sha256'].get(case_id)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('missing actual plan SHA for an audited acquisition')
    return value
