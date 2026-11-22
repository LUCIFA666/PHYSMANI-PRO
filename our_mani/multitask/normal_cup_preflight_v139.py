"""Same normal-cup policy, with batched immutable fit-source verification."""
from pathlib import Path
import json
from .normal_cup_native_v111 import binding as primary_binding,library_scope
from .normal_cup_lag_native_v118 import binding as lag_binding
from .normal_cup_diagonal_native_v123 import binding as diagonal_binding
from .normal_cup_primary_fitting_v107 import TASK
from .task_actor_backend_v69 import persistent_reference_actor_model_preflight
from .frozen_subset_batch_v132 import checked_subset


def model_preflight(root,prepared):
    root=Path(root).resolve();args=prepared['args'];protocol=prepared['config']
    if args.split!='train' or args.task!=TASK or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('normal-cup explicit-domain TRAIN consumer required')
    primary=primary_binding(root,protocol);lag=lag_binding(root,protocol);diagonal=diagonal_binding(root,protocol)
    if (protocol.get('normal_cup_primary_v111')!=primary or protocol.get('normal_cup_lag_v118')!=lag
            or protocol.get('normal_cup_diagonal_v123')!=diagonal):raise ValueError('actual normal-cup policy/source binding differs')
    expected={}
    for proof in (primary,lag,diagonal):
        expected.update({str(root/name):sha for name,sha in proof['sources_sha256'].items()})
    expected[str(root/primary['report']['path'])]=primary['report']['sha256']
    checked_subset(root,prepared['files'],expected)
    report=json.loads((root/primary['report']['path']).read_text())
    checked_subset(root,prepared['files'],report['verified_files_sha256'])
    pointer=protocol['cup_public_association_v133'];checked_subset(root,prepared['files'],{str(root/pointer['path']):pointer['sha256']})
    association=json.loads((root/pointer['path']).read_text())
    if protocol['attachment_association_distance_m']!=association['association_radius_m']:raise ValueError('actual cup association radius differs')
    checked_subset(root,prepared['files'],association['verified_files_sha256'])
    with library_scope(root,protocol,args.task):result=persistent_reference_actor_model_preflight(root,prepared)
    return dict(result,explicit_initial_sequence_policy_domain=primary,
        fitted_lag_budget_overlay=lag,bounded3D_overlay=diagonal,
        batched_fit_source_verification=True,old_full_scope_library_release_claimed=False,formal_VAL_TEST=False)
