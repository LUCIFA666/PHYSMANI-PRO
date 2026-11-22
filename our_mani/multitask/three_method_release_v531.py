"""Full16 VAL/TEST completeness with common author RML and motion diagnostics.

Old frozen releases retain their original rules. This new protocol implements
the user's explicit choice to report motion uniformly and score native tasks.
"""
from .evaluation import digest
from .evaluation_roster16_v283 import METHODS,TASKS
from .three_method_release16 import policy_fingerprint as route_fingerprint,validate_grid
from .common_rml_protocol_v530 import fingerprint,require_attestation

SCHEMA='r10_full16_common_author_RML_validation_release_v531'


def policy_fingerprint(routes):
    profile_sha=fingerprint(routes.get('common_author_RML_profile',{}))
    for route in routes.get('routes',[]):
        if route.get('common_RML_profile_sha256')!=profile_sha:
            raise ValueError('all48 routes must bind the same author RML profile')
    return digest(dict(routes_sha256=route_fingerprint(routes),common_RML_profile_sha256=profile_sha))


def audit_grid(cases,manifest,split,routes,*,completed_VAL_audit=False):
    cases=list(cases)
    policy_sha=policy_fingerprint(routes)
    profile_sha=fingerprint(routes['common_author_RML_profile'])
    proof=validate_grid(cases,manifest,split,policy_sha,completed_VAL_audit=completed_VAL_audit)
    for case in cases:
        require_attestation(case.get('common_author_RML_parameters',{}),profile_sha)
    return dict(proof,common_RML_profile_sha256=profile_sha,
        motion_is_diagnostic_only=True,motion_metrics_required_for_all_methods=True,
        native_task_success_unchanged=True)


def validation_release(cases,manifest,routes):
    proof=audit_grid(cases,manifest,'validation',routes)
    if proof['source_cases']!=315 or proof['method_cases']!=945:
        raise ValueError('complete315-source/945-case VAL required')
    return dict(proof,schema=SCHEMA,formal_test_permitted=True,TEST_used=False)


def require_test_release(release,routes):
    if (release.get('schema')!=SCHEMA or release.get('complete') is not True
        or release.get('split')!='validation' or release.get('TEST_used') is not False
        or release.get('TEST_used_for_fitting') is not False
        or release.get('formal_test_permitted') is not True
        or release.get('source_cases')!=315 or release.get('method_cases')!=945
        or release.get('completed_method_cases')!=945 or release.get('methods')!=list(METHODS)
        or release.get('tasks')!=list(TASKS) or release.get('policy_sha256')!=policy_fingerprint(routes)
        or release.get('common_RML_profile_sha256')!=fingerprint(routes['common_author_RML_profile'])
        or release.get('motion_is_diagnostic_only') is not True
        or release.get('motion_metrics_required_for_all_methods') is not True
        or release.get('native_task_success_unchanged') is not True):
        raise ValueError('complete unchanged common-author-RML full16 VAL release required before TEST selection')
    return True
