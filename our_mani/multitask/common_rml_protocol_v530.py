"""The user-authorized common author RML profile for the three-method suite.

The six-rad/s² observed-window threshold is reported for every method. It is
not an extra release gate on the geometry methods. This module only reads
sources and runtime parameters; it never changes the native arm.
"""
from pathlib import Path
import math
from .evaluation import digest, file_sha
from .evaluation_roster16_v283 import METHODS

AUTHORIZATION = '三组沿用作者相同 RML 参数，统一报告运动指标、按原任务判成功（推荐）'
PARAMETERS = dict(max_velocity=3., max_acceleration=12., max_jerk=1000.,
                  counted_dt=.05, decision_ticks=10, budget_seconds=10.)
LOCAL_SOURCES = (
    'our_mani/ablations/physmani_executor_v4.py',
    'our_mani/sensor_alignment/physmani_executor_v5.py',
    'our_mani/sensor_alignment/physmani_executor_v6.py',
    'our_mani/multitask/physmani_executor_v7.py',
)
NATIVE_SOURCES = (
    'third_party/RLBench/rlbench/environment.py',
    'third_party/PyRep/pyrep/robots/arms/arm.py',
    'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',
    'third_party/PyRep/pyrep/backend/sim.py',
)


def binding(root, native_repo):
    return dict(schema='r10_common_author_RML_profile_v530', methods=list(METHODS),
        parameters=dict(PARAMETERS), user_authorization=AUTHORIZATION,
        original_waypoint_filter_and_RML_algorithm=True,
        motion_metrics_required_for_all_methods=True,
        measured_acceleration6_is_diagnostic_only=True,
        native_task_success_unchanged=True,
        sources_sha256={p:file_sha(Path(root)/p) for p in LOCAL_SOURCES},
        native_sources_sha256={p:file_sha(Path(native_repo)/p) for p in NATIVE_SOURCES})


def fingerprint(profile):
    if (profile.get('schema')!='r10_common_author_RML_profile_v530'
        or profile.get('methods')!=list(METHODS) or profile.get('parameters')!=PARAMETERS
        or profile.get('user_authorization')!=AUTHORIZATION
        or any(profile.get(k) is not True for k in (
            'original_waypoint_filter_and_RML_algorithm', 'motion_metrics_required_for_all_methods',
            'measured_acceleration6_is_diagnostic_only', 'native_task_success_unchanged'))):
        raise ValueError('user-authorized unchanged common author RML profile required')
    for key, names in (('sources_sha256',LOCAL_SOURCES),('native_sources_sha256',NATIVE_SOURCES)):
        hashes=profile.get(key,{})
        if set(hashes)!=set(names) or any(not isinstance(s,str) or len(s)!=64 for s in hashes.values()):
            raise ValueError('all common RML source hashes required')
    return digest(profile)


def attest(env, config, profile):
    """Read actual arm/counted-clock settings before any policy action."""
    profile_sha=fingerprint(profile)
    actual={k:float(getattr(env.arm,k)) for k in ('max_velocity','max_acceleration','max_jerk')}
    actual.update(counted_dt=float(env.ledger.dt), decision_ticks=config['baseline_decision_ticks'],
                  budget_seconds=float(env.ledger.budget))
    if any(not math.isfinite(v) or not math.isclose(v,PARAMETERS[k],rel_tol=0.,abs_tol=1e-10)
           for k,v in actual.items()):
        raise ValueError('runtime arm or counted-clock parameters differ from common author RML')
    if config['dt']!=PARAMETERS['counted_dt']:
        raise ValueError('configured counted dt differs from native clock')
    return dict(schema='r10_observed_common_author_RML_parameters_v530',
        profile_sha256=profile_sha, actual_parameters=actual, complete=True,
        native_parameters_mutated=False, actual_motion_bound_claimed=False)


def require_attestation(receipt, profile_sha):
    if (receipt.get('schema')!='r10_observed_common_author_RML_parameters_v530'
        or receipt.get('profile_sha256')!=profile_sha or receipt.get('complete') is not True
        or receipt.get('actual_parameters')!=PARAMETERS
        or receipt.get('native_parameters_mutated') is not False
        or receipt.get('actual_motion_bound_claimed') is not False):
        raise ValueError('complete unchanged runtime common RML parameter attestation required')
    return True
