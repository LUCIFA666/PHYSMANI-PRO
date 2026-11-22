"""Optional probe-only on-policy joint response; original fingers stay intact.

The new TRAIN6 fit improves a small internal-probe cohort, while general800ms
error worsens. It is therefore not a global response replacement. This helper
contains no simulator calls, fitting, object state inference, or task success.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path

import numpy as np

from ..local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel
from ..local_mpc_onpolicy_response_20261103.ridge import OnPolicyCoupledResponse
from .coupled_response import CoupledWithFingerResponse

ROOT=Path(__file__).resolve().parents[2]
FIT='artifacts/framework/local_mpc_20261102/P2_onpolicy_response_TRAIN8_fit_v1'
DATA='artifacts/framework/local_mpc_20261102/P2_onpolicy_response_TRAIN8_dataset_v1'
PLAN='config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/plan.json'
PROTOCOL='config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/response_fit_protocol.json'
PARAMETER_SHA256='f876fbba9f2351ca6457627ab68726be2eea9f2ae435db525fde346cd5f50c32'
EXPECTED={FIT+'/parameters.json':PARAMETER_SHA256,
    FIT+'/training_status.json':'1aa19caa8056ffed7fdac867945c6f1d158e1704ea09d425bdb74db74704aa95',
    FIT+'/metrics.json':'2b4d7420323e7e59fe85711d08a26652c00bb612597f0667c4ee6c4c72ecdc7f',
    FIT+'/old_B_matched_cohort_metrics.json':'001a429adcbd679743a7edc9caf09da77c2e7bfd31fdd1b424936b4d00730b87',
    DATA+'/manifest.json':'5eb9d6f6f82d6883dfb56ce82f86bef6c1cd412dc170efc35e964ba31e6a4a3c',
    PLAN:'ffb27596d46f63afefb309b29f48986954534ea0f8ed39312f2026ab1e0f07d0',
    PROTOCOL:'486efe03532896b8928306bda88bb719fc5ab28305eff76d74933c9e5421d83c'}


def _checked(path,digest):
    body=Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest()!=digest:raise ValueError('frozen probe-response evidence SHA mismatch: '+str(path))
    return body


def _verified():
    raw={relative:_checked(ROOT/relative,digest) for relative,digest in EXPECTED.items()}
    status=json.loads(raw[FIT+'/training_status.json']);dataset=json.loads(raw[DATA+'/manifest.json'])
    protocol=json.loads(raw[PROTOCOL]);parameters=json.loads(raw[FIT+'/parameters.json'])
    identity=status.get('identity',{})
    if (status.get('complete') is not True or status.get('one_predeclared_ridge_fit') is not True
            or status.get('parameter_sha256')!=PARAMETER_SHA256
            or status.get('internal_val_used_for_selection') is not False
            or status.get('official_or_development_VAL_or_TEST_for_fit') is not False
            or identity.get('dataset_manifest_sha256')!=EXPECTED[DATA+'/manifest.json']
            or identity.get('plan_sha256')!=EXPECTED[PLAN]
            or identity.get('protocol_sha256')!=EXPECTED[PROTOCOL]
            or dataset.get('all_sources_dataset_use_train') is not True
            or dataset.get('official_or_development_VAL_or_TEST_opened') is not False
            or parameters.get('fit_source',{}).get('source_identity')!=identity):
        raise ValueError('one fixed TRAIN-only probe-response fit identity required')
    fit=set(protocol['fit_parent_ids']);internal=set(protocol['internal_val_parent_ids'])
    rows=dataset['episodes']
    if (len(fit)!=6 or len(internal)!=2 or fit&internal or len(rows)!=8
            or len({r['parent_id'] for r in rows})!=8
            or {r['parent_id'] for r in rows if r['role']=='fit'}!=fit
            or {r['parent_id'] for r in rows if r['role']=='internal_val'}!=internal):
        raise ValueError('exact six fit plus two independent internal TRAIN parents required')
    sources={str((ROOT/name).resolve()):digest for name,digest in EXPECTED.items()}
    for path,digest in identity['source_sha256'].items():
        path=Path(path).resolve()
        if ROOT not in path.parents:raise ValueError('frozen code evidence must remain inside project')
        _checked(path,digest);sources[str(path)]=digest
    for row in rows:
        path=ROOT/DATA/row['file'];_checked(path,row['sha256']);sources[str(path.resolve())]=row['sha256']
    driver=ROOT/'scripts/fit_onpolicy_response_20261103.py'
    _checked(driver,identity['fit_driver_sha256']);sources[str(driver.resolve())]=identity['fit_driver_sha256']
    fk=ROOT/'runtime/framework/local_mpc_20261102/p2_fk_canary_v1/kinematics.json'
    _checked(fk,status['frozen_FK_sha256']);sources[str(fk.resolve())]=status['frozen_FK_sha256']
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metrics=json.loads(raw[FIT+'/metrics.json'])['roles']['internal_val']
    old=json.loads(raw[FIT+'/old_B_matched_cohort_metrics.json'])['roles']['internal_val']
    def cohort(value,horizon,name):
        return deepcopy(value['rollouts']['regenerated_reference']['by_horizon'][str(horizon)]['cohorts'][name])
    evidence=dict(protocol='probe_only_onpolicy_joint_response_v55',parameter_sha256=PARAMETER_SHA256,
        source_TRAIN_parents=8,fit_parents=6,internal_TRAIN_parents=len(metrics['parents']),
        internal_probe_ticks=metrics['probe_ticks'],
        internal_pure_probe_400ms=cohort(metrics,8,'pure_probe_window'),
        old_B_matched_internal_pure_probe_400ms=cohort(old,8,'pure_probe_window'),
        internal_pure_probe_800ms=cohort(metrics,16,'pure_probe_window'),
        internal_all_800ms=cohort(metrics,16,'all'),old_B_matched_internal_all_800ms=cohort(old,16,'all'),
        pure_probe_800ms_qualified=False,global_800ms_qualified=False,
        cohort_caveat='only two internal TRAIN parents and14probe ticks; no pure800ms probe samples; all-cohort800ms worsens',
        parameter_selection='single predeclared fit; internal reporting only',
        deployment_scope_selected_using_internal_TRAIN_report=True,
        parameter_fit_hyperparameter_selection_used_internal=False,
        gripper_response_requalified=False,contact_or_task_success_qualified=False,formal_TEST_used=False,
        globally_qualified=False)
    return sources,evidence


def probe_response_sources():
    """Exact absolute byte identities for the owner's conditional sourcefreeze."""
    sources,_=_verified();return sources


class ProbeOnlyOnPolicyExecution(JointResidualFKExecutionModel):
    """Stateless model; every plan must supply current actual unheld/closed h."""
    def initialize_batch(self,actual,batch_size):
        h=actual.get('actuator_state') if isinstance(actual,dict) else None
        now=actual.get('time') if isinstance(actual,dict) else None
        clock=h.get('reference_clock') if isinstance(h,dict) else None
        if (not isinstance(h,dict) or actual.get('held') is not False
                or actual.get('requested_opening')!=0. or h.get('gripper_target')!=0.
                or isinstance(now,(bool,np.bool_)) or not isinstance(now,(float,int,np.number)) or not np.isfinite(now)
                or isinstance(clock,(bool,np.bool_)) or not isinstance(clock,(float,int,np.number))
                or not np.isfinite(clock) or abs(float(clock)-float(now))>1e-8
                or ('predicted' in h and (not isinstance(h['predicted'],(bool,np.bool_)) or bool(h['predicted'])))
                or 'virtual' in str(h.get('source','')).lower()):
            raise ValueError('probe-only model requires fresh actual unheld state/h and known close request')
        return super().initialize_batch(actual,batch_size)

    def step_batch(self,x,u,h,dt=.1):
        if np.any(x['held']) or not np.all(np.asarray(u['opening'])==0.):
            raise ValueError('probe-only response cannot be used for held/reopen phases')
        return super().step_batch(x,u,h,dt)


def build_onpolicy_probe_execution(base_execution):
    """Return a new probe model and evidence; never mutate the global core.

    The caller's optional flag must route here only for an admitted verification
    probe. Unsupported outer response wrappers are rejected, not flattened.
    """
    if type(base_execution) is not JointResidualFKExecutionModel:
        raise TypeError('explicit unwrapped JointResidualFKExecutionModel base required')
    transition=base_execution.transition_batch
    if type(transition) is not CoupledWithFingerResponse:
        raise TypeError('explicit coupled joint plus original finger response required; do not flatten other wrappers')
    sources,evidence=_verified()
    joint=OnPolicyCoupledResponse.from_json(ROOT/FIT/'parameters.json',expected_sha256=PARAMETER_SHA256)
    response=CoupledWithFingerResponse(joint,transition.finger_response)
    model=ProbeOnlyOnPolicyExecution(base_execution.reference_step,base_execution.limits,response,
        base_execution.fk_batch,config=base_execution.config,latent_initializer=base_execution.latent_initializer,
        model_id='probe_only_onpolicy_'+PARAMETER_SHA256)
    receipt=dict(**evidence,base_model_id=base_execution.model_id,model_id=model.model_id,
        reference_step_preserved=True,limits_preserved=True,fk_preserved=True,config_preserved=True,
        original_finger_response_object_preserved=True,latent_initializer_preserved=True,
        actual_state_or_h_cached=False,new_weights_used_in_other_phases=False,
        source_sha256=sources)
    return model,receipt
