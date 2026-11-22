"""Default check only; --fit makes one predeclared TRAIN6 ridge solution."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_onpolicy_response_20261103.data import load_dataset,oracle_reference_audit,sha
from our_mani.local_mpc_onpolicy_response_20261103.ridge import fit_parameters,OnPolicyCoupledResponse
from our_mani.local_mpc_onpolicy_response_20261103.evaluation import evaluate
from our_mani.local_mpc_control_20261103.coupled_response import FrozenCoupledResponse
from our_mani.local_mpc_p2_20261103.kinematics import PoEKinematics

BASE_SHA='d7c7fac5f8de7a1bf3d37c10b94a20585728a7c03c7d1a05920799e69a501f92'

def write_new(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fit',action='store_true')
    args=parser.parse_args();episodes,manifest=load_dataset(args.dataset)
    audits={e['episode_id']:oracle_reference_audit(e) for e in episodes}
    if not args.fit:
        print(json.dumps(dict(mode='check_only_no_fit',complete=True,episodes=len(episodes),reference_audits=audits,
            parameter_updates=0,Native_started=False),indent=2));return
    if args.output is None:parser.error('--fit requires a fresh --output; existing outputs are never replaced')
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(dataset_manifest_sha256=sha(args.dataset/'manifest.json'),plan_sha256=manifest['plan_sha256'],
        protocol_sha256=manifest['protocol_sha256'],source_sha256=manifest['source_sha256'],
        fit_driver_sha256=sha(__file__))
    parameters=fit_parameters(episodes,manifest['protocol'],source_identity=identity)
    path=args.output/'parameters.json';write_new(path,parameters)
    model=OnPolicyCoupledResponse.from_json(path,expected_sha256=sha(path))
    fkpath=ROOT/'runtime/framework/local_mpc_20261102/p2_fk_canary_v1/kinematics.json'
    fk=PoEKinematics(json.loads(fkpath.read_text()))
    metrics=evaluate(model,episodes,fk);write_new(args.output/'metrics.json',metrics)
    oldpath=ROOT/'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json'
    old=FrozenCoupledResponse.from_json(oldpath,expected_sha256=BASE_SHA)
    write_new(args.output/'old_B_matched_cohort_metrics.json',evaluate(old,episodes,fk))
    status=dict(complete=True,one_predeclared_ridge_fit=True,fit_parents=manifest['protocol']['fit_parent_ids'],
        internal_val_parents=manifest['protocol']['internal_val_parent_ids'],internal_val_used_for_selection=False,
        parameter_path=str(path.resolve()),parameter_sha256=sha(path),identity=identity,
        frozen_FK_sha256=sha(fkpath),old_B_parameter_sha256=BASE_SHA,
        reference_audits=audits,Native_started=False,NN_fitting_performed=False,
        official_or_development_VAL_or_TEST_for_fit=False,old_frozen_parameters_changed=False,
        gripper_or_contact_qualified=False,public_task_success_claimed=False,globally_qualified=False)
    write_new(args.output/'training_status.json',status)
    print(json.dumps(status,indent=2,allow_nan=False))

if __name__=='__main__':main()
