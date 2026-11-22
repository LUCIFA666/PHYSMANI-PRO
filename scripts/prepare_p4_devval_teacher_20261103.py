"""Materialize the already preassigned8 newdevVAL parents; CPU only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.local_mpc_20261102.records import freeze_sources,sha


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def prepare(directory):
    directory=Path(directory).resolve()
    if ROOT/'config/framework' not in directory.parents or directory.exists():
        raise ValueError('fresh config/framework directory required')
    proposal_path=ROOT/'artifacts/framework/local_mpc_20261102/P4_new_development_VAL_plan_proposal_v1/plan_proposal.json'
    proposal=json.loads(proposal_path.read_text())
    base_path=ROOT/proposal['shared_config_base']['path']
    if sha(base_path)!=proposal['shared_config_base']['sha256']:
        raise RuntimeError('frozen shared TRAIN-only calibration/teacher base changed')
    base=json.loads(base_path.read_text())
    train_path=ROOT/'config/framework/P4_teacher_train_20261103_v1/plan.json'
    train=json.loads(train_path.read_text())
    if len(proposal['jobs'])!=8 or len(train['jobs'])!=16:
        raise ValueError('preassigned eightVAL and sixteenTRAIN required')
    directory.mkdir(parents=True,exist_ok=False)
    jobs=[];sources={}
    for original in proposal['jobs']:
        job=deepcopy(original);cfg=deepcopy(base);cfg.update(deepcopy(job.pop('config_overrides')))
        cfg['schema']='P4_newdevVAL_truthB_public50ms_shadow_teacher_20261103_v1'
        cfg['information_setting']='new_development_VAL_truthB_teacher_with_passive_public50ms_GRU_shadow'
        cfg['P4_collection_metadata'].update(variant_id=job['variant_id'],new_development_VAL_only=True,
            official_VAL_or_TEST_source=False,initial_z_is_placeholder_replaced_by_Native_default=True,
            tau_definition='firstreal50msNativeheld in continuous segment with ownlaterGoal+liftedhold')
        path=directory/(job['episode_id']+'.json');write_new(path,cfg)
        job['config']=str(path.relative_to(ROOT));job['config_sha256']=sha(path)
        job['frozen_GRU_checkpoint_sha256']='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'
        sources.update({k:v for k,v in freeze_sources(path,'p1').items() if k!=job['config']})
        jobs.append(job)
    for relative in ['scripts/collect_p4_teacher_train_20261103.py',
        'scripts/collect_p4_devval_teacher_20261103.py','scripts/prepare_p4_devval_teacher_20261103.py',
        'our_mani/local_mpc_control_20261103/native_task_evaluation.py',
        'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt']:
        sources[relative]=sha(ROOT/relative)
    plan=dict(schema='P4_newdevVAL_truthB_shadow50_teacher_20261103_v1',split='val',
        prepared_only=True,Native_started=False,official_VAL_or_TEST_read=False,fit_started=False,
        parent_split_assigned_before_generation_and_fit=True,
        preassigned_proposal=dict(path=str(proposal_path.relative_to(ROOT)),sha256=sha(proposal_path)),
        TRAIN_plan=dict(path=str(train_path.relative_to(ROOT)),sha256=sha(train_path),owner=train['suggested_owner']),
        batch_output_parent='runtime/framework/local_mpc_20261102/P4_teacher_devVAL_20261103_v1',
        suggested_owner='runtime/framework/local_mpc_20261102/P4_teacher_devVAL_20261103_v1_owner',
        execution_requires_TRAIN16_all_actual_exits_and_model_freeze=True,
        model_freeze_manifest_schema='P4_model_freeze_before_newdevVAL_v1',
        teacher_policy='unchanged truth B joint and oldNNfinger undercompletionlatch',
        student_candidate_model_executed_in_teacher_collection=False,
        old_NN_finger_response_requalified=False,
        control=dict(native_dt=.05,control_dt=.1,H=8,CEM_sequences=256),
        failed_task_episodes_retained=True,minimum_success_gate_used_to_select_VAL_cases=False,
        semantic_task_count=1,automatic_task_classifier_claimed=False,
        Native_locks='unchanged original run_local_mpc shell bothshared exclusionlocks',
        STOP_contract='newVALowner STOP propagates only to newVALbatch; originalworker checks100msboundary andactualchild reaped',
        source_sha256=sources,jobs=jobs)
    write_new(directory/'plan.json',plan)
    return directory/'plan.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir',type=Path,default=ROOT/'config/framework/P4_teacher_devVAL_20261103_v1')
    args=parser.parse_args();path=prepare(args.config_dir)
    print(json.dumps(dict(prepared_plan=str(path),newdevVALparents=8,Native_started=False,fit_started=False)))


if __name__=='__main__':main()
