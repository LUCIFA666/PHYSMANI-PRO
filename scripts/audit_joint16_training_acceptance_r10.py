"""Verify completed joint16 TRAIN coverage independently of executor success."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.executor_suite16 import TASKS,OURS,CHECKPOINT_SHA
from our_mani.multitask.context_se3 import MultiBodyContextDynamics
from our_mani.multitask.joint16_checkpoint_v172 import check_training_metadata_and_roster
from our_mani.multitask.evaluation import atomic_json,file_sha


def main():
    base=ROOT/'artifacts/framework/multitask_r10';out=base/'joint16_training_acceptance_v438'
    if out.exists():raise FileExistsError('preserve prior audit')
    checkpoint=base/'context_joint16_hold77_v157/dynamics.pt';model=MultiBodyContextDynamics(checkpoint)
    if model.checkpoint_sha256!=CHECKPOINT_SHA:raise ValueError('shared checkpoint changed')
    metadata=model.metadata;source_path=checkpoint.parent/metadata['training_source_manifest']['path_relative_to_checkpoint']
    if file_sha(source_path)!=metadata['training_source_manifest']['sha256']:raise ValueError('TRAIN source manifest changed')
    source=json.loads(source_path.read_text());roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json'
    roster=json.loads(roster_path.read_text())['episodes']
    proof=check_training_metadata_and_roster(metadata,source['entries'],roster)
    if set(proof['tasks'])!=set(TASKS):raise ValueError('full16 training required')
    report_path=checkpoint.parent/'report.json';report=json.loads(report_path.read_text())
    if report['checkpoint_sha256']!=CHECKPOINT_SHA or report['metadata']!=metadata:
        raise ValueError('saved training report and actual checkpoint disagree')
    artifacts=[];inputs={str(checkpoint):CHECKPOINT_SHA,str(source_path):file_sha(source_path),
        str(roster_path):file_sha(roster_path),str(report_path):file_sha(report_path),str(Path(__file__)):file_sha(__file__)}
    for task in TASKS:
        route=OURS[task];config_path=base/route.configuration(task);config=json.loads(config_path.read_text())
        if config['context_checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('task context differs')
        inputs[str(config_path)]=file_sha(config_path)
        learned={}
        if task in config.get('learned_artifacts',{}):
            for name in ('geometry','interaction','rigid_layout','colour'):
                pointer=config['learned_artifacts'][task].get(name)
                if pointer is None:continue
                path=ROOT/pointer['path']
                if file_sha(path)!=pointer['sha256']:raise ValueError('task TRAIN artifact changed')
                inputs[str(path)]=pointer['sha256'];learned[name]=pointer
        artifacts.append(dict(task=task,context_checkpoint=config['context_checkpoint'],
            existing_TRAIN_operation_artifacts=learned,
            block_uses_original_geometry_kernel=task.startswith('pick_moving_target_on_the_table')))
    atomic_json(out/'report.json',dict(schema='r10_joint16_completed_training_independent_acceptance_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),training_complete=True,training=proof,
        checkpoint=dict(path=str(checkpoint.relative_to(ROOT)),sha256=CHECKPOINT_SHA),
        operation_artifacts=artifacts,inputs_sha256=inputs,
        training_selection_disclosure=report['selection_disclosure'],
        existing_dynamics_reused_without_executor_only_retraining=True,
        executor_acceptance_granted=False,formal_VAL_TEST_started=False,TEST_read=False,
        Value_NGFF_FASTER_opened_or_started=False))
    print(json.dumps(dict(training_complete=True,TRAIN_original_parents=1600,tasks=16,
        completed_epochs=proof['completed_epochs'],executor_acceptance_granted=False)),flush=True)


if __name__=='__main__':main()
