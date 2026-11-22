"""Independent acceptance of the requested fresh post-repair joint16 training."""
from datetime import datetime,timezone
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.context_se3 import MultiBodyContextDynamics
from our_mani.multitask.joint16_checkpoint_v172 import check_training_metadata_and_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base=ROOT/'artifacts/framework/multitask_r10';out=base/'joint16_after_executor_training_acceptance_v672/report.json'
    if out.exists():raise FileExistsError('preserve the independent training acceptance')
    runtime=ROOT/'runtime/framework/joint16_after_executor_training_v671'
    terminal=json.loads((runtime/'terminal.json').read_text());launch=json.loads((runtime/'launch.json').read_text())
    if terminal.get('complete') is not True or launch.get('previous_v157_checkpoint_reused_as_final_training') is not False:
        raise ValueError('actual fresh completed post-repair training required')
    freeze_pointer=launch['executor_freeze'];freeze_path=ROOT/freeze_pointer['path']
    if file_sha(freeze_path)!=freeze_pointer['sha256']:raise ValueError('executor freeze changed after training started')
    freeze=json.loads(freeze_path.read_text());verify_inputs(ROOT,freeze['inputs_sha256'])
    checkpoint_pointer=terminal['checkpoint'];checkpoint=ROOT/checkpoint_pointer['path']
    model=MultiBodyContextDynamics(checkpoint)
    if model.checkpoint_sha256!=checkpoint_pointer['sha256']:raise ValueError('actual new checkpoint differs from completed training')
    metadata=model.metadata;source_path=checkpoint.parent/metadata['training_source_manifest']['path_relative_to_checkpoint']
    if file_sha(source_path)!=metadata['training_source_manifest']['sha256']:raise ValueError('actual new TRAIN source roster changed')
    source=json.loads(source_path.read_text());roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json'
    proof=check_training_metadata_and_roster(metadata,source['entries'],json.loads(roster_path.read_text())['episodes'])
    if proof['tasks']!=list(TASKS):raise ValueError('full16 training required')
    report_path=checkpoint.parent/'report.json';report=json.loads(report_path.read_text())
    if report['checkpoint_sha256']!=model.checkpoint_sha256 or report['metadata']!=metadata:raise ValueError('saved new training report differs from actual model')
    artifacts=[];inputs={str(p):file_sha(p) for p in [checkpoint,source_path,roster_path,report_path,freeze_path,runtime/'terminal.json',runtime/'launch.json',Path(__file__)]}
    for task in TASKS:
        current=('cup_input_repair_v712_setup' if task.startswith('place_cups_on_rotating_frame') else
            'remove_time_scene_v706_setup' if task.startswith('remove_cups_from_rotating_frame') else 'rml_suite16_smoke_v677_setup')
        path=base/current/f'{task}_ours_configuration.json';c=json.loads(path.read_text())
        inputs[str(path)]=file_sha(path);learned={}
        if task not in BLOCKS:
            for name in ('geometry','interaction','rigid_layout','colour'):
                pointer=c['learned_artifacts'][task].get(name)
                if pointer is None:continue
                material=ROOT/pointer['path']
                if file_sha(material)!=pointer['sha256']:raise ValueError('original all-TRAIN operation artifact changed')
                learned[name]=pointer;inputs[str(material)]=pointer['sha256']
        artifacts.append(dict(task=task,context_checkpoint=checkpoint_pointer,
            existing_TRAIN_operation_artifacts=learned,block_uses_original_geometry_kernel=task in BLOCKS))
    atomic_json(out,dict(schema='r10_joint16_completed_training_independent_acceptance_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),training_complete=True,training=proof,checkpoint=checkpoint_pointer,
        operation_artifacts=artifacts,inputs_sha256=inputs,training_selection_disclosure=report['selection_disclosure'],
        executor_freeze=freeze_pointer,fresh_training_after_executor_repairs=True,
        existing_dynamics_reused_without_executor_only_retraining=False,
        existing_TRAIN_geometry_and_operation_models_reused=True,executor_acceptance_granted=False,
        formal_VAL_TEST_started=False,TEST_read=False,Value_NGFF_FASTER_opened_or_started=False))
    print(json.dumps(dict(training_complete=True,TRAIN_original_parents=1600,tasks=16,completed_epochs=30,
        new_checkpoint=checkpoint_pointer,final_fixed_four_readiness_required=True)),flush=True)


if __name__=='__main__':main()
