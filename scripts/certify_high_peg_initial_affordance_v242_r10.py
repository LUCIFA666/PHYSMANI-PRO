"""Bind the new highpeg initial library to all original100 native grasp sources."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import high_peg_affordance_v241 as proof
from our_mani.multitask.evaluation import atomic_json,file_sha

B=ROOT/'artifacts/framework/multitask_r10'
OUT=B/'high_peg_initial_affordance_v242'
TASK='insert_onto_rotating_peg_high_speed'


def pointer(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('local proof pointer required')
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    if OUT.exists():raise FileExistsError('preserve earlier proof attempt')
    OUT.mkdir();began=time.perf_counter()
    try:
        old_report=json.loads((B/'ring_full_TRAIN_affordance_v21/report.json').read_text())
        old=next(r for r in old_report['entries'] if r['task']==TASK)
        old_permission=json.loads((ROOT/old['permission']['path']).read_text())
        symmetry=old_permission['finite_symmetry']
        manifest=pointer(B/'high_peg_primary_minimax_v232/TRAIN_source_manifest.json')
        library=pointer(B/'high_peg_primary_minimax_v232/interaction.json')
        permission=proof.build_train_permission(ROOT,symmetry,manifest,
            native_gripper_source=old_permission['native_gripper_source'],
            native_task_source=old_permission['native_task_source'],library_pointer=library,development_only=False)
        path=OUT/(TASK+'__ring_phase_permission.json');atomic_json(path,permission)
        loaded=proof.VerifiedRingAffordance.load(ROOT,pointer(path),allow_development=False)
        assert loaded.permission==permission
        row=dict(task=TASK,permission=pointer(path),status=permission['status'],
            exact_action_orbit_schema=proof.EXACT_ORBIT_SCHEMA,all_permission_fields_replayed_exactly=True,
            actual_TRAIN_grasp_parents=len(permission['actual_TRAIN_grasp_rows']),
            phase_support=permission['phase_support'],policy_skill_authorized=permission['policy_skill_authorized'],
            reproducible_loaded_permission=True,full_orientation_observable=False)
        report=dict(schema='r10_highpeg_initial_sequence_full100_D4_proof_v242',complete=True,
            source_manifest=manifest,source_script=pointer(__file__),entries=[row],
            original_full_scope_permission_not_replaced=old['permission'],
            first_completed_initial_operation_per_source=True,released_recovery_unconditionally_excluded=True,
            original_all100_sources_preserved=True,TRAIN_only=True,VAL_read=False,TEST_read=False,
            original8_parent_gate_unchanged=True,original_native_grasp_and_collision_gates_unchanged=True,
            model_fitted=False,native_or_GPU_launched=False,formal_comparison_ready=False,
            seconds=time.perf_counter()-began)
        atomic_json(OUT/'report.json',report)
        print(json.dumps(dict(complete=True,authorized=row['policy_skill_authorized'],
            support={k:v['completed_parent_count'] for k,v in row['phase_support'].items()},seconds=report['seconds'])),flush=True)
    except BaseException as error:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),native_started=False,GPU_started=False))
        raise


if __name__=='__main__':main()
