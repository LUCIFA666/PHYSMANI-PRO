"""Select TRAIN-only cached parents and fixed new generated evaluation seeds."""
from collections import Counter
import hashlib

from .common import ROOT, OUT, PLAN, HISTORICAL_STOP, read, write, pointer, stopped
from ..multitask_execution_pipeline_20261104_v1.registry import TASKS
from ..multitask.specs import get_task_spec
from ..trained_world_reference_live_20261104_v1.runtime import E2, E2_SHA256, WORLD, WORLD_SHA256


def prepare():
    stopped()
    if PLAN.exists():
        raise FileExistsError('Prospective plan already fixed')
    cache = ROOT / 'artifacts/framework/local_mpc_20261102/world_command_v3_deferred_run_v1/artifacts/public_cache/cache_manifest.json'
    manifest = ROOT / 'artifacts/framework/local_mpc_20261102/current_network16_data_manifest_v1/manifest.json'
    public = read(cache)
    inventory = read(manifest)
    train = {r['source_parent_group']: r for r in inventory['episodes']
             if r['split'] == 'train' and r['offline_eligible']}
    tasks = [t for t in TASKS if not t.startswith('pick_moving_target_on_the_table')]
    selected, cases = [], []
    for ordinal, task in enumerate(tasks):
        chunks = [(i, c) for i, c in enumerate(public['chunks'])
                  if c['role'] == 'fit' and c['source_parent_group'] in train
                  and train[c['source_parent_group']]['task'] == task]
        chunks.sort(key=lambda x: hashlib.sha256(('20261112:' + x[1]['source_parent_group']).encode()).hexdigest())
        if len(chunks) < 4:
            raise ValueError('Four actual TRAIN cached parents required: ' + task)
        for i, (index, chunk) in enumerate(chunks[:4]):
            selected.append(dict(role='fit' if i < 3 else 'selection', task=task, chunk_index=index,
                                 source_parent_group=chunk['source_parent_group'], public_chunk=chunk['file'],
                                 source_split='train', source=train[chunk['source_parent_group']],
                                 original_pretrained_world_fit_exposed=True))
        variations = get_task_spec(task).supported_variations
        for role, count, offset in (('fit', 3, 0), ('selection', 1, 100), ('validation', 1, 200), ('test', 1, 300)):
            for replicate in range(count):
                case = dict(id=f'{ordinal:02d}_{task}_{role}_{replicate}', task=task, role=role,
                            seed=2611130000 + ordinal * 1000 + offset + replicate,
                            variation=variations[replicate % len(variations)],
                            calibration_episode=train[chunks[0][1]['source_parent_group']]['source_path'],
                            task_time_limit_seconds=60., budget_ticks=1200,
                            generated_reset=True, archived_official_source=False)
                case['parent_id'] = 'new_native_seed:' + str(case['seed'])
                cases.append(case)
    assert len({c['seed'] for c in cases}) == len(cases) == 84
    plan = dict(schema='fourteen_task_small_executor_cycle_20261112_v1',
                user_authorization='修好其他几个任务执行器，像方块一样做小型TRAIN VAL TEST；按其余14版本推进',
                tasks=tasks, historical_global_stop=HISTORICAL_STOP,
                cached_TRAIN_parent_counts=dict(Counter(x['role'] for x in selected)),
                cached_TRAIN_windows_per_parent=16, cached_TRAIN=selected, cached_public_manifest=pointer(cache),
                inventory_manifest=pointer(manifest), cases=cases,
                Native_case_counts=dict(Counter(c['role'] for c in cases)),
                original_encoder=dict(path=str(E2), sha256=E2_SHA256),
                original_world=dict(path=str(WORLD), sha256=WORLD_SHA256),
                training=dict(epochs=150, steps_per_epoch=8, windows_per_parent_per_step=2,
                              learning_rate=3e-5, weight_decay=.01, seed=20261112,
                              fitted_components=['position_head', 'rotation_head', 'opening_head', 'demo_likelihood_head'],
                              frozen_visual_encoder_and_world_backbone=True),
                policy_family='existing_public_RGBD_world_command_model; not Cube_GRU_candidate_model',
                language_branch_used=False, Native_success_supervision_used_for_BC=False,
                training_targets='recorded_expert_future_command_intent_not_actual_contact',
                source_clock='recorded_frame_order; physical_dt_unverified',
                fresh_native_generated_VAL_TEST=True, official_archived_VAL_TEST=False,
                base_model_previously_selected_on_old_validation=True,
                new_VAL_TEST_resets_not_used_for_training_or_selection=True,
                Native_at_most=1, GPU_fitting_Native_overlap=False,
                success_authority='original_task.success_first_true_at_actual50ms',
                historical_cube_baselines_untouched=True)
    write(PLAN, plan)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'prospective_plan.json', dict(plan=pointer(PLAN), native_cases=84,
                                             cached_TRAIN_parents=56, raw_TEST_payloads_read=0))
    return plan


if __name__ == '__main__':
    print(prepare()['Native_case_counts'])
