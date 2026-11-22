"""Create a fresh sixteen-parent TRAIN teacher plan/config set, CPU only."""
from copy import deepcopy
import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.local_mpc_20261102.records import freeze_sources


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def specifications():
    z, d = .775, 1 / sqrt(2)
    return [
        ('static', [.17, -.23, z], [0., 0., 0.], None, None, 'fixed static offset A'),
        ('uniform', [.27, -.28, z], [0., .02, 0.], None, None, 'positiveY low speed'),
        ('uniform', [.21, .28, z], [0., -.04, 0.], None, None, 'negativeY normal speed'),
        ('speed_change', [.24, -.30, z], [0., .02, 0.], 24, [0., .08, 0.], 'positiveY acceleration at1.2s'),
        ('static', [.29, .20, z], [0., 0., 0.], None, None, 'fixed static offset B'),
        ('uniform', [.10, -.16, z], [.04, 0., 0.], None, None, 'positiveX normal speed'),
        ('uniform', [.30, .16, z], [-.08, 0., 0.], None, None, 'negativeX faster motion; finite table boundary'),
        ('uniform', [.18, -.32, z], [0., .08, 0.], None, None, 'positiveY faster motion'),
        ('uniform', [.28, .32, z], [0., -.12, 0.], None, None, 'negativeY high speed'),
        ('uniform', [.075, -.08, z], [.12, 0., 0.], None, None, 'positiveX high speed; unheld straight path reachesXedge near2.29s'),
        ('uniform', [.25, -.32, z], [0., .12, 0.], None, None, 'positiveY high speed'),
        ('uniform', [.29, .24, z], [-.02*d, -.02*d, 0.], None, None, 'negativeXY diagonal with speednorm.02'),
        ('uniform', [.11, -.24, z], [.04*d, .04*d, 0.], None, None, 'positiveXY diagonal with speednorm.04'),
        ('speed_change', [.22, .32, z], [0., -.04, 0.], 24, [0., -.12, 0.], 'negativeY acceleration at1.2s'),
        ('speed_change', [.18, -.20, z], [.02, 0., 0.], 28, [-.04, 0., 0.], 'one declaredX reversal at1.4s'),
        ('speed_change', [.12, .24, z], [.02*d, -.02*d, 0.], 30, [.08*d, -.08*d, 0.], 'diagonal acceleration at1.5s'),
    ]


def prepare(config_dir):
    directory = Path(config_dir).resolve()
    family = ROOT / 'config/framework'
    if family not in directory.parents or directory.exists():
        raise ValueError('fresh project config/framework directory required')
    base_path = ROOT / 'config/framework/local_mpc_coupled_fixed_regression_20261103_v1/04_train_uniform.json'
    base = json.loads(base_path.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    batch_relative = 'runtime/framework/local_mpc_20261102/P4_teacher_train_20261103_v1'
    evaluations = 'artifacts/framework/local_mpc_20261102/P4_teacher_train_20261103_v1'
    jobs, frozen = [], {}
    for index, (scene, position, velocity, switch, changed, description) in enumerate(specifications()):
        episode_id = f'{index:02d}_train_{scene}'
        seed = 261103000 + index
        parent = f'p4_new_TRAIN_20261103_{index:02d}_seed{seed}'
        config = deepcopy(base)
        config.update(schema='P4_TRAIN_truthB_public50ms_shadow_teacher_20261103_v1', seed=seed,
            information_setting='truth_state_teacher_plus_strictly_separate_public50ms_RGBD_GRU_shadow',
            formal_evaluation=False, dataset_use='train', simulation_seconds=8.,
            initial_target_position=position, uniform_velocity=velocity,
            speed_change_at_tick=24 if switch is None else switch,
            changed_velocity=velocity if changed is None else changed,
            preserve_native_height=True, public_motion_shadow=True, public_shadow_save_frames=True,
            gripper_completion_latch=True, public_grasp_prototype=False, public_pregrasp_control=False,
            public_grasp_active_visual_v4=False, public_grasp_active_visual_v41=False,
            execution_model_P1='TRAIN-only B coupled joint response plus frozen old NN two-finger response; completion-latch contact/finger dynamics not requalified',
            finger_response_requalified_for_runtime_protocol=False,
            completion_response_approximation_used=False,
            Native_labels_used_by_public_shadow=False,
            Transformer_training_started=False,
            automatic_task_latent_classification_claimed=False,
            P4_collection_metadata=dict(parent_id=parent, split='train', episode_id=episode_id,
                variant_id='truthB_completion_latched_public50_shadow_v1',
                parent_assigned_before_episode_generation=True, randomized_initialization=False,
                fixed_XY_scenario_description=description,
                initial_z_is_placeholder_replaced_by_Native_default=True,
                old_NN_finger_response_is_not_requalified=True,
                future_truth_only_for_posthoc_contact_tau_and_outcomes=True,
                tau_definition='first real50ms Nativeheld observation after actual close dispatch; API done is separate'))
        config_path = directory / (episode_id + '.json')
        write_new(config_path, config)
        frozen.update({k:v for k,v in freeze_sources(config_path, 'p1').items() if k != str(config_path.relative_to(ROOT))})
        jobs.append(dict(episode_id=episode_id, parent_id=parent, split='train', scene=scene,
            seed=seed, tranche='pilot' if index < 4 else 'continuation',
            variant_id='truthB_completion_latched_public50_shadow_v1',
            config=str(config_path.relative_to(ROOT)), config_sha256=sha(config_path),
            path=batch_relative + '/' + episode_id, evaluation=evaluations + '/' + episode_id,
            frozen_GRU_checkpoint_sha256='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef',
            independent_new_parent=True, original_parent_variant_of=None,
            supported_scenario=dict(initial_position=position, initial_velocity=velocity,
                single_switch_tick=switch, changed_velocity=changed, no_random_initialization=True),
            Native_boundary_note=description if 'boundary' in description or 'edge' in description else
                'original Native table boundary handling retained; uniform is an initial motion segment, not a no-bounce guarantee'))
    extras = ['scripts/prepare_p4_teacher_train_20261103.py', 'scripts/collect_p4_teacher_train_20261103.py',
        'our_mani/local_mpc_control_20261103/native_task_evaluation.py',
        'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt']
    for relative in extras:
        frozen[relative] = sha(ROOT / relative)
    old_jobs = json.loads((ROOT / 'artifacts/framework/local_mpc_20261102/P4_candidate_training_data_audit_v1/selected_train_jobs.json').read_text())['jobs']
    plan = dict(schema='P4_preassigned_TRAIN_truthB_shadow50_teacher_20261103_v1',
        prepared_only=True, Native_started=False, formal_TEST_read=False, training_started=False,
        single_task='pick_moving_target_on_the_table',
        parent_partition_assigned_before_generation=True, independent_new_TRAIN_parents=16,
        original_eight_parents_not_counted_as_new=[j['parent_id'] for j in old_jobs],
        original_parent_variants_in_this_batch=[],
        batch_output_parent=batch_relative, suggested_owner=batch_relative + '_owner',
        teacher='current truth-state B joint controller; original Native task; explicit grasp cost; no learned value',
        public_input='passive causal shadow only; actualinitial plus every50ms RGBD/robot; frozen v671 GRU/context',
        controller_uses_public_forecast=False, current_finger_NN_completion_protocol_qualified=False,
        candidate_Transformer_used_or_trained=False, automatic_task_classification_claimed=False,
        pilot_gate=dict(pilot_jobs=4, fixed_continuation_jobs=12,
            all_actual_50ms_public_capture_required=True, all_source_Native_receipts_required=True,
            minimum_independent_positive_contact_parents=2, minimum_positive_causal_macro_anchors=1,
            positive_contact_definition='firstNativeheld in one uninterrupted held segment, with originalNativeGoal and .5s/.04m/.02m hold evidence in that same segment; never borrow across drops',
            task_failures_retained=True, no_gate_threshold_tuning_on_new_results=True,
            gate_failed='stop continuation and preserve all pilot records; no silent repeat/resplit'),
        supported_worker_fields=['static/uniform/speed_change', 'fixed initial_target_position XY',
            'arbitrary planar uniform_velocity vector', 'one speed_change_at_tick/changed_velocity',
            'preserve_native_height', 'public_motion_shadow/public_shadow_save_frames'],
        unsupported_not_claimed=['random initialization distributions', 'forced alternative grasp yaw',
            'rotation/angular-velocity scenarios', 'changed object geometry', 'occlusion/camera/latency perturbations',
            'multiple semantic tasks'],
        STOP_contract='original worker checks case-parent STOP each100ms; owner propagates its STOP to the new batch and waits actual launcher/child exit; not an immediate midtick kill',
        Native_serial_contract='existing run_local_mpc_20261102.sh acquires both unchanged shared exclusion locks',
        preparation_base_config=dict(path=str(base_path.relative_to(ROOT)),sha256=sha(base_path)),
        source_sha256=frozen, jobs=jobs)
    write_new(directory / 'plan.json', plan)
    return directory / 'plan.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'config/framework/P4_teacher_train_20261103_v1')
    args = parser.parse_args()
    path = prepare(args.config_dir)
    print(json.dumps(dict(prepared_plan=str(path), parents=16, Native_started=False, training_started=False)))


if __name__ == '__main__':
    main()
