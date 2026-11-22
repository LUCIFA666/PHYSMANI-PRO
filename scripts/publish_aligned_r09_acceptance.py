"""Publish a separate human delivery after all frozen R09 evaluations finish.

This entry point reads the continuation's final report and preserved evidence.
It does not rerun evaluation, change frozen inputs, or rewrite the old summary.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.benchmark_protocol import file_sha256, verify_lock, wilson

BASE = ROOT / 'artifacts/framework/aligned_r09'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
VARIANTS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')
LABELS = {'cv_geometry': 'CV + geometry', 'cv_value': 'CV + success',
          'context_geometry': 'Context + geometry', 'context_value': 'Context + success',
          'physmani': 'PhysMani (aligned V6)'}
CASE_ISOLATION = 'fresh_simulator_process_per_episode'
DISCLOSURE = ('VAL screen failed; stopping rule changed after VAL; fixed candidates; '
              'not benefit validation')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def inside_artifacts(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'invalid evidence path')
    path = (ROOT / name).resolve()
    require(path.is_relative_to(BASE.resolve()), 'evidence must remain within R09 artifacts')
    return path


def check_hashes(files):
    for name, digest in files.items():
        require(file_sha256(ROOT / name) == digest, 'evidence changed: ' + name)


def parallel_evidence(report, extra):
    declared = report.get('parallel_evaluation_inputs')
    if declared is None:
        require(not report.get('parallel_execution') and
                not report.get('parallel_evaluation_inputs_sha256'), 'parallel metadata lacks its lock')
        return None, None, None
    path = inside_artifacts(declared)
    require(path == (BASE / 'parallel_evaluation_inputs.json').resolve(), 'unexpected parallel lock path')
    require(report.get('parallel_evaluation_inputs_sha256') == file_sha256(path), 'parallel lock SHA differs')
    lock = read(path)
    verify_lock(ROOT, lock)
    require(all(lock['files'].get(name) == digest for name, digest in extra['files'].items()),
            'parallel lock must preserve every previous frozen input digest')
    execution = report.get('parallel_execution', {})
    require(lock.get('parallel_execution') == execution and
            lock.get('development_performance_screen_passed') is False and
            lock.get('stopping_rule_changed_after_VAL') is True and
            lock.get('fixed_model_comparison_not_benefit_validation') is True,
            'parallel lock scheduling/disclosure differs from final report')
    require(execution.get('version') == 'r09_per_case_parallel_queues_v1' and
            execution.get('cpu_workers') == 4 and execution.get('gpu_workers') == 4 and
            execution.get('baseline_validation_gpu_workers') == 4 and
            execution.get('scheduling_unit') == 'one manifest case' and
            execution.get('validation_must_complete_before_TEST') is True,
            'parallel scheduling specification differs')
    expected = (
        ('wrapper', 'wrapper_sha256', ROOT / 'scripts/continue_parallel_fixed_evaluation_r09.py'),
        ('transition', 'transition_sha256', BASE / 'scheduling_transition.json'),
        ('previous_fixed_evaluation_inputs', 'previous_fixed_evaluation_inputs_sha256',
         BASE / 'fixed_evaluation_extra_inputs.json'))
    for field, sha_field, source in expected:
        name = relative(source)
        require(execution.get(field) == name and
                execution.get(sha_field) == file_sha256(source) == lock['files'].get(name),
                'parallel scheduling source/transition binding differs: ' + field)
    transition = read(BASE / 'scheduling_transition.json')
    require(transition.get('models_changed') is False and transition.get('thresholds_changed') is False and
            transition.get('evaluators_or_physics_changed') is False and
            transition.get('existing_results_preserved') is True and
            transition.get('validation_must_complete_before_TEST') is True and
            transition.get('replacement_wrapper') == execution['wrapper'] and
            transition.get('replacement_wrapper_sha256') == execution['wrapper_sha256'],
            'transition must preserve model/evaluator/physics inputs and original case evidence')
    return path, lock, execution


def case_evidence(row, variant, config, locked):
    require(row.get('case_isolation') == CASE_ISOLATION, 'case isolation missing')
    receipt = row['case_process_receipt']
    require(receipt.get('version') == 'fresh_process_r09_v1', 'unexpected process receipt')
    evaluator = ('scripts/evaluate_physmani_aligned_r09_v6.py' if variant == 'physmani'
                 else 'scripts/evaluate_aligned_r09.py')
    require(receipt.get('child_evaluator') == evaluator and
            receipt.get('child_evaluator_sha256') == locked.get(evaluator), 'child source differs')
    child_path = inside_artifacts(receipt['child_record_path'])
    inputs_path = inside_artifacts(receipt['child_inputs_path'])
    require(file_sha256(child_path) == receipt['child_record_sha256'], 'child record changed')
    require(file_sha256(inputs_path) == receipt['child_inputs_sha256'], 'child inputs changed')
    child, inputs = read(child_path), read(inputs_path)
    require(all(key in row and row[key] == value for key, value in child.items()),
            'original child fields were changed')
    require(type(child.get('evaluation_process_pid')) is int and child['evaluation_process_pid'] > 0,
            'native child process evidence missing')
    identity = hashlib.sha256(json.dumps({k: v for k, v in inputs.items() if k != 'sha256'},
                                        sort_keys=True).encode()).hexdigest()
    require(inputs.get('sha256') == identity, 'declared child input identity differs')
    require(inputs.get('config') == config and inputs.get('raw_episode_id') == row['raw_episode_id'],
            'child configuration or raw episode differs')
    require(child.get('raw_episode_filter') == row['raw_episode_id'], 'child raw filter missing')
    require(inputs.get('files', {}).get(evaluator) == locked.get(evaluator), 'child input source differs')
    for name, digest in inputs.get('files', {}).items():
        if name in locked:
            require(digest == locked[name], 'child input differs from frozen source: ' + name)
    if variant != 'physmani':
        require(row.get('inputs_sha256') == identity and inputs.get('split') == row['split'],
                'child record/input binding differs')


def collect_cases(split, variants, expected, config, baseline_config, locked):
    result, hashes = {}, {}
    for variant in variants:
        directory = BASE / (split + '_physmani_v6' if variant == 'physmani' else split + '_' + variant)
        counts = {task: dict(episodes=0, successes=0) for task in TASKS}
        seen = set()
        for path in sorted(directory.glob('pick*__episode*.json')):
            if path.name.endswith('infrastructure_error.json'):
                continue
            row = read(path)
            key = (row['task'], row['raw_episode_id'])
            require(key in expected and key not in seen, 'unexpected/duplicate case: ' + str(path))
            require(row.get('split') == split and row.get('formal_test_used') is (split == 'test'),
                    'case split differs')
            require(type(row.get('official_task_success')) is bool, 'native success label missing')
            require(not row.get('one_decision_diagnostic', False), 'partial diagnostic is not evaluation')
            case_evidence(row, variant, baseline_config if variant == 'physmani' else config, locked)
            seen.add(key)
            counts[row['task']]['episodes'] += 1
            counts[row['task']]['successes'] += int(row['official_task_success'])
            hashes[relative(path)] = file_sha256(path)
        require(seen == expected, 'incomplete ' + split + '/' + variant)
        result[variant] = counts
    return result, hashes


def check_usage(usage, split, counts):
    require(usage.get('split') == split and usage.get('complete') is True and
            usage.get('coverage_complete') is True and usage.get('diagnostics_valid') is True and
            not usage.get('errors'), 'incomplete/invalid ' + split + ' selection diagnostics')
    per_group = sum(r['episodes'] for r in counts['cv_value'].values())
    overrides = {}
    for predictor in ('cv', 'context'):
        for ranking in ('geometry', 'value'):
            variant = predictor + '_' + ranking
            group = usage['groups'][split + '_' + variant]
            require(group['coverage']['complete'] is True and
                    group['coverage']['completed_episodes'] == per_group and
                    group['coverage']['expected_episodes'] == per_group and
                    group['matching_complete'] is True and not group['input_errors'],
                    'selection group incomplete: ' + variant)
            for task in TASKS:
                data = group['per_task'][task]
                require(data['complete_coverage'] is True and
                        data['episodes'] == counts[variant][task]['episodes'] and
                        data['native_successes'] == counts[variant][task]['successes'] and
                        data['matching_issue_count'] == 0 and
                        data['matched_selection_events'] == data['candidate_attempts'],
                        'selection counts/trace differ: ' + variant + '/' + task)
                if ranking == 'value':
                    require(data['unmatched_attempts'] == 0 and data['unknown_override_status_selections'] == 0,
                            'unknown override status')
                    events = data['selection_events']
                    actual = sum(e.get('actual_override') is True for e in events)
                    episodes = len({e['episode'] for e in events if e.get('actual_override') is True})
                    require(actual == data['actual_override_selections'] and
                            episodes == data['episodes_with_actual_override'], 'override trace/count mismatch')
                    overrides.setdefault(predictor, {})[task] = dict(
                        actual_override_selections=actual, episodes_with_actual_override=episodes,
                        candidate_attempts=data['candidate_attempts'],
                        matched_selection_events=data['matched_selection_events'])
    provenance = usage['provenance']
    for field in ('analysis_and_current_code_sha256', 'model_files_sha256', 'input_episode_files_sha256',
                  'per_case_inputs_files_sha256', 'preserved_child_episode_files_sha256'):
        require(isinstance(provenance.get(field), dict) and provenance[field], 'missing usage provenance: ' + field)
        check_hashes(provenance[field])
    return overrides


def decode_video(path):
    import imageio_ffmpeg
    require(path.is_file() and path.stat().st_size > 0, 'video missing: ' + str(path))
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    decoded = 0
    try:
        metadata = next(frames)
        width, height = metadata['size']
        fps, duration = float(metadata['fps']), float(metadata['duration'])
        require(width > 0 and height > 0 and math.isfinite(fps) and fps > 0 and
                math.isfinite(duration) and duration > 0, 'invalid video metadata: ' + str(path))
        for frame in frames:
            require(len(frame) == width * height * 3, 'incomplete RGB frame: ' + str(path))
            decoded += 1
    finally:
        frames.close()
    require(decoded > 0, 'no decoded video frames: ' + str(path))
    return dict(decoded_frame_count=decoded, size=[width, height], fps=fps,
                duration_seconds=duration, decoded_duration_seconds_at_reported_fps=decoded / fps,
                duration_note='duration is container metadata; frame count is from complete RGB24 decoding',
                ffmpeg_executable=str(executable), ffmpeg_executable_sha256=file_sha256(executable),
                ffmpeg_version=metadata.get('ffmpeg_version'),
                validation='all frames decoded to RGB24, each byte count checked; generator closed')


def videos(manifest):
    result = []
    for task in TASKS:
        row = next(r for r in manifest if r['task'] == task)
        path = BASE / 'test_context_value' / (task + '__' + row['raw_episode_id'] + '.mp4')
        decoded = decode_video(path)
        result.append(dict(task=task, raw_episode_id=row['raw_episode_id'], method='context_value',
                           selection='fixed first manifest case; not selected by outcome',
                           path=relative(path), sha256=file_sha256(path), **decoded))
    return result


def plot(counts, path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for axis, task, title in zip(axes, TASKS, ('Normal speed', 'High speed')):
        cells = [counts[name][task] for name in VARIANTS]
        rates = [cell['successes'] / cell['episodes'] for cell in cells]
        intervals = [wilson(cell['successes'], cell['episodes']) for cell in cells]
        errors = np.maximum(np.array([[v - lo for v, (lo, hi) in zip(rates, intervals)],
                                      [hi - v for v, (lo, hi) in zip(rates, intervals)]]), 0.)
        axis.bar(np.arange(5), rates, color=['#6386a6', '#406e96', '#61a18a', '#267f67', '#ba8553'],
                 yerr=errors, capsize=4)
        axis.set_xticks(np.arange(5), ['CV\ngeometry', 'CV\nsuccess', 'Context\ngeometry',
                                      'Context\nsuccess', 'PhysMani\naligned V6'])
        axis.set_title(title)
        axis.set_ylim(0, 1.16)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.15)
        for index, (value, cell) in enumerate(zip(rates, cells)):
            axis.text(index, value + .045, f"{cell['successes']}/{cell['episodes']}", ha='center')
    axes[0].set_ylabel('Native task success rate (Wilson 95% interval)')
    figure.suptitle('R09 fixed-candidate comparison under the shared simulator protocol')
    figure.text(.5, .035,
        'VAL screen failed (Context normal: geometry 20/20, success 18/20).\n'
        'Stopping rule changed after VAL; fixed candidates; not benefit validation.\n'
        'Adapted PhysMani protocol; previously exposed public TEST; 40 episodes/task.',
        ha='center', va='bottom', fontsize=10, color='#8b2531')
    figure.tight_layout(rect=(0, .17, 1, .94))
    temporary = path.with_name(path.stem + '.tmp.png')
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(path)


def main():
    report_path = BASE / 'final_report.json'
    release_path = BASE / 'fixed_candidate_evaluation_release.json'
    gate_path = BASE / 'success_acceptance.json'
    report, release, gate = read(report_path), read(release_path), read(gate_path)
    require(report.get('complete') is True and report.get('protocol_acceptance_passed') is True and
            report.get('total_test_runs') == 400 and report.get('expected_test_runs') == 400 and
            not report.get('acceptance_errors'), 'all 400 TEST cases must pass the physical/input protocol first')
    require(gate.get('passed') is False and gate.get('formal_test_used') is False and
            gate.get('failures') == ['context pick_moving_target_on_the_table selection VAL regression'],
            'expected original failed performance screen is missing')
    require(release.get('original_performance_screen_sha256') == file_sha256(gate_path) and
            release.get('original_performance_failures') == gate['failures'] and
            release.get('original_VAL_results') == gate['results'], 'original failed screen binding differs')
    require(release.get('evaluation_allowed') is True and
            release.get('deployment_or_benefit_validated') is False and
            release.get('development_performance_screen_passed') is False and
            release.get('model_or_threshold_retuned_after_VAL') is False and
            release.get('TEST_success_labels_used_for_release') is False, 'fixed-candidate release flags differ')
    require(report.get('development_performance_screen_passed') is False and
            report.get('fixed_candidate_evaluation_release') == release,
            'run continuation to preserve the failed-screen disclosure in the final report first')
    locked = {}
    lock_paths = [BASE / 'frozen_inputs.json', BASE / 'comparison_v6_inputs.json',
                  BASE / 'fixed_evaluation_extra_inputs.json']
    for path in lock_paths:
        lock = read(path)
        verify_lock(ROOT, lock)
        for name, digest in lock['files'].items():
            require(name not in locked or locked[name] == digest, 'conflicting freeze identity: ' + name)
            locked[name] = digest
    extra = read(lock_paths[-1])
    require(extra.get('development_performance_screen_passed') is False and
            extra.get('stopping_rule_changed_after_VAL') is True and
            extra.get('fixed_model_comparison_not_benefit_validation') is True, 'extra freeze lacks disclosure')
    parallel_path, parallel_lock, parallel_execution = parallel_evidence(report, extra)
    if parallel_path is not None:
        lock_paths.append(parallel_path)
        for name, digest in parallel_lock['files'].items():
            require(name not in locked or locked[name] == digest, 'conflicting parallel freeze identity: ' + name)
            locked[name] = digest
    for path in (release_path, gate_path, BASE / '固定候选评估说明.txt',
                 ROOT / 'scripts/continue_fixed_evaluation_r09.py'):
        require(locked.get(relative(path)) == file_sha256(path), 'release/continuation evidence is not frozen')
    config = read(ROOT / 'config/framework/aligned_protocol_r09.json')
    baseline_config = read(ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json')
    train = {}
    for predictor in ('cv', 'context'):
        info = report['frozen_success_predictors'][predictor]
        path = ROOT / config[predictor + '_local_value']
        require(locked.get(relative(path)) == file_sha256(path) == info['checkpoint_sha256'], 'success model changed')
        require(locked.get(relative(path.with_name('report.json'))) == info['report_sha256'] and
                locked.get(relative(path.with_name('value_support.json'))) == info['support_sha256'],
                'success report/support differs from final summary')
        require(info.get('classifier_retrained_for_r09') is False and
                info.get('calibration_refitted_for_r09') is False and
                info['metadata'].get('all_200_TRAIN_source_episodes_executed') is True, 'TRAIN/frozen lineage differs')
        train[predictor] = {}
        for task in TASKS:
            coverage = info['source_coverage'][task]
            require(coverage['TRAIN_executed_sources'] == 100 and
                    coverage['all_observed_TRAIN_sources_have_positive_ensemble_weights'] is True,
                    'incomplete TRAIN source execution/support')
            train[predictor][task] = dict(executed_sources=100, expected_sources=100)
    checkpoint = Path(baseline_config['baseline_checkpoint'])
    require(locked.get(os.path.relpath(checkpoint, ROOT)) ==
            file_sha256(checkpoint) == report['baseline_checkpoint_sha256'], 'released baseline checkpoint changed')
    test_manifest_path = ROOT / 'artifacts/framework/pipeline_v1/official_test_manifest.json'
    test_manifest = [r for r in read(test_manifest_path)['episodes'] if r['task'] in TASKS]
    validation_manifest = [r for r in read(ROOT / 'artifacts/framework/protocol/validation_manifest.json')['episodes'] if r['task'] in TASKS]
    test_expected = {(r['task'], r['raw_episode_id']) for r in test_manifest}
    validation_expected = {(r['task'], r['raw_episode_id']) for r in validation_manifest}
    require(len(test_manifest) == len(test_expected) == 80 and
            len(validation_manifest) == len(validation_expected) == 40, 'manifest coverage differs')
    test_counts, test_hashes = collect_cases('test', VARIANTS, test_expected, config, baseline_config, locked)
    validation_counts, validation_hashes = collect_cases('validation', VARIANTS, validation_expected,
                                                        config, baseline_config, locked)
    for variant in VARIANTS:
        for task in TASKS:
            cell, saved = test_counts[variant][task], report['variants'][variant][task]
            require(cell['episodes'] == 40 and cell['episodes'] == saved['episodes'] and
                    cell['successes'] == saved['successes'] and
                    abs(saved['success_rate'] - cell['successes'] / 40) < 1e-12, 'final success counts differ')
            require(validation_counts[variant][task]['episodes'] == 20, 'VAL coverage differs')
    usage_path = BASE / 'success_selection_usage_test.json'
    usage = read(usage_path)
    require(usage['provenance']['manifest_sha256'] == file_sha256(test_manifest_path), 'TEST usage manifest changed')
    test_overrides = check_usage(usage, 'test', test_counts)
    validation_usage_path = BASE / 'success_selection_usage_validation.json'
    validation_overrides = check_usage(read(validation_usage_path), 'validation', validation_counts)
    for predictor in ('cv', 'context'):
        for task in TASKS:
            screen = gate['results'][predictor][task]
            require(screen['geometry_successes'] == validation_counts[predictor + '_geometry'][task]['successes'] and
                    screen['value_successes'] == validation_counts[predictor + '_value'][task]['successes'] and
                    screen['episodes'] == 20 and screen['actual_learned_overrides'] ==
                    validation_overrides[predictor][task]['actual_override_selections'],
                    'original VAL screening numbers differ from preserved cases')
    require(validation_counts['context_geometry'][TASKS[0]]['successes'] == 20 and
            validation_counts['context_value'][TASKS[0]]['successes'] == 18 and
            validation_overrides['context'][TASKS[0]]['actual_override_selections'] == 4 and
            validation_overrides['context'][TASKS[0]]['episodes_with_actual_override'] == 2,
            'documented R09 failed-screen example differs')
    video_info = videos(test_manifest)
    limits = [
        '本轮为共同协议下的本地发布权重对比，非论文Table 1复现；论文数字仅作原协议参考。',
        '公开TEST在R04/R05已曝光；不能称从未见过的新TEST，或独立于此前开发的盲测。',
        '两任务使用已知红色方块、抓取姿态、区域与目标标定先验；基线为发布的多任务策略，不能外推多物体通用优势。',
        '双方允许当前仿真实例mask和静态地图；非纯RGB-D无仿真辅助，也不使用真实未来轨迹。',
        '相同原生机器人参数、作者OPENGL传感器与10秒/.05秒物理预算；正式TEST每例独立官方reset，VAL采用HDF5 source/home reset而非官方target_state replay。统一计时与增量夹爪改变了原作者执行协议。',
        '条件验收仅表示已记录的重置和外部条件在预设容差内匹配；正式TEST使用相同官方运动参数文件和初始化程序，不证明原生全状态相同或位级完全一致。',
        'VAL public_initial_conditions只核对robot/map/cameras/sensor，未直接核验target初始位置、朝向、线角速度、加速度或task phase。TEST额外核对官方source/variation/index/target_state.pkl SHA及target实际初始位置等，但未记录actual target orientation/linear-angular velocity/active schedule/t/t0/step_id；未观测项及原生微小随机性保留为归因限制。',
        '我们每50ms控制、100ms重规划；基线每10个counted tick作策略决策，其间更新WM。各方法控制频率与算法不同。',
        '我们裁剪参考关节位置；V6基线保留作者direct RML目标与原抽点规则，参考处理属于各自算法；原生标称速度可能有数值/瞬态超出。',
        '同步仿真在模型计算期间暂停物理时间；并行运行墙钟耗时不能证明实时硬件性能。',
        'R09使用冻结R08预测头、嵌套CAL校准及TRAIN支持规则；本轮不重训或重校准。R06 GL3与R07 OPENGL训练域历史保留于冻结来源。',
        '未使用TEST成功标签或未来信息进行离线训练、校准、选模型或调阈值；PhysMani仍按作者算法用当前/过去观测在线拟合world model，属于合法在线适配。',
        '成功率模块预测候选当前尝试在替换前的成功，而非跨重试最终成功。概率质量、实际override和任务收益分别报告。',
        'zero override仅说明所访问状态的选择未改变，不能将独立原生轨迹差异归为模块收益；任务成功点估计也不能识别因果贡献。',
        '单固定checkpoint、每任务40回合；Wilson区间不是多seed方差，配对检验为探索性且未做多重比较校正。',
        '本轮未使用FASTER或NGFF；Context版本为轻量context与SimNorm动力学。',
    ]
    lines = ['R09 固定候选评估验收说明',
        '400/400正式TEST完成，已记录的重置/外部条件及计时检查通过；五组完整VAL为200/200。',
        '开发性能筛选未通过：Context普通速度 geometry 20/20，success 18/20，实际override 4次（2个episode）。',
        '原门槛要求value至少geometry-1/20；观察VAL后依据用户指令改变停止规则，继续评估固定候选。',
        '原failed筛选、模型、阈值与负面结果原样保留；不能称性能门槛通过、部署放行或成功率模块收益已验证。', '',
        '方法 | 普通速度成功数/回合 | 高速成功数/回合']
    for variant in VARIANTS:
        cells = [test_counts[variant][task] for task in TASKS]
        lines.append(LABELS[variant] + ' | ' + ' | '.join(
            f"{c['successes']}/{c['episodes']} ({100*c['successes']/c['episodes']:.1f}%)" for c in cells))
    lines += ['', 'TRAIN覆盖：每个预测器两任务各100/100源，共200源已执行；不把重复采集或bootstrap副本当独立源。',
              'VAL覆盖：每方法两任务各20回合，共五方法200回合；其中四组ours为160回合。',
              'TEST覆盖：每方法两任务各40回合，共五方法400回合。']
    if parallel_execution is not None:
        lines += ['调度变更：4 CPU + 4 GPU worker按单个manifest case排队；基线VAL GPU worker上限4。',
                  '新增并行锁完整保留旧extra锁的每个输入digest，只改变队列调度，不改变冻结模型或每例进程隔离。',
                  '已完成合法case沿用原过程证据，不重标为新worker执行；并行墙钟耗时不代表实时性能。']
    lines += ['', '实际override（改变同次选择中的几何最优候选）：']
    for split, groups in (('VAL', validation_overrides), ('TEST', test_overrides)):
        for predictor in ('cv', 'context'):
            for task, task_label in zip(TASKS, ('普通速度', '高速')):
                data = groups[predictor][task]
                lines.append(f"{split} {predictor} {task_label}：{data['actual_override_selections']}/{data['candidate_attempts']}次候选选择；"
                             f"涉及{data['episodes_with_actual_override']}个episode。")
    lines += ['', '已知条件与限制：', *['- ' + item for item in limits], '', '固定首case视频（未按结果挑选）：']
    lines += [str(ROOT / item['path']) for item in video_info]
    figure_path = BASE / 'success_rates_with_disclosure.png'
    lines += ['', '带披露结果图：' + str(figure_path), '机器验收交付：' + str(BASE / 'acceptance_delivery.json')]
    plot(test_counts, figure_path)
    text_path = ROOT / '验收说明.txt'
    temporary = text_path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(text_path)
    delivery = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=True,
        protocol_acceptance_passed=True, evaluation_completed=True, development_performance_screen_passed=False,
        deployment_or_benefit_validated=False, stopping_rule_changed_after_VAL=True,
        disclosure=DISCLOSURE, fixed_candidate_release=release,
        coverage=dict(TRAIN=train, VAL=validation_counts, TEST=test_counts, total_VAL_cases=200, total_TEST_cases=400),
        actual_override_statistics=dict(VAL=validation_overrides, TEST=test_overrides),
        videos=video_info, limitations=limits,
        scheduling=dict(parallel_execution=parallel_execution,
            parallel_lock=relative(parallel_path) if parallel_path is not None else None,
            parallel_lock_sha256=file_sha256(parallel_path) if parallel_path is not None else None,
            previous_extra_digests_preserved=True if parallel_path is not None else None,
            transition_changes_queue_scheduling_only=True if parallel_path is not None else None,
            transition_document=read(BASE / 'scheduling_transition.json') if parallel_path is not None else None,
            case_isolation=CASE_ISOLATION,
            completed_case_reuse='accepted original child/process evidence retained; not relabeled as a new worker run'),
        initial_condition_matching=dict(
            claim='recorded reset/external conditions match within preset tolerances; not complete native-state equivalence',
            VAL_reset='HDF5 source/home reset; not official target_state replay',
            VAL_compared=['robot', 'static map', 'cameras', 'sensor protocol'],
            VAL_target_initial_pose_velocity_acceleration_phase_directly_checked=False,
            TEST_reset='official reset in a fresh simulator for every episode',
            TEST_additional_checks=['official source', 'variation', 'index', 'target_state.pkl SHA256',
                                    'actual target initial position', 'recorded fixture/reset fields'],
            TEST_unrecorded=['actual target orientation', 'linear/angular velocity', 'active schedule',
                             't', 't0', 'step_id'],
            full_native_state_or_bitwise_equality_claimed=False),
        input_sha256={relative(p): file_sha256(p) for p in
                      [report_path, release_path, gate_path, usage_path, validation_usage_path, *lock_paths,
                       ROOT / 'config/framework/aligned_protocol_r09.json',
                       ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json', Path(__file__)]},
        evaluated_parent_record_sha256=dict(VAL=validation_hashes, TEST=test_hashes),
        output_sha256={relative(p): file_sha256(p) for p in (text_path, figure_path)})
    destination = BASE / 'acceptance_delivery.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(delivery, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(complete=True, text=relative(text_path), figure=relative(figure_path),
                          delivery=relative(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
