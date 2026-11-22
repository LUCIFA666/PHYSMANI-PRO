"""Pure CPU, postfreeze fresh-TRAIN192 audit for an explicit repaired graph.

No candidate is imported, model constructed, simulation launched, formal source
read or graph rewritten. Native False/fatal and failed motion diagnostics remain
measured outcomes. Missing, spoofed or incomplete execution cannot grant readiness.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from . import repaired_policy_graph_v1143 as graphs

SOURCE = 'our_mani/multitask/repaired_readiness_v1144.py'
READINESS_SCHEMA = 'r10_fresh_TRAIN192_readiness_v1144'
WITNESS_SCHEMA = 'r10_actual_repaired_graph_execution_witness_v1144'
COHORT_SCHEMA = 'r10_fresh_TRAIN192_execution_cohort_v1144'
PLAN_SCHEMA = 'r10_fresh_TRAIN192_execution_plan_v1144'
CORE_SCHEMA = 'r10_explicit_author_action_outcome_sampling_v741'
BLOCK_SCHEMA = 'r10_original_no_failure_block_action_sampling_v744'
METHODS, TASKS = graphs.METHODS, graphs.TASKS
BLOCKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
HASHES = ('physical_policy_sha256', 'control_graph_sha256', 'policy_sha256')
PARAMETERS = dict(max_velocity=3., max_acceleration=12., max_jerk=1000.,
                  counted_dt=.05, decision_ticks=10, budget_seconds=10.)
RML_AUTHORIZATION = '三组沿用作者相同 RML 参数，统一报告运动指标、按原任务判成功（推荐）'
RML_LOCAL_SOURCES = ('our_mani/ablations/physmani_executor_v4.py',
    'our_mani/sensor_alignment/physmani_executor_v5.py',
    'our_mani/sensor_alignment/physmani_executor_v6.py', 'our_mani/multitask/physmani_executor_v7.py')
RML_NATIVE_SOURCES = ('third_party/RLBench/rlbench/environment.py',
    'third_party/PyRep/pyrep/robots/arms/arm.py',
    'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',
    'third_party/PyRep/pyrep/backend/sim.py')


class ReadinessError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ReadinessError('repaired readiness: ' + message)


def finite_json(value):
    kind = type(value)
    require(kind in (dict, list, tuple, str, int, float, bool, type(None)), 'plain JSON values required')
    if kind is float:
        require(math.isfinite(value), 'nonfinite JSON value')
    elif kind is dict:
        require(all(type(key) is str for key in value), 'plain JSON keys required')
        for item in value.values():
            finite_json(item)
    elif kind in (list, tuple):
        for item in value:
            finite_json(item)


def digest(value):
    finite_json(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def same(left, right):
    return digest(left) == digest(right)


def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def pointer(value):
    require(type(value) is dict and set(value) in ({'path', 'sha256'}, {'root', 'path', 'sha256'}), 'explicit pointer required')
    require(type(value['path']) is str and type(value['sha256']) is str
        and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None, 'actual SHA256 pointer required')
    return dict(root=value.get('root', 'project'), path=value['path'], sha256=value['sha256'])


def read_json(root, ref):
    try:
        value, normalized = graphs.read_project_json(root, pointer(ref))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ReadinessError('repaired readiness: missing or invalid explicit evidence: ' + str(error)) from error
    finite_json(value)
    return value, normalized


def _stamp(value, graph_ref, graph):
    require(same(pointer(value.get('graph')), pointer(graph_ref)), 'graph artifact pointer differs')
    require(all(value.get(key) == graph[key] for key in HASHES), 'physical/control/policy fingerprint differs')


def _outcome(value):
    require(type(value) is dict and set(value) == {'success', 'terminate', 'terminal_failure'}
        and all(type(v) is bool for v in value.values()), 'explicit bool Native result triple required')
    require(tuple(value[k] for k in ('success', 'terminate', 'terminal_failure')) in
        ((False, False, False), (False, True, True), (True, True, False)), 'inconsistent Native result')
    return value


def validate_sampling_case(case, expected_protocol, *, block, case_version):
    """Validate original741/core or744/block accounting under a NEW exact schema."""
    finite_json(case); finite_json(expected_protocol)
    require(type(block) is bool and type(case_version) is str and bool(case_version), 'explicit branch and case schema required')
    require(type(case) is dict and case.get('version') == case_version, 'actual graph-declared case schema required')
    require(case.get('complete') is True and case.get('cleanup_completed') is True
        and case.get('infrastructure_error') is None, 'clean completed execution required')
    require((case.get('task') in BLOCKS) is block and case.get('method') in METHODS, 'registered method/core/block route required')
    require(expected_protocol.get('schema') == 'r10_author_action_outcome_protocol_v741'
        and same(case.get('author_outcome_protocol'), expected_protocol), 'exact graph-frozen original741 protocol required')
    for key, expected in dict(pose_action_max_ticks=10, public_hold_max_ticks=1,
            block_native_fail_conditions=0, block_extra_task_success_calls=0).items():
        require(type(expected_protocol.get(key)) is int and expected_protocol[key] == expected, 'original741 numeric contract: '+key)
    require(expected_protocol.get('failure_sampling') == 'current_native_result_at_explicit_action_end'
        and expected_protocol.get('success_sampling') == 'original_per_tick_success_latch'
        and expected_protocol.get('budget_partial_action_consumed') is True
        and expected_protocol.get('native_RML_parameters_unchanged') is True, 'original741 sampling semantics required')
    success, fatal = case.get('official_task_success'), case.get('native_terminal_failure')
    require(type(success) is bool and type(fatal) is bool and not (success and fatal), 'actual consistent Native bool outcomes required')
    terminal = success or fatal
    clock = case.get('counted_clock', {}); ticks = clock.get('ticks')
    require(integer(ticks, maximum=200) and number(clock.get('dt')) and clock['dt'] == .05
        and number(clock.get('seconds')) and abs(clock['seconds'] - ticks*.05) <= 1e-8, 'finite original200tick/50ms clock required')
    alignment = case.get('alignment_receipt', {}); own = alignment.get('clock', {})
    require(type(own.get('ticks')) is int and own['ticks'] == ticks and number(own.get('dt')) and own['dt'] == .05
        and number(own.get('simulation_seconds')) and abs(own['simulation_seconds']-ticks*.05) <= 1e-8
        and number(own.get('budget_seconds')) and own['budget_seconds'] == 10.
        and alignment.get('outcomes_are_policy_observations', False) is False, 'alignment and physical clock differ')
    require(all(number(alignment.get(field)) and abs(alignment[field]-ticks*.05) <= 2e-5
        for field in ('actual_physics_seconds','actual_task_seconds')), 'original native/task clocks differ from counted time')
    receipt = case.get('native_outcome_sampling', {})
    require(receipt.get('schema') == (BLOCK_SCHEMA if block else CORE_SCHEMA)
        and receipt.get('active_action', 'missing') is None and receipt.get('aborted') is False
        and receipt.get('aborted_actions', [] if block else None) == []
        and type(receipt.get('last_tick')) is int and receipt['last_tick'] == ticks
        and receipt.get('global_tick_modulus_used') is False, 'complete original action sampling required')
    require(same(alignment.get('native_outcome_sampling'), receipt)
        and alignment.get('native_success_latched') is success
        and alignment.get('native_terminal_failure_latched') is fatal, 'alignment and case Native samples differ')
    final = dict(success=success, terminate=terminal, terminal_failure=fatal)
    if block:
        require(not fatal and receipt.get('original_native_success_latch_preserved') is True, 'original block success-only sampler required')
        for key in ('native_failure_conditions', 'extra_task_success_calls'):
            require(type(receipt.get(key)) is int and receipt[key] == 0, 'block original no-failure/new-extra-call contract required')
        # These zero counters describe the new action sampler, not the inherited
        # LocalSimulation.observe task.success call. Never demand only-tick calls.
    else:
        require(all(receipt.get(key) is True for key in ('success_sampled_each_tick',
            'failure_sampled_at_explicit_action_end', 'reset_failure_is_diagnostic_only',
            'reset_success_semantics_preserved')), 'original741 flags required')
        require(same(_outcome(receipt.get('sampled_outcome')), final)
            and alignment.get('native_terminal_latched') is terminal
            and same(alignment.get('native_outcome'), final), 'sampled terminal outcome differs')
        require(integer(receipt.get('intermediate_failure_samples')), 'diagnostic intermediate failure count required')
        initial = _outcome(receipt.get('initial_native_result'))
    rows = receipt.get('completed_actions'); require(type(rows) is list, 'actual complete action list required')
    cursor = 0; last = None; previous_terminal = False
    allowed = {'baseline_execute'} if case['method'] == 'physmani' else {'geometry_execute', 'public_hold'}
    for index, row in enumerate(rows):
        require(type(row) is dict and row.get('complete') is True and not previous_terminal, 'incomplete action or action after Native terminal')
        label = row.get('label'); start, end, count, maximum = (row.get(k) for k in ('start_tick', 'end_tick', 'ticks', 'max_ticks'))
        require(label in allowed and integer(maximum, 1, 10) and (label != 'public_hold' or maximum == 1)
            and type(start) is int and start == cursor and integer(end, start+1, ticks)
            and integer(count, 1, maximum) and count == end-start, 'original bounded continuous action interval required')
        last = _outcome(row.get('sampled_outcome'))
        if block:
            require(type(row.get('extra_task_success_calls')) is int and row['extra_task_success_calls'] == 0
                and last['terminal_failure'] is False, 'block action cannot invent fatal or new outcome calls')
        else:
            require(type(row.get('action_index')) is int and row['action_index'] == index, 'sequential original action index required')
            current = _outcome(row.get('current_native_result'))
            require(last['success'] or same(last, current), 'failure must be current action-end result, not a past failure OR')
        cursor = end; previous_terminal = last['terminate']
    require(cursor == ticks, 'every counted tick must be covered once')
    if rows:
        require(same(last, final) and (block or initial['success'] is False), 'final action/initial-success accounting differs')
    else:
        require(ticks == 0 and not fatal and (block or initial['success'] is success), 'zero-action Native outcome differs')
    if not block:
        when = alignment.get('native_terminal_tick')
        require(type(when) is int and when == ticks if terminal else when is None, 'actual terminal tick differs')
    return receipt


def make_graph_witness(graph_ref, graph, route, source_row, *, split):
    """Pure metadata helper shared by the future frozen TRAIN/EVAL worker."""
    require(split in ('train', 'validation'), 'TRAIN or genuine validation purpose required')
    declaration = route['train' if split == 'train' else 'evaluation']
    branch = 'physmani' if route['method'] == 'physmani' else ('block_geometry' if route['task'] in BLOCKS else 'core_geometry')
    return deepcopy(dict(schema=WITNESS_SCHEMA, graph=pointer(graph_ref), **{k:graph[k] for k in HASHES},
        purpose=declaration['purpose'], actual_split=split, task=route['task'], method=route['method'],
        source_raw_episode_id=source_row['raw_episode_id'], source_row_sha256=digest(source_row),
        actual_worker=declaration['worker'], actual_preparer=declaration['preparer'], native_consumer=declaration['consumer'],
        physical_factory=route['physical_factory'], factory_binding=graph['physical']['factories'][route['physical_factory']],
        source_overlay_lock=graph['physical']['source_overlay_lock'], configuration=route['configuration'],
        frozen_inputs=route['frozen_inputs'], models=route['models'], checkpoints=route['checkpoints'], model_branch=branch))


def bind_graph_identity(identity, witness):
    result = deepcopy(identity); result.pop('sha256', None)
    result.update(graph_sha256=witness['graph']['sha256'], **{key:witness[key] for key in HASHES})
    result['sha256'] = digest(result)
    return result


def _process(value):
    require(type(value) is dict and integer(value.get('pid'), 1) and integer(value.get('start_ticks'), 1), 'actual PID/start identity required')
    return dict(pid=value['pid'], start_ticks=value['start_ticks'])


def _not_live(process):
    process = _process(process)
    try:
        fields = Path('/proc/'+str(process['pid'])+'/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return True
    except OSError as error:
        raise ReadinessError('cannot establish owner process death') from error
    return int(fields[19]) != process['start_ticks'] or fields[0] in ('Z', 'X')


def _key(value):
    require(type(value) is dict and all(type(value.get(k)) is str for k in ('task', 'raw_episode_id', 'method')), 'explicit source/method key required')
    return value['task'], value['raw_episode_id'], value['method']


def _sources(manifest):
    require(manifest.get('split') == 'train' and type(manifest.get('episodes')) is list, 'actual TRAIN manifest required')
    rows = manifest['episodes']
    require(all(type(r) is dict and type(r.get('task')) is str and type(r.get('raw_episode_id')) is str
        and type(r.get('path')) is str and bool(r['path']) for r in rows), 'explicit actual TRAIN source rows required')
    keys = [(r['task'], r['raw_episode_id']) for r in rows]
    require(len(keys) == 1600 and len(set(keys)) == 1600
        and Counter(r.get('task') for r in rows) == Counter({t:100 for t in TASKS})
        and all(type(r.get('raw_episode_id')) is str and type(r.get('path')) is str and r['path'] for r in rows),
        'complete unique original1600-source TRAIN manifest required')
    return {(t, r['raw_episode_id']):r for t in TASKS for r in [v for v in rows if v['task'] == t][:4]}


def _profile(graph, root):
    profile, _ = read_json(root, graph['physical']['common_RML']['profile'])
    protocol, _ = read_json(root, graph['physical']['outcome_sampling']['protocol'])
    require(profile.get('schema') == 'r10_common_author_RML_profile_v530'
        and profile.get('methods') == list(METHODS) and same(profile.get('parameters'), PARAMETERS)
        and profile.get('user_authorization') == RML_AUTHORIZATION
        and all(profile.get(k) is True for k in ('original_waypoint_filter_and_RML_algorithm',
            'motion_metrics_required_for_all_methods','measured_acceleration6_is_diagnostic_only','native_task_success_unchanged')),
        'graph-frozen original530 profile required')
    pinned = {(p['path'],p['sha256']) for p in graph['physical']['sources_map'].values()}
    for field, names in (('sources_sha256', RML_LOCAL_SOURCES), ('native_sources_sha256', RML_NATIVE_SOURCES)):
        require(type(profile.get(field)) is dict and set(profile[field]) == set(names)
            and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
                and (p,v) in pinned for p,v in profile[field].items()), 'complete graph-pinned530 source hashes required')
    require(same(pointer(protocol.get('implementation')), pointer(graph['physical']['outcome_sampling']['source'])), 'graph741 implementation source differs')
    require(pointer(protocol.get('original_runtime')) in list(graph['physical']['sources_map'].values()),
        'original shared runtime must be in the frozen physical graph')
    return profile, protocol


def _attestation(case, profile):
    receipt = case.get('common_author_RML_parameters', {})
    require(receipt.get('schema') == 'r10_observed_common_author_RML_parameters_v530'
        and receipt.get('complete') is True and receipt.get('profile_sha256') == digest(profile)
        and same(receipt.get('actual_parameters'), PARAMETERS)
        and receipt.get('native_parameters_mutated') is False and receipt.get('actual_motion_bound_claimed') is False,
        'actual original530 attestation required')


def _motion(case):
    report = case.get('all_stage_counted_joint_motion', {})
    require(type(report) is dict and type(report.get('events')) is list
        and len(report['events']) == case['counted_clock']['ticks']
        and type(report.get('all_stage_actual_window_checks_passed')) is bool
        and report.get('schema') == 'r10_original_baseline_measured6_independent_of_unused_plan_model_v345'
        and same(case['alignment_receipt'].get('all_stage_counted_joint_motion_v191'),report)
        and type(report.get('extra_physics_ticks')) is int and report['extra_physics_ticks'] == 0
        and type(report.get('extra_camera_queries')) is int and report['extra_camera_queries'] == 0
        and report.get('policy_or_physics_modified') is False
        and report.get('original_planning_model_acceleration_unchanged') is True
        and report.get('planning_metadata_used_as_actual_acceleration_cap') is False
        and number(report.get('actual_window_acceleration_criterion_rad_s2'))
        and report['actual_window_acceleration_criterion_rad_s2'] == 6.,
        'one original motion diagnostic per counted tick required')
    violations=[];previous=None;peak=0.
    for index,event in enumerate(report['events']):
        require(type(event) is dict and type(event.get('tick_before')) is int and event['tick_before'] == index
            and type(event.get('tick_after')) is int and event['tick_after'] == index+1
            and number(event.get('actual_issued_at')) and abs(event['actual_issued_at']-index*.05) <= 1e-8
            and number(event.get('reference_target_at')) and abs(event['reference_target_at']-(index+1)*.05) <= 1e-8
            and type(event.get('phase')) is str and event.get('diagnostic_only') is True
            and event.get('native_parameters_changed') is False
            and all(type(event.get(key)) is bool for key in ('position_passed','speed_passed')),
            'actual continuous original motion diagnostic interval required')
        vectors=[event.get(key) for key in ('measured_q_before','measured_q_after','backward_window_velocity')]
        require(all(type(v) is list and len(v) == 7 and all(number(x) for x in v) for v in vectors),
            'finite actual seven-joint windows required')
        q0,q1,velocity=vectors
        require(all(abs((b-a)/.05-v) <= 1e-8 for a,b,v in zip(q0,q1,velocity)), 'actual measured velocity differs')
        acceleration=event.get('backward_window_acceleration')
        if previous is None:
            require(acceleration is None and event.get('acceleration_passed') is None, 'first window cannot invent an acceleration')
        else:
            require(type(acceleration) is list and len(acceleration) == 7 and all(number(x) for x in acceleration)
                and all(abs((v-p)/.05-a) <= 1e-7 for v,p,a in zip(velocity,previous,acceleration))
                and type(event.get('acceleration_passed')) is bool
                and event['acceleration_passed'] is all(abs(v) <= 6.+1e-6 for v in acceleration),
                'actual adjacent-window acceleration differs')
            peak=max(peak,*[abs(v) for v in acceleration])
        if not event['position_passed'] or not event['speed_passed'] or event.get('acceleration_passed') is False:
            violations.append(index+1)
        previous=velocity
    require(same(report.get('violations'),violations)
        and report['all_stage_actual_window_checks_passed'] is (not violations)
        and number(report.get('peak_actual_window_acceleration'))
        and abs(report['peak_actual_window_acceleration']-peak) <= 1e-7,
        'motion diagnostic summary differs from original measured windows')
    return report['all_stage_actual_window_checks_passed']


def _option(argv, flag, expected, *, required=True):
    require(type(argv) is list and all(type(v) is str for v in argv), 'actual argv list required')
    if flag not in argv and not required:
        return
    require(argv.count(flag) == 1, 'one actual '+flag+' required')
    index = argv.index(flag)
    require(index+1 < len(argv) and argv[index+1] == expected, 'actual argv differs: '+flag)


def _argv(root, argv, job, route, graph_ref, graph, *, check):
    worker = pointer(route['train']['worker'])
    require(worker['root'] == 'project' and type(argv) is list
        and sum(v in (worker['path'], str(Path(root)/worker['path'])) for v in argv) == 1,
        'actual graph-declared worker must occur once in child argv')
    wrapper = graph['control']['components']['environment_wrapper']
    require(wrapper['root'] == 'project'
        and sum(v in (wrapper['path'], str(Path(root)/wrapper['path'])) for v in argv) == 1,
        'actual graph-frozen820 wrapper must occur once in child argv')
    _option(argv, '--manifest-sha256', graph['physical']['environment_manifest']['sha256'])
    launcher = 'scripts/run_aligned_gpu.sh' if route['GPU_required'] else 'scripts/run_local_sim.sh'
    require(any(p['root'] == 'project' and p['path'] == launcher for p in graph['control']['sources_map'].values())
        and sum(v in (launcher, str(Path(root)/launcher)) for v in argv) == 1,
        'actual method-specific Native launcher must be frozen in the control graph')
    for flag, expected in (('--graph', graph_ref['path']), ('--task', job['task']),
            ('--raw-episode-id', job['raw_episode_id']), ('--method', job['method']),
            ('--split', 'train'), ('--out', job['out']), ('--runtime', job['runtime'])):
        _option(argv, flag, expected)
    for flag, expected in (('--config', route['configuration']['path']), ('--frozen-inputs', route['frozen_inputs']['path'])):
        _option(argv, flag, expected, required=False)
    require(argv.count('--check') == int(check), 'CPU/native argv check mode differs')


def _phase(root, refs, job, route, graph_ref, graph, witness, *, check):
    require(type(refs) is dict and set(refs) == {'launch','terminal','sidecar'}, 'actual child launch/terminal/sidecar pointers required')
    launch, launch_ref = read_json(root, refs['launch']); terminal, terminal_ref = read_json(root, refs['terminal'])
    sidecar, sidecar_ref = read_json(root, refs['sidecar'])
    outer = _process(launch.get('process')); inner = _process(sidecar.get('process'))
    require(same(_process(terminal.get('process')), outer)
        and type(terminal.get('returncode')) is int and terminal['returncode'] == 0
        and terminal.get('automatic_retry') is False, 'actual child exit0 without retry required')
    require(same(pointer(launch.get('actual_worker')), pointer(route['train']['worker'])), 'child launch worker differs')
    _stamp(launch, graph_ref, graph); _argv(root, launch.get('argv'), job, route, graph_ref, graph, check=check)
    _stamp(sidecar, graph_ref, graph)
    require(same(pointer(sidecar.get('actual_worker')), pointer(route['train']['worker']))
        and same(pointer(sidecar.get('actual_preparer')), pointer(route['train']['preparer'])),
        'actual inner worker/preparer source differs')
    execution = sidecar.get('execution', {})
    require(sidecar.get('complete') is True and execution.get('complete') is True
        and same(sidecar.get('graph_witness'), witness) and same(execution.get('graph_witness'), witness),
        'complete actual CPU/native graph witness required')
    require(execution.get('native_started') is (not check), 'actual CPU/native started mode differs')
    if check:
        require(execution.get('mode') == 'CPU_check' and execution.get('actual_split') == 'train'
            and execution.get('GPU_started') is False, 'CPU preflight cannot start a GPU or change purpose')
    else:
        restored = execution.get('runtime_scope_restoration', {})
        require(execution.get('mode') == 'graph_bound_native_TRAIN' and execution.get('actual_split') == 'train'
            and execution.get('native_false_preserved') is True
            and restored.get('complete') is True and restored.get('restored') is True,
            'actual native purpose and completed factory/writer restoration required')
    namespace = sidecar.get('frozen_OS_environment', {})
    require(namespace.get('complete') is True
        and all(type(namespace.get(k)) is int for k in ('outer_wrapper_pid','outer_wrapper_start_ticks','observer_pid','observer_start_ticks'))
        and namespace.get('outer_wrapper_pid') == outer['pid']
        and namespace.get('outer_wrapper_start_ticks') == outer['start_ticks']
        and namespace.get('observer_pid') == inner['pid']
        and namespace.get('observer_start_ticks') == inner['start_ticks'], 'actual820 outer/inner process binding required')
    expected_sidecar = job['runtime'] + ('/CPU_check/worker_sidecar.json' if check else '/native/worker_sidecar.json')
    require(sidecar_ref['path'] == expected_sidecar, 'sidecar escaped its planned fresh case namespace')
    return sidecar, dict(launch=launch_ref, terminal=terminal_ref, sidecar=sidecar_ref), (inner['pid'], inner['start_ticks'])


def _owner(root, cohort, graph_ref, graph, plan_ref):
    refs = cohort.get('owner', {})
    require(type(refs) is dict and set(refs) == {'launch','terminal','exec_completion'}, 'actual completed owner evidence required')
    launch, _ = read_json(root, refs['launch']); terminal, _ = read_json(root, refs['terminal']); completion, _ = read_json(root, refs['exec_completion'])
    process = _process(launch.get('process'))
    require(same(pointer(launch.get('source')), pointer(graph['control']['components']['train_owner']))
        and same(pointer(launch.get('plan')), plan_ref), 'actual frozen train owner/plan binding required')
    _stamp(launch, graph_ref, graph)
    require(terminal.get('complete') is True and type(terminal.get('completed')) is int and terminal['completed'] == 192
        and type(terminal.get('attempted')) is int and terminal['attempted'] == 192
        and type(terminal.get('infrastructure_failed')) is int and terminal['infrastructure_failed'] == 0
        and terminal.get('automatic_retry') is False and terminal.get('active') is None,
        'all192 original attempts must naturally complete with no active/infra case')
    require(same(_process(completion.get('process')), process)
        and completion.get('actual_tool_exit_observed') is True and completion.get('natural_completion') is True
        and type(completion.get('returncode')) is int and completion['returncode'] == 0
        and same(pointer(completion.get('plan')), plan_ref)
        and same(pointer(completion.get('terminal')), pointer(refs['terminal'])) and _not_live(process),
        'actual root-observed owner exit0 and dead process identity required')


def audit_fresh_train192(root, graph_ref, cohort_ref):
    """Return deterministic readiness only after ALL192 hash-bound fresh cases.

    The frozen graph contains no cohort/proof pointer. This operation reads
    explicit TRAIN metadata and existing per-case receipts only, and writes none.
    """
    root = Path(root).resolve()
    _, graph_ref = read_json(root, graph_ref)
    try:
        graph = graphs.read_policy_graph(root, graph_ref)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ReadinessError('repaired readiness: invalid frozen graph: '+str(error)) from error
    require(graph['control']['contracts']['readiness_schema'] == READINESS_SCHEMA, 'new readiness contract required')
    self_pointer = dict(root='project', path=SOURCE, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(same(pointer(graph['control']['components']['train_auditor']), self_pointer), 'actual1144 auditor must have been frozen before execution')
    manifest, manifest_ref = read_json(root, graph['train_source_manifest']); sources = _sources(manifest)
    expected = {(t, raw, method) for t, raw in sources for method in METHODS}
    profile, protocol = _profile(graph, root)
    cohort, cohort_ref = read_json(root, cohort_ref)
    require(cohort.get('schema') == COHORT_SCHEMA and cohort.get('complete') is True
        and cohort.get('automatic_retry') is False and cohort.get('source_outcome_filtering') is False
        and cohort.get('reused_cases') == [], 'complete fresh unfiltered no-retry cohort required')
    _stamp(cohort, graph_ref, graph)
    plan, plan_ref = read_json(root, cohort.get('plan'))
    require(plan.get('schema') == PLAN_SCHEMA and plan.get('automatic_retry') is False
        and plan.get('source_outcome_filtering') is False and plan.get('reused_cases') == [], 'prelaunch fresh192 plan required')
    _stamp(plan, graph_ref, graph)
    jobs, records = plan.get('jobs'), cohort.get('records')
    require(type(jobs) is list and len(jobs) == 192 and type(records) is list and len(records) == 192,
        'all192 planned and completed attempts required; partial is not ready')
    job_keys, record_keys = [_key(v) for v in jobs], [_key(v) for v in records]
    require(len(set(job_keys)) == len(set(record_keys)) == 192 and set(job_keys) == set(record_keys) == expected,
        'exact original16 by first4 by3 grid, once without replacement required')
    require(len({j.get('out') for j in jobs}) == len({j.get('runtime') for j in jobs}) == 192,
        'fresh unique case output/runtime namespaces required')
    _owner(root, cohort, graph_ref, graph, plan_ref)
    routes = {(r['task'], r['method']):r for r in graph['routes']}; record_map = {_key(r):r for r in records}
    result, processes, case_paths, evidence_paths = [], set(), set(), set()
    for job in jobs:
        key = _key(job); row = record_map[key]; route = routes[(key[0], key[2])]
        require(type(row.get('attempt')) is int and row['attempt'] == 1, 'exactly one actual attempt per source required')
        for field, prefix in (('out','artifacts/framework/multitask_r10/'), ('runtime','runtime/framework/')):
            name = job.get(field)
            require(type(name) is str and name.startswith(prefix) and Path(name).as_posix() == name
                and '..' not in Path(name).parts, 'explicit fresh project case namespaces required')
        source = sources[key[:2]]
        witness = make_graph_witness(graph_ref, graph, route, source, split='train')
        cpu, cpu_refs, cpu_process = _phase(root, row.get('CPU_check'), job, route, graph_ref, graph, witness, check=True)
        native, native_refs, native_process = _phase(root, row.get('native'), job, route, graph_ref, graph, witness, check=False)
        require(cpu_process not in processes and native_process not in processes and cpu_process != native_process,
            'a fresh actual CPU/native worker process is required for every case')
        processes.update((cpu_process, native_process))
        for phase_refs in (cpu_refs, native_refs):
            for reference in phase_refs.values():
                require(reference['path'] not in evidence_paths, 'per-case execution evidence cannot be reused')
                evidence_paths.add(reference['path'])
        case, case_ref = read_json(root, row.get('case'))
        require(case_ref['path'] == job['out']+'/case.json' and case_ref['path'] not in case_paths, 'fresh planned native case required')
        case_paths.add(case_ref['path'])
        require(case.get('task') == key[0] and case.get('method') == key[2] and case.get('split') == 'train'
            and case.get('source_raw_episode_id') == key[1] and type(case.get('process_id')) is int and case['process_id'] == native_process[0]
            and case.get('automatic_retry') is False and case.get('TEST_used_for_fitting') is False
            and same(case.get('graph_witness'), witness), 'actual case source/worker/graph branch differs')
        identity = case.get('identity', {}); no_sha = dict(identity); no_sha.pop('sha256', None)
        require(identity.get('version') == 'r10_fresh_case_identity_v1'
            and identity.get('split') == 'train' and identity.get('task') == key[0] and identity.get('method') == key[2]
            and identity.get('source_raw_episode_id') == key[1] and identity.get('parent_raw_episode_id') == key[1]
            and identity.get('raw_episode_id') == key[1]+'__policy_aug_r0'
            and type(identity.get('replica')) is int and identity['replica'] == 0
            and identity.get('exploration_seed') is None and identity.get('demo_only') is True
            and identity.get('development_demo_only') is True and identity.get('fitting_source') is False
            and identity.get('formal_validation_metric') is False
            and case.get('raw_episode_id') == identity['raw_episode_id'] and case.get('parent_raw_episode_id') == key[1]
            and case.get('case_isolation') == 'fresh_simulator_process_per_episode'
            and case.get('formal_test_used') is False and case.get('original_source_motion_replayed') is False
            and case.get('recorded_initial_conditions',{}).get('metric_only') is True
            and case.get('recorded_initial_conditions',{}).get('policy_use_allowed') is False,
            'actual unaugmented first-parent original TRAIN identity and metric-only reset required')
        execution_protocol = case.get('execution_protocol', {})
        require(type(execution_protocol.get('schema')) is str and bool(execution_protocol['schema'])
            and identity.get('execution_protocol_schema') == execution_protocol['schema']
            and identity.get('execution_protocol_sha256') == digest(execution_protocol)
            and same(cpu['execution'].get('execution_protocol'),execution_protocol)
            and same(native['execution'].get('execution_protocol'),execution_protocol),
            'CPU/native/case actual new execution protocol identity differs')
        for field,expected_value in (('actual_worker',route['train']['worker']),
                ('actual_preparer',route['train']['preparer']),('inherited_consumer',route['train']['consumer']),
                ('source_overlay_lock',graph['physical']['source_overlay_lock'])):
            require(same(pointer(execution_protocol.get(field)),pointer(expected_value)), 'new execution source differs: '+field)
        require(execution_protocol.get('selected_physical_factory') == route['physical_factory']
            and same(execution_protocol.get('factory_binding'),witness['factory_binding'])
            and execution_protocol.get('execution_branch') == witness['model_branch']
            and execution_protocol.get('actual_split') == 'train'
            and execution_protocol.get('component_purpose') == 'train_qualification'
            and execution_protocol.get('automatic_retry') is False
            and execution_protocol.get('original_745_execution_acceptance_granted') is False
            and execution_protocol.get('qualification_granted_by_worker') is False
            and execution_protocol.get('formal_score_contribution') is False
            and type(execution_protocol.get('additional_native_getter_observe_predict_or_success_calls')) is int
            and execution_protocol['additional_native_getter_observe_predict_or_success_calls'] == 0
            and same(execution_protocol.get('unchanged_action_outcome_sampling'),protocol),
            'new declared factory/purpose cannot claim legacy acceptance, change sampling or add Native observations')
        _stamp(execution_protocol, graph_ref, graph)
        require(same(pointer(case.get('source_overlay_lock')),graph['physical']['source_overlay_lock'])
            and case.get('original_745_execution_acceptance_granted') is False
            and case.get('qualification_granted_by_worker') is False
            and case.get('formal_score_contribution') is False,
            'honest new case namespace/protocol cannot impersonate previous acceptance')
        require(identity.get('sha256') == digest(no_sha)
            and identity.get('config_sha256') == route['configuration']['sha256']
            and identity.get('frozen_inputs_sha256') == route['frozen_inputs']['sha256']
            and identity.get('source_row_sha256') == digest(source)
            and identity.get('graph_sha256') == graph_ref['sha256']
            and all(identity.get(k) == graph[k] for k in HASHES)
            and same(cpu['execution'].get('identity'), identity) and same(native['execution'].get('identity'), identity),
            'CPU/native/case actual graph/config/source identity differs')
        domain = case.get('component_domain', {})
        require(domain.get('schema') == 'r10_source_bound_RML_component_purpose_v640'
            and domain.get('purpose') == 'train_qualification' and domain.get('split') == 'train'
            and domain.get('task') == key[0] and domain.get('method') == key[2] and domain.get('raw_episode_id') == key[1],
            'genuine fixed-TRAIN component purpose required')
        require(domain.get('VAL_TEST_relabelled_as_TRAIN') is False and domain.get('policy_sha256') is None
            and same(cpu['execution'].get('component_domain'), domain)
            and same(native['execution'].get('component_domain'), domain), 'CPU/native genuine640 TRAIN domain differs')
        require(same(pointer(native['execution'].get('case')), case_ref)
            and native['execution'].get('native_success') is case.get('official_task_success')
            and native['execution'].get('native_failure') is case.get('native_terminal_failure'), 'native sidecar/outcome differs')
        require(same(pointer(protocol.get('runtime_consumer')), pointer(route['train']['consumer'])), 'actual original consumer differs from741 protocol')
        validate_sampling_case(case, protocol, block=key[0] in BLOCKS, case_version=route['train']['case_schema'])
        _attestation(case, profile); motion_pass = _motion(case)
        hooks, hook_ref = read_json(root, row.get('owned_hooks'))
        require(hook_ref['path'] == job['out']+'/owned_hook_receipt.json' and hooks.get('actual_split') == 'train'
            and type(hooks.get('reset')) is list and type(hooks.get('paths')) is list
            and all(v.get('complete') is True for v in hooks['reset'])
            and all(v.get('close_complete') is True for v in hooks['paths']), 'actual reset/RML cleanup evidence required')
        if key[0] not in BLOCKS:
            require(len(hooks['reset']) == 1, 'one original captured core reset required')
        result.append(dict(task=key[0], raw=key[1], method=key[2], status='completed',
            native_success=case['official_task_success'], native_terminal_failure=case['native_terminal_failure'],
            motion_pass=motion_pass, counted_ticks=case['counted_clock']['ticks'], case=case_ref,
            owned_hooks=hook_ref, CPU_check=cpu_refs, native=native_refs, old_protocol_case_reused=False))
    # Recheck every graph source at the transaction's end, after all receipts.
    require(same(graphs.read_policy_graph(root, graph_ref), graph), 'frozen source graph changed during audit')
    # Receipts are independently hash-bound, including the cohort and owner
    # which name all per-case pointers. Reject a concurrent evidence replacement.
    inputs = [cohort_ref, plan_ref, *cohort['owner'].values()]
    for row in records:
        inputs.extend((row['case'], row['owned_hooks']))
        for phase in ('CPU_check', 'native'):
            inputs.extend(row[phase].values())
    for ref in inputs:
        read_json(root, ref)
    result.sort(key=lambda v:(v['task'],v['raw'],v['method']))
    return dict(schema=READINESS_SCHEMA, graph=graph_ref, cohort=cohort_ref, **{k:graph[k] for k in HASHES},
        TRAIN_manifest=manifest_ref, complete=True, completed=192, method_case_count=192,
        source_count=64, task_ready_count=16, cases=result,
        groups=[dict(task=t, method=m, completed=4, native_successes=sum(r['native_success'] for r in result
            if r['task'] == t and r['method'] == m)) for t in TASKS for m in METHODS],
        native_success_is_not_an_admission_threshold=True, motion_is_diagnostic_only=True,
        all_failures_retained=True, automatic_retry=False, source_outcome_filtering=False,
        old_protocol_case_reused=False,
        block_inherited_observe_success_calls_are_not_new_sampler_calls=True,
        block_only_tick_success_call_claimed=False, VAL_read=False, TEST_read=False,
        Native_started_by_auditor=False, GPU_started_by_auditor=False)


def require_readiness_proof(root, graph_ref, proof_ref):
    """Reaudit the proof's separate cohort; never trust a readiness flag alone."""
    saved, _ = read_json(root, proof_ref)
    require(saved.get('schema') == READINESS_SCHEMA, 'new repaired readiness proof schema required')
    actual = audit_fresh_train192(root, graph_ref, saved.get('cohort'))
    require(same(saved, actual), 'saved readiness differs from full fresh192 reaudit')
    return actual
