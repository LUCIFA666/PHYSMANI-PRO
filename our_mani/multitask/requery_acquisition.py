"""Separate all100 high-cup TRAIN augmentation with counted planning requeries.

No original acquisition, Context budget, raw label or fitting gate is changed.
This auditor owns a new typed manifest. A future versioned fitting adapter must
use this strict auditor before admitting it. Plan checking performs no native work.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math

from . import acquisition_lineage as lineage
from .train_teacher_requery import PlanningRequeryConfig
from .train_teacher_requery_hook import TASK, BEHAVIOR, VERSION as HOOK_VERSION

REGISTRATION = 'r10_fixed_high_cup_counted_requery_execution_registration_v2'
PLAN_SCOPE = 'r10_fixed_high_cup_counted_requery_augmentation_v2'
RUNNER_VERSION = 'r10_fixed_high_cup_counted_requery_first_execution_v2'
MANIFEST_VERSION = 'r10_fixed_high_cup_counted_requery_TRAIN_acquisition_v2'
PROPOSAL_VERSION = 'r10_fixed_all100_high_cup_failed_planning_requery_proposal_v2'
RAW_AUDIT_VERSION = 'r10_counted_high_cup_actual_feedback_array_audit_v2'
BUDGET = dict(parent_sources=100, teacher_rollouts=100, replica=1,
              maximum_ticks_per_case=200, maximum_total_ticks=20000,
              hold_rollouts=0, Context_primary_view=False,
              parent_folds='same original TRAIN parent')
OWNED_SOURCES = (
    'scripts/collect_TRAIN_high_cup_requery_r10.py',
    'our_mani/multitask/requery_acquisition.py',
    'our_mani/multitask/train_teacher_requery.py',
    'our_mani/multitask/train_teacher_requery_hook.py',
    'scripts/collect_multitask_training_r10.py',
    'scripts/collect_multitask_recovery_r10.py',
    'scripts/run_local_sim.sh',
    'our_mani/multitask/acquisition_lineage.py',
    'our_mani/multitask/development_reset_recovery.py',
    'our_mani/multitask/simulation.py',
    'our_mani/multitask/rml_path_repair.py',
    'our_mani/multitask/rml_path_recovery.py',
    'our_mani/multitask/train_teacher.py',
    'our_mani/multitask/training_view.py',
    'our_mani/multitask/acquisition_manifest.py',
    'our_mani/multitask/motion_labels.py',
    'our_mani/multitask/teacher_supplement.py',
    'our_mani/multitask/counted_teacher_bundle.py',
)
require = lineage.require


def expected_cases(primary):
    require(primary.get('split') == 'train' and len(primary['cases']) == 2800,
            'complete original2800 roster required')
    rows = [row for row in primary['cases'] if row['task'] == TASK and row['behavior'] == BEHAVIOR]
    require(len(rows) == 100 and len({row['parent_source_group'] for row in rows}) == 100,
            'all100 high-cup original parents required')
    cases = []
    for row in rows:
        require(row['replica'] == row['behavior_replica'] == 0
                and row['teacher_used'] is True and row['context_primary_view'] is False
                and row['interaction_eligible'] is True and row['maximum_collection_ticks'] == 200,
                'original high-cup teacher scope differs')
        case = deepcopy(row); raw = row['parent_raw_episode_id']
        case.update(replica=1, behavior_replica=1, raw_episode_id=raw + '__' + BEHAVIOR + '_r1')
        case['case_id'] = lineage.canonical_sha(dict(split='train', task=TASK, raw=raw,
                                                    replica=1, behavior=BEHAVIOR))[:16]
        cases.append(case)
    require(len({c['case_id'] for c in cases}) == 100 and
            not ({c['case_id'] for c in cases} & {c['case_id'] for c in primary['cases']}),
            'new fixed identities overlap original K1')
    return cases


def verify_registration(root, registry_pointer, *, live=False, require_ready=False):
    root = Path(root).resolve(); files = {}
    if not isinstance(registry_pointer, dict):
        registry_pointer = lineage.pointer(root, registry_pointer)
    path = lineage.checked_pointer(root, registry_pointer, files); registry = lineage.read(path)
    require(set(registry) == {'schema', 'proposal', 'shared_profile_reference_registration',
            'execution', 'execution_ready', 'fixed_budget'} and registry['schema'] == REGISTRATION,
            'exact counted-requery registration required')
    require(type(registry['execution_ready']) is bool and
            (not require_ready or registry['execution_ready']), 'candidate registration is not execution ready')
    require(registry['fixed_budget'] == BUDGET, 'fixed separate100 teacher-only budget differs')
    proposal = lineage.read(lineage.checked_pointer(root, registry['proposal'], files))
    require(proposal.get('schema') == PROPOSAL_VERSION and proposal.get('proposal_only') is True
            and proposal.get('native_or_GPU_started') is False and proposal.get('task') == TASK
            and proposal.get('augmentation_budget') == BUDGET
            and proposal.get('requery_configuration') == asdict(PlanningRequeryConfig())
            and proposal.get('raw_audit_version') == RAW_AUDIT_VERSION,
            'original all100 proposal/source policy differs')
    preserved = proposal.get('preserved_unexecuted_v1_candidate', {})
    require(set(preserved) == {'proposal','plan','execution_registration','source_snapshot','audit_counterexamples'},
            'v2 must preserve and bind the unexecuted v1 candidate and audit evidence')
    for value in preserved.values():
        lineage.checked_pointer(root, value, files)
    old_registry = lineage.read(lineage.local(root, preserved['execution_registration']['path']))
    require(old_registry.get('schema') == 'r10_fixed_high_cup_counted_requery_execution_registration_v1'
            and old_registry.get('execution_ready') is False
            and old_registry.get('proposal') == preserved['proposal']
            and old_registry['execution']['plan'] == preserved['plan']
            and old_registry['execution']['source_snapshot'] == preserved['source_snapshot'],
            'preserved v1 candidate identity/readiness differs')
    old_actual = lineage.load_archived_plan(root, old_registry['execution'], files)
    primary_path = lineage.checked_pointer(root, proposal['original_plan'], files)
    primary = lineage.read(primary_path); cases = expected_cases(primary)
    require([r['proposed_case'] for r in proposal['proposed_cases']] == cases,
            'proposal is not exact all100 original-parent derivation')
    for row in proposal['proposed_cases']:
        require(row['proposed_reset'] == lineage.expected_reset(row['proposed_case'], primary['protocol']),
                'proposal source-derived reset seed/variation differs')
    for key in ('complete_primary_lineage', 'actual_coverage_diagnosis', 'primary_registrations', 'source_snapshot'):
        lineage.checked_pointer(root, proposal[key], files)
    primary_proof = lineage.read(lineage.local(root, proposal['complete_primary_lineage']['path']))
    require(primary_proof.get('audit_passed') is True and primary_proof.get('complete') is True,
            'original complete2800 primary audit required')
    # The existing accepted basketball registry is only a shared physics/RML
    # reference. Its cases, budget and execution identity never become ours.
    from . import teacher_supplement
    shared = teacher_supplement.verify_registration(root, registry['shared_profile_reference_registration'],
                                                    require_ready=True)
    files.update(shared['files'])
    require(shared['registry']['primary_plan'] == proposal['original_plan'],
            'shared physical profile belongs to another original primary plan')
    current = lineage.load_archived_plan(root, registry['execution'], files)
    plan = current['plan']
    require(plan.get('counted_requery_augmentation_scope') == PLAN_SCOPE
            and plan.get('counted_requery_proposal') == registry['proposal']
            and plan.get('counted_requery_collection') == registry['execution']['collection']
            and plan.get('augmentation_budget') == BUDGET and plan.get('cases') == cases
            and plan.get('counted_planning_requery_configuration') == asdict(PlanningRequeryConfig())
            and plan.get('counted_requery_raw_audit_version') == RAW_AUDIT_VERSION
            and plan.get('primary_K1_roster_unmodified') is True
            and plan.get('primary_Context_hold_view_unmodified') is True,
            'counted-requery actual fixed roster/policy/budget differs')
    require(not any(key in plan for key in ('auxiliary_teacher_scope', 'supplement_proposal', 'auxiliary_collection')),
            'high-cup augmentation cannot impersonate basketball supplement')
    require(plan['protocol'] == shared['plan']['protocol'] and 'recovery_input' not in plan['protocol']
            and old_actual['plan']['protocol'] == plan['protocol'] and old_actual['plan']['cases'] == cases
            and plan['protocol'].get('draft') is False
            and plan['protocol'].get('native_execution_authorized') is True,
            'new teacher policy cannot alter shared physics/RML/camera/gripper protocol')
    require(current['collection'].is_relative_to(root / 'artifacts/framework/multitask_r10')
            and current['collection'] != shared['current']['collection'], 'new independent collection namespace required')
    for name in OWNED_SOURCES:
        require(str((root / name).resolve()) in current['sources'], 'new executable missing from actual source archive: ' + name)
    if live:
        from scripts import collect_multitask_training_r10 as collection
        require(collection.verify_plan(current['path']) == plan, 'current plan inputs changed')
        from scripts import collect_multitask_recovery_r10 as recovery
        recovery.load_rml_permission(plan['protocol']['RML_factory_hook'], plan)
        from .development_reset_recovery import load_permission
        permission = load_permission(plan['protocol']['development_reset_profile'], root=root,
                                     native_repo=plan['protocol']['baseline_repo'])
        for name, checksum in permission['native_inputs_sha256'].items():
            require(plan['inputs'].get(str((Path(plan['protocol']['baseline_repo']) / name).resolve())) == checksum,
                    'shared native physical initialization source is unbound')
    return dict(root=root, registry=registry, registry_pointer=registry_pointer,
                registry_path=path, current=current, plan=plan, primary=primary,
                files=files, expected_cases={c['case_id']: c for c in cases})


def validate_counted_commands(ledger, scope, *, ticks, arrays=None):
    """Check ledger semantics; production additionally requires actual arrays.

    ``arrays=None`` is only a CPU ledger diagnostic. ``validate_result`` always
    supplies the hash-verified original NPZ to prove actual feedback intervals.
    """
    require(type(ticks) is int and 0 <= ticks <= 200, 'actual counted tick budget differs')
    maximum_time = ticks * .05
    def frame_time(value):
        require(type(value) in (int, float) and math.isfinite(value)
                and 0 <= value <= maximum_time + .000001
                and abs(value - round(value / .05) * .05) <= 1e-6,
                'command time/real waits must be finite, on the actual50ms grid and within the counted clock')
        return float(value)
    commands = ledger['commands']; configuration = asdict(PlanningRequeryConfig())
    require(scope.get('version') == HOOK_VERSION and scope.get('complete') is True
            and scope.get('original_factory_restored') is True
            and scope.get('hook_owner_unchanged') is True and scope.get('source_unchanged') is True
            and scope.get('follower_instances') == 1 and scope.get('raw_commands') == len(commands)
            and scope.get('native_or_physics_steps_called_by_hook') is False,
            'actual counted-requery scope did not complete/restore')
    indexed = {row['command_id']: row for row in commands}
    require(all(type(row['command_id']) is int for row in commands)
            and len(indexed) == len(commands)
            and [row['command_id'] for row in commands] == list(range(len(commands))),
            'raw command identities are not consecutive unique real queries')
    count = 0
    time_fields = {'generated_at','last_executed_at','measured_arrival_at','gripper_completed_at',
        'command_completed_at','cutoff_at','planning_requery_wait_started_at',
        'planning_requery_wait_completed_at','planning_requery_last_actual_feedback_at',
        'command_terminal_reference_metric_time'}
    fresh_fields = {'fresh_query_after_failed_command_id','actual_planning_requery_index','same_episode_after_counted_wait'}
    for row in commands:
        generated = frame_time(row['generated_at'])
        require(not any((key.endswith('_at') or key.endswith('_time')) and key not in time_fields for key in row),
                'unknown command timestamp semantics')
        for key in time_fields & set(row):
            require(frame_time(row[key]) >= generated - 1e-6, 'command event precedes its actual issue time')
        chain = [row[key] for key in ('measured_arrival_at','gripper_completed_at','command_completed_at') if key in row]
        require(all(a <= b + 1e-6 for a,b in zip(chain,chain[1:])), 'actual arrival/gripper/completion clock order differs')
        if 'command_completed_at' in row:
            require('cutoff_at' not in row and 'last_executed_at' in row
                    and abs(row['last_executed_at']-row['command_completed_at']) <= 1e-6,
                    'completed command must end at its last actual execution feedback')
        if 'command_terminal_reference_metric_time' in row:
            require('command_completed_at' in row
                    and abs(row['command_terminal_reference_metric_time']-row['command_completed_at']) <= 1e-6,
                    'terminal reference label clock differs from actual completion')
        if 'cutoff_at' in row:
            require(all(row[key] <= row['cutoff_at'] + 1e-6 for key in
                ('last_executed_at','planning_requery_last_actual_feedback_at','planning_requery_wait_completed_at') if key in row),
                'command cutoff precedes actual execution or wait feedback')
        require(row.get('TRAIN_planning_requery_configuration') == configuration,
                'a raw command lacks its explicit fixed teacher policy')
        if 'planning_failure' in row:
            require('command_completed_at' not in row and row.get('planning_failure_is_completed_skill') is False
                    and type(row.get('executed_tick_count')) is int and row['executed_tick_count'] == 0,
                    'failed planning cannot become a completed command')
            if 'planning_requery_counted_wait_ticks' in row:
                wait_ticks = row['planning_requery_counted_wait_ticks']
                require(type(wait_ticks) is int and 0 <= wait_ticks <= ticks
                        and row.get('planning_requery_wait_started_at') == generated
                        and abs(frame_time(row['planning_requery_last_actual_feedback_at']) - generated - wait_ticks * .05) < 1e-6,
                        'failed wait count/time differs from its actual issued clock')
        cid = row['command_id']; previous = indexed.get(cid - 1)
        if previous is None:
            require(not fresh_fields & set(row), 'first command cannot claim a previous fresh query')
            continue
        require(generated >= previous['generated_at'] + .05 - 1e-6,
                'successive queries do not follow actual counted feedback')
        if 'planning_failure' not in previous:
            require(not fresh_fields & set(row), 'nonfailed command cannot authorize a fresh-failure marker')
            require('command_completed_at' in previous and previous['command_completed_at'] <= generated + 1e-6,
                    'new normal query requires the previous actual command completion')
            continue
        require(fresh_fields <= set(row), 'new query after failure requires all fresh-query markers')
        count += 1
        require(type(row['fresh_query_after_failed_command_id']) is int
                and row['fresh_query_after_failed_command_id'] == previous['command_id']
                and 'planning_failure' in previous and 'command_completed_at' not in previous
                and previous.get('planning_failure_preserved') is True
                and previous.get('command_incomplete_at_cutoff') is True
                and previous.get('cutoff_reason') == 'failed_planning_fresh_query_after_counted_wait'
                and previous.get('planning_requery_counted_wait_ticks') == 10
                and previous.get('planning_requery_wait_started_at') == previous['generated_at']
                and abs(previous['planning_requery_wait_completed_at'] - previous['generated_at'] - .5) < 1e-6
                and previous['planning_requery_last_actual_feedback_at'] == previous['planning_requery_wait_completed_at']
                and previous['cutoff_at'] == previous['planning_requery_wait_completed_at']
                and row['generated_at'] >= previous['planning_requery_wait_completed_at'] - 1e-8
                and type(row.get('actual_planning_requery_index')) is int
                and row['actual_planning_requery_index'] == count
                and row.get('same_episode_after_counted_wait') is True,
                'fresh query is not bound to the immediately preceding preserved failure/actual wait')
    failed = {row['command_id'] for row in commands if 'planning_failure' in row}
    require(not any(row['origin_command_id'] in failed for row in ledger['phase_completion_events']),
            'failed command has a positive phase completion event')
    require(type(scope.get('actual_requeries')) is int
            and count == scope['actual_requeries'] and count <= 20 and count * 10 <= ticks,
            'recorded fresh-query count exceeds real waits or fixed budget')
    proof = dict(actual_requeries=count, raw_commands=len(commands), failed_commands=len(failed),
                 failed_commands_marked_completed=0, real_wait_ticks_per_requery=10,
                 raw_audit_version=RAW_AUDIT_VERSION, actual_feedback_arrays_verified=False)
    if arrays is not None:
        proof.update(validate_actual_feedback_arrays(arrays, commands, ticks=ticks))
    return proof


def validate_actual_feedback_arrays(arrays, commands, *, ticks):
    """Reconstruct each issued command and failed wait from saved motor rows.

    A cache row stores its pre-action time/command and the feedback timestamp
    after exactly one real step. Its final row has no action and no command.
    No simulator object or future native pose is read by this audit.
    """
    import numpy as np
    require(type(ticks) is int and 0 <= ticks <= 200, 'actual feedback tick budget differs')
    required = ('times','teacher_command_id','teacher_command_is_new','teacher_command_generated_at',
        'teacher_command_age_seconds','actuator_feedback_timestamp','teacher_execution_stage',
        'teacher_execution_valid','teacher_action_valid','teacher_command_completed','teacher_joint_limit_recovery')
    require(all(name in arrays for name in required), 'actual requery command/feedback arrays missing')
    n = ticks + 1
    require(all(np.asarray(arrays[name]).shape == (n,) for name in required), 'actual command/feedback array shapes differ')
    times = np.asarray(arrays['times']); ids = np.asarray(arrays['teacher_command_id'])
    new = np.asarray(arrays['teacher_command_is_new']); generated = np.asarray(arrays['teacher_command_generated_at'])
    feedback = np.asarray(arrays['actuator_feedback_timestamp']); stages = np.asarray(arrays['teacher_execution_stage'])
    valid = np.asarray(arrays['teacher_execution_valid']); action = np.asarray(arrays['teacher_action_valid'])
    completed = np.asarray(arrays['teacher_command_completed']); recovery = np.asarray(arrays['teacher_joint_limit_recovery'])
    age = np.asarray(arrays['teacher_command_age_seconds'])
    require(ids.dtype.kind in 'iu' and all(np.asarray(arrays[name]).dtype == np.bool_ for name in
        ('teacher_command_is_new','teacher_execution_valid','teacher_action_valid','teacher_command_completed','teacher_joint_limit_recovery')),
        'actual command IDs/masks have incorrect types')
    require(np.isfinite(times).all() and np.allclose(times, np.arange(n)*.05, atol=1e-6, rtol=0)
        and np.isfinite(feedback).all() and np.allclose(feedback[:-1], times[1:], atol=1e-6, rtol=0)
        and abs(feedback[-1]-times[-1]) <= 1e-6,
        'actual motor feedback is not one distinct consecutive50ms step per action row')
    require(np.isfinite(generated).all() and np.isfinite(age).all()
        and ids[-1] == -1 and not any((new[-1],valid[-1],action[-1],completed[-1]))
        and stages[-1] == 'no_teacher', 'terminal cache row cannot issue or execute a teacher command')
    known = set(range(len(commands)))
    require(set(map(int,ids[ids>=0])) == known and np.all(ids >= -1), 'actual array command roster differs from ledger')
    observed_new = np.flatnonzero(new)
    require(len(observed_new) == len(commands) and list(ids[observed_new]) == list(range(len(commands))),
            'actual fresh-query rows are not one-to-one with the complete command ledger')
    active = ids >= 0
    require(not np.any((new|valid|action|completed) & ~active), 'unissued command has actual action/completion evidence')
    if commands:
        first = int(observed_new[0])
        require(np.all(ids[first:ticks]>=0) and np.all(recovery[:first])
                and np.all(stages[:first]=='public_joint_limit_recovery'),
                'missing teacher command rows outside explicit initial public joint recovery')
    else:
        require(np.all(recovery[:ticks]) and np.all(stages[:ticks]=='public_joint_limit_recovery'),
                'zero-command rollout has unexplained action rows')
    reconstructed = []
    for command in commands:
        cid = command['command_id']; rows = np.flatnonzero(ids == cid)
        require(len(rows)>0 and np.array_equal(rows,np.arange(rows[0],rows[-1]+1))
            and rows[0] == observed_new[cid]
            and abs(times[rows[0]]-command['generated_at']) <= 1e-6
            and np.allclose(generated[rows],command['generated_at'],atol=1e-6,rtol=0)
            and np.allclose(age[rows],times[rows]-command['generated_at'],atol=1e-6,rtol=0),
            'actual command ID/issue time/age rows differ from raw ledger')
        execution_rows = rows[valid[rows] & ~recovery[rows]]
        require(type(command.get('executed_tick_count')) is int
            and command['executed_tick_count'] == len(execution_rows),
            'raw executed tick count differs from actual motor command rows')
        completion_rows = rows[completed[rows]]
        require(len(completion_rows) == int('command_completed_at' in command)
            and (not len(completion_rows) or abs(feedback[completion_rows[0]]-command['command_completed_at'])<=1e-6),
            'raw command completion differs from actual feedback mask/time')
        if 'planning_failure' not in command:
            continue
        require(not np.any(valid[rows]|action[rows]|completed[rows])
                and np.all(np.isin(stages[rows],('failed_no_resample','public_joint_limit_recovery'))),
                'failed planning was promoted to actual executed/completed supervision')
        wait_ticks = command.get('planning_requery_counted_wait_ticks',0)
        require(type(wait_ticks) is int and 0 <= wait_ticks <= len(rows), 'failed wait exceeds actual motor feedback rows')
        wait_rows = rows[:wait_ticks]
        require(wait_ticks > 0 and abs(feedback[wait_rows[-1]]-command['planning_requery_last_actual_feedback_at']) <= 1e-6,
                'failed wait lacks its actual last motor feedback row')
        if 'planning_requery_wait_completed_at' in command:
            require(wait_ticks == 10 and abs(feedback[wait_rows[-1]]-command['planning_requery_wait_completed_at']) <= 1e-6
                and np.all(recovery[rows[wait_ticks:]])
                and np.all(stages[rows[wait_ticks:]]=='public_joint_limit_recovery'),
                'failed wait completion or delayed-query recovery rows differ')
        else:
            require(wait_ticks == len(rows), 'partial failed wait omitted actual feedback rows')
        if cid + 1 < len(commands):
            next_start = int(observed_new[cid+1])
            require(wait_ticks == 10 and next_start >= int(wait_rows[-1])+1
                    and abs(times[next_start]-commands[cid+1]['generated_at']) <= 1e-6,
                    'fresh query was issued before ten actual feedback rows completed')
            reconstructed.append(dict(previous_command_id=cid,new_command_id=cid+1,
                issue_frame=int(rows[0]),wait_action_frames=list(map(int,wait_rows)),
                feedback_frame_times=list(map(float,feedback[wait_rows])),
                actual_wait_ticks=10,next_issue_frame=next_start))
    return dict(actual_feedback_arrays_verified=True,reconstructed_requery_waits=reconstructed,
                actual_feedback_rows=ticks,actual_command_issue_rows=list(map(int,observed_new)))


def validate_result(root, path, context, case, *, files=None, validation_proofs=None):
    root = Path(root).resolve(); files = {} if files is None else files
    path = lineage.local(root, path); record = lineage.read(path)
    require(record.get('status') == 'completed' and record.get('case_id') == case['case_id']
            and record.get('plan_sha256') == context['plan']['plan_sha256']
            and record.get('split') == 'train' and record.get('task') == TASK
            and not any(k in record for k in ('recovery_execution', 'teacher_supplement_execution',
                                             'development_initialization_execution')),
            'new counted-requery raw record identity differs')
    own = record.get('counted_requery_execution', {})
    require(set(own) == {'version', 'hook_receipt', 'base_collector_receipt', 'execution_registration'}
            and own['version'] == RUNNER_VERSION and own['execution_registration'] == context['registry_pointer'],
            'new raw record needs its exact counted-requery registration')
    hooks_path = lineage.checked_pointer(root, own['hook_receipt'], files)
    base_path = lineage.checked_pointer(root, own['base_collector_receipt'], files)
    hooks = lineage.read(hooks_path); base = lineage.read(base_path)
    require(hooks_path.parent == base_path.parent and hooks.get('version') == RUNNER_VERSION
            and hooks.get('complete') is True and hooks.get('error') is None
            and hooks.get('case_id') == case['case_id']
            and hooks.get('plan') == context['current']['registration']['plan']
            and hooks.get('execution_registration') == context['registry_pointer']
            and hooks.get('first_execution') is True and hooks.get('automatic_retry') is False
            and hooks.get('is_infrastructure_recovery') is False and hooks.get('outcome_selected') is False,
            'new hook identity/completion differs')
    require(hooks.get('permissions') == {key: context['plan']['protocol'][key]
            for key in ('development_reset_profile', 'RML_factory_hook')},
            'new hook did not use the exact shared physical/RML permissions')
    scope = hooks['requery_factory']
    require(scope.get('case_id') == case['case_id'] and scope.get('task') == TASK
            and scope.get('split') == 'train' and scope.get('replica') == 1
            and scope.get('source_sha256') == context['plan']['inputs'][str((root / 'our_mani/multitask/train_teacher.py').resolve())]
            and lineage.instant(hooks['started_utc']) <= lineage.instant(scope['started_utc'])
            <= lineage.instant(scope['finished_utc']) <= lineage.instant(hooks['finished_utc']),
            'new follower scope/source/clock differs from the actual case')
    runner = str((root / hooks['actual_runner']['path']).resolve())
    require(runner == str((root / OWNED_SOURCES[0]).resolve())
            and context['current']['sources'].get(runner, {}).get('sha256') == hooks['actual_runner']['sha256'],
            'actual new runner is not its archived source')
    stripped = deepcopy({k: v for k, v in record.items() if k not in ('counted_requery_execution', 'parent_process_receipt')})
    stripped['files_sha256'] = {k: v for k, v in stripped['files_sha256'].items()
                               if k not in (own['hook_receipt']['path'], own['base_collector_receipt']['path'])}
    require(stripped == base, 'new wrapper rewrote original collector labels/bytes')
    require(lineage.instant(hooks['started_utc']) <= lineage.instant(base['started_utc'])
            <= lineage.instant(base['finished_utc']) <= lineage.instant(hooks['finished_utc']),
            'actual hook/base execution clocks reversed')
    for key in ('hook_receipt', 'base_collector_receipt'):
        require(record['files_sha256'].get(own[key]['path']) == own[key]['sha256'], 'raw new witness hash missing')
    for name, checksum in record['files_sha256'].items():
        output = lineage.checked_pointer(root, dict(path=name, sha256=checksum), files)
        require(output.is_relative_to(base_path.parent), 'new output reused another case or attempt')
    entry = record['entry']
    require(record.get('teacher_actions_used') is True and record.get('policy_history_contains_native_GT') is False
            and entry.get('augmentation_replica') == 1 and entry.get('context_primary_view') is False
            and entry.get('context_training_eligible') is False,
            'new teacher data cannot enter primary Context or runtime observations')
    for field in ('task', 'raw_episode_id', 'parent_raw_episode_id', 'parent_source_group', 'behavior',
                  'behavior_replica', 'maximum_collection_ticks', 'teacher_used', 'interaction_eligible'):
        require(entry.get(field) == case[field], 'new entry identity/budget differs: ' + field)
    resets = hooks['reset_recovery_receipts']; expected = lineage.expected_reset(case, context['plan']['protocol'])
    require(len(resets) == 1 and resets[0].get('complete') is True
            and resets[0].get('permission') == context['plan']['protocol']['development_reset_profile']
            and record['native_reset_receipt'].get('reset_recovery_receipt') == resets[0]
            and all(resets[0].get(k) == v for k, v in expected.items())
            and resets[0].get('source_resampled') is False,
            'new raw same-source shared physical reset differs')
    paths = hooks['RML_path_recovery_receipts']
    require(all(p.get('close_complete') is True and p.get('physics_step_called_by_adapter') is False
                and p.get('real_native_step_motion_replaced') is False for p in paths),
            'new RML paths did not close or changed physics')
    clock = record['alignment']['clock']; ticks = clock['ticks']
    require(type(ticks) is int and 0 <= ticks <= 200 and clock.get('dt') == .05
            and abs(clock['simulation_seconds'] - ticks * .05) < 1e-6,
            'new fixed actor collection clock differs')
    names = [name for name in record['files_sha256'] if Path(name).name == 'TRAIN_teacher_commands.json']
    require(len(names) == 1, 'new raw command ledger missing')
    ledger = lineage.read(lineage.local(root, names[0]))
    from .training_view import validate_arrays
    import numpy as np
    with np.load(lineage.local(root, entry['cache']), allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    validate_arrays(arrays, entry, record)
    feedback_proof = validate_counted_commands(ledger, hooks['requery_factory'], ticks=ticks, arrays=arrays)
    require(feedback_proof['actual_feedback_arrays_verified'] is True,
            'production requery audit needs actual motor feedback arrays')
    if validation_proofs is not None:
        require(case['case_id'] not in validation_proofs, 'duplicate raw feedback audit')
        validation_proofs[case['case_id']] = feedback_proof
    require(not np.count_nonzero(arrays['teacher_action_valid']) or bool(paths),
            'new executed teacher references need actual RML witnesses')
    return record


def audit_collection(root, registry_pointer, *, require_complete=False):
    root = Path(root).resolve(); context = verify_registration(root, registry_pointer, require_ready=True)
    directory = context['current']['collection']; files = dict(context['files']); known = context['expected_cases']
    entries, receipts, proofs, errors, pending = [], [], [], [], []
    feedback_proofs = {}
    for attempt_root in (directory / 'infrastructure_attempts').glob('*'):
        require(attempt_root.name in known and attempt_root.is_dir()
                and {p.name for p in attempt_root.iterdir()} <= {'0'},
                'unknown source or additional hidden infrastructure attempt')
        if not (attempt_root / '0/parent_receipt.json').exists():
            pending.append(attempt_root.name)
    for path in (directory / 'cases/train').glob('*/*/case.json'):
        require(path.parent.name in known and path.parent.parent.name == TASK, 'unknown new canonical case')
    for cid, case in known.items():
        parent = directory / 'infrastructure_attempts' / cid / '0/parent_receipt.json'
        canonical = directory / 'cases/train' / TASK / cid / 'case.json'
        if not parent.exists():
            continue
        raw_parent = lineage.read(parent)
        if raw_parent['status'] != 'completed':
            errors.append(dict(case_id=cid, parent_receipt=lineage.pointer(root, parent))); continue
        if not canonical.exists():
            pending.append(cid); continue
        record = validate_result(root, canonical, context, case, files=files, validation_proofs=feedback_proofs)
        row = dict(case_id=cid, canonical_receipt=lineage.pointer(root, canonical),
                   parent_receipt=lineage.pointer(root, parent))
        proof = lineage.verify_completed(root, row, context['current'], case, files)
        entries.append(record['entry']); proofs.append(proof)
        receipts.append(dict(case_id=cid, **lineage.pointer(root, canonical),
                             actual_plan_sha256=context['plan']['plan_sha256']))
    complete = len(entries) == 100 and not errors and not pending
    require(not require_complete or complete, 'all100 committed new teacher sources required; no outcome selection')
    require(all(lineage.file_sha(name) == checksum for name, checksum in files.items()), 'new audit inputs changed')
    projection = dict(version=MANIFEST_VERSION, split='train', TEST_used=False,
        complete=complete, expected=100, completed=len(entries), entries=entries, case_receipts=receipts,
        missing_case_ids=sorted(set(known) - {p['case_id'] for p in receipts}),
        plan_sha256=context['plan']['plan_sha256'], actual_plan=context['registry']['execution']['plan'],
        execution_registration=context['registry_pointer'], augmentation_budget=BUDGET,
        primary_K1_roster_unmodified=True, primary_Context_view_unmodified=True,
        new_teacher_policy_is_not_original_author_reproduction=True)
    return dict(schema=MANIFEST_VERSION, audit_passed=True, complete=complete,
        acquisition_projection=projection, verified_files_sha256=files,
        per_case_actual_plan_sha256={p['case_id']: context['plan']['plan_sha256'] for p in receipts},
        per_case_feedback_audit=feedback_proofs, raw_audit_version=RAW_AUDIT_VERSION,
        case_proofs=proofs, infrastructure_errors=errors, pending=pending,
        formal_model_or_policy_release=False)


def audit_manifest(root, path, *, require_complete=True):
    """Exact typed new100 projection; original primary2800 remains separate."""
    root = Path(root).resolve(); path = lineage.local(root, path)
    before = lineage.file_sha(path); value = lineage.read(path)
    require(value.get('version') == MANIFEST_VERSION, 'typed counted-requery v2 manifest required')
    proof = audit_collection(root, value['execution_registration'], require_complete=require_complete)
    require(value == proof['acquisition_projection'], 'counted-requery manifest differs from committed raw100 projection')
    context = verify_registration(root, value['execution_registration'], require_ready=True)
    require(value['actual_plan'] == context['registry']['execution']['plan']
            and value['plan_sha256'] == context['plan']['plan_sha256']
            and value['primary_K1_roster_unmodified'] is True
            and value['primary_Context_view_unmodified'] is True,
            'typed new100 manifest changed actual plan or original Context scope')
    require(lineage.file_sha(path) == before, 'typed new100 manifest changed during audit')
    for name,checksum in context['files'].items():
        require(name not in proof['verified_files_sha256'] or proof['verified_files_sha256'][name] == checksum,
                'typed new100 archived source hash conflict')
        proof['verified_files_sha256'][name] = checksum
    proof['verified_files_sha256'][str(path)] = before
    proposal = lineage.read(lineage.local(root,context['registry']['proposal']['path']))
    proof.update(base_plan_sha256=context['primary']['plan_sha256'],
        primary_plan=deepcopy(proposal['original_plan']),
        primary_Context_view_unmodified=True, primary_K1_roster_unmodified=True)
    return proof
