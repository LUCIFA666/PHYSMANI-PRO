"""Draft TRAIN-only semantic acquire evidence; no production fitter hook.

Original command completion/censoring, cache arrays and native outcomes remain
unchanged.  This additive view separates a contiguous acquire path from its
actual external/attached mode segments.  It cannot be passed off as a legacy
single-command TRAINPhaseCompletionEvent.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'r10_TRAIN_derived_semantic_acquire_attempt_v1'
RULE_SCHEMA = 'r10_TRAIN_semantic_acquire_derivation_rule_v1'
TASKS = ('moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed')
MASKS = ('modes', 'mode_valid', 'native_pose_valid', 'target_valid',
         'target_twist_valid', 'target_pose_valid_before_twist_mask', 'valid',
         'rotation_observable', 'target_rotation_observable', 'dynamics_segment_valid')


def require(condition, message):
    if not condition:
        raise ValueError('semantic acquire: '+message)


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AcquireModeSegment:
    mode: str
    frame_indices: tuple[int, ...]
    raw_command_ids: tuple[int, ...]
    start_time: float
    end_time: float
    terminal_frame: int
    terminal_kind: str
    original_phase_completion_command_id: int

    def __post_init__(self):
        require(self.mode in ('external', 'attached') and self.frame_indices
                and self.frame_indices == tuple(range(self.frame_indices[0], self.frame_indices[-1]+1)),
                'a fitted segment must contain actual contiguous same-mode frames')
        require(self.raw_command_ids and len(set(self.raw_command_ids)) == len(self.raw_command_ids)
                and all(type(i) is int and i >= 0 for i in self.raw_command_ids),
                'segment must retain its actual unique raw command IDs')
        require(np.isfinite([self.start_time, self.end_time]).all()
                and self.end_time > self.start_time and self.terminal_frame > self.frame_indices[-1],
                'explicit acquired start/terminal axis required')
        require(self.terminal_kind == ('public_external_to_attached_boundary' if self.mode == 'external'
                                      else 'original_acquire_to_lift_event'),
                'segment terminal evidence differs from its actual mode')


def derive_acquire(entry, arrays, commands, *, raw_bindings):
    """Derive additive evidence from already verified raw TRAIN inputs.

    ``raw_bindings`` must be the canonical/cache/command path-SHA witnesses.
    The file auditor below verifies them; a caller's in-memory fixture alone
    does not establish production provenance or model readiness.
    """
    require(entry.get('task') in TASKS and entry.get('split') == 'train'
            and entry.get('teacher_used') is True and entry.get('interaction_eligible') is True
            and entry.get('source_kind') == 'TRAIN_native_generated_augmentation',
            'only explicit basketball TRAIN teacher labels are accepted')
    parent = 'train:'+entry['task']+':'+entry['parent_raw_episode_id']
    require(entry.get('parent_source_group') == parent, 'original parent grouping changed')
    require(set(raw_bindings) == {'canonical_receipt', 'cache', 'commands'}
            and all(set(x) == {'path', 'sha256'} and len(x['sha256']) == 64
                    for x in raw_bindings.values()), 'exact raw path/SHA bindings required')
    require(all(name in arrays for name in MASKS), 'every original mask must be preserved')
    before = {name: array_sha(arrays[name]) for name in MASKS}
    t = np.asarray(arrays['times']); n = len(t); bodies = list(map(str, arrays['body_ids']))
    require(t.shape == (n,) and n > 0 and np.isfinite(t).all()
            and np.allclose(np.diff(t), .05, rtol=0., atol=1e-6), 'actual acquisition clock/gap differs')
    require(bodies.count('ball') == 1, 'one persistent public ball binding required')
    j = bodies.index('ball')
    for name in ('mode_valid', 'native_pose_valid'):
        require(arrays[name].shape == (n, len(bodies)) and arrays[name].dtype == np.bool_,
                'explicit original boolean pose/mode masks required')
    require(arrays['robot_contact'].shape == (n,) and arrays['robot_contact'].dtype == np.bool_
            and arrays['teacher_action_valid'].shape == (n,)
            and arrays['teacher_action_valid'].dtype == np.bool_, 'public contact/action masks required')
    require(commands.get('version') == 'r10_TRAIN_committed_waypoint_commands_v1',
            'original command schema missing')
    ledger = commands['commands']; ids = [x['command_id'] for x in ledger]
    require(all(type(i) is int and i >= 0 for i in ids) and len(ids) == len(set(ids))
            and ids == sorted(ids), 'raw command identities repeated or reordered')
    issue_times = np.asarray([x['generated_at'] for x in ledger], float)
    require(np.isfinite(issue_times).all() and np.all(np.diff(issue_times) >= 0),
            'raw command issue clock reordered or nonfinite')
    ac_positions = [i for i, x in enumerate(ledger) if x['raw_phase'] == 'acquire']
    ac = [ledger[i] for i in ac_positions]
    require(not ac_positions or ac_positions == list(range(ac_positions[0], ac_positions[-1]+1)),
            'acquire commands are not one contiguous same-phase run')
    for command in ac:
        require(command.get('reference_body') == command.get('manipulated_body') == 'ball'
                and command.get('reference_role') == command.get('manipulated_role') == 'ball'
                and command.get('reference_kind') == 'translation_only',
                'unknown or changed acquire reference/association')
    events = [x for x in commands['phase_completion_events'] if x['phase'] == 'acquire']
    require(len(events) <= 1, 'duplicate semantic acquire completion evidence')
    result = dict(schema=SCHEMA, task=entry['task'], parent_source_group=parent,
        source_id=entry['source_id'], augmentation_id=entry['augmentation_id'],
        raw_bindings=deepcopy(raw_bindings), raw_acquire_commands=deepcopy(ac),
        original_acquire_completion_events=deepcopy(events), original_masks_sha256=before,
        original_command_success_and_censoring_relabelled=False, raw_arrays_modified=False,
        native_task_outcome_used=False, formal_fitter_acceptance_proven=False,
        production_consumer_enabled=False, mode_segments=[])
    if not events:
        result.update(status='censored_no_original_acquire_completion', physical_acquire_evidence=False)
    else:
        event = events[0]
        require(ac and event['origin_command_id'] == ac[-1]['command_id']
                and event['eligible_command_ids'] == [event['origin_command_id']]
                and event.get('from_phase') == 'acquire' and event.get('to_phase') == 'lift'
                and event.get('evidence') == 'completed_command_then_ordered_teacher_phase_transition'
                and event.get('label_namespace') == 'TRAIN_metric_only'
                and event['attempt_start_time'] == ac[-1]['generated_at']
                and event.get('origin_command_completed') is True
                and event.get('origin_executed_tick_count') == ac[-1]['executed_tick_count']
                and 'command_completed_at' in ac[-1], 'original phase event/last command binding differs')
        start, end = float(ac[0]['generated_at']), float(event['attempt_end_time'])
        require(np.isfinite([start, end]).all() and end > start, 'phase interval is invalid')
        terminal = np.flatnonzero(np.isclose(t, end, rtol=0., atol=1e-8))
        require(len(terminal) == 1, 'phase terminal is not one actual acquired frame')
        terminal = int(terminal[0])
        selected = np.flatnonzero((t >= start-1e-8) & (t < end-1e-8)
                                 & arrays['teacher_action_valid'])
        require(len(selected) and all(str(arrays['teacher_phase'][i]) == 'acquire' for i in selected)
                and set(map(int, arrays['teacher_command_id'][selected])) == {x['command_id'] for x in ac},
                'phase executed frames/commands differ from actual contiguous ledger')
        require(np.allclose(arrays['measured_tcp_xyzw'][terminal],
                            event['current_measured_tcp_xyzw'], rtol=0., atol=1e-8),
                'phase terminal TCP differs from raw acquired label')
        runs = []
        for i in selected:
            key = (int(arrays['modes'][i, j]), bool(arrays['mode_valid'][i, j]),
                   bool(arrays['native_pose_valid'][i, j]))
            if not runs or runs[-1]['key'] != key or int(i) != runs[-1]['frames'][-1]+1:
                runs.append(dict(key=key, frames=[int(i)]))
            else:
                runs[-1]['frames'].append(int(i))
        result.update(phase_start_time=start, phase_end_time=end,
            phase_command_ids=[x['command_id'] for x in ac],
            original_mode_runs=[dict(mode=x['key'][0], mode_valid=x['key'][1],
                native_pose_valid=x['key'][2], frame_indices=x['frames']) for x in runs])
        simple = [x['key'] for x in runs] == [(0, True, True), (1, True, True)]
        has_grasp = bool(simple and arrays['robot_contact'][runs[1]['frames'][0]]
                         and arrays['robot_contact'][runs[1]['frames']].all())
        terminal_valid = bool(arrays['mode_valid'][terminal, j]
                              and arrays['native_pose_valid'][terminal, j]
                              and arrays['modes'][terminal, j] == 1
                              and arrays['robot_contact'][terminal])
        if not simple:
            result.update(status='pending_original_mode_invalid_gap_or_non_simple_path',
                          physical_acquire_evidence=False)
        elif not has_grasp or not terminal_valid:
            result.update(status='pending_missing_public_grasp_or_ball_association',
                          physical_acquire_evidence=False)
        else:
            boundary = runs[1]['frames'][0]
            segments = []
            for mode, run, terminal_frame, terminal_kind in (
                    ('external', runs[0], boundary, 'public_external_to_attached_boundary'),
                    ('attached', runs[1], terminal, 'original_acquire_to_lift_event')):
                frames = tuple(run['frames'])
                command_ids = tuple(dict.fromkeys(map(int, arrays['teacher_command_id'][list(frames)])))
                segments.append(asdict(AcquireModeSegment(mode, frames, command_ids,
                    float(t[frames[0]]), float(t[terminal_frame]), terminal_frame,
                    terminal_kind, event['origin_command_id'])))
            result.update(status='derived_mode_segments_pending_explicit_fitter_interface',
                physical_acquire_evidence=True, mode_segments=segments,
                boundary=dict(frame=boundary, time=float(t[boundary]),
                    target_valid=bool(arrays['target_valid'][boundary, j]),
                    target_twist_valid=bool(arrays['target_twist_valid'][boundary, j]),
                    metric_terminal_is_exogenous_dynamics_target=False))
    require(before == {name: array_sha(arrays[name]) for name in MASKS}, 'derivation mutated original masks')
    return result


def summarize(derived):
    """Distinct original parents, never rows, commands or replicas as support."""
    results = {}
    for task in TASKS:
        rows = [x for x in derived if x['task'] == task]
        ready = [x for x in rows if x['physical_acquire_evidence']]
        results[task] = dict(acquisitions=len(rows), original_parents=len({x['parent_source_group'] for x in rows}),
            statuses=dict(Counter(x['status'] for x in rows)),
            external_parent_sources=sorted({x['parent_source_group'] for x in ready
                if any(s['mode'] == 'external' for s in x['mode_segments'])}),
            attached_parent_sources=sorted({x['parent_source_group'] for x in ready
                if any(s['mode'] == 'attached' for s in x['mode_segments'])}),
            existing_minimum_completed_parents=8, existing_position_cap_m=.03,
            existing_orientation_cap_rad=.15, formal_fitter_acceptance_proven=False)
    return results


def legacy_fitter_interface():
    return dict(compatible=False, legacy_single_command_constraints_preserved=True,
        necessary_additive_interface='TRAINInteractionSource.semantic_acquire_segments: verified AcquireModeSegment tuple plus raw derivation receipt; do not relax TRAINPhaseCompletionEvent.',
        necessary_fitter_change='For acquire only, select verified actual frame_indices and segment terminal snapshot; key each segment by its actual mode. Preserve old command/event/censor ledger as audit evidence, avoid counting legacy and derived copies of the same raw frames twice.',
        necessary_provenance_check='Reload canonical/cache/command SHA and recompute the derived receipt before accepting segments; never accept an arbitrary caller supplied frame list.',
        existing_gate_changes=False, raw_ids_or_arrays_rewritten=False,
        current_legacy_fitter_consumed_derived_sources=False)


def verify_derived_receipt(root, receipt, *, registered, expected_case,
                           predecessor=None, preceding=(), auxiliary_registration=None):
    """Recompute a draft from actual archived-plan raw evidence, fail closed.

    This is a diagnostic verification interface, not production authorization.
    Consumers must retain the source/derivation SHA and explicitly integrate a
    new typed segment branch rather than relaxing the legacy event type.
    """
    from . import acquisition_lineage as lineage
    from .training_view import validate_arrays
    root=Path(root).resolve();files={};bindings=receipt['raw_bindings']
    canonical=lineage.checked_pointer(root,bindings['canonical_receipt'],files)
    record=lineage.read(canonical);entry=record['entry']
    output=Path(lineage.command_argument(record['parent_process_receipt']['command'],'--output')).resolve()
    row=dict(canonical_receipt=bindings['canonical_receipt'],
             parent_receipt=lineage.pointer(root,output.parent/'parent_receipt.json'))
    completion=lineage.verify_completed(root,row,registered,expected_case,files)
    if 'teacher_supplement_execution' in record:
        require(auxiliary_registration is not None, 'auxiliary semantic evidence needs its independent first-execution registration')
        from . import teacher_supplement
        auxiliary=teacher_supplement.verify_registration(root,auxiliary_registration,live=False,require_ready=True)
        require(auxiliary['current']['registration']==registered['registration'], 'auxiliary semantic actual registration differs')
        teacher_supplement.validate_result(root,canonical,auxiliary,expected_case,files=files)
    else:
        require(auxiliary_registration is None, 'original semantic source cannot impersonate auxiliary evidence')
    if 'recovery_input' in registered['plan']['protocol']:
        require(predecessor is not None and preceding, 'recovery-derived evidence needs its actual predecessor failures')
        lineage.verify_recovery_execution(root,completion,registered,predecessor,preceding,files)
    cache=lineage.checked_pointer(root,bindings['cache'],files)
    commands=lineage.checked_pointer(root,bindings['commands'],files)
    require(cache==lineage.local(root,entry['cache']) and bindings['cache']['sha256']==entry['cache_sha256']
            and commands==lineage.local(root,entry['teacher_command_receipt'])
            and entry['teacher_command_receipt']==entry['phase_completion_events_receipt']
            and record['files_sha256'][entry['teacher_command_receipt']]==bindings['commands']['sha256'],
            'derived source bindings differ from the original verified raw receipt')
    with np.load(cache,allow_pickle=False) as stored:
        arrays={key:stored[key].copy() for key in stored.files}
    validate_arrays(arrays,entry,record)
    expected=derive_acquire(entry,arrays,lineage.read(commands),raw_bindings=bindings)
    expected.update(actual_plan_sha256=registered['plan']['plan_sha256'],
                    original_source_archive=registered['registration']['source_snapshot'])
    require(json.loads(json.dumps(receipt,allow_nan=False))==json.loads(json.dumps(expected,allow_nan=False)),
            'derived semantic receipt was altered or belongs to another raw/source identity')
    return dict(verified_for_diagnostic_only=True,raw_evidence_verified=True,receipt=expected,
                segments=tuple(AcquireModeSegment(**{**value,
                    'frame_indices':tuple(value['frame_indices']),
                    'raw_command_ids':tuple(value['raw_command_ids'])}) for value in expected['mode_segments']),
                verified_arrays=arrays,
                validation_context=dict(root=str(root),actual_registration=registered['registration'],
                    expected_case_id=expected_case['case_id'],
                    predecessor_registration=None if predecessor is None else predecessor['registration'],
                    preceding=tuple(preceding),auxiliary_registration=auxiliary_registration),
                verified_files_sha256=files,production_fitter_or_policy_authorized=False)


@dataclass(frozen=True)
class SemanticAcquireRule:
    root: str
    path: str
    sha256: str
    tasks: tuple[str, ...]
    helper: dict
    contexts: dict
    auxiliary_contexts: dict
    evidence_report: dict
    verified_files_sha256: dict


def load_rule(root, pointer):
    """Explicit optional TRAIN rule bound to executable sources and archives."""
    from . import acquisition_lineage as a
    root=Path(root).resolve();files={}
    def checked(value):return a.checked_pointer(root,value,files)
    path=checked(pointer);value=a.read(path)
    require(set(value)=={'schema','TRAIN_only','VAL_used','TEST_used','tasks','source_bindings',
        'actual_registration_list','auxiliary_execution_registrations','evidence_report'}, 'unexpected semantic rule keys')
    require(value['schema']==RULE_SCHEMA and value['TRAIN_only'] is True and value['VAL_used'] is False
        and value['TEST_used'] is False and value['tasks']==list(TASKS), 'unsupported semantic TRAIN rule scope')
    expected={'helper':Path(__file__).resolve(),'fitter':root/'our_mani/multitask/interaction.py',
              'adapter':root/'our_mani/multitask/artifact_fitting.py'}
    require(set(value['source_bindings'])==set(expected), 'exact helper/fitter/adapter source binding required')
    for key,source in expected.items():require(checked(value['source_bindings'][key])==source, 'semantic source owner changed')
    evidence=a.read(checked(value['evidence_report']))
    require(evidence.get('schema')=='r10_original_K1_200_semantic_acquire_CPU_audit_v1'
        and evidence.get('verified_original_teacher_acquisitions')==200
        and evidence.get('all_original_parents_and_raw_events_preserved') is True
        and evidence.get('original_full2800_or_Context_view_changed') is False
        and evidence.get('native_GPU_or_fitting_started') is False,
        'semantic rule needs complete original teacher200 TRAIN raw design evidence')
    require(evidence.get('derivation_module')==value['source_bindings']['helper']
        and evidence.get('actual_registration_list')==value['actual_registration_list'],
        'semantic design evidence belongs to another helper or actual archive roster')
    contexts={}
    for registration in a.read(checked(value['actual_registration_list'])):
        current=a.load_archived_plan(root,registration,files);key=current['plan']['plan_sha256']
        require(key not in contexts, 'duplicate semantic actual plan')
        if 'predecessor_plan_sha256' in registration:
            a.verify_recovery_transition(root,contexts[registration['predecessor_plan_sha256']],current,files)
        contexts[key]=current
    require(contexts, 'actual archived acquisition contexts required')
    auxiliary_contexts={}
    for registration_pointer in value['auxiliary_execution_registrations']:
        from . import teacher_supplement
        checked(registration_pointer)
        auxiliary=teacher_supplement.verify_registration(root,registration_pointer,live=False,require_ready=True)
        proof=teacher_supplement.audit_collection(root,registration_pointer,require_complete=True)
        files.update(proof['verified_files_sha256'])
        current=auxiliary['current'];key=current['plan']['plan_sha256']
        require(key not in contexts and key not in auxiliary_contexts, 'duplicate auxiliary semantic plan')
        auxiliary_contexts[key]=dict(pointer=registration_pointer,context=auxiliary)
    return SemanticAcquireRule(str(root),str(path),pointer['sha256'],TASKS,value['source_bindings']['helper'],
        contexts,auxiliary_contexts,value['evidence_report'],files)


def derive_source(source, case, rule):
    """Optional adapter hook; unknown actual plans never fall back to legacy."""
    from . import acquisition_lineage as a
    from .interaction import bind_verified_semantic_acquire
    require(isinstance(rule,SemanticAcquireRule), 'verified semantic TRAIN rule required')
    if source.task_name not in rule.tasks:
        return source,dict(rule_applied=False)
    root=Path(rule.root);record=case['record'];key=record['plan_sha256'];previous=None;preceding=[];auxptr=None
    if key in rule.contexts:
        current=rule.contexts[key]
        if 'recovery_input' in current['plan']['protocol']:
            previous=rule.contexts[current['registration']['predecessor_plan_sha256']]
            payload=a.read(a.local(root,current['plan']['protocol']['recovery_input']['path']))
            preceding=[dict(actual_plan_sha256=previous['plan']['plan_sha256'],parent_receipt=p)
                for p in payload['failed_parent_receipts'] if a.read(a.local(root,p['path']))['case_id']==record['case_id']]
    elif key in rule.auxiliary_contexts:
        auxiliary=rule.auxiliary_contexts[key];current=auxiliary['context']['current'];auxptr=auxiliary['pointer']
    else:
        raise ValueError('semantic acquire: actual plan is outside the explicitly bound archived TRAIN rule')
    expected=next((c for c in current['plan']['cases'] if c['case_id']==record['case_id']),None)
    require(expected is not None and expected['teacher_used'] is True, 'semantic source is not an actual fixed teacher case')
    bindings=dict(canonical_receipt=a.pointer(root,Path(case['path'])),
        cache=a.pointer(root,a.local(root,case['entry']['cache'])),
        commands=a.pointer(root,a.local(root,case['entry']['teacher_command_receipt'])))
    receipt=derive_acquire(case['entry'],case['arrays'],case['command_receipt'],raw_bindings=bindings)
    receipt.update(actual_plan_sha256=key,original_source_archive=current['registration']['source_snapshot'])
    verified=verify_derived_receipt(root,receipt,registered=current,expected_case=expected,
        predecessor=previous,preceding=preceding,auxiliary_registration=auxptr)
    result=bind_verified_semantic_acquire(source,verified,rule_sha256=rule.sha256,
        rule_pointer=dict(path=rule.path,sha256=rule.sha256))
    evidence=result.semantic_acquire_evidence
    return result,dict(rule_applied=True,rule=dict(path=rule.path,sha256=rule.sha256),
        helper=rule.helper,derivation_schema=SCHEMA,derivation_sha256=evidence.derivation_sha256,
        raw_bindings=evidence.raw_bindings,status=evidence.status,segment_ids=[x.segment_id for x in evidence.segments],
        verified_files_sha256={**rule.verified_files_sha256,**verified['verified_files_sha256']},
        raw_masks_and_events_preserved=True)
