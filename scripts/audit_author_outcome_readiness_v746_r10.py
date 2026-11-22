"""Independently admit all192 new fixed TRAIN cases under author sampling."""
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, digest, validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS, METHODS, validate_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint, require_attestation
from our_mani.multitask.three_method_release_v531 import policy_fingerprint
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding
from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    base = ROOT/'artifacts/framework/multitask_r10'
    setup = base/'author_outcome_qualification_v745_setup'
    out = base/'author_outcome_qualification_v745/readiness.json'
    if args.freeze and out.exists():
        raise FileExistsError('preserve the independently frozen readiness')
    plan = json.loads((setup/'plan.json').read_text())
    routes = json.loads((setup/'routes.json').read_text())
    files = json.loads((setup/'frozen_inputs.json').read_text())['files']
    verify_inputs(ROOT, files)
    protocol = outcome_protocol_binding(ROOT)
    if plan.get('reused') != [] or len(plan['planned']) != 192 or routes.get('author_outcome_protocol') != protocol:
        raise ValueError('all192 new runtime cases must bind the corrected protocol')
    policy = policy_fingerprint(routes)
    profile = fingerprint(routes['common_author_RML_profile'])
    roster = validate_roster(json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text()), 'train')
    sources = {(t, r['raw_episode_id']): r for t in TASKS
        for r in [x for x in roster if x['task'] == t][:4]}
    expected = {(t, raw, method) for t, raw in sources for method in METHODS}
    rows, inputs = [], {}
    for job in plan['planned']:
        key = (job['task'], job['raw'], job['method'])
        route = next(r for r in routes['routes'] if (r['task'], r['method']) == (key[0], key[2]))
        argv = job['argv']
        if argv[2] != route['worker']['path']:
            raise ValueError('actual case worker differs from the new frozen physical route')
        path = ROOT/argv[argv.index('--out')+1]/'case.json'
        row = dict(task=key[0], raw=key[1], method=key[2], status='pending')
        error = path.parent/'infrastructure_error.json'
        if error.exists():
            row.update(status='infrastructure_error', error=json.loads(error.read_text()))
        elif path.exists():
            case = json.loads(path.read_text())
            validate_completed_case(case, case['identity'])
            identity = case['identity']
            if (case.get('cleanup_completed') is not True or case.get('split') != 'train'
                    or case.get('source_raw_episode_id') != key[1] or case.get('method') != key[2]
                    or identity['config_sha256'] != route['configuration']['sha256']
                    or identity['frozen_inputs_sha256'] != file_sha(setup/'frozen_inputs.json')
                    or identity['source_row_sha256'] != digest(sources[key[:2]])
                    or case.get('TEST_used_for_fitting') is not False):
                raise ValueError('native source, configuration or frozen scope differs')
            domain = case.get('component_domain', {})
            if (domain.get('purpose') != 'train_qualification' or domain.get('task') != key[0]
                    or domain.get('raw_episode_id') != key[1] or domain.get('method') != key[2]):
                raise ValueError('actual TRAIN component domain differs')
            require_attestation(case.get('common_author_RML_parameters', {}), profile)
            require_sampling_case(case, protocol, block=key[0] in BLOCKS)
            if (len(case.get('all_stage_counted_joint_motion', {}).get('events', [])) != case['counted_clock']['ticks']
                    or type(measured_motion(case)) is not bool):
                raise ValueError('every actual tick requires the common motion diagnostic')
            if key[2] != 'physmani':
                config = json.loads((ROOT/route['configuration']['path']).read_text())
                if (case.get('shared_context_checkpoint') != config['context_checkpoint']
                        or config['context_checkpoint']['sha256'] != routes['shared_context_checkpoint_sha256']):
                    raise ValueError('every geometry consumer must retain the completed joint16 model')
            inputs[str(path)] = file_sha(path)
            row.update(status='completed', native_success=case['official_task_success'],
                motion_pass=measured_motion(case), common_author_RML_parameters=case['common_author_RML_parameters'],
                counted_ticks=case['counted_clock']['ticks'],
                case=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path)),
                author_outcome_protocol=protocol, old_protocol_case_reused=False)
        rows.append(row)
    keys = [(r['task'], r['raw'], r['method']) for r in rows]
    if len(keys) != 192 or len(set(keys)) != 192 or set(keys) != expected:
        raise ValueError('exact first-four source grid required without replacements')
    complete = all(r['status'] == 'completed' for r in rows)
    groups = [dict(task=t, method=m,
        completed=sum(r['status'] == 'completed' for r in rows if (r['task'], r['method']) == (t, m)),
        native_successes=sum(r.get('native_success', False) for r in rows if (r['task'], r['method']) == (t, m)))
        for t in TASKS for m in METHODS]
    status = dict(complete=complete, completed=sum(r['status'] == 'completed' for r in rows), expected=192,
        infrastructure_errors=sum(r['status'] == 'infrastructure_error' for r in rows),
        native_successes=sum(r.get('native_success', False) for r in rows))
    if args.freeze:
        if not complete:
            raise ValueError('all192 clean fixed-source cases required: '+json.dumps(status))
        for path in [setup/'plan.json', setup/'routes.json', setup/'frozen_inputs.json', Path(__file__)]:
            inputs[str(path)] = file_sha(path)
        atomic_json(out, dict(schema='r10_exact16_fixed_four_common_author_RML_readiness_v532',
            created_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=policy,
            all16_fixed_four_runtime_ready=True, completed=192, method_case_count=192,
            source_count=64, task_ready_count=16, common_RML_profile_sha256=profile,
            motion_is_diagnostic_only=True, cases=rows, groups=groups, source_sha256=file_sha(__file__),
            inputs_sha256=inputs, all_failures_retained=True, native_success_is_not_an_admission_threshold=True,
            baseline_observed_parameter_attestations_not_fabricated=True, reused_actual_baseline_first_cases=0,
            newly_observed_baseline_cases=64, old_protocol_case_reuse=False,
            author_outcome_protocol=protocol, TEST_read=False, formal_VAL_TEST_started=False))
        status['readiness'] = str(out.relative_to(ROOT))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
