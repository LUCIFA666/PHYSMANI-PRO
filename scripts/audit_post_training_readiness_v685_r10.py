"""Exact192 fixed-four runtime admission for the fresh full16 common RML suite."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release_v531 import policy_fingerprint
from our_mani.multitask.three_method_release16 import measured_motion


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'post_training_qualification_v680_setup'
    out=base/'post_training_qualification_v680/readiness.json'
    if args.freeze and out.exists():raise FileExistsError('preserve exact final readiness')
    plan=json.loads((setup/'plan.json').read_text());routes=json.loads((setup/'routes.json').read_text())
    files=json.loads((setup/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    policy=policy_fingerprint(routes);profile=fingerprint(routes['common_author_RML_profile'])
    roster=validate_roster(json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text()),'train')
    source={(t,r['raw_episode_id']):r for t in TASKS for r in [x for x in roster if x['task']==t][:4]}
    expected={(t,raw,m) for t,raw in source for m in METHODS};rows=[];inputs={};reused_count=0
    jobs=[dict(j,reused=False) for j in plan['planned']]+[dict(j,reused=True) for j in plan['reused']]
    for job in jobs:
        key=(job['task'],job['raw'],job['method']);route=next(r for r in routes['routes'] if (r['task'],r['method'])==(key[0],key[2]))
        if job['reused']:
            path=ROOT/job['case']['path']
            if file_sha(path)!=job['case']['sha256']:raise ValueError('reused actual baseline case changed')
            oldlock=ROOT/job['original_input_lock']['path']
            if file_sha(oldlock)!=job['original_input_lock']['sha256']:raise ValueError('reused actual input lock changed')
            oldfiles=json.loads(oldlock.read_text())['files']
            if any(files.get(k)!=v for k,v in oldfiles.items()):raise ValueError('baseline reuse requires exact original input-map subset')
            if key[2]!='physmani' or key[1]!=next(r['raw_episode_id'] for r in roster if r['task']==key[0]):
                raise ValueError('only unchanged observed baseline first cases may reuse source-scope evidence')
            actual_argv=job['original_argv'];expected_lock=file_sha(oldlock);reused_count+=1
        else:
            actual_argv=job['argv'];path=ROOT/actual_argv[actual_argv.index('--out')+1]/'case.json'
            expected_lock=file_sha(setup/'frozen_inputs.json')
        row=dict(task=key[0],raw=key[1],method=key[2],status='pending')
        error=path.parent/'infrastructure_error.json'
        if error.exists():row.update(status='infrastructure_error',error=json.loads(error.read_text()))
        elif path.exists():
            case=json.loads(path.read_text());validate_completed_case(case,case['identity'])
            if (case.get('cleanup_completed') is not True or case.get('split')!='train'
                or case.get('source_raw_episode_id')!=key[1] or case.get('method')!=key[2]
                or case['identity']['config_sha256']!=route['configuration']['sha256']
                or case['identity']['frozen_inputs_sha256']!=expected_lock
                or case['identity']['source_row_sha256']!=digest(source[key[:2]])
                or case.get('TEST_used_for_fitting') is not False):raise ValueError('case or original source/configuration/lock differs')
            require_attestation(case.get('common_author_RML_parameters',{}),profile)
            motion=case.get('all_stage_counted_joint_motion',{})
            if len(motion.get('events',[]))!=case['counted_clock']['ticks'] or type(measured_motion(case)) is not bool:
                raise ValueError('every actual tick requires the shared measured-motion diagnostics')
            domain=case.get('component_domain',{})
            if (domain.get('purpose')!='train_qualification' or domain.get('task')!=key[0]
                or domain.get('raw_episode_id')!=key[1] or domain.get('method')!=key[2]):raise ValueError('actual TRAIN runtime domain changed')
            if key[2]!='physmani':
                c=json.loads((ROOT/route['configuration']['path']).read_text())
                if (case.get('shared_context_checkpoint')!=c['context_checkpoint']
                    or c['context_checkpoint']['sha256']!=routes['shared_context_checkpoint_sha256']):
                    raise ValueError('all geometry consumers must use the fresh completed shared16 model')
            if job['reused']:
                runtime=job['native_runtime_unchanged'];runtime_path=ROOT/runtime['path']
                if file_sha(runtime_path)!=runtime['sha256'] or route['consumer']!=runtime:
                    raise ValueError('baseline reuse requires the identical frozen native runtime')
                if case['actor_metrics'].get('original_baseline_algorithm_unchanged') is not True:
                    raise ValueError('original neural/WM/RML baseline must remain unchanged')
            inputs[str(path)]=file_sha(path)
            row.update(status='completed',native_success=case['official_task_success'],motion_pass=measured_motion(case),
                common_author_RML_parameters=case['common_author_RML_parameters'],counted_ticks=case['counted_clock']['ticks'],
                case=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path)),
                observed_baseline_reuse_with_input_scope_extension=bool(job['reused']))
        rows.append(row)
    if len(rows)!=192 or {(r['task'],r['raw'],r['method']) for r in rows}!=expected or reused_count!=16:
        raise ValueError('exact192 grid, no duplicates/source replacement')
    complete=all(r['status']=='completed' for r in rows)
    groups=[dict(task=t,method=m,completed=sum(r['status']=='completed' for r in rows if (r['task'],r['method'])==(t,m)),
        native_successes=sum(r.get('native_success',False) for r in rows if (r['task'],r['method'])==(t,m))) for t in TASKS for m in METHODS]
    status=dict(complete=complete,completed=sum(r['status']=='completed' for r in rows),expected=192,
        infrastructure_errors=sum(r['status']=='infrastructure_error' for r in rows),native_successes=sum(r.get('native_success',False) for r in rows))
    if args.freeze:
        if not complete:raise ValueError('all192 clean native fixed-source cases required: '+json.dumps(status))
        for path in [setup/'plan.json',setup/'routes.json',setup/'frozen_inputs.json',Path(__file__)]:inputs[str(path)]=file_sha(path)
        atomic_json(out,dict(schema='r10_exact16_fixed_four_common_author_RML_readiness_v532',
            created_utc=datetime.now(timezone.utc).isoformat(),policy_sha256=policy,
            all16_fixed_four_runtime_ready=True,completed=192,method_case_count=192,source_count=64,task_ready_count=16,
            common_RML_profile_sha256=profile,motion_is_diagnostic_only=True,cases=rows,groups=groups,
            source_sha256=file_sha(__file__),inputs_sha256=inputs,all_failures_retained=True,
            native_success_is_not_an_admission_threshold=True,baseline_observed_parameter_attestations_not_fabricated=True,
            reused_actual_baseline_first_cases=16,remaining_baseline_sources_newly_observed=48,
            old64_baselines_without_parameter_readback_not_promoted=True,TEST_read=False,formal_VAL_TEST_started=False))
        status['readiness']=str(out.relative_to(ROOT))
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
