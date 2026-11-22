"""Post-exit paired Cube candidate development report; default checks metadata only.

No simulator/model is started. All eight owner waits/Native closures and source
archives are checked before any initial truth snapshot or Native tick is read.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT_PLAN=ROOT/'config/framework/cube_v72_paired_development_owner_20261104_v1/plan.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def pointer(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(item):
    p=Path(item['path']).resolve(strict=True)
    if sha(p)!=item['sha256']:raise ValueError('Frozen pointer changed: '+str(p))
    return p

def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def norm(x):return math.sqrt(sum(float(v)**2 for v in x))
def stats(values):
    a=sorted(float(v) for v in values)
    if any(not math.isfinite(v) for v in a):raise ValueError('Finite diagnostic values required')
    def q(p):
        k=(len(a)-1)*p;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
    return dict(count=len(a),mean=None if not a else sum(a)/len(a),p50=None if not a else q(.5),
                p95=None if not a else q(.95),max=None if not a else a[-1])


def plan_metadata(path,expected_sha256=None):
    path=Path(path).resolve(strict=True)
    if expected_sha256 and sha(path)!=expected_sha256:raise ValueError('External owner plan SHA changed')
    owner=read(path);execution_path=checked(owner['execution_plan']);plan=read(execution_path)
    freeze_path=checked(owner['execution_source_freeze']);freeze=read(freeze_path)
    expected=[(p,c['id']) for p in plan['profiles_order'] for c in plan['cases']]
    if len(expected)!=8 or [(j['profile'],j['case_id'])for j in owner['jobs']]!=expected:
        raise ValueError('Exactly both profiles on the four fixed cases required')
    if plan['formal_VAL_TEST'] is not False:raise ValueError('Development-only plan required')
    return owner,plan,freeze,dict(owner_plan=pointer(path),execution_plan=pointer(execution_path),
                                execution_source_freeze=pointer(freeze_path))


def closed_gate(owner,plan,freeze):
    """Only lifecycle/public result/source files here; no initial snapshot/ticks."""
    root=Path(owner['output_root']);prepared=[];live={};archives=0
    case_specs={c['id']:c for c in plan['cases']}
    for job in owner['jobs']:
        case=Path(job['output']).resolve(strict=True)
        actual_path=root/'owners'/job['id']/'actual_exit.json'
        actual=read(actual_path);worker=read(case/'worker_exit.json')
        if (type(actual.get('actual_process_returncode'))is not int or actual['actual_process_returncode']!=0
            or actual.get('actual_wait_completed')is not True or actual.get('source_unchanged')is not True
            or actual.get('unreaped_descendants')!=[] or actual.get('stop_reason')is not None
            or type(worker.get('exit_code'))is not int or worker['exit_code']!=0
            or worker.get('native_closed')is not True):
            raise ValueError('Case actual closed0/wait/source/nochildren required before GT: '+job['id'])
        run=read(case/'run_inputs.json');result=read(case/'result.json');spec=case_specs[job['case_id']]
        if (result.get('complete')is not True or result.get('source_unchanged')is not True
            or run.get('TEST_read')is not False or run.get('Native_labels_used_by_policy')is not False
            or run.get('paired_case_id')!=job['case_id'] or run.get('candidate_profile')!=job['profile']
            or result.get('native_task')!=spec['task'] or result.get('seed')!=spec['seed']):
            raise ValueError('Matching completed paired case required: '+job['id'])
        sources=run['source_sha256']
        if sources!=freeze['files_sha256']:raise ValueError('Common captured source map changed')
        for name,digest in sources.items():
            p=Path(name)
            if p.is_absolute() or '..'in p.parts:raise ValueError('Repo-relative source path required')
            if name not in live:
                if sha(ROOT/p)!=digest:raise ValueError('Live source changed: '+name)
                live[name]=digest
            if sha(case/'source_snapshot'/p)!=digest:raise ValueError('Archived source changed: '+name)
            archives+=1
        native_root=Path(run['config']['native_repo'])
        for name,digest in run['native_sources_sha256'].items():
            absolute=str(native_root/name)
            if freeze['native_files_sha256'].get(absolute)!=digest or sha(absolute)!=digest:
                raise ValueError('Original task/source/success condition changed: '+absolute)
        cp=plan['candidate_profiles'][job['profile']]
        if result['candidate_transformer']['checkpoint_sha256']!=cp['sha256']:
            raise ValueError('Actual candidate checkpoint identity differs')
        prepared.append(dict(job=job,case=case,spec=spec,actual=actual,result=result,run=run,
            evidence=dict(actual_exit=pointer(actual_path),worker_exit=pointer(case/'worker_exit.json'),
                          result=pointer(case/'result.json'),run_inputs=pointer(case/'run_inputs.json'))))
    return prepared,dict(complete=True,all_eight_actual_closed0_wait_source=True,
        live_repo_source_files=len(live),archived_source_files_checked=archives,
        Native_source_files=len(freeze['native_files_sha256']),GT_opened_before_all_case_gate=False)


def decision_summary(rows,result):
    modes=Counter();reasons=Counter();rejected=Counter();selected=[];executed=[]
    lazy_queries=final_queries=0;qual_walls=[]
    for r in rows:
        d=r['decision'];modes[d['mode']]+=1
        if d.get('reason'):reasons[str(d['reason'])]+=1
        if r.get('action_rejected_reason'):rejected[str(r['action_rejected_reason'])]+=1
        action=d.get('action')
        if isinstance(action,dict):selected.append(dict(time=r['time'],candidate_id=action.get('candidate_id'),mode=d['mode']))
        if r['executed_action']['source']=='exact_selected_modeled_q_and_opening':
            executed.append(dict(time=r['time'],candidate_id=None if not isinstance(action,dict)else action.get('candidate_id'),mode=d['mode']))
        q=d.get('V71_static_collision_query_counts',{})
        lazy_queries+=q.get('lazy_first_action_queries',0);final_queries+=q.get('final_runner_action_queries',0)
        plan=d.get('plan')or{};diag=plan.get('diagnostics')or{}
        for key in ('V71_lazy_free_qualification',):
            value=diag.get(key)or{}
            for attempt in value.get('attempts',[]):
                qualification=attempt.get('qualification')or{}
                costs=[qualification.get('wall_seconds',0.)]+[probe.get('wall_seconds',0.)for probe in qualification.get('collision_probes',[])]
                qual_walls.append(sum(float(v)for v in costs))
    receipt=result['candidate_transformer'];calls=receipt.get('inference_calls')
    if type(calls)is not int or calls<0:raise ValueError('Actual candidate inference counter required')
    return dict(mode_counts=dict(modes),failure_or_unavailability_reasons=dict(reasons),
        action_rejection_reasons=dict(rejected),selected_action_count=len(selected),
        exact_model_action_dispatch_count=len(executed),selected_actions=selected,dispatched_actions=executed,
        candidate_NN_forward_calls=calls,Transformer_used=result['Transformer_used'],
        actual_candidates_generated=calls>0,selected_or_dispatched_is_not_task_success=True,
        candidate_NN_used_for_control=bool(calls and selected),
        candidate_exact_dispatch_observed=bool(calls and executed),
        other_NN_forward_calls=None,other_NN_calls_note='Not inferred from frame counts; no aggregate counter in result',
        lazy_collision_queries=lazy_queries,final_collision_queries=final_queries,
        total_lazy_plus_final_queries=lazy_queries+final_queries,
        excludes_nominal_queries=True,lazy_qualification_wall_seconds=stats(qual_walls),
        cycle_wall_seconds=stats(result['full_cycle_wall_seconds_samples']),
        canary_stop_reason=result['canary_stop_reason'])


def initial_comparison(a,b):
    fields=('object_position','object_quaternion','object_native_linear_velocity','object_native_angular_velocity',
        'tcp_position','tcp_quaternion','q','dq','servo_target_positions','finger_positions','finger_velocities',
        'opening','requested_opening','lift_goal_position')
    differences={};exact=True
    def flat(x):
        if isinstance(x,list):return [v for y in x for v in flat(y)]
        return [float(x)]
    for key in fields:
        x,y=flat(a['state'][key]),flat(b['state'][key])
        if len(x)!=len(y):raise ValueError('Paired initial shape differs')
        d=max((abs(v-w)for v,w in zip(x,y)),default=0.)
        differences[key]=dict(max_absolute_difference=d,exact=x==y);exact&=x==y
    return dict(fields=fields,all_fields_bitwise_equal=exact,
        all_fields_within_1e_8=all(x['max_absolute_difference']<=1e-8 for x in differences.values()),
        differences=differences,full_snapshot_byte_equal=canonical(a)==canonical(b),
        absolute_sim_task_clocks_and_staging_not_expected_byte_equal=True,
        note='Initial matching is checked, never repaired or resampled. Differences limit paired causal interpretation.')


def summarize(owner_plan,output,expected_sha256=None):
    owner,plan,freeze,inputs=plan_metadata(owner_plan,expected_sha256)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh report-only output required')
    prepared,gate=closed_gate(owner,plan,freeze)
    # Only after every completed wait/source/closed gate: open privileged trajectories.
    from our_mani.local_mpc_control_20261103.native_task_evaluation import evaluate_states
    from scripts.evaluate_public_position_closed_cases_20261103 import summarize_public
    from scripts.evaluate_public_position_closed_cases_with_alignment_20261103 import alignment_from_records
    summaries=[];snapshots={};all_inputs=[]
    for item in prepared:
        case=item['case'];job=item['job'];result=item['result']
        snap=read(case/'initial_snapshot.json');ticks=list(lines(case/'native_ticks.jsonl'))
        decisions=list(lines(case/'public_candidate_position_v72_decisions.jsonl'))
        native,_=evaluate_states(snap['state'],ticks)
        public_result=dict(result)
        if public_result.get('public_goal_completion')is None:public_result['public_goal_completion']={}
        public=summarize_public(case,public_result)
        if len(decisions)!=result['cycles'] or len(result['full_cycle_wall_seconds_samples'])!=len(decisions):
            raise ValueError('All decisions and cycle durations required')
        if sum(r['executor']['executed_ticks']for r in decisions)!=len(ticks):raise ValueError('Tick/decision counts differ')
        speed=[norm(t['object_causal_fd_velocity'])for t in ticks if t['object_causal_twist_valid']]
        free_speed=[norm(t['object_causal_fd_velocity'])for t in ticks if t['object_causal_twist_valid'] and not t['held']]
        success=dict(Native_goal_ever=native['native_success_any'],Native_goal_final=native['native_success_final'],
            Native_held_final=native['final']['held'],Native_lifted_hold_final=native['final']['current_lifted_hold_passed'],
            Native_final_goal_AND_final_held=native['native_success_final']and native['final']['held'],
            Native_final_goal_AND_current_lifted_hold=native['native_success_final']and native['final']['current_lifted_hold_passed'],
            Native_ever_goal_AND_current_lifted_hold=native['strict_current_joint_condition']['final'],
            P1_lifted_hold_ever=native['p1_hold_metric']['passed'],
            public_lifted_hold_ever=public['public_lift_hold_ever'],public_lifted_hold_final=public['public_lift_hold_final'],
            public_goal_final=public['public_goal_complete_final'])
        tracepaths=[case/x for x in ('initial_snapshot.json','native_ticks.jsonl','public_candidate_position_v72_decisions.jsonl','public_shadow/public_outputs.jsonl')]
        evidence=dict(item['evidence'],**{str(p.relative_to(case)):pointer(p)for p in tracepaths});all_inputs.extend(evidence.values())
        summaries.append(dict(id=job['id'],case_id=job['case_id'],profile=job['profile'],task=item['spec']['task'],
            speed_group='high_speed'if item['spec']['task'].endswith('_high_speed')else'ordinary',seed=item['spec']['seed'],
            variation=0,declared_seconds=8.,metrics=success,original_native_hold_metrics=native,public=public,
            first_close_alignment=alignment_from_records(snap['state'],ticks,decisions),
            control=decision_summary(decisions,result),actual_speed_m_per_s=stats(speed),actual_free_speed_m_per_s=stats(free_speed),
            actual_elapsed_simulation_seconds=native['final']['time'],actual_ticks=len(ticks),
            owner_elapsed_wall_seconds=(datetime.fromisoformat(item['actual']['finished_utc'])-datetime.fromisoformat(item['actual']['started_utc'])).total_seconds(),
            evidence=evidence))
        snapshots[(job['profile'],job['case_id'])]=snap
    paired=[]
    for case in plan['cases']:
        a=next(r for r in summaries if r['case_id']==case['id']and r['profile']==plan['profiles_order'][0])
        b=next(r for r in summaries if r['case_id']==case['id']and r['profile']==plan['profiles_order'][1])
        paired.append(dict(case_id=case['id'],task=case['task'],seed=case['seed'],
            initial_conditions=initial_comparison(snapshots[(a['profile'],case['id'])],snapshots[(b['profile'],case['id'])]),
            baseline_metrics=a['metrics'],new_metrics=b['metrics'],
            baseline_first_close=a['first_close_alignment'],new_first_close=b['first_close_alignment']))
    aggregate={}
    for profile in plan['profiles_order']:
        aggregate[profile]={}
        for group in ('ordinary','high_speed','all'):
            rows=[r for r in summaries if r['profile']==profile and(group=='all'or r['speed_group']==group)]
            aggregate[profile][group]=dict(case_denominator=len(rows),counts={k:sum(r['metrics'][k]for r in rows)for k in rows[0]['metrics']},
                candidate_NN_forward_calls=sum(r['control']['candidate_NN_forward_calls']for r in rows),
                total_lazy_plus_final_queries=sum(r['control']['total_lazy_plus_final_queries']for r in rows),
                modes=dict(sum((Counter(r['control']['mode_counts'])for r in rows),Counter())))
    report=dict(schema='Cube_candidate_paired_generated_TRAIN_development_posthoc_20261104_v1',complete=True,
        inputs=inputs,source_audit=gate,cases=summaries,paired=paired,aggregate=aggregate,
        definitions=dict(Native_goal='Original Task.success Boolean from original conditions; no centre-distance replacement',
            final_goal_AND_current_lifted_hold='Final Native task flag AND current original P1 .5s/.04m/.02m hold',
            final_goal_AND_final_held='Final Native task flag AND current Native attachment; separate from lifted duration',
            public_unknown='Public proof unavailable; never relabelled as physical drop',
            drops='Native held True-to-False, including intentional reopen; not force-sensor slipping',
            initial_conditions='Generated same-seed scenes compared, not HDF scene replay; differences reported without replacement'),
        development_only=True,formal_VAL_TEST=False,task_success_rate_generalization_claimed=False,
        GT_used_only_posthoc=True,Native_started=False,NN_forward_calls=0,GPU_used=False,training_started=False,
        script=pointer(__file__),original_native_judge=pointer(ROOT/'our_mani/local_mpc_control_20261103/native_task_evaluation.py'),
        original_hold_implementation=pointer(ROOT/'our_mani/local_mpc_20261102/records.py'))
    for item in all_inputs:
        if sha(item['path'])!=item['sha256']:raise ValueError('Completed input changed during report')
    out.mkdir(parents=True,exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    receipt=dict(complete=True,report=pointer(out/'report.json'),script=pointer(__file__),inputs=inputs,
        all_eight_cases_retained=True,original_sources_modified=False,NN_forward_calls=0,Native_started=False,TEST_read=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(report=pointer(out/'report.json'),receipt=pointer(out/'receipt.json'),aggregate=aggregate,paired=paired)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--expected-sha256');p.add_argument('--output',type=Path);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if not a.evaluate:
        owner,plan,_,inputs=plan_metadata(a.owner_plan,a.expected_sha256)
        print(json.dumps(dict(ready_to_attempt_after_all_eight_closed=True,actual_results_read=False,GT_read=False,
            cases=len(owner['jobs']),profiles=plan['profiles_order'],inputs=inputs,Native_started=False,NN_forward_calls=0)));return 0
    if a.output is None:p.error('--evaluate requires fresh --output')
    print(json.dumps(summarize(a.owner_plan,a.output,a.expected_sha256),indent=2,allow_nan=False));return 0

if __name__=='__main__':raise SystemExit(main())
