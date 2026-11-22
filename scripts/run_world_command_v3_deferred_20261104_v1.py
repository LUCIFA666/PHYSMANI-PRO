"""Default source-only owner; Root later executes the fixed deferred BC chain.

Only actual child waits are asserted. Root13223 parent wait is never inferred.
Children own the correct global resource locks; this parent owns only its
family lock and never passes a resource-lock bypass. No retries/resume/TEST.
"""
import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'scripts/run_large_network_training_continuation_20261104_v1.py'
spec=importlib.util.spec_from_file_location('world_v3_frozen_lifecycle',HELPER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
SCHEMA='world_command_v3_deferred_owner_20261104_v1'
DEFAULT_PLAN=ROOT/'config/framework/world_command_v3_deferred_20261104_v1/plan.json'
LOCK=ROOT/'runtime/framework/world_command_v3_deferred_20261104_v1.lock'
STAGES=('world_GPU_canary4','world_GPU_public_cache','world_CPU_join_commands',
        'world_GPU_fit5','world_GPU_selection_predict5168','world_CPU_selection_score5168')
read=base.read;sha=base.sha;pointer=base.pointer;checked_pointer=base.checked_pointer;write_new=base.write_new
_ORIGINAL_STOP_PATHS=base.stop_paths


def stop_paths(plan):
    # Extend only this imported helper instance; the frozen helper file and
    # other owners' module instances are untouched. Its actual wait/cleanup
    # loop now sees this new family's STOP during a running child too.
    return tuple(_ORIGINAL_STOP_PATHS(plan))+(DEFAULT_PLAN.parent/'STOP',)


base.stop_paths=stop_paths


def prior_helper(plan):
    path=checked_pointer(plan['e2_owner_source'])
    spec=importlib.util.spec_from_file_location('world_frozen_e2_deferred_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_plan(path,expected_sha=None):
    path=Path(path).resolve(strict=True)
    if expected_sha is not None and sha(path)!=expected_sha:raise ValueError('External exact Root plan SHA required')
    plan=read(path)
    if (plan.get('schema')!=SCHEMA or tuple(plan.get('stages',()))!=STAGES
            or plan.get('component_only')is not True or plan.get('all_task_Native_VAL_TEST_authorized')is not False
            or plan.get('child_owns_global_resource_locks')is not True or plan.get('automatic_OOM_fallback')is not False):
        raise ValueError('Exact component-only six-stage child-owned/no-retry protocol required')
    for key in ('e2_owner_source','e2_plan','world_method_sources','world_config','subset','method_sources'):
        checked_pointer(plan[key])
    if sha(HELPER)!=plan['lifecycle_helper']['sha256']or str(HELPER)!=plan['lifecycle_helper']['path']:
        raise ValueError('Exact original actual-wait/cleanup helper required')
    for key in ('python','output_root','e2_output_root','CF_output_root'):
        if not isinstance(plan.get(key),str)or not Path(plan[key]).is_absolute():raise ValueError('Explicit absolute owner paths required')
    if set(plan['stage_timeout_seconds'])!=set(STAGES):raise ValueError('Every stage needs a bounded timeout')
    for value in (plan['wait_timeout_seconds'],plan['poll_seconds'],plan['stop_grace_seconds'],*plan['stage_timeout_seconds'].values()):
        if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:
            raise ValueError('Finite positive lifecycle timing bounds required')
    world=read(checked_pointer(plan['world_method_sources']))
    world=world.get('source_sha256',world)
    if len(world)!=48 or any(sha(p)!=s for p,s in world.items()):raise ValueError('Final exact world48 method source required')
    if plan['subset']['sha256']!='eeb22774a62e548e45353b00fc17900bb56e6d438dc24e02c0fea4954ac06212':
        raise ValueError('Immutable fixed all-parent stage16 subset required')
    helper=prior_helper(plan);prior=helper.load_plan(checked_pointer(plan['e2_plan']),plan['e2_plan']['sha256'])
    if prior['output_root']!=plan['e2_output_root']or prior['CF_output_root']!=plan['CF_output_root']:
        raise ValueError('Exact prior e2/CF owner identity required')
    plan['_config_path']=str(path);plan['_config_sha256']=sha(path)
    base.verify_sources(plan)
    return plan


def assert_running(plan):
    base.assert_running(plan)
    if(DEFAULT_PLAN.parent/'STOP').exists():raise InterruptedError('World deferred family STOP')


def wait_prior(plan):
    root=Path(plan['e2_output_root']);start=time.monotonic()
    while True:
        assert_running(plan);base.verify_sources(plan)
        if(root/'failure.json').exists():raise ValueError('e2 failed; preserve failure, no automatic retry or successor')
        if(root/'completion.json').is_file()and(root/'terminal_status.json').is_file():return
        if time.monotonic()-start>plan['wait_timeout_seconds']:raise TimeoutError('Waiting for actual e2 completion/terminal')
        time.sleep(plan['poll_seconds'])


def validate_prior(plan):
    helper=prior_helper(plan);prior=helper.load_plan(checked_pointer(plan['e2_plan']),plan['e2_plan']['sha256'])
    facts=helper.validate_prior(prior)
    root=Path(plan['e2_output_root']);completion=read(root/'completion.json');terminal=read(root/'terminal_status.json')
    for value in (completion,terminal):
        if(value.get('complete')is not True or value.get('completed_stages')!=list(helper.STAGES)
                or value.get('current_network_end_to_end_ready')is not False or value.get('Native_VAL_TEST_started')is not False):
            raise ValueError('All three completed e2 stages, no Native/runtime qualification required')
    if {k:v for k,v in completion.items()if k!='finished_utc'}!={k:v for k,v in terminal.items()if k!='finished_utc'}:
        raise ValueError('Actual e2 completion and terminal status disagree')
    request=read(root/'request.json')
    if request.get('plan')!=plan['e2_plan']or request.get('source_map')!=prior['method_sources']:
        raise ValueError('Actual prior owner request differs from frozen e2 plan/source map')
    expected_sources=helper.verify_sources(prior)
    rows=[json.loads(line)for line in(root/'stage_ledger.jsonl').read_text().splitlines()if line]
    owners=[];releases=[]
    if len(rows)!=3 or [x.get('stage')for x in rows]!=list(helper.STAGES):raise ValueError('Exact three-stage e2 ledger required')
    for stage,row in zip(helper.STAGES,rows):
        op=root/'owners'/stage/'actual_exit.json';owner=read(op);helper.exact_zero_wait(owner,stage=stage)
        if row!=owner:raise ValueError('e2 actual owner and ledger disagree')
        child=read(root/'owners'/stage/'request.json')
        if child.get('source_sha256')!=expected_sources or child.get('driver_holds_resource_locks')is not True:
            raise ValueError('Actual source-bound e2 stage ownership required')
        rp=root/'resource_ownership'/stage/'released.json';release=read(rp)
        if(release.get('locks_released')is not True or release.get('actual_child_wait_completed')is not True
                or release.get('unreaped_descendants')!=[]or release.get('stage_actual_exit')!=pointer(op)):
            raise ValueError('e2 correct global lock release and actual stage wait proof required')
        owners.append(pointer(op));releases.append(pointer(rp))
    fit=root/'artifacts/fit10';trained=read(fit/'completion.json')
    if(trained.get('complete')is not True or trained.get('source_unchanged')is not True
            or trained.get('fitted_epochs')!=10 or trained.get('optimizer_steps')!=1930
            or trained.get('source_sha256')!=read(checked_pointer(prior['v2_method_sources']))
            or trained.get('selected_model')!=completion.get('selected_model')):
        raise ValueError('Actual complete1930-step10epoch source-bound e2 selected artifact required')
    selected=checked_pointer(trained['selected_model'])
    if selected!=fit/'selected.pt':raise ValueError('Actual selected e2 sibling required, no arbitrary checkpoint')
    evaluated=read(root/'artifacts/selected646_evaluation/completion.json')
    metrics=read(checked_pointer(evaluated['metrics']))
    if(evaluated.get('complete')is not True or evaluated.get('source_unchanged')is not True
            or evaluated.get('selected_model')!=pointer(selected)or metrics.get('windows')!=646
            or len(metrics.get('per_task',{}))!=16):raise ValueError('Complete16task646 e2 selected evaluation required; no quality threshold invented')
    # Check that the exact shared inodes are currently free, then release them.
    # Children later acquire their own locks. The parent never holds/bypasses
    # them during waiting or child work.
    streams=[]
    try:
        for path in helper.resource_paths():
            stream=path.open('a+');streams.append(stream);fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        for stream in reversed(streams):fcntl.flock(stream,fcntl.LOCK_UN);stream.close()
    return dict(prior_facts=facts,e2_completion=pointer(root/'completion.json'),e2_terminal=pointer(root/'terminal_status.json'),
        e2_ledger=pointer(root/'stage_ledger.jsonl'),e2_actual_stage_owners=owners,e2_released_locks=releases,
        e2_fit_completion=pointer(fit/'completion.json'),selected_encoder=pointer(selected),
        e2_selection_evaluation=pointer(root/'artifacts/selected646_evaluation/completion.json'),
        Root13223_parent_wait_claimed=False,current_network_end_to_end_ready=False)


def build_authorization(plan,facts):
    f=facts['prior_facts'];owners=f['actual_stage_owners']+f['prior_training_owners']+facts['e2_actual_stage_owners']
    owners=list({x['path']:x for x in owners}.values())
    return dict(schema='world_command_fine_v3_GPU_authorization_20261104_v1',execution_authorized=True,
        source_sha256=read(checked_pointer(plan['world_method_sources'])).get('source_sha256',read(checked_pointer(plan['world_method_sources']))),
        config=plan['world_config'],subset=plan['subset'],encoder_checkpoint=facts['selected_encoder'],encoder_family='semantic_e207M_v2',
        native_lifecycle=dict(all_native_children_reaped=True,owners=f['CF_native_owners'],results=f['CF_native_results']),
        prior_training_lifecycle=dict(all_training_children_reaped=True,owners=owners),
        allow_first4_fit_canary_command_labels=True,all_task_Native_VAL_TEST_authorized=False,
        Root13223_parent_wait_claimed=False,current_network_end_to_end_ready=False,
        authorization_basis='Root executes exact externalSHA deferred plan after actual child waits/source/releases; no inferred Root parent wait',
        deferred_plan=dict(path=plan['_config_path'],sha256=plan['_config_sha256']))


def execute(path,expected_sha):
    plan=load_plan(path,expected_sha);out=Path(plan['output_root'])
    with base.driver_lock(LOCK):
        if out.exists():raise FileExistsError('Fresh owner only; no retry/resume/overwrite')
        assert_running(plan);out.mkdir(parents=True,exist_ok=False);(out/'owners').mkdir();(out/'artifacts').mkdir()
        base.enable_subreaper();write_new(out/'request.json',dict(plan=pointer(path),source_map=plan['method_sources'],child_owns_global_resource_locks=True))
        previous={s:signal.signal(s,lambda n,f:base.SIGNAL_REQUEST.update(signal=n))for s in (signal.SIGINT,signal.SIGTERM)}
        status=dict(complete=False,completed_stages=[],current_network_end_to_end_ready=False,Native_VAL_TEST_started=False,Root13223_parent_wait_claimed=False)
        try:
            wait_prior(plan);facts=validate_prior(plan);write_new(out/'prior_chain_validation.json',facts)
            auth=out/'Root_GPU_authorization.json';write_new(auth,build_authorization(plan,facts));ap=pointer(auth)
            enc=facts['selected_encoder'];artifact=out/'artifacts';cacheCLI='scripts/cache_world_tcp_command_warmstart_20261104_v3.py'
            shared=['--subset',plan['subset']['path'],'--encoder-checkpoint',enc['path'],'--encoder-sha256',enc['sha256'],
                '--root-authorization',str(auth),'--root-authorization-sha256',ap['sha256']]
            def run(stage,args,*,cpu=False):
                assert_running(plan);checked_pointer(ap)
                value=base.run_stage(plan,stage,args,cpu=cpu)
                prior_helper(plan).exact_zero_wait(value,stage=stage)
                status['completed_stages'].append(stage);base.append(out/'stage_ledger.jsonl',value)
            canary=artifact/'GPU_canary4'
            run(STAGES[0],[cacheCLI,'--gpu-canary',*shared,'--microbatch','4','--output',str(canary)])
            result=read(canary/'canary.json')
            if any(result.get(k)is not True for k in ('complete','passed','source_unchanged'))or result.get('microbatch')!=4 or result.get('candidate_optimizer_steps')!=1 or result.get('canary_weights_reused')is not False:
                raise ValueError('Actual successful discarded-micro4 canary required; no automatic OOM retry')
            public=artifact/'public_cache'
            run(STAGES[1],[cacheCLI,'--build-cache',*shared,'--preflight-report',str(canary/'canary.json'),
                '--preflight-sha256',sha(canary/'canary.json'),'--output',str(public)])
            freeze=read(public/'public_feature_freeze.json')
            if freeze.get('complete')is not True or freeze.get('public_features_frozen')is not True:raise ValueError('Whole public cache freeze required before labels')
            joined=artifact/'joined_commands'
            run(STAGES[2],[cacheCLI,'--join-labels','--public-cache-manifest',str(public/'cache_manifest.json'),'--output',str(joined)],cpu=True)
            jm=joined/'joined_manifest.json';j=read(jm)
            if j.get('complete')is not True or j.get('command_target_join_after_whole_public_freeze')is not True or j.get('role_counts')!={'fit':24592,'selection':5168}:raise ValueError('Complete fixed29760 target join required')
            trained=artifact/'fit5'
            run(STAGES[3],['scripts/train_world_tcp_command_warmstart_20261104_v3.py','--train','--cache-manifest',str(jm),
                '--output',str(trained),'--preflight-report',str(canary/'canary.json'),'--preflight-sha256',sha(canary/'canary.json'),
                '--root-authorization',str(auth),'--root-authorization-sha256',ap['sha256']])
            fit=read(trained/'fit_status.json')
            if fit.get('training_complete')is not True or fit.get('epochs_completed')!=5 or fit.get('actual_optimizer_steps')!=7685 or fit.get('source_unchanged')is not True:raise ValueError('Actual full5epoch7685step fit required')
            selected=checked_pointer(fit['selected_checkpoint']);prediction=artifact/'selected5168_evaluation'
            run(STAGES[4],['scripts/evaluate_world_tcp_command_warmstart_20261104_v3.py','--predict','--cache-manifest',str(jm),
                '--selected',str(selected),'--expected-sha256',sha(selected),'--device','cuda:0','--output',str(prediction)])
            frozen=read(prediction/'prediction_freeze.json')
            if frozen.get('complete')is not True or frozen.get('rows')!=5168:raise ValueError('Complete selected5168 numeric prediction freeze required')
            run(STAGES[5],['scripts/evaluate_world_tcp_command_warmstart_20261104_v3.py','--score','--output',str(prediction)],cpu=True)
            report=read(prediction/'report.json')
            if report.get('rows')!=5168 or len(report.get('per_task',{}))!=16 or report.get('runtime_or_control_admitted')is not False:raise ValueError('Fixed16task5168 command metrics only required')
            status.update(complete=True,selected_model=pointer(selected),selection_report=pointer(prediction/'report.json'),
                joined_manifest=pointer(jm),public_cache=pointer(public/'cache_manifest.json'))
            write_new(out/'completion.json',status);return status
        except BaseException as error:
            status.update(error_type=type(error).__name__,error=str(error));write_new(out/'failure.json',status);raise
        finally:
            write_new(out/'terminal_status.json',dict(status,finished_utc=base.now()))
            for s,handler in previous.items():signal.signal(s,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--plan-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if a.execute:
        if not a.plan_sha256:p.error('Root --execute requires exact external --plan-sha256')
        value=execute(a.plan,a.plan_sha256)
    else:
        plan=load_plan(a.plan,a.plan_sha256)
        value=dict(complete=True,preparation_only=True,plan=pointer(a.plan),stages=list(STAGES),
            waiting_for=str(Path(plan['e2_output_root'])/'completion.json'),child_processes_started=0,
            GPU=False,Native=False,model_forward_calls=0,resource_locks_acquired=False,Root13223_parent_wait_claimed=False)
    print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
