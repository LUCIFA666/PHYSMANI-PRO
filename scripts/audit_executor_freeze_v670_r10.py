"""Freeze complete first-TRAIN full16 runtime repair evidence, not native scores.

All outcomes remain in the record. This gate does not grant formal VAL/TEST;
new joint16 training and exact final fixed-four admission are still required.
"""
import argparse,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,digest,validate_completed_case
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from our_mani.multitask.common_rml_protocol_v530 import fingerprint,require_attestation
from our_mani.multitask.three_method_release16 import measured_motion
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.rml_case_preflight_v710 import SOURCES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');args=p.parse_args()
    base=ROOT/'artifacts/framework/multitask_r10';setup=base/'rml_suite16_smoke_v677_setup'
    out=base/'rml_executor_freeze_v670/report.json'
    if args.freeze and out.exists():raise FileExistsError('preserve the original executor freeze')
    replacements=[base/'cup_input_repair_v712_setup',base/'remove_time_scene_v706_setup']
    accepted=json.loads((setup/'CPU_acceptance.json').read_text())
    if accepted.get('complete') is not True:raise ValueError('complete actual full16 worker CPU admission required')
    files=json.loads((setup/'frozen_inputs.json').read_text())['files']
    replaced=lambda j:j['method']!='physmani' and j['task'].startswith(('place_cups_on_rotating_frame','remove_cups_from_rotating_frame'))
    selected_jobs=[j for j in accepted['actual_argv_preflights'] if not replaced(j)]
    for replacement in replacements:
        while not (replacement/'CPU_acceptance.json').exists():
            if not args.freeze:raise ValueError('current input/planner repair CPU proof pending')
            if list(replacement.glob('*_CPU_failed.json')):raise ValueError('current repair CPU failures retained; do not freeze')
            if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
            time.sleep(5.)
        receipt=json.loads((replacement/'CPU_acceptance.json').read_text())
        if receipt.get('complete') is not True:raise ValueError('complete current repair CPU proof required')
        for name,sha in json.loads((replacement/'frozen_inputs.json').read_text())['files'].items():
            if name in files and files[name]!=sha:raise ValueError('immutable current repair inputs conflict: '+name)
            files[name]=sha
        selected_jobs.extend(receipt['actual_argv_preflights'])
        if args.freeze:
            while True:
                pending=[]
                for job in receipt['actual_argv_preflights']:
                    folder=ROOT/job['argv'][job['argv'].index('--out')+1]
                    if (folder/'infrastructure_error.json').exists():raise ValueError('current repair native error retained; do not freeze')
                    if not (folder/'case.json').exists():pending.append(str(folder))
                if not pending:break
                if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
                time.sleep(5.)
    verify_inputs(ROOT,files)
    roster=validate_roster(json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text()),'train')
    expected={(t,next(r['raw_episode_id'] for r in roster if r['task']==t),m) for t in TASKS for m in METHODS}
    rows=[];profiles=set();inputs={}
    for job in selected_jobs:
        key=(job['task'],job['raw'],job['method']);argv=job['argv'];folder=ROOT/argv[argv.index('--out')+1]
        result=dict(task=key[0],raw=key[1],method=key[2],status='pending')
        if (folder/'infrastructure_error.json').exists():
            result.update(status='infrastructure_error',error=json.loads((folder/'infrastructure_error.json').read_text()))
        elif (folder/'case.json').exists():
            case_path=folder/'case.json';case=json.loads(case_path.read_text());validate_completed_case(case,case['identity'])
            if (case['task'],case['source_raw_episode_id'],case['method'])!=key:raise ValueError('case/source/method changed')
            if case.get('cleanup_completed') is not True or case.get('component_domain',{}).get('purpose')!='train_qualification':
                raise ValueError('clean actual TRAIN component execution required')
            config=ROOT/argv[argv.index('--config')+1];c=json.loads(config.read_text());profile=fingerprint(c['common_author_RML_profile'])
            require_attestation(case.get('common_author_RML_parameters',{}),profile);profiles.add(profile)
            motion=case.get('all_stage_counted_joint_motion',{})
            if len(motion.get('events',[]))!=case['counted_clock']['ticks'] or type(measured_motion(case)) is not bool:
                raise ValueError('one actual motion measurement per counted tick required')
            if type(case.get('official_task_success')) is not bool or case.get('TEST_used_for_fitting') is not False:
                raise ValueError('native outcome and no TEST fitting required')
            inputs[str(case_path)]=file_sha(case_path);inputs[str(config)]=file_sha(config)
            result.update(status='completed',native_success=case['official_task_success'],motion_pass=measured_motion(case),
                counted_ticks=case['counted_clock']['ticks'],common_author_RML_parameters=case['common_author_RML_parameters'],
                case=dict(path=str(case_path.relative_to(ROOT)),sha256=file_sha(case_path)),
                configuration=dict(path=str(config.relative_to(ROOT)),sha256=file_sha(config)))
        rows.append(result)
    if len(rows)!=48 or {(r['task'],r['raw'],r['method']) for r in rows}!=expected:raise ValueError('exact first-original full16x3 grid required')
    complete=all(r['status']=='completed' for r in rows)
    if len(profiles)>1:raise ValueError('one unchanged author RML profile across all methods')
    status=dict(complete=complete,completed=sum(r['status']=='completed' for r in rows),expected=48,
        infrastructure_errors=sum(r['status']=='infrastructure_error' for r in rows),
        native_successes=sum(r.get('native_success',False) for r in rows),formal_VAL_TEST_started=False)
    if args.freeze:
        known_success={(t,m) for t in ('pick_moving_target_on_the_table','moving_basketball_in_hoop',
            'place_cups_on_rotating_frame','put_rubbish_in_moving_bin') for m in ('cv_geometry','context_geometry')}
        known_success.update({('push_moving_button','context_geometry'),('push_moving_button_high_speed','context_geometry'),
            ('moving_basketball_in_hoop_high_speed','cv_geometry')})
        regressions=[(r['task'],r['method']) for r in rows if (r['task'],r['method']) in known_success and r.get('native_success') is not True]
        if regressions:raise ValueError('known fixed TRAIN functional successes regressed; repair before freeze: '+str(regressions))
        if not complete:raise ValueError('all native TRAIN consumer cases must be clean before executor freeze: '+json.dumps(status))
        for name in (*SOURCES,'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py','scripts/run_task_executor_smoke_rml_inputs_v709_r10.py','scripts/audit_executor_freeze_v670_r10.py'):
            path=ROOT/name;inputs[str(path)]=file_sha(path)
        for path in [d/n for d in [setup,*replacements] for n in ('plan.json','CPU_acceptance.json','frozen_inputs.json')]:
            inputs[str(path)]=file_sha(path)
        report=dict(schema='r10_full16_executor_first_TRAIN_runtime_freeze_v670',created_utc=datetime.now(timezone.utc).isoformat(),
            **status,tasks=list(TASKS),methods=list(METHODS),cases=rows,inputs_sha256=inputs,
            common_RML_profile_sha256=next(iter(profiles)),source_sha256=file_sha(__file__),
            native_success_threshold_applied=False,known_ordinary_cup_input_regression_restored=True,known_fixed_TRAIN_functional_successes_preserved=11,
            superseded_failed_cup_runtime_namespaces=['rml_suite16_smoke_v677','cup_input_repair_v700'],
            current_cup_input_repair_namespace='cup_input_repair_v712',current_remove_time_scene_namespace='remove_time_scene_v706',motion_is_diagnostic_only=True,all_failures_retained=True,
            requested_new_unified_training_permitted=True,new_training_required_before_formal_VAL=True,
            final_fixed_four_192_readiness_granted=False,TEST_read=False,Value_NGFF_FASTER_used=False)
        report['executor_freeze_sha256']=digest(report);atomic_json(out,report)
        status['freeze_report']=str(out.relative_to(ROOT))
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
