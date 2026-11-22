"""All100 original bin TRAIN full curves in source-exact intention coordinates."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,interaction
from our_mani.multitask import command_goal_reference as goal,bin_command_goal_reference as bin_goal


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(manifest_path,configuration_path,permission_path,output):
    begun=time.perf_counter();output=a.local(ROOT,output);a.require(not output.exists(),'fresh draft output required');output.mkdir()
    files={};manifest=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,manifest_path)),files))
    a.require(manifest['task']==bin_goal.TASK and len(manifest['case_proofs'])==100,'all100 original teachers required')
    configuration=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,configuration_path)),files))
    config=interaction.InteractionFitConfig(**configuration['interaction'])
    base=goal.load_rule(ROOT,configuration['command_goal_reference_rule'],purpose='production_fitting')
    permission=bin_goal.load_permission(ROOT,a.pointer(ROOT,a.local(ROOT,permission_path)),purpose='CPU_diagnostic')
    sources=[];evidence=[];proofs={};opening=closed=frames=0;code={str(Path(m.__file__).resolve()):a.file_sha(m.__file__)
        for m in (adapter,interaction,goal,bin_goal)}
    for row in manifest['case_proofs']:
        path=a.checked_pointer(ROOT,row['canonical_receipt'],files);record=a.read(path)
        with np.load(a.checked_pointer(ROOT,row['cache'],files),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        command=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
        case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=command,
            command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
        source,_=adapter.interaction_source(case);proof=bin_goal.derive_source(case,source,base,permission)
        sources.append(source);evidence.append(proof);proofs[source.source_id]=bin_goal.evidence_proof(proof)
        files.update(proof.derivation_proof['verified_files_sha256']);opening+=proof.derivation_proof['opening_forecast_commands']
        closed+=proof.derivation_proof['closed_issue_commands'];frames+=proof.derivation_proof['actual_bin_frames']
    proof_path=output/'TRAIN_derivation_proof_before_fit.json'
    write(proof_path,dict(schema=bin_goal.SCHEMA,permission=permission.pointer,all_original100_sources=proofs,
        original100_parents_preserved=True,all_opening_commands=opening,all_closed_commands=closed,all_actual_frames=frames,
        source_sha256=code,production_enabled=False))
    proof_sha=a.file_sha(proof_path);result=goal.fit_diagnostic(sources,evidence,config)
    a.require(a.file_sha(proof_path)==proof_sha,'derivation source proof changed after fit')
    old=a.read(a.local(ROOT,Path(manifest_path).parent/'real_production_fit_report.json'))
    old_deposit=next(v for k,v in old['phase_fits'].items() if k.startswith('deposit|'))
    new_deposit=next(v for k,v in result['phase_fits'].items() if k.startswith('deposit|'))
    tail={k:v for k,v in old_deposit['terminal_parent_residuals'].items() if v['position']>.03}
    a.require(len(tail)==5 and set(tail)<=set(new_deposit['terminal_parent_residuals']),
        'all five original tail parents must remain in the actual fit')
    for key,value in result['phase_fits'].items():
        if key.startswith('deposit|'):continue
        prior=old['phase_fits'][key]
        for name in ('selected_lag','position_tolerance_m','orientation_tolerance_rad'):
            a.require(abs(value[name]-prior[name])<1e-12,'unaffected production phase changed')
    for name,sha in code.items():a.require(a.file_sha(name)==sha,'draft helper/core source changed during fit')
    details=output/'real_full_curve_fit_report.json';write(details,result)
    report=dict(schema='r10_bin_exact_intention_all100_full_curve_draft_fit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun,permission=permission.pointer,original_manifest=a.pointer(ROOT,a.local(ROOT,manifest_path)),
        numeric_configuration=a.pointer(ROOT,a.local(ROOT,configuration_path)),derivation_proof_frozen_before_fit=a.pointer(ROOT,proof_path),
        detailed_fit=a.pointer(ROOT,details),all_original_teacher_sources=100,all_opening_commands=opening,
        all_closed_current_issue_commands=closed,all_original_bin_frames=frames,
        original_deposit_completed_parents=len(old_deposit['terminal_parent_residuals']),
        new_deposit_completed_parents=len(new_deposit['terminal_parent_residuals']),
        original_tail5={k:dict(original_actual_metric_crossfit=v,new_intention_tracking_crossfit=new_deposit['terminal_parent_residuals'][k])
            for k,v in tail.items()},all_required_phases_pass=result['all_required_phases_and_observed_partitions_passed'],
        deposit_summary={k:new_deposit.get(k) for k in ('passed','completed_parent_count','selected_lag','grouped_TRAIN_CV_error',
            'position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')},
        all_failed_censored_commands_and_actual_metrics_preserved=True,original_current_metric_verifier_calibrated=False,
        numerical_gates_unchanged_8_03_15=True,historical_execution_source_closure_claimed=False,
        production_InteractionLibrary_exported=False,production_core_modified=False,native_GPU_or_collection_started=False,
        source_sha256=code,verified_files_sha256=files)
    p=output/'report.json';write(p,report)
    return dict(report=a.pointer(ROOT,p),passed=report['all_required_phases_pass'],deposit=report['deposit_summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--fit-config',required=True)
    p.add_argument('--permission',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    print(json.dumps(run(args.manifest,args.fit_config,args.permission,args.out),ensure_ascii=False))
