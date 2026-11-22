"""Rebuild actual goal-source proofs through the production release consumer."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,release_preflight as release
from our_mani.multitask import command_goal_reference as goal,interaction,artifact_fitting as adapter


def run(report_path,output):
    begun=time.perf_counter();output=a.local(ROOT,output)
    a.require(not output.exists(),'fresh actual consumer audit namespace required');output.mkdir()
    report=a.read(a.local(ROOT,report_path));config=a.read(a.local(ROOT,report['configuration']['path']))
    rule=goal.load_rule(ROOT,config['command_goal_reference_rule'],purpose='production_fitting')
    cases=[];proofs={};files={};code={str(Path(m.__file__).resolve()):a.file_sha(m.__file__) for m in (release,goal,interaction,adapter)}
    for task,row in report['tasks'].items():
        if task not in rule.tasks:continue
        manifest=a.read(a.checked_pointer(ROOT,row['fit_manifest'],files));proofs.update(manifest['command_goal_reference_derivation']['per_source'])
        for source in manifest['case_proofs']:
            path=a.checked_pointer(ROOT,source['canonical_receipt'],files);record=a.read(path)
            cache=a.checked_pointer(ROOT,source['cache'],files)
            with np.load(cache,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
            commands=a.read(a.checked_pointer(ROOT,source['command_receipt'],files))
            cases.append(dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=commands,
                command_receipt_verified=True,command_receipt_sha256=source['command_receipt']['sha256']))
    manifest=dict(command_goal_reference_derivation=dict(rule=rule.pointer,helper=adapter.pointer(Path(goal.__file__)),
        derivation_schema=goal.SCHEMA,original_raw_metric_events_and_masks_preserved=True,per_source=proofs,
        verified_files_sha256=rule.verified_files_sha256))
    inputs=release._Inputs(ROOT)
    actual_rule,sources=release._command_goal_sources(inputs,config,manifest,cases,None,{})
    a.require(len(sources)==len(proofs)==500,'all500 configured actual teacher sources required')
    checks={};negatives=[]
    for task,row in report['tasks'].items():
        if 'library' not in row or task not in rule.tasks:continue
        library=interaction.InteractionLibrary.load(a.checked_pointer(ROOT,row['library'],files))
        selected=[sources[c['entry']['source_id']] for c in cases if c['entry']['task']==task]
        release._command_goal_library_metadata(library.metadata,selected,actual_rule)
        checks[task]=len(selected)
        for field in ('rule','helper','source','raw_metric','public_scope'):
            metadata=deepcopy(library.metadata);proof=metadata['command_goal_reference_derivation']
            if field=='rule':proof['rule_sha256_values']=['f'*64]
            elif field=='helper':proof['helper_sha256']='f'*64
            elif field=='source':proof['per_source'].pop(selected[-1].source_id)
            elif field=='raw_metric':proof['original_raw_completion_metric_preserved']=False
            else:metadata['public_completion_verifier_calibrated']=True
            try:release._command_goal_library_metadata(metadata,selected,actual_rule)
            except ValueError:negatives.append(task+':'+field)
            else:raise AssertionError('tampered metadata accepted: '+field)
    case=cases[0];source=adapter.interaction_source(case)[0]
    for field in ('arrays','commands'):
        bad=deepcopy(case)
        if field=='arrays':bad['arrays']['measured_tcp_xyzw'][0,0]+=.1
        else:bad['command_receipt']['commands'][0]['raw_waypoint_xyzw'][0]+=.1
        try:goal.derive_source(bad,source,rule)
        except ValueError:negatives.append('actual_raw_memory:'+field)
        else:raise AssertionError('tampered actual source accepted')
    inputs.include(files);inputs.rehash_all()
    for name,sha in code.items():a.require(a.file_sha(name)==sha,'consumer source changed during audit')
    result=dict(schema='r10_command_goal_production_consumer_actual500_audit_v2',created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun,production_refit=a.pointer(ROOT,a.local(ROOT,report_path)),
        actual_source_rebuilds=500,full_1400_release_claimed=False,actual_raw_parent_archives_arrays_clock_reverified=True,
        producer_consumer_proofs_identical=True,passing_libraries_checked=checks,negative_cases_rejected=negatives,
        source_sha256=code,verified_files_sha256=inputs.files,native_GPU_or_collection_started=False)
    p=output/'report.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(report=a.pointer(ROOT,p),actual_sources=500,negative_cases=len(negatives))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(run(args.report,args.out),ensure_ascii=False))
