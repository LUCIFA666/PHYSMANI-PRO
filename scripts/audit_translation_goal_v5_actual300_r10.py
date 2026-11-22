"""Independent raw reconstruction/consumer boundaries for actual300 candidate."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,translation_command_goal_reference as trans,release_preflight as release,interaction


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(fitreport,out):
    begun=time.perf_counter();out=a.local(ROOT,out);a.require(not out.exists(),'fresh actual300 consumer namespace required');out.mkdir()
    inputs=release._Inputs(ROOT);fitreport=a.local(ROOT,fitreport);report=inputs.read(str(fitreport),a.file_sha(fitreport))
    config=inputs.read(report['configuration']['path'],report['configuration']['sha256'])
    for field in ('fitting_core_source_snapshot','fitting_producer_source_snapshot'):inputs.source_snapshot(config[field])
    inputs.file(__file__,a.file_sha(__file__))
    manifest=inputs.read(report['prefit300_source_proof']['path'],report['prefit300_source_proof']['sha256'])
    cases=[]
    for row in manifest['case_proofs']:
        record=inputs.read(row['canonical_receipt']['path'],row['canonical_receipt']['sha256'])
        cache=inputs.pointer(row['cache']);command_path=inputs.pointer(row['command_receipt'])
        with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        cases.append(dict(path=inputs.pointer(row['canonical_receipt']),record=record,entry=record['entry'],arrays=arrays,
            command_receipt=a.read(command_path),command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256']))
    a.require(len(cases)==len({c['entry']['parent_source_group'] for c in cases})==300,'exact original300 parents required')
    rule,sources=release._command_goal_sources(inputs,config,manifest,cases,None,{})
    a.require(len(sources)==300 and len(rule.contexts)==9 and not rule.auxiliary_contexts,'complete original300 consumer reconstruction required')
    originals={c['entry']['source_id']:adapter.interaction_source(c)[0] for c in cases}
    for sid,s in sources.items():
        o=originals[sid];a.require(s.references==o.references and np.array_equal(s.command_ids,o.command_ids)
            and list(s.command_ledger)==list(o.command_ledger),'actual raw source was changed')
        a.require(len(s.completion_events)==len(o.completion_events) and all(np.array_equal(x.terminal_reference_xyzw,y.terminal_reference_xyzw)
            and np.array_equal(x.terminal_tcp_xyzw,y.terminal_tcp_xyzw) and x.command_completed==y.command_completed
            for x,y in zip(s.completion_events,o.completion_events)),'raw actual metrics/events were changed')
    task=trans.HIGH_BIN;libptr=report['tasks'][task]['library'];library=interaction.InteractionLibrary.load(inputs.pointer(libptr))
    high=[s for s in sources.values() if s.task_name==task]
    release._command_goal_library_metadata(library.metadata,high,rule)
    supports={}
    for template in library.templates:
        key,parents,attempts=release._completion_support(high,template);release._command_goal_geometry(high,template)
        a.require(len(parents)>=8,'actual library lost completion parent support')
        supports['|'.join(key)]=dict(parents=len(parents),attempts=len(attempts),position_tolerance_m=template.position_tolerance_m,
            orientation_tolerance_rad=template.orientation_tolerance_rad)
    negatives=[]
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):negatives.append(dict(name=name,rejected=True));return
        raise ValueError('consumer accepted negative: '+name)
    for mutation in ('missing','raw_binding','masks','forecast_delay','taskscope'):
        m=deepcopy(library.metadata);der=m['command_goal_reference_derivation'];sid=next(iter(der['per_source']))
        if mutation=='missing':der['per_source'].pop(sid)
        elif mutation=='raw_binding':der['per_source'][sid]['raw_bindings']['cache']['sha256']='f'*64
        elif mutation=='masks':der['per_source'][sid]['original_masks_sha256']['target_valid']='e'*64
        elif mutation=='forecast_delay':der['per_source'][sid]['translation_source_intention']['delay_counts']['1.0']=999
        else:der['rule_sha256_values']=['d'*64]
        rejects('library_'+mutation,lambda m=m:release._command_goal_library_metadata(m,high,rule))
    template=next(t for t in library.templates if t.phase=='deposit')
    bad_geometry=template.completion_geometry_xyzw.copy();bad_geometry[0]+=.01
    rejects('canonical_geometry_changed',lambda:release._command_goal_geometry(high,replace(template,completion_geometry_xyzw=bad_geometry)))
    rejects('support_rule_sha_changed',lambda:release._completion_support(high,replace(template,reference_time_rule_sha256='c'*64)))
    case=next(c for c in cases if c['entry']['task']==task)
    for mutation in ('actual_plan','raw_arrays','raw_commands','initial_driver'):
        bad=deepcopy(case)
        if mutation=='actual_plan':bad['record']['plan_sha256']='b'*64
        elif mutation=='raw_arrays':bad['arrays']['target_positions'][1,0,0]+=.01
        elif mutation=='raw_commands':next(c for c in bad['command_receipt']['commands'] if c['raw_phase']=='deposit')['raw_waypoint_xyzw'][0]+=.01
        else:bad['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']['v'][0]+=.01
        rejects(mutation,lambda bad=bad:adapter.interaction_source(bad,command_goal_rule=rule))
    inputs.rehash_all()
    result=dict(schema='r10_actual300_translation_goal_v5_independent_consumer_v1',actual_fit_report=a.pointer(ROOT,fitreport),
        configuration=report['configuration'],actual_sources_reconstructed=300,actual_original_parent_groups=300,
        actual_high_bin_library_support=supports,negative_boundaries=negatives,negative_count=len(negatives),
        original_raw_references_events_metrics_commands_censoring_preserved=True,
        consumers_are_real_production_functions_not_mocked=True,canonical_geometry_rebuilt_from_actual_commands=True,
        final_direct_all_input_rehash_passed=True,verified_files_sha256=inputs.files,
        native_or_GPU_started=False,formal_policy_or_SR_release=False,seconds=time.perf_counter()-begun)
    write(out/'report.json',result);return a.pointer(ROOT,out/'report.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-report',required=True);p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(run(args.fit_report,args.out)))
