"""Read-only analysis of strict full14 pending-reference diagnosis.
No producer/model/runtime/source labels are changed; proposed rules are draft.
"""
import ast,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from our_mani.multitask import acquisition_lineage as a
from diagnose_TRAIN_reference_times_r10 import stats
from diagnose_full14_pending_references_r10 import write
BASE=ROOT/'artifacts/framework/multitask_r10'


def run(out):
    began=time.perf_counter();out=a.local(ROOT,out);a.require(not out.exists(),'fresh analysis namespace required');out.mkdir(parents=True)
    p=BASE/'full14_pending_reference_readonly_diagnostic_v6';report=a.read(p/'report.json');files={}
    def checked(path,sha=None):
        path=a.local(ROOT,path);value=a.file_sha(path);a.require(sha is None or sha==value,'bound diagnostic bytes changed')
        a.require(str(path) not in files or files[str(path)]==value,'input sha conflict');files[str(path)]=value;return path
    checked(p/'report.json');checked(__file__)
    a.require(report['actual_original_teacher_cases']==800 and report['target_tasks']==8
        and report['production_rules_or_sources_changed'] is False,'complete readonly800 diagnosis required')
    old_snapshot=a.read(checked(BASE/'command_goal_reference_rule_v4_revision2_setup/source_snapshot_v2.json'))
    old_fitter=old_snapshot['sources']['our_mani/multitask/artifact_fitting.py']['archived_source']
    oldpath=checked(old_fitter['path'],old_fitter['sha256']);newpath=checked(ROOT/'our_mani/multitask/artifact_fitting.py')
    def function(tree,name):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    before=ast.parse(oldpath.read_text());after=ast.parse(newpath.read_text())
    unchanged={name:function(before,name)==function(after,name) for name in ('interaction_source','colour_samples','colour_parent_crossfit')}
    a.require(all(unchanged.values()),'readonly adapter or colour numeric logic changed')
    events={};cases={};records={};arrays={};commands={}
    for task in report['tasks']:
        events[task]=a.read(checked(p/task/'events.json'));cases[task]=a.read(checked(p/task/'cases.json'))
        for case in cases[task]:
            sid=case['source_id'];r=a.read(checked(case['canonical_receipt']['path'],case['canonical_receipt']['sha256']));records[sid]=r
            commands[sid]=a.read(checked(case['command_receipt']['path'],case['command_receipt']['sha256']))
    press={};drift={};remove={}
    def load_arrays(sid,task):
        if sid not in arrays:
            c=next(x for x in cases[task] if x['source_id']==sid);cache=checked(c['cache']['path'],c['cache']['sha256'])
            with np.load(cache,allow_pickle=False) as z:arrays[sid]={k:z[k].copy() for k in z.files}
        return arrays[sid]
    for task in ('push_moving_button','push_moving_button_high_speed'):
        evs=[e for e in events[task] if e['phase']=='press' and e['originally_admitted_in_fit']];proofs=[]
        for ev in evs:
            sid=ev['source_id'];r=records[sid];z=load_arrays(sid,task);align=r['alignment'];raw=commands[sid]
            c=next(x for x in raw['commands'] if x['command_id']==ev['command_id']);native=align['native_outcome']
            re=next(x for x in raw['phase_completion_events'] if x['origin_command_id']==ev['command_id'])
            valid=(native['success'] is True and align['native_success_latched'] is True
                and align['native_terminal_latched'] is True and align['native_terminal_failure_latched'] is False
                and align['native_terminal_tick']==len(z['times'])-1
                and abs(align['native_terminal_tick']*.05-re['attempt_end_time'])<1e-8
                and abs(z['times'][-1]-re['attempt_end_time'])<1e-8
                and re['phase']==c['raw_phase']=='press' and re['evidence']=='TRAIN_native_success_during_executed_final_required_phase'
                and re['origin_command_completed']==('command_completed_at' in c))
            a.require(valid,'native task success witness is not exact actual final event')
            proofs.append(dict(source_id=sid,parent=ev['parent'],case_id=ev['case_id'],command_id=ev['command_id'],
                native_task_success_witness_verified=True,terminal_tick=align['native_terminal_tick'],
                task_success_time_seconds=re['attempt_end_time'],raw_command_completed=('command_completed_at' in c),
                actual_waypoint_arrival_observed=('measured_arrival_at' in c),
                arrival_time_seconds=c.get('measured_arrival_at'),
                terminal_tracking_error_m=ev['terminal_to_committed_goal']['position_m'],
                native_task_success_is_waypoint_or_gripper_completion=False,
                raw_command_or_event_rewritten=False))
        press[task]=dict(actual_success_event_count=len(evs),verified_native_success_count=len(proofs),
            original_raw_command_completed_count=sum(x['raw_command_completed'] for x in proofs),
            actual_waypoint_arrival_count=sum(x['actual_waypoint_arrival_observed'] for x in proofs),
            distinct_parents_with_true_arrival=len({x['parent'] for x in proofs if x['actual_waypoint_arrival_observed']}),
            terminal_tracking_error_m=stats([x['terminal_tracking_error_m'] for x in proofs]),
            no_geometry_gate_override_from_native_success=True,success_label_can_be_separate_TRAIN_typed_witness=True,
            native_success_is_never_runtime_public_observation=True)
        write(out/task/'verified_native_task_success_witnesses.json',proofs)
    task='insert_onto_rotating_peg_high_speed';evs=[e for e in events[task] if e['phase']=='insert' and e['originally_admitted_in_fit']]
    drift_rows=[]
    for ev in evs:
        z=load_arrays(ev['source_id'],task);c=next(x for x in commands[ev['source_id']]['commands'] if x['command_id']==ev['command_id'])
        a.require('measured_arrival_at' in c and 'command_completed_at' in c,'actual highpeg completed insert lost arrival')
        begin,end=ev['origin_and_terminal_frames'];arrival=int(np.flatnonzero(np.isclose(z['times'],c['measured_arrival_at'],atol=1e-8,rtol=0))[0])
        goal=np.asarray(c['raw_waypoint_xyzw'])[:3];trace=[dict(frame=i,time=float(z['times'][i]),
            stage=str(z['teacher_execution_stage'][i]),tracking_error_m=float(np.linalg.norm(z['measured_tcp_xyzw'][i,:3]-goal)),
            joint_limit_recovery=bool(z['teacher_joint_limit_recovery'][i])) for i in range(arrival,end+1)]
        drift_rows.append(dict(parent=ev['parent'],case_id=ev['case_id'],source_id=ev['source_id'],command_id=ev['command_id'],
            actual_arrival_error_m=c['arrival_position_error_m'],actual_terminal_error_m=ev['terminal_to_committed_goal']['position_m'],
            drift_seconds=float(z['times'][end])-c['measured_arrival_at'],
            after_arrival_tracking_trace=trace,gripper_desired_opening=c['desired_opening'],
            command_completed_true=True,raw_tracking_or_completion_times_unchanged=True))
    drift=dict(completed_insert_attempts=len(drift_rows),
        arrival_errors_m=stats([x['actual_arrival_error_m'] for x in drift_rows]),
        terminal_errors_m=stats([x['actual_terminal_error_m'] for x in drift_rows]),
        arrival_to_terminal_tracking_error_change_m=stats([x['actual_terminal_error_m']-x['actual_arrival_error_m'] for x in drift_rows]),
        arrival_to_terminal_seconds=stats([x['drift_seconds'] for x in drift_rows]),
        terminal_errors_over_3cm=sum(x['actual_terminal_error_m']>.03 for x in drift_rows),
        any_joint_limit_recovery_after_arrival=any(y['joint_limit_recovery'] for x in drift_rows for y in x['after_arrival_tracking_trace']),
        worst_actual_case=max(drift_rows,key=lambda x:x['actual_terminal_error_m']),
        classification='actual_post_arrival_contact_tracking_drift_not_reference_clock_or_public_pose_error',
        fix_requires_new_executor_or_precise_phase_completion_definition_with_new_artifact_not_cap_clipping=True)
    write(out/'highpeg_actual_post_arrival_drift.json',drift_rows)
    for task in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'):
        summaries={};rows=[]
        for ev in events[task]:
            if ev['phase'] not in ('approach','acquire','detach'):continue
            z=load_arrays(ev['source_id'],task);ids=list(z['body_ids']);start,end=ev['origin_and_terminal_frames']
            chosen=ev['nearest_spoke_to_manipulated_cup_at_issue'];cup=next(x for x in commands[ev['source_id']]['commands'] if x['command_id']==ev['command_id'])['manipulated_body']
            ci,si=ids.index(cup),ids.index(chosen);spokes=[b for b in ids if b.startswith('spoke')]
            def valid(i,j):return bool(z['valid'][i,j])
            def full(i,j):return bool(z['valid'][i,j] and z['rotation_observable'][i,j])
            row=dict(source_id=ev['source_id'],parent=ev['parent'],case_id=ev['case_id'],command_id=ev['command_id'],phase=ev['phase'],
                cup_reference=cup,nearest_native_spoke_label_TRAIN_only=chosen,
                cup_issue_public_valid=valid(start,ci),cup_issue_public_fullR=full(start,ci),
                cup_terminal_public_valid=valid(end,ci),cup_terminal_public_fullR=full(end,ci),
                spoke_issue_public_valid=valid(start,si),spoke_issue_public_fullR=full(start,si),
                spoke_terminal_public_valid=valid(end,si),spoke_terminal_public_fullR=full(end,si),
                all_three_spokes_and_cup_public_valid_at_issue=valid(start,ci) and all(valid(start,ids.index(s)) for s in spokes))
            if row['all_three_spokes_and_cup_public_valid_at_issue']:
                public_selected=min(spokes,key=lambda s:np.linalg.norm(z['positions'][start,ids.index(s)]-z['positions'][start,ci]))
                row.update(causal_public_nearest_spoke=public_selected,causal_public_nearest_spoke_matches_TRAIN_label=(public_selected==chosen))
            for moment,i in (('issue',start),('terminal',end)):
                if full(i,si):row['spoke_'+moment+'_public_rotation_error_rad']=float(Rotation.from_matrix(z['rotations'][i,si]@z['target_rotations'][i,si].T).magnitude())
            rows.append(row)
        for phase in ('approach','acquire','detach'):
            rr=[x for x in rows if x['phase']==phase];summaries[phase]=dict(actual_events=len(rr),
                **{key:sum(x[key] for x in rr) for key in ('cup_issue_public_valid','cup_issue_public_fullR','cup_terminal_public_valid','cup_terminal_public_fullR',
                    'spoke_issue_public_valid','spoke_issue_public_fullR','spoke_terminal_public_valid','spoke_terminal_public_fullR','all_three_spokes_and_cup_public_valid_at_issue')},
                public_nearest_spoke_eligible=sum('causal_public_nearest_spoke' in x for x in rr),
                public_nearest_spoke_agree=sum(x.get('causal_public_nearest_spoke_matches_TRAIN_label',False) for x in rr),
                public_spoke_orientation_error_issue_rad=stats([x['spoke_issue_public_rotation_error_rad'] for x in rr if 'spoke_issue_public_rotation_error_rad' in x]),
                public_spoke_orientation_error_terminal_rad=stats([x['spoke_terminal_public_rotation_error_rad'] for x in rr if 'spoke_terminal_public_rotation_error_rad' in x]))
        remove[task]=summaries;write(out/task/'public_reference_availability.json',rows)
    prior=a.read(checked(BASE/'actual700_TRAIN_reference_root_cause_v2/report.json'))
    checks=[dict(task=task,classification=kind,evidence=ev,minimal_next_action=action) for task,kind,ev,action in [
        ('insert_onto_rotating_peg_high_speed','genuine_post_arrival_tracking_error_and_released_parent_scarcity','43 inserts; cap30.447mm;3 released parents','keep pending; inspect counted closed-gripper contact servo/phase endpoint then new fixed collection if executor changes; no trim'),
        ('place_cups_on_rotating_frame','released_parent_scarcity','1 released parent in second teacher cycle; acquire released reachable after failed own-close','keep exact-mode pending; fixed development retry acquisition or reviewed unsupported-mode abort; do not remove released rows'),
        ('place_cups_on_rotating_frame_high_speed','downstream_completed_phase_scarcity','4 align parents; endpoint geometry fits caps but support<8','ROOT independent all100 requery supplement addresses progression without outcome filtering; do not mass repeat unchanged follower'),
        ('push_moving_button','approach_reference_time_error_plus_press_completion_semantics','approach300.314mm→issue5.225mm; press68 verified task-success,55 without arrival; intent tracking79.266mm','new optional approach issue rule; independent native-success typed labels; press geometry remains pending'),
        ('push_moving_button_high_speed','approach_reference_time_error_plus_source_specific_press_intent','approach current296.671mm→issue5.364mm; press exact.25+clamp, not.5;38 successes only3 arrivals','new optional approach issue rule; keep .25+clamp TRAIN source provenance; no physical success claim from intent'),
        ('put_rubbish_in_moving_bin_high_speed','source_goal_reference_time_semantics','all739deposit commands:100current/606forecast1s/33forecast.5s;639forecasts exact0;32 completed endpoint5mm','new explicit source-backed current/1/.5 intent helper; complete100 real fullcurve/CV required before release'),
        ('remove_cups_from_rotating_frame','wrong_acquire_anchor_plus_approach_rotating_intent_and_occlusion','acquire cup→nearestspoke endpoint13.183mm/.05638rad; approach forecast11.033mm/.07485rad; terminalcup0/68visible','new explicit source-bound spoke issue acquire+rotating approach; retain public freshness/fullR/contact gates and rare π errors'),
        ('remove_cups_from_rotating_frame_high_speed','wrong_acquire_anchor_plus_approach_detach_rotating_intent_and_occlusion','acquire spoke15.631mm/.06961rad; approach forecast12.787mm/.05508rad;detach forecast5.341mm/.03;cup0/56terminalvisible','same source-backed rule; verify public spoke candidate support; no native-yaw runtime access')]]
    rule=dict(schema='r10_pending_task_minimal_TRAIN_reference_rule_PROPOSAL_v1',approved=False,production_enabled=False,
        TRAIN_only=True,VAL_or_TEST_labels_used=False,raw_and_current_labels_masks_censors_preserved=True,
        requested_optional_branches={
            'button_approach':dict(tasks=['push_moving_button','push_moving_button_high_speed'],phase='approach',reference='original issued button translation',
                source_scope='original wp0 delay0 current waypoint canonical offset; measured executed curve unchanged'),
            'remove_acquire':dict(tasks=['remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'],phase='acquire',
                reference='nearest spoke to original selected cup at command issue; observed_SE3',identity_proof='original expert nearest-cup/spoke definitions + exact all100 raw goal relative spoke geometry',
                runtime='causal public cup/spoke geometry association only; fresh fullR; existing own-close/contact/identity checks retained'),
            'remove_rotating_phases':dict(tasks=['remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed'],phases=['approach','detach'],
                source_delays='approach1s first wp1 then.5;detach.5',TRAIN_driver_only=True,
                evidence_limit='frame rigid-rotation endpoint hypothesis passes; exact source hierarchy/delay proof and actual complete100 fullcurve must be reviewed before production'),
            'high_bin_deposit':dict(task='put_rubbish_in_moving_bin_high_speed',phase='deposit',reference='issue actual bin on initial wp2→wp3 closed; original unreflected polynomial intent1s later closed and.5s open',
                all_commands=739,canonical_current_waypoint_offset_xy_m=.0019967117322543986,source_formula_exact_forecasts=639,
                source_forecasts_are_not_future_observations=True,runtime_forecast='selected causal CV/Context pose fixed; no source-private1s/.5s lookup at runtime')},
        verified_native_task_success_proposed_type=dict(type_name='TRAINVerifiedNativeTaskSuccessCompletion',
            fields=['task_name','source_id','parent_source_group','command_id','terminal_frame','terminal_time','native_task_source_sha256','canonical_raw_binding','commands_sha256','cache_sha256','actual_native_outcome','raw_command_completed','raw_waypoint_arrival_observed','original_actual_TCP','original_actual_reference'],
            validation='strict raw success/latched terminal tick=last frame=final-required-phase event time; own actualplan/archive/raw/arrays witness',
            supplies='independent TRAIN success/censor supervision; never command_completed_at nor a runtime observation',
            geometry_fitting='original measured geometry and8/.03/.15 remain; no successBoolean or desired goal substitutes an unobserved arrival',
            actual_arrival_parent_support=dict(normal=press['push_moving_button']['distinct_parents_with_true_arrival'],high=press['push_moving_button_high_speed']['distinct_parents_with_true_arrival'])),
        minimum_completed_parent_sources=8,position_cap_m=.03,orientation_cap_rad=.15,
        no_native_or_GPU_start=True,no_unreviewed_data_collection=True,no_formal_release=True)
    write(out/'minimal_TRAIN_rule_proposal.json',rule)
    color=report['colour'];colour_actions=dict(hard_data_scarcity={'orange':15},minimum_disjoint_twofold_total=16,
        yellow_current_folds=[7,9],yellow_existing_whole_parent_exchange_candidates=len(color['readonly_feasible_yellow_whole_parent_move_candidates']),
        yellow_can_rebalance_without_lowering_any_nonorange_class_below8=bool(color['readonly_feasible_yellow_whole_parent_move_candidates']),
        orange_need_at_least_one_new_independent_TRAIN_parent=True,same_parent_r1_not_sufficient=True,
        minimal_proposal='explicit colour-only TRAIN supplement with distinct source parent, fixed before outcomes; keep K1 Context roster unchanged; deterministic exact parent fold feasibility after support; rerun original.95 accuracy gate',
        cannot_assume_accuracy_will_pass_when_support_fixed=True,no_annotation_filtered=True)
    for path,sha in files.items():a.require(a.file_sha(path)==sha,'analysis input changed at final direct rehash')
    value=dict(schema='r10_full14_pending_root_cause_and_minimal_rule_review_v1',raw800_diagnosis=a.pointer(ROOT,p/'report.json'),
        full14_fit=a.pointer(ROOT,BASE/'TRAIN_artifacts_full_goalv4_semanticv6_v1/report.json'),
        source_current_vs_archived=dict(archived_fitter=old_fitter,current_fitter=a.pointer(ROOT,newpath),
            exact_AST_unchanged_functions=unchanged,current_change='ROOT typed-counted-bundle admission/provenance only; historical full14 source identity retained'),
        verified_native_task_success=press,highpeg_actual_post_arrival_drift=drift,remove_public_reference_evidence=remove,
        per_task_classification=checks,colour=colour_actions,rare_partition_runtime_evidence=prior['runtime_released_partition_review'],
        minimal_rule_proposal=a.pointer(ROOT,out/'minimal_TRAIN_rule_proposal.json'),
        verified_files_sha256=files,production_sources_changed=False,native_or_GPU_started=False,
        thresholds_outliers_or_parent_support_relaxed=False,diagnostic_only=True,fullcurve_new_pass_or_formal_release_claimed=False,
        seconds=time.perf_counter()-began)
    write(out/'report.json',value)
    lines=['Actual full14 pending diagnosis; readonly, no production/rule/raw change.',
        'Colour: orange15<16 hard parent scarcity; yellow7/9 has9 feasible whole-parent exchanges; accuracy still unknown.',
        'Button approach: ordinary300.314mm→issue5.225mm; high current296.671mm→issue5.364mm (original fullfit high350.164mm atlag.2 preserved).',
        'Button press: normal68/high38 exact native-success witnesses; no completed commands; true arrivals13/3; keep task-success labels independent of geometry caps.',
        'Remove acquire: wrong cup anchor; source wp2 is nearest-spoke-relative. Endpoint normal13.183mm/.05638rad,high15.631mm/.06961rad; fullcurve not refitted.',
        'High bin: all100parents/all739deposit commands retained;100current,606one-second forecasts,33half-second; all639raw forecast goals exactly reproduced.',
        'High peg: real post-arrival actuator/contact drift; worst15.545mm→39.978mm over.4s, no joint-limit recovery. Cap30.447mm stayspending.',
        'Cuphigh4align parents; normalcup released1parent and highpeg released3parents remainpending.',
        'All8task original100sources retained; no public pose injected into TRAIN fits; no native state permitted at runtime.']
    write(out/'summary.json',dict(lines=lines));(out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('REPORT',out/'report.json')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();run(args.out)
