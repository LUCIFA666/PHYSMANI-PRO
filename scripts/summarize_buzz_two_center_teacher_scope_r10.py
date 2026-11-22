"""Readonly tail attribution for the fixed TRAIN teacher crossfit predictions.

No model is fit, no gate is tuned, and native labels are diagnostic outcomes,
never estimator inputs. All teacher parents/rows, including rejected rows,
remain represented in the saved residual arrays and summaries.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from scripts import diagnose_buzz_public_position_pose_r10 as d

BASE=ROOT/'artifacts/framework/multitask_r10'
SOURCE=BASE/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/report.json'
SOURCE_SHA='0136c7f482e59d7090e52d7ed32dedeb2db2402c29749451ec34c841c92da5e4'


def summarize(rows):
    if not rows:return dict(frames=0)
    return dict(frames=len(rows),accepted=sum(r['accepted'] for r in rows),
        frame_error_rad=d.describe([r['frame_error'] for r in rows]),
        wand_error_rad=d.describe([r['wand_error'] for r in rows]),
        native_relative_rotation_from_static_prior_rad=d.describe([r['native_relative_R_error'] for r in rows]),
        native_local_offset_from_static_prior_m=d.describe([r['native_offset_error'] for r in rows]),
        public_minus_native_centroid_delta_m=d.describe([r['centroid_delta_error'] for r in rows]),
        native_frame_axis_from_static_world_axis_rad=d.describe([r['native_axis_error'] for r in rows]),
        original_gate_wand_tails_above_point15=sum(r['wand_error']>.15 for r in rows),
        original_gate_frame_tails_above_point15=sum(r['frame_error']>.15 for r in rows),
        public_contact_true=sum(r['contact'] for r in rows),public_gripper_open_true=sum(r['open'] for r in rows),
        actual_teacher_action_rows=sum(r['teacher_action'] for r in rows),
        execution_stage_counts=dict(Counter(r['execution_stage'] for r in rows)),
        public_wand_mode_counts=dict(Counter(r['mode'] for r in rows)))


def main(output):
    output=Path(output).resolve()
    if not output.is_relative_to(BASE) or output.exists():raise ValueError('fresh ROOT artifact directory required')
    output.mkdir(parents=True);files={}
    source=a.read(a.checked_pointer(ROOT,dict(path=str(SOURCE.relative_to(ROOT)),sha256=SOURCE_SHA),files))
    report={};archives=[]
    for task,task_report in source['teacher_parent_crossfit'].items():
        archive=next(p for p in source['prediction_archives'] if p['path'].endswith(task+'__all100_teacher_parent_crossfit_predictions.npz'))
        path=a.checked_pointer(ROOT,archive,files)
        with np.load(path,allow_pickle=False) as stored:predictions={k:stored[k].copy() for k in stored.files}
        by_case={p['case_id']:p for p in task_report['per_parent']};rows=[];parents=[]
        for case in [p for p in source['actual_case_provenance'] if p['task']==task and p['behavior']=='TRAIN_teacher_committed_waypoint']:
            selected=np.flatnonzero(predictions['case_id']==case['case_id']);indices=predictions['frame_index'][selected]
            assert np.array_equal(indices,np.arange(len(indices)))
            model=a.read(a.checked_pointer(ROOT,by_case[case['case_id']]['model'],files))
            prior=model['static_prior'];cache=a.checked_pointer(ROOT,case['cache'],files)
            with np.load(cache,allow_pickle=False) as stored:arrays={k:stored[k].copy() for k in stored.files}
            body_ids=list(map(str,arrays['body_ids']));fi=body_ids.index('frame');wi=body_ids.index('wand')
            R=arrays['target_rotations'][:,fi];Q=np.einsum('tji,tjk->tik',R,arrays['target_rotations'][:,wi])
            native_delta=arrays['target_positions'][:,wi]-arrays['target_positions'][:,fi]
            local=np.einsum('tji,tj->ti',R,native_delta)
            public_delta=arrays['positions'][:,wi]-arrays['positions'][:,fi]
            rotation_errors=d.angles(Q,np.asarray(prior['frame_to_wand_rotation']))
            offset_errors=np.linalg.norm(local-np.asarray(prior['local_frame_to_wand_offset']),axis=1)
            centroid_errors=np.linalg.norm(public_delta-native_delta,axis=1)
            native_axes=R@np.asarray(prior['local_axis'])
            axis_errors=np.arccos(np.clip(native_axes@np.asarray(prior['world_axis']),-1.,1.))
            case_rows=[]
            for j,index in enumerate(indices):
                error=predictions['SO3_error_rad'][selected[j]]
                case_rows.append(dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],
                    frame_index=int(index),time=float(arrays['times'][index]),phase=str(arrays['teacher_phase'][index]),
                    mode=('external','attached','released')[int(arrays['modes'][index,wi])],
                    accepted=bool(predictions['prediction_valid'][selected[j]]),reason=str(predictions['reason'][selected[j]]),
                    frame_error=float(error[0]),wand_error=float(error[1]),
                    native_relative_R_error=float(rotation_errors[index]),native_offset_error=float(offset_errors[index]),
                    centroid_delta_error=float(centroid_errors[index]),native_axis_error=float(axis_errors[index]),
                    contact=bool(arrays['robot_contact'][index]),open=bool(arrays['robot_gripper_open'][index]),
                    teacher_action=bool(arrays['teacher_action_valid'][index]),execution_stage=str(arrays['teacher_execution_stage'][index]),
                    command_id=int(arrays['teacher_command_id'][index])))
            rows.extend(case_rows)
            tail=[r for r in case_rows if r['wand_error']>.15 or r['frame_error']>.15]
            parents.append(dict(case_id=case['case_id'],parent_source_group=case['parent_source_group'],cache=case['cache'],
                all_rows=summarize(case_rows),accepted_rows=summarize([r for r in case_rows if r['accepted']]),
                tail_rows=summarize(tail),first_tail=None if not tail else {k:tail[0][k] for k in ('frame_index','time','phase','mode','contact','open','teacher_action','execution_stage','command_id')},
                tail_frame_indices=[r['frame_index'] for r in tail]))
        assert len(parents)==100 and len(rows)==task_report['total_frames']
        phase=defaultdict(list)
        for row in rows:phase[row['phase']].append(row)
        tail=[r for r in rows if r['wand_error']>.15 or r['frame_error']>.15]
        report[task]=dict(all_rows=summarize(rows),accepted_rows=summarize([r for r in rows if r['accepted']]),
            tail_rows=summarize(tail),per_phase={name:dict(all_rows=summarize(values),
                accepted_rows=summarize([r for r in values if r['accepted']]),
                original_gate_tails=summarize([r for r in values if r['wand_error']>.15 or r['frame_error']>.15]))
                for name,values in sorted(phase.items())},
            all100_parents=parents,tail_parent_count=sum(bool(p['tail_frame_indices']) for p in parents),
            worst10_parent_sources_for_diagnostic_display=sorted(parents,key=lambda p:len(p['tail_frame_indices']),reverse=True)[:10])
        path=output/(task+'__all_teacher_residuals.npz')
        np.savez_compressed(path,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
        archives.append(a.pointer(ROOT,path))
    for path,checksum in files.items():assert a.file_sha(path)==checksum
    result=dict(schema='r10_buzz_teacher_two_center_phase_and_tail_source_readonly_audit_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),source_teacher_report=a.pointer(ROOT,SOURCE),script=a.pointer(ROOT,Path(__file__)),
        fixed_original_teacher_parents=200,estimator_or_prior_refit=False,parameters_or_thresholds_changed=False,
        all_invalid_rejected_censored_and_tail_rows_retained=True,native_labels_used_only_for_residual_attribution=True,
        runtime_permission_none_unchanged=True,live_source_modified=False,native_or_GPU_started=False,
        tasks=report,residual_archives=archives,all_input_bytes_stable_before_after=True,verified_files_sha256=files,
        interpretation=[
            'A hold-derived approximate rigid relation does not establish wand orientation during manipulation. The actual teacher-domain q95 failures must remain visible.',
            'Native relative-R/offset residuals diagnose physical relation changes; public-minus-native centroid residuals diagnose observation error. These labels are not passed to the estimator.',
            'Contact/open/phase/action fields are retrospective source attribution only, not a new runtime gate. No outlier-selected phase or threshold is promoted.',
            'Attached/released observations are already rejected. Public external alone is insufficient; later work needs independent wand orientation evidence or a separately verified affordance.',
            'Empty teacher phase is retained exactly as recorded, including terminal/no-command rows. Task outcomes are not used for selection or qualification.'])
    path=output/'report.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['Buzz teacher-only phase/tail attribution; fixed original200 parents, no new fit or gate.']
    for task,item in report.items():
        lines.append(task+': accepted wand q95='+str(item['accepted_rows']['wand_error_rad']['q95'])+', tail parents='+str(item['tail_parent_count']))
        for name,values in item['per_phase'].items():
            item=values['accepted_rows']
            lines.append('  phase '+repr(name)+': '+json.dumps({k:item.get(k) for k in ('frames','wand_error_rad','native_relative_rotation_from_static_prior_rad','native_local_offset_from_static_prior_m','public_minus_native_centroid_delta_m')},ensure_ascii=False))
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=a.pointer(ROOT,path),task_summary={t:dict(tail_parents=v['tail_parent_count'],tails=v['tail_rows']['frames']) for t,v in report.items()}),ensure_ascii=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
