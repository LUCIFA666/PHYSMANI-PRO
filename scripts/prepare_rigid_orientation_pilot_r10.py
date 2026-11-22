"""SHA-audit completed TRAIN rigid priors; prepare a later same-frame pilot.

Historical V4 files are read as immutable evidence. This does not revalidate
the old V4 source paths against a newer runtime, execute native code, transfer
priors between tasks, fit current-GT relations, or update orientation labels.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import pilot_orientation_registration_r10 as pilot
from our_mani.multitask.specs import TASK_SPECS


def audit_priors(source_directory,output):
    source_directory=Path(source_directory).resolve();inputs={};layouts={};status={}
    for task in TASK_SPECS:
        paths=sorted((source_directory/'cases/train'/task).glob('*/case.json'))
        completed=[]
        for path in paths:
            case=json.loads(path.read_text())
            if case.get('status')=='completed':completed.append((path,case))
        if not completed:
            status[task]='no completed TRAIN case; no inferred prior';continue
        pilot.require(len(completed)==1,'fixed first-source prior audit expects one completed TRAIN case per task')
        path,case=completed[0]
        pilot.require(case['split']=='train' and case['task']==task and case['parent_source_group'].startswith('train:')
            and case['policy_history_contains_native_GT'] is False,'TRAIN prior provenance boundary differs')
        inputs[str(path)]=pilot.sha(path)
        parent=case['parent_process_receipt'];child=ROOT/parent['child_case_path']
        pilot.require(pilot.sha(child)==parent['child_case_sha256'],'original child case changed')
        original=json.loads(child.read_text())
        pilot.require(all(case.get(k)==v for k,v in original.items()),'canonical completed case changed original child fields')
        inputs[str(child.resolve())]=pilot.sha(child)
        for name,expected in case['files_sha256'].items():
            value=(ROOT/name).resolve();pilot.require(pilot.sha(value)==expected,'prior source evidence changed: '+name)
            inputs[str(value)]=expected
        candidates=[ROOT/name for name in case['files_sha256'] if Path(name).name=='TRAIN_rigid_layout.json']
        pilot.require(len(candidates)==1,'completed TRAIN case must expose its rigid-fit audit')
        layout_path=candidates[0].resolve();layout=json.loads(layout_path.read_text())
        pilot.require(layout.get('version')=='r10_TRAIN_observed_rigid_layout_v1' and layout.get('split')=='train'
            and layout.get('task')==task and layout.get('shared_with_baseline') is True
            and layout.get('evidence_sha256')==pilot.digest({k:v for k,v in layout.items() if k!='evidence_sha256'}),
            'rigid artifact fingerprint/task/split differs')
        tracks=(ROOT/case['entry']['cache']).resolve()
        with np.load(tracks,allow_pickle=False) as arrays:
            fingerprint=pilot.digest({key:arrays[key].tolist() for key in
                ('body_ids','times','target_positions','target_rotations','target_valid')})
        pilot.require(fingerprint==layout['arrays_sha256'],'rigid fit no longer matches raw TRAIN labels')
        if not layout['accepted']:
            status[task]='no accepted permanent rigid relation';continue
        for relation in layout['accepted']:
            p,c=relation['parent'],relation['child']
            pilot.require((p=='frame' and c.startswith(('peg','spoke'))) or (p=='button' and c=='wrap'),
                'articulated/attachable body cannot become a permanent rigid child')
            limits=layout['thresholds']
            pilot.require(relation['samples']>=limits['minimum_samples']
                and relation['position_error_max_m']<=limits['position_tolerance_m']
                and relation['orientation_error_max_rad']<=limits['orientation_tolerance_rad'],
                'accepted relation violates its original TRAIN validation thresholds')
        layouts[task]=dict(path=str(layout_path.relative_to(ROOT)),sha256=pilot.sha(layout_path),
            validated_on_TRAIN=True,source_group=layout['source_group'],source_case=str(path.relative_to(ROOT)),
            accepted_relations=[dict(parent=r['parent'],child=r['child']) for r in layout['accepted']])
        status[task]='SHA and raw TRAIN-fit validation passed'
    report=dict(version='r10_completed_TRAIN_rigid_prior_manifest_v1',TRAIN_only=True,TEST_used=False,
        source_directory=str(source_directory.relative_to(ROOT)),layout_map=layouts,task_status=status,
        inputs_sha256=inputs,source_script_sha256=pilot.sha(__file__),native_run_performed=False,
        task_success_used_for_prior_selection=False,per_task_transfer_permitted=False,
        old_V4_current_source_reverification_performed=False)
    pilot.write(output,report);return report


def prepare_plan(collection_plan,prior_manifest,output):
    prior_manifest=Path(prior_manifest).resolve();prior=json.loads(prior_manifest.read_text())
    pilot.require(prior['TRAIN_only'] and prior['TEST_used'] is False,'only audited TRAIN priors permitted')
    for name,expected in prior['inputs_sha256'].items():pilot.require(pilot.sha(name)==expected,'TRAIN prior changed: '+name)
    output=Path(output).resolve();map_path=output.with_name(output.stem+'.rigid_map.json')
    pilot.write(map_path,prior['layout_map'])
    # Base preparation supplies the unchanged tracker/runtime protocol; append
    # every raw calibration proof to the final input lock before execution.
    draft=output.with_name(output.stem+'.base_draft.json')
    plan=pilot.build_plan(collection_plan,draft,frames=16,rigid_layout_map=map_path)
    plan['inputs_sha256'].update(prior['inputs_sha256'])
    plan['inputs_sha256'][str(prior_manifest)]=pilot.sha(prior_manifest)
    plan['inputs_sha256'][str(Path(__file__).resolve())]=pilot.sha(__file__)
    plan.update(rigid_prior_manifest=str(prior_manifest.relative_to(ROOT)),
        rigid_prior_manifest_sha256=pilot.sha(prior_manifest),
        direct_vs_group_same_frame=True,missing_rigid_prior_tasks=[r['task'] for r in plan['cases'] if r['rigid_layout'] is None],
        indirect_pose_disclosure='group child inferred from currently observed parent and frozen TRAIN relation is never declared directly visible',
        annotation_changes_written=False)
    plan['plan_sha256']=pilot.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    pilot.write(output,plan);pilot.verify_plan(output);return plan


def indirect_evidence(directory,output):
    directory=Path(directory).resolve();queue=json.loads((directory/'queue_report.json').read_text())
    pilot.require(queue['complete'] and len(queue['results'])==14,'full queue required')
    cases={}
    for row in queue['results']:
        if not row['completed']:continue
        path=ROOT/row['report'];pilot.require(pilot.sha(path)==row['report_sha256'],'native pilot report changed')
        report=json.loads(path.read_text());comparisons={}
        for variant in report['variants']:
            if not variant.endswith('__rigid_group'):continue
            direct=variant.removesuffix('__rigid_group')+'__direct';roles=report['frames'][0]['metric_only']['body_ids']
            per_body={}
            for role in roles:
                pairs=[(f['tracker_diagnostics'][direct]['bodies'][role],f['tracker_diagnostics'][variant]['bodies'][role])
                       for f in report['frames']]
                per_body[role]=dict(frames=len(pairs),
                    direct_registration_valid=sum(a['registration_valid'] for a,b in pairs),
                    group_registration_valid=sum(b['registration_valid'] for a,b in pairs),
                    direct_visible_frames=sum(a['directly_visible'] for a,b in pairs),
                    group_visible_frames=sum(b['directly_visible'] for a,b in pairs),
                    group_indirect_not_visible_frames=sum(b['registration_valid'] and not b['directly_visible'] for a,b in pairs),
                    group_pose_source_counts={source:sum(b['pose_source']==source for a,b in pairs)
                        for source in {b['pose_source'] for a,b in pairs}},
                    inferred_child_was_declared_directly_visible=False)
                pilot.require(per_body[role]['direct_visible_frames']==per_body[role]['group_visible_frames'],
                    'same sensor input changed direct visibility merely by adding a rigid prior')
            comparisons[variant]=per_body
        cases[report['task']]=comparisons
    result=dict(version='r10_rigid_pilot_indirect_evidence_v1',TRAIN_only=True,cases=cases,
        annotation_changes_written=False,GT_used_for_runtime_pose_inference=False,
        evidence='current observed group + frozen independently TRAIN-validated rigid layout; no hidden native child pose')
    pilot.write(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-priors',metavar='DIRECTORY');mode.add_argument('--prepare-plan',metavar='COLLECTION_PLAN')
    mode.add_argument('--indirect-analysis',metavar='PILOT_DIRECTORY')
    parser.add_argument('--prior-manifest');parser.add_argument('--out',required=True);args=parser.parse_args()
    if args.audit_priors:value=audit_priors(ROOT/args.audit_priors,ROOT/args.out)
    elif args.prepare_plan:
        pilot.require(args.prior_manifest,'frozen audited TRAIN prior manifest required')
        value=prepare_plan(ROOT/args.prepare_plan,ROOT/args.prior_manifest,ROOT/args.out)
    else:value=indirect_evidence(ROOT/args.indirect_analysis,ROOT/args.out)
    print(json.dumps(dict(prepared=True,native_run_performed=False,layout_tasks=len(value.get('layout_map',{}))),ensure_ascii=False))


if __name__=='__main__':main()
