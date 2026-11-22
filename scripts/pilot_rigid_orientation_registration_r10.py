"""Five preregistered TRAIN same-frame direct versus rigid-group cases.

The historical 14-case pilot is imported read-only. Its verifier is substituted
only in this owned fresh process while its unchanged causal tracker/metric
loop runs. No orientation annotation is promoted and no native run is started
by preparing a plan. A later collection freeze is required before preparation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pilot_orientation_registration_r10 as original

VERSION = 'r10_TRAIN_five_task_rigid_registration_pilot_v1'
TASKS = ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed',
         'place_cups_on_rotating_frame', 'place_cups_on_rotating_frame_high_speed',
         'remove_cups_from_rotating_frame')


def prepare(collection_plan, prior_manifest, output):
    """Called only after root freezes the new component-filter runtime."""
    from prepare_rigid_orientation_pilot_r10 import prepare_plan
    output = Path(output).resolve(); prior = json.loads(Path(prior_manifest).read_text())
    original.require(set(prior['layout_map']) == set(TASKS), 'exact five independently TRAIN-verified group tasks required')
    full = output.with_name(output.stem + '.fourteen_case_base_draft.json')
    base = prepare_plan(collection_plan, prior_manifest, full)
    base['version'] = VERSION
    base['cases'] = [next(row for row in base['cases'] if row['task'] == task) for task in TASKS]
    base['inputs_sha256'][str(Path(__file__).resolve())] = original.sha(__file__)
    process_helper=ROOT/'scripts/audit_train_component_flags_r10.py'
    base['inputs_sha256'][str(process_helper.resolve())]=original.sha(process_helper)
    base.update(expected_cases=5,maximum_CPU_workers=2,
        missing_rigid_prior_tasks=[],
        source_selection='fixed first TRAIN source/replica0 in each of five tasks with independently verified rigid priors',
        excluded_tasks='no button/wrap relation or remove-high relation is inferred or transferred',
        geometry_filter_required='r10_shared_native_TRAIN_mesh_v2',
        role_visibility='native-nonrenderable spoke2 remains directly_visible=false; any inferred pose is indirect',
        native_run_performed=False,automatic_annotation_promotion=False)
    base['plan_sha256'] = original.digest({k:v for k,v in base.items() if k != 'plan_sha256'})
    original.write(output,base);verify(output);return base


def verify(pointer):
    from collect_multitask_training_r10 import verify_plan as verify_collection, load_rigid_constraints
    from audit_train_resets_r10 import native_inputs
    plan = json.loads(Path(pointer).read_text())
    original.require(plan.get('version') == VERSION and plan.get('split') == 'train'
        and plan.get('plan_sha256') == original.digest({k:v for k,v in plan.items() if k != 'plan_sha256'}),
        'five-task rigid pilot identity changed')
    original.require(plan.get('expected_cases') == 5 and plan.get('maximum_CPU_workers') == 2
        and [row['task'] for row in plan['cases']] == list(TASKS), 'exact five preregistered tasks required')
    collection = verify_collection(plan['collection_plan_path'])
    original.require(plan['protocol'] == collection['protocol']
        and original.sha(plan['collection_plan_path']) == plan['collection_plan_sha256'], 'new collection/runtime identity changed')
    original.require(plan['frames'] == 16 and plan['hold_ticks'] == 15
        and plan['teacher_used'] is False and plan['GPU_used'] is False
        and plan['TEST_used'] is False and plan['automatic_annotation_promotion'] is False,
        'fixed TRAIN diagnostic budget/boundary changed')
    required = {**collection['inputs'], **native_inputs(plan['protocol']),
        str(Path(__file__).resolve()):original.sha(__file__),
        str(ROOT/'scripts/audit_train_component_flags_r10.py'):original.sha(ROOT/'scripts/audit_train_component_flags_r10.py'),
        str(Path(original.__file__).resolve()):original.sha(original.__file__)}
    original.require(all(plan['inputs_sha256'].get(path) == value for path,value in required.items()),
                     'pilot omitted a current runtime/collector/native source')
    prior_path = (ROOT / plan['rigid_prior_manifest']).resolve()
    original.require(original.sha(prior_path) == plan['rigid_prior_manifest_sha256'], 'TRAIN rigid prior manifest changed')
    prior = json.loads(prior_path.read_text())
    original.require(set(prior['layout_map']) == set(TASKS) and prior['TRAIN_only']
        and prior['TEST_used'] is False, 'unvalidated/inherited TRAIN relation forbidden')
    original.require(all(plan['inputs_sha256'].get(k) == v for k,v in prior['inputs_sha256'].items()),
                     'pilot omitted raw rigid calibration proof')
    for path,expected in plan['inputs_sha256'].items():
        original.require(original.sha(path) == expected, 'pilot source/asset changed: ' + path)
    for row in plan['cases']:
        first = next(c for c in collection['cases'] if c['task'] == row['task'] and c['split'] == 'train' and c['replica'] == 0)
        original.require(row['source'] == first['source'] and row['fixed_collection_case_id'] == first['case_id'],
                         'pilot fixed first TRAIN source changed')
        prior_row = prior['layout_map'][row['task']]; layout = row['rigid_layout']
        original.require(layout and Path(layout['path']).resolve() == (ROOT / prior_row['path']).resolve()
            and layout['sha256'] == prior_row['sha256'], 'pilot group does not match independent TRAIN evidence')
        original.require(load_rigid_constraints(layout['path'], row['task']), 'empty TRAIN rigid relation')
    return plan


def run_case(pointer, task, output):
    """Reuse unchanged tracker/metric loop; process-local verifier scope only."""
    plan = verify(pointer); original.require(task in TASKS, 'unknown rigid-pilot task')
    from our_mani.multitask.training_geometry import ASSET_VERSION
    original.require(ASSET_VERSION == plan['geometry_filter_required'], 'new component-filter geometry runtime required')
    previous_verifier, previous_version = original.verify_plan, original.VERSION
    try:
        original.verify_plan, original.VERSION = verify, VERSION
        result = original.run_case(pointer, task, output)
        from audit_train_component_flags_r10 import process_ancestry
        original.write(Path(output)/'native_process_receipt.json',dict(task=task,
            process_id=os.getpid(),process_ancestry=process_ancestry(),plan_sha256=plan['plan_sha256'],
            wrapper_sha256=original.sha(__file__),inherited_loop_sha256=original.sha(original.__file__)))
        return result
    finally:
        original.verify_plan, original.VERSION = previous_verifier, previous_version


def run_queue(pointer, output, *, workers=2):
    from audit_train_resets_r10 import require_completed_delivery
    plan = verify(pointer); gate = require_completed_delivery(); output = Path(output).resolve()
    original.require(type(workers) is int and 1 <= workers <= 2, 'at most two fresh CPU native processes')
    original.require(output.is_relative_to(original.BASE.resolve()) and not output.exists(), 'fresh rigid pilot output required')
    output.mkdir(parents=True)
    def one(task):
        verify(pointer); original.require(require_completed_delivery() == gate, 'delivery guard changed')
        child_out = output / task; log = output / (task + '.native.log')
        command = ['bash',str(ROOT/'scripts/run_local_sim.sh'),str(Path(__file__).resolve()),
            '--plan',str(Path(pointer).resolve()),'--child-task',task,'--out',str(child_out)]
        with log.open('wb') as stream:
            process = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'})
            timeout = False
            try:code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timeout = True;os.killpg(process.pid,signal.SIGTERM)
                try:code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        report = child_out/'report.json'
        process_path=child_out/'native_process_receipt.json'
        completed=code == 0 and report.exists() and process_path.exists()
        process_record=json.loads(process_path.read_text()) if process_path.exists() else None
        if completed:
            actual=json.loads(report.read_text())
            original.require(actual['version']==VERSION and actual['complete'] is True and actual['task']==task
                and actual['plan_sha256']==plan['plan_sha256'] and actual['planned_frames']==16
                and actual['counted_hold_ticks']==actual['actual_frames']-1
                and actual['teacher_used'] is False and actual['GPU_used'] is False
                and process_record['plan_sha256']==plan['plan_sha256']
                and process.pid in process_record['process_ancestry'],'rigid-pilot child identity/clock differs')
        result = dict(task=task,completed=completed,returncode=code,timed_out=timeout,
            child_launcher_pid=process.pid,launch_command=command,report=str(report.relative_to(ROOT)),
            native_python_pid=process_record['process_id'] if process_record else None,
            native_process_receipt=str(process_path.relative_to(ROOT)),
            native_process_receipt_sha256=original.sha(process_path) if process_path.exists() else None,
            report_sha256=original.sha(report) if report.exists() else None,
            log=str(log.relative_to(ROOT)),log_sha256=original.sha(log),counted_as_policy_failure=False)
        original.write(output/(task+'.parent_receipt.json'),result);return result
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one,task) for task in TASKS]
        for future in as_completed(futures):
            row = future.result();results.append(row);print(json.dumps(row),flush=True)
    verify(pointer);original.require(require_completed_delivery() == gate,'delivery guard changed')
    native_pids=[row['native_python_pid'] for row in results if row['completed']]
    original.require(len(native_pids)==len(set(native_pids)),'one fresh Python process per rigid TRAIN case')
    results.sort(key=lambda row:TASKS.index(row['task']))
    result = dict(version=VERSION,complete=len(results) == 5 and all(r['completed'] for r in results),
        expected=5,results=results,workers=workers,teacher_used=False,GPU_used=False,
        plan_sha256=plan['plan_sha256'],release_gate=gate,annotation_changes_written=False,
        automatic_annotation_promotion=False,indirect_children_are_not_visible=True)
    original.write(output/'queue_report.json',result);return result


def summarize(directory, pointer, output):
    """Retain raw accuracy/ambiguity and report indirect support separately."""
    from analyze_orientation_pilot_r10 import summarize_body
    plan=verify(pointer);directory=Path(directory).resolve()
    queue_path=directory/'queue_report.json';queue=json.loads(queue_path.read_text())
    original.require(queue.get('complete') is True and queue.get('expected')==5
        and queue['plan_sha256']==plan['plan_sha256']
        and [row['task'] for row in queue['results']]==list(TASKS),'five complete fixed pilot records required')
    task_bodies={};indirect={};reports={}
    for row in queue['results']:
        path=ROOT/row['report'];original.require(original.sha(path)==row['report_sha256'],'native pilot record changed')
        record=json.loads(path.read_text());sensor=record['causal_sensor_observations']
        original.require(original.sha(ROOT/sensor['path'])==sensor['sha256'],'current RGBD archive changed')
        original.require(record['complete'] and record['TRAIN_only'] and record['TEST_used'] is False
            and record['task']==row['task'] and record['plan_sha256']==plan['plan_sha256']
            and record['teacher_used'] is False and record['GPU_used'] is False
            and record['annotation_changes_written'] is False,'pilot boundary differs')
        ids=record['frames'][0]['metric_only']['body_ids'];variants={};pairs={}
        for variant in record['variants']:
            variants[variant]={role:summarize_body([frame['metric_only']['registration_errors'][variant][role]
                for frame in record['frames']]) for role in ids}
            if not variant.endswith('__rigid_group'):continue
            direct=variant.removesuffix('__rigid_group')+'__direct';pair={}
            for role in ids:
                values=[(frame['tracker_diagnostics'][direct]['bodies'][role],
                         frame['tracker_diagnostics'][variant]['bodies'][role]) for frame in record['frames']]
                original.require(all(a['directly_visible']==b['directly_visible'] for a,b in values),
                    'adding rigid prior changed actual direct visual evidence')
                pair[role]=dict(frames=len(values),direct_visible_frames=sum(a['directly_visible'] for a,b in values),
                    direct_valid_frames=sum(a['registration_valid'] for a,b in values),
                    group_valid_frames=sum(b['registration_valid'] for a,b in values),
                    group_indirect_frames=sum(b['registration_valid'] and not b['directly_visible'] for a,b in values),
                    indirect_pose_is_directly_visible=False)
            pairs[variant]=pair
        task_bodies[row['task']]=dict(actual_frames=record['actual_frames'],planned_frames=16,
            counted_hold_ticks=record['counted_hold_ticks'],shortened_coverage=record['shortened_coverage'],variants=variants,
            CAD_symmetry_diagnostic=record['CAD_symmetry_diagnostic'])
        indirect[row['task']]=pairs;reports[row['task']]=dict(path=str(path.relative_to(ROOT)),sha256=original.sha(path))
    result=dict(version=VERSION,complete=True,TRAIN_only=True,TEST_used=False,expected_cases=5,
        task_bodies=task_bodies,indirect_evidence=indirect,native_reports=reports,
        plan_sha256=plan['plan_sha256'],queue_sha256=original.sha(queue_path),analysis_source_sha256=original.sha(__file__),
        annotation_changes_written=False,automatic_annotation_promotion=False,orientation_verified_for_policy=False,
        interpretation=['full is a diagnostic hypothesis, not an observability certificate',
            'child poses supported by parent TRAIN rigid calibration remain indirect and non-visible',
            'raw SO3, directed/undirected axes, ambiguity and registration gaps remain metric-only',
            'square ring/peg finite symmetry is not continuous rotational invariance',
            'one source/short hold cannot certify across-source grasp/occlusion observability'])
    original.write(output,result);return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--collection-plan');parser.add_argument('--prior-manifest');parser.add_argument('--plan')
    parser.add_argument('--child-task');parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--analyze',metavar='PILOT_DIRECTORY')
    args = parser.parse_args();original.require(sum((args.prepare,args.run,args.child_task is not None,args.analyze is not None)) == 1,'choose one mode')
    if args.prepare:
        original.require(args.collection_plan and args.prior_manifest,'frozen new collection and TRAIN rigid prior required')
        value = prepare(args.collection_plan,args.prior_manifest,args.out)
    elif args.analyze:
        original.require(args.plan,'frozen plan required');value=summarize(args.analyze,args.plan,args.out)
    else:
        original.require(args.plan,'frozen five-case pilot plan required')
        value = run_case(args.plan,args.child_task,args.out) if args.child_task else run_queue(args.plan,args.out,workers=args.workers)
    print(json.dumps(dict(complete=value.get('complete'),plan_sha256=value.get('plan_sha256'))))


if __name__ == '__main__':main()
