"""One independent second-TRAIN-source rigid registration validation.

No teacher, policy, GPU or collision query. Uses the unchanged same-frame
orientation pilot loop, with a process-local verifier and version scope only.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import pilot_orientation_registration_r10 as original
from our_mani.multitask.perception import RegistrationConfig

TASK='remove_cups_from_rotating_frame_high_speed'
VERSION='r10_high_remove_second_TRAIN_source_rigid_pilot_v1'


def selected_source(manifest):
    rows=[row for row in manifest['episodes'] if row['split']=='train' and row['task']==TASK]
    original.require(len(rows)==100,'audited original one hundred TRAIN parents required')
    return rows[1]


def required_sources(protocol):
    from audit_train_resets_r10 import native_inputs
    paths=[Path(__file__),Path(original.__file__),ROOT/'scripts/collect_multitask_training_r10.py',
        ROOT/'scripts/audit_train_resets_r10.py',ROOT/'scripts/audit_train_component_flags_r10.py',
        ROOT/'scripts/run_local_sim.sh',ROOT/'our_mani/sensor_alignment/author_rgbd.py',
        *sorted((ROOT/'our_mani').glob('*.py')),*sorted((ROOT/'our_mani/aligned').glob('*.py')),
        *[ROOT/'our_mani/multitask'/name for name in ('perception.py','mode_observer.py','safe_native_mesh.py',
            'component_flags.py','collision_geometry.py','training_geometry.py','maps.py','simulation.py',
            'reset_parity.py','native_tasks.py','specs.py','protocol.py')]]
    values={str(path.resolve()):original.sha(path) for path in paths};values.update(native_inputs(protocol));return values


def prepare(manifest_path,protocol_path,prior_path,output):
    manifest_path,protocol_path,prior_path=map(lambda p:Path(p).resolve(),(manifest_path,protocol_path,prior_path))
    manifest=json.loads(manifest_path.read_text());protocol=json.loads(protocol_path.read_text());prior=json.loads(prior_path.read_text())
    from our_mani.multitask.protocol import validate_protocol
    validate_protocol(protocol);source=selected_source(manifest)
    original.require(prior.get('TRAIN_only') is True and prior.get('TEST_used') is False,'independent TRAIN prior required')
    for path,expected in prior['inputs_sha256'].items():original.require(original.sha(path)==expected,'TRAIN prior evidence changed')
    layout=prior['layout_map'][TASK];group='train:'+TASK+':'+source['source_episode']
    original.require(group!=layout['source_group'],'validation must use a different TRAIN parent from fitted layout')
    cache=(ROOT/source['npz_path']).resolve();hdf=Path(source['source_path']).resolve()
    original.require(original.sha(cache)==source['npz_sha256'] and 'train_package_compressed' in hdf.parts,'TRAIN cache/source changed')
    inputs=required_sources(protocol);inputs.update(prior['inputs_sha256'])
    inputs.update({str(p):original.sha(p) for p in (manifest_path,protocol_path,prior_path,cache,hdf)})
    case=dict(task=TASK,source=dict(task=TASK,raw_episode_id=source['source_episode'],path=str(hdf),variation=source['variation']),
        replica=0,parent_source_group=group,offline_demonstration=dict(path=source['npz_path'],sha256=source['npz_sha256']),
        rigid_layout=dict(path=str((ROOT/layout['path']).resolve()),sha256=layout['sha256'],accepted_constraints=len(layout['accepted_relations'])))
    plan=dict(version=VERSION,split='train',cases=[case],protocol=protocol,frames=16,hold_ticks=15,
        inputs_sha256=inputs,manifest_path=str(manifest_path),protocol_path=str(protocol_path),prior_manifest=str(prior_path),
        fitted_source_group=layout['source_group'],source_index=1,expected_cases=1,maximum_CPU_workers=1,
        registration_config=asdict(RegistrationConfig()),hypotheses=['none_baseline','candidate_full_except_ball_none'],
        teacher_used=False,GPU_used=False,TEST_used=False,automatic_annotation_promotion=False,
        source_selection='second fixed original TRAIN registry source; independent from first-source same-task fitted relation',
        collision_asset_registration_covered=False,native_run_performed=False)
    plan['plan_sha256']=original.digest(plan);original.write(output,plan);verify(output);return plan


def verify(pointer):
    plan=json.loads(Path(pointer).read_text())
    original.require(plan.get('version')==VERSION and plan.get('split')=='train'
        and plan.get('plan_sha256')==original.digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'independent pilot plan changed')
    original.require(plan['expected_cases']==1 and plan['maximum_CPU_workers']==1 and len(plan['cases'])==1
        and plan['cases'][0]['task']==TASK and plan['frames']==16 and plan['hold_ticks']==15
        and plan['source_index']==1 and not any(plan[k] for k in ('teacher_used','GPU_used','TEST_used','automatic_annotation_promotion')),
        'independent TRAIN scope/budget changed')
    original.require(plan['protocol']==json.loads(Path(plan['protocol_path']).read_text())
        and plan['registration_config']==original.clean(asdict(RegistrationConfig())),'protocol/registration parameters changed')
    source=selected_source(json.loads(Path(plan['manifest_path']).read_text()));case=plan['cases'][0]
    original.require(case['source']==dict(task=TASK,raw_episode_id=source['source_episode'],path=source['source_path'],variation=source['variation'])
        and case['parent_source_group']!=plan['fitted_source_group'],'independent fixed source changed')
    prior=json.loads(Path(plan['prior_manifest']).read_text());layout=prior['layout_map'][TASK]
    original.require(layout['source_group']==plan['fitted_source_group']
        and case['rigid_layout']['sha256']==layout['sha256']
        and Path(case['rigid_layout']['path']).resolve()==(ROOT/layout['path']).resolve(),'independent same-task relation changed')
    required={**required_sources(plan['protocol']),**prior['inputs_sha256']}
    original.require(all(plan['inputs_sha256'].get(k)==v for k,v in required.items()),'runtime/raw prior source omitted')
    for path,expected in plan['inputs_sha256'].items():original.require(original.sha(path)==expected,'pilot source/input changed: '+path)
    return plan


def native_case(pointer,output):
    plan=verify(pointer);previous_verify,previous_version=original.verify_plan,original.VERSION
    try:
        original.verify_plan,original.VERSION=verify,VERSION
        result=original.run_case(pointer,TASK,output)
        from audit_train_component_flags_r10 import process_ancestry
        original.write(Path(output)/'native_process_receipt.json',dict(process_id=os.getpid(),
            process_ancestry=process_ancestry(),plan_sha256=plan['plan_sha256'],source_sha256=original.sha(__file__)))
        return result
    finally:original.verify_plan,original.VERSION=previous_verify,previous_version


def run(pointer,output):
    from audit_train_resets_r10 import require_completed_delivery
    plan=verify(pointer);gate=require_completed_delivery();output=Path(output).resolve()
    original.require(output.is_relative_to(original.BASE.resolve()) and not output.exists(),'fresh one-case pilot directory required')
    output.mkdir(parents=True);log=output/'native.log';child_out=output/TASK
    command=['bash',str(ROOT/'scripts/run_local_sim.sh'),str(Path(__file__).resolve()),
        '--plan',str(Path(pointer).resolve()),'--child','--out',str(child_out)]
    with log.open('wb') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1'})
        code=process.wait()
    report=child_out/'report.json';identity=child_out/'native_process_receipt.json'
    complete=code==0 and report.exists() and identity.exists()
    if complete:
        record=json.loads(report.read_text());process_record=json.loads(identity.read_text())
        original.require(record['complete'] is True and record['actual_frames']==16 and record['counted_hold_ticks']==15
            and record['plan_sha256']==plan['plan_sha256'] and process.pid in process_record['process_ancestry'],
            'independent native child identity/coverage differs')
    verify(pointer);original.require(require_completed_delivery()==gate,'R09 delivery gate changed')
    result=dict(version=VERSION,complete=complete,expected=1,workers=1,returncode=code,
        report=str(report.relative_to(ROOT)),report_sha256=original.sha(report) if report.exists() else None,
        native_process_receipt=str(identity.relative_to(ROOT)),native_process_receipt_sha256=original.sha(identity) if identity.exists() else None,
        launch_command=command,launcher_pid=process.pid,native_python_pid=process_record['process_id'] if complete else None,
        log=str(log.relative_to(ROOT)),log_sha256=original.sha(log),plan_sha256=plan['plan_sha256'],release_gate=gate,
        teacher_used=False,GPU_used=False,counted_as_policy_failure=False,annotation_changes_written=False)
    original.write(output/'queue_report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--child',action='store_true')
    parser.add_argument('--manifest');parser.add_argument('--protocol');parser.add_argument('--prior');parser.add_argument('--plan')
    parser.add_argument('--out',required=True);args=parser.parse_args()
    if args.prepare:value=prepare(args.manifest,args.protocol,args.prior,args.out)
    elif args.run:value=run(args.plan,args.out)
    else:value=native_case(args.plan,args.out)
    print(json.dumps(dict(complete=value.get('complete'),plan_sha256=value.get('plan_sha256'))))


if __name__=='__main__':main()
