"""One fresh, SHA-gated R10 native case; planning/imports do not start a simulator.

The parent queue owns the frozen lock and stage release. This child never tunes
on TEST, supplies metrics to an actor, reuses another native environment or
turns infrastructure errors into task failures. Videos copy already observed
frames; encoding happens after successful native cleanup.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import (METHODS,atomic_json,case_identity,digest,
    file_sha,native_completion_record,verify_inputs)
from our_mani.multitask.protocol import (MultitaskInfrastructureError,
    MultitaskProtocolError,validate_protocol)
from our_mani.multitask.reset_parity import profile_problems
from our_mani.multitask.specs import TASK_SPECS,get_task_spec

BASE=ROOT/'artifacts/framework/multitask_r10'
FFMPEG=ROOT/'.conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
CAMERAS=('front','8','16','36')
EXECUTOR='released_waypoints_counted_actuator_author_rgbd_terminal_v7'


class ExplicitPolicyFailure(Exception):
    """Only malformed/nonfinite policy outputs; backend exceptions are not this."""


def validate_actor_output(command,robot):
    """Validate the returned public value before entering the native backend."""
    if not isinstance(command,dict) or command.get('command') not in ('hold','joint_trajectory'):
        raise ExplicitPolicyFailure('unknown_public_actor_command')
    if command['command'] == 'joint_trajectory':
        if not isinstance(command.get('opening'),(int,float,np.number)):
            raise ExplicitPolicyFailure('malformed_actor_gripper_reference')
        try:
            joints=np.asarray(command.get('joints'),dtype=float)
            opening=float(command.get('opening'))
        except (TypeError,ValueError) as error:
            raise ExplicitPolicyFailure('malformed_actor_joint_or_gripper_reference') from error
        if joints.shape != np.asarray(robot.joints).shape or not np.isfinite(joints).all():
            raise ExplicitPolicyFailure('invalid_actor_joint_reference')
        if not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ExplicitPolicyFailure('invalid_actor_gripper_reference')


def require(value,message):
    if not value:
        raise MultitaskProtocolError(message)


def clean(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {key:clean(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return [clean(item) for item in value]
    return value


def read(path): return json.loads(Path(path).read_text())


def absolute(path):
    return (ROOT/Path(path)).resolve()


def locked_digest(path,files):
    path=Path(path).resolve()
    matches={expected for name,expected in files.items() if absolute(name) == path}
    require(len(matches) == 1,'used input is missing or conflicting in explicit frozen lock: '+str(path))
    expected=next(iter(matches))
    require(path.is_file() and file_sha(path) == expected,'frozen input changed: '+str(path))
    return expected


def locked_pointer(pointer,files):
    if isinstance(pointer,str):
        path=absolute(pointer);expected=locked_digest(path,files)
    else:
        require(isinstance(pointer,dict) and {'path','sha256'} <= set(pointer),'frozen file pointer needs path/sha256')
        path=absolute(pointer['path']);expected=locked_digest(path,files)
        require(pointer['sha256'] == expected,'pointer and comparison lock disagree')
    return path,expected


def required_local_sources(method, *, split=None):
    paths=[Path(__file__),*[ROOT/'our_mani/multitask'/name for name in (
        'evaluation.py','protocol.py','simulation.py','native_tasks.py','specs.py','maps.py',
        'reset_parity.py','actuation.py','safe_native_mesh.py')],ROOT/'our_mani/aligned/protocol.py',
        ROOT/'our_mani/sensor_alignment/author_rgbd.py',ROOT/'our_mani/contracts.py',
        ROOT/'our_mani/reachability.py',ROOT/'scripts/run_local_sim.sh']
    # PhysMani TRAIN/VAL never executes the offline five-model TEST coverage
    # checker. TEST and all ours methods still bind that checker explicitly.
    # An unspecified split remains conservative for existing callers.
    if method != 'physmani' or split not in ('train','validation'):
        paths.append(ROOT/'our_mani/multitask/release_preflight.py')
    if method == 'physmani':
        paths += [ROOT/'our_mani/aligned/physmani.py',ROOT/'our_mani/multitask/physmani_executor_v7.py',
            ROOT/'our_mani/ablations/physmani_executor_v4.py',
            *[ROOT/'our_mani/sensor_alignment'/name for name in ('physmani_executor_v5.py','physmani_executor_v6.py')],
            ROOT/'scripts/run_aligned_gpu.sh']
    else:
        paths += [ROOT/'our_mani/multitask'/name for name in ('factory.py','pipeline.py','interaction.py',
            'candidate_success.py','forecasting.py','mode_observer.py','perception.py','public_roles.py','training_geometry.py',
            'component_flags.py','collision_geometry.py')]
        paths += [ROOT/'our_mani/intercept_planner.py']
        if method.startswith('context_'):
            paths += [ROOT/'our_mani/multitask/context_se3.py']
    return paths


def r09_release(config,files):
    path,expected=locked_pointer(config.get('r09_completed_release',
        'artifacts/framework/aligned_r09/final_report.json'),files)
    report=read(path)
    require(report.get('complete') is True and report.get('protocol_acceptance_passed') is True
        and report.get('total_test_runs') == report.get('expected_test_runs') == 400
        and not report.get('acceptance_errors'),'native R10 is locked until accepted formal R09 400/400')
    return dict(path=str(path),sha256=expected,formal_runs=400,
        development_performance_screen_passed=report.get('development_performance_screen_passed'),
        success_rate_benefit_inferred=False)


def test_release(config,files,config_sha):
    require(config.get('formal_comparison_ready') is True,'TEST requires the frozen formal R10 protocol')
    path,expected=locked_pointer(config.get('r10_validation_release'),files)
    release=read(path)
    require(release.get('complete') is True and release.get('protocol_acceptance_passed') is True
        and release.get('split') == 'validation' and release.get('TEST_used') is False
        and release.get('formal_test_permitted') is True
        and release.get('expected_source_cases') == 275
        and release.get('expected_method_cases') == release.get('completed_method_cases') == 1375,
        'full fixed five-method R10 VAL release is missing or incompatible')
    val_path,val_sha=locked_pointer(release.get('validated_input_lock'),files)
    val_lock=read(val_path)
    require(isinstance(val_lock.get('files'),dict) and val_lock['files'],'original validated VAL file lock is empty')
    verify_inputs(ROOT,val_lock['files'])
    from our_mani.multitask.evaluation import comparison_protocol_sha256
    validated_path,validated_sha=locked_pointer(release.get('validated_config'),val_lock['files'])
    validated_config=read(validated_path)
    require(release.get('config_sha256') == validated_sha
        and release.get('comparison_protocol_sha256') == comparison_protocol_sha256(validated_config)
        == comparison_protocol_sha256(config),
        'TEST changed a policy/model/external condition from the complete five-method VAL')
    for name,checksum in val_lock['files'].items():
        require(locked_digest(absolute(name),files) == checksum,'TEST lock changed or omitted a validated VAL input')
    from our_mani.multitask.release_preflight import require_frozen_coverage_certificate
    coverage=require_frozen_coverage_certificate(ROOT,release.get('model_coverage_certificate'),config,files)
    profile_path,profile_sha=locked_pointer(config.get('initial_condition_tolerance_profile'),files)
    profile=read(profile_path)
    problems=profile_problems(profile,evidence_root=ROOT)
    require(not problems,'TRAIN initial-condition profile is not frozen/valid: '+str(problems))
    evidence_path,evidence_sha=locked_pointer(profile['TRAIN_evidence'],files)
    return dict(path=str(path),sha256=expected,validated_input_lock=dict(path=str(val_path),sha256=val_sha),
        validated_config=dict(path=str(validated_path),sha256=validated_sha),
        comparison_protocol_sha256=release['comparison_protocol_sha256'],
        model_coverage=coverage,
        frozen_profile=dict(path=str(profile_path),sha256=profile_sha),
        TRAIN_repeat_evidence=dict(path=str(evidence_path),sha256=evidence_sha),
        performance_screen=release.get('development_performance_screen'),
        performance_screen_is_separate_from_protocol=True)


def select_source(manifest,split,task,raw):
    require(task in TASK_SPECS,'only the remaining fourteen native R10 tasks are supported')
    values=manifest.get('episodes',manifest.get('entries',[]))
    require(isinstance(values,list) and values,'source manifest is empty')
    if split == 'test':
        require(manifest.get('split') == 'test' and manifest.get('mapping_status') == 'verified'
            and manifest.get('episode_count') == 560 and len(values) == 560
            and manifest.get('training_calibration_or_parameter_selection_used') is False,
            'TEST requires the complete verified 560-source author-reset manifest')
        require(len({(row['task'],row['raw_episode_id']) for row in values}) == 560
            and {row['task'] for row in values} == set(TASK_SPECS)
            and all(sum(row['task'] == name for row in values) == 40 for name in TASK_SPECS),
            'official TEST coverage/uniqueness differs')
    selected=[]
    for original in values:
        row_split=original.get('split',manifest.get('split'))
        if row_split != split:
            continue
        source_raw=original.get('raw_episode_id',original.get('source_episode'))
        if original.get('task') == task and source_raw == raw:
            require(original.get('raw_episode_id') == source_raw and 'path' in original,
                'policy source manifest needs canonical raw_episode_id/path; preserve the exact original row SHA')
            row=dict(original)
            require(isinstance(row['path'],str) and row['path'],'source lacks its real calibration/archive path')
            selected.append((row,original))
    require(len(selected) == 1,'--task/--raw-episode-id must select one exact manifest source')
    return selected[0],values


def preflight(args):
    config_path,manifest_path,lock_path=map(absolute,(args.config,args.manifest,args.frozen_inputs))
    config,manifest,lock=read(config_path),read(manifest_path),read(lock_path)
    validate_protocol(config)
    files=lock.get('files')
    verify_inputs(ROOT,files)
    for path in (config_path,manifest_path,*required_local_sources(args.method,split=args.split)):
        locked_digest(path,files)
    require(type(args.replica) is int and args.replica >= 0,'replica is a nonnegative explicit budget')
    require(args.exploration_seed is None or (args.split == 'train' and args.method.endswith('_geometry')),
        'first-candidate exploration is TRAIN geometry only')
    (row,original),values=select_source(manifest,args.split,args.task,args.raw_episode_id)
    release=r09_release(config,files)
    formal_release=test_release(config,files,file_sha(config_path)) if args.split == 'test' else None
    if args.split == 'test':
        require(args.replica == 0 and args.exploration_seed is None,'official TEST cannot be an augmented/explored replica')
        require(Path(row['path']).is_dir() and row.get('mapping_status') == 'verified','official reset alias source missing')
    else:
        package='train_package_compressed' if args.split == 'train' else 'val_package_compressed'
        source=Path(row['path']).resolve()
        require(source.is_file() and source.name == 'data.h5' and package in source.parts,
            'TRAIN/VAL must use their real HDF5 camera source; native motion is generated, not replayed')
        if 'source_file_stat' in row:
            stat=source.stat();actual=dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)
            require(actual == row['source_file_stat'],'development source metadata changed')
    if args.demo_only or args.record_demo:
        require(args.split == 'test','videos are restricted to fixed first official TEST sources')
        first=next(item for item in values if item['task'] == args.task)
        require(row['raw_episode_id'] == first['raw_episode_id'],'one demo per method/task uses the fixed first source, not outcome selection')
        locked_digest(FFMPEG,files)
        policy_path=ROOT/'config/framework/demo_policy.json';locked_digest(policy_path,files)
        policy=read(policy_path)
        require(policy.get('videos_per_method_per_task') == 1 and policy.get('methods') == list(METHODS),
                'shared one-demo policy differs')
    if args.method == 'physmani':
        checkpoint=absolute(config.get('baseline_checkpoint',''))
        require(checkpoint.name == 'epoch_99999.pth' and config.get('baseline_decision_ticks') == 10
            and config.get('baseline_executor') == EXECUTOR,'use original released epoch99999 policy and terminal V7 actuator')
        locked_digest(checkpoint,files)
    else:
        pointers=config.get('learned_artifacts',{}).get(args.task,{})
        for name in ('geometry','interaction'):
            locked_pointer(pointers.get(name),files)
        for name in ('rigid_layout','colour'):
            if pointers.get(name) is not None:
                locked_pointer(pointers[name],files)
        if args.task.startswith(('insert_onto_rotating_peg','push_moving_button')):
            locked_pointer(pointers.get('colour'),files)
        predictor,ranking=args.method.split('_',1)
        if predictor == 'context': locked_pointer(config.get('context_checkpoint'),files)
        if ranking == 'value': locked_pointer(pointers.get(predictor+'_value'),files)
    out=absolute(args.out)
    require(out.is_relative_to(BASE.resolve()) and out != BASE.resolve(),'fresh child outputs must stay inside new desktop R10 artifacts')
    require(not out.exists() or not any(out.iterdir()),'preserve completed/partial case; choose a fresh child output directory')
    if args.demo_only:
        require('demo_only' in out.relative_to(BASE.resolve()).parts,'independent demo replays require an explicit demo_only output directory')
    else:
        require('demo_only' not in out.relative_to(BASE.resolve()).parts,'formal cases cannot write into demo-only output directories')
    source_raw=row['raw_episode_id']
    raw=source_raw if args.split == 'test' else source_raw+f'__policy_aug_r{args.replica}'
    identity=case_identity(split=args.split,task=args.task,raw_episode_id=raw,method=args.method,
        source_row=row,config_sha256=file_sha(config_path),inputs_sha256=file_sha(lock_path),
        parent_raw_episode_id=None if args.split == 'test' else source_raw,replica=args.replica,
        demo_only=args.demo_only,exploration_seed=args.exploration_seed)
    return dict(args=args,config=config,config_path=config_path,manifest_path=manifest_path,
        manifest_sha256=file_sha(manifest_path),row=row,original_row=original,out=out,identity=identity,
        files=files,lock_path=lock_path,lock_sha256=file_sha(lock_path),
        r09_release=release,r10_validation_release=formal_release,record_demo=bool(args.demo_only or args.record_demo))


def verify_preflight(prepared):
    require(file_sha(prepared['lock_path']) == prepared['lock_sha256'],'external frozen lock bytes changed during case')
    verify_inputs(ROOT,prepared['files'])
    require(file_sha(prepared['config_path']) == prepared['identity']['config_sha256']
        and file_sha(prepared['manifest_path']) == prepared['manifest_sha256'],'case config/source manifest changed')


def video_frame(observation):
    images=[]
    for name in CAMERAS:
        value=np.asarray(observation['observation/images'][name])
        require(value.shape == (256,256,3) and value.dtype == np.uint8,'demo requires existing four-view uint8 current frames')
        images.append(value.copy())
    return np.concatenate((np.concatenate(images[:2],axis=1),np.concatenate(images[2:],axis=1)),axis=0)


def save_mp4(path,frames):
    path=Path(path);require(frames,'video contains no actual observation frames')
    require(not path.exists() and FFMPEG.is_file() and os.access(FFMPEG,os.X_OK),'fresh video path/local FFmpeg required')
    temporary=path.with_suffix('.tmp.mp4')
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','512x512','-r','20','-i','pipe:0','-an','-c:v','libx264',
        '-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    failure=None
    try:
        for frame in frames:
            require(frame.shape == (512,512,3) and frame.dtype == np.uint8,'inconsistent demo frame')
            process.stdin.write(frame.tobytes())
    except Exception as error:
        failure=error
    finally:
        try: process.stdin.close()
        except BrokenPipeError as error: failure=failure or error
    stderr=process.stderr.read().decode(errors='replace');code=process.wait();process.stderr.close()
    require(failure is None and code == 0,'video infrastructure encoding failed: '+repr(failure)+' '+stderr[-1000:])
    decoded=subprocess.run([str(FFMPEG),'-hide_banner','-v','error','-nostats','-i',str(temporary),
        '-map','0:v:0','-progress','pipe:1','-f','null','-'],capture_output=True,text=True,check=True)
    count=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    require(count and count[-1] == len(frames),'not all recorded video frames decode')
    temporary.replace(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path),decoded_frame_count=count[-1],
        recorded_frame_count=len(frames),fps=20.,resolution=[512,512],reset_frame_included=True,
        video_duration_seconds=len(frames)/20.,native_simulation_duration_seconds=(len(frames)-1)/20.,
        extra_sensor_or_physics_calls_for_recording=False,ffmpeg_path=str(FFMPEG),ffmpeg_sha256=file_sha(FFMPEG))


class NativeBackend:
    """Delayed imports: instantiated only after preflight has passed all gates."""
    def environment(self,p):
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation,AlignedOfficialSimulation
        args,config,row=p['args'],p['config'],p['row']
        if args.split == 'test':
            env=AlignedOfficialSimulation(args.task,row['path'],protocol=config,
                seed=int(row.get('seed',config['seed'])),servo_lead=config.get('local_servo_lead',1.4))
        else:
            source=dict(task=args.task,raw_episode_id=row['raw_episode_id'],path=row['path'])
            if 'variation' in row: source['variation']=row['variation']
            env=AlignedDevelopmentSimulation(args.task,row['path'],source=source,split=args.split,
                protocol=config,seed=config['seed'],replica=args.replica,servo_lead=config.get('local_servo_lead',1.4))
        try:
            descriptions,_=(env.reset_official(row) if args.split == 'test' else
                env.reset_development(variation=row.get('variation')))
        except Exception as reset_error:
            # A failed reset never returns env to run_case; release it here.
            try:
                env.close()
            except Exception as cleanup_error:
                raise MultitaskInfrastructureError('native reset failed and cleanup failed: '
                    +repr(reset_error)+'; '+repr(cleanup_error)) from reset_error
            raise
        return env,descriptions

    def initial_conditions(self,env,observation,robot):
        from our_mani.multitask.reset_parity import capture_recorded_initial_conditions
        return capture_recorded_initial_conditions(env,observation,robot)

    def actor(self,p,env,descriptions):
        from our_mani.multitask.factory import create_actor
        return create_actor(ROOT,env,p['config'],p['args'].method,descriptions,
            purpose='train_collection' if p['args'].split == 'train' else
                    'validation_collection' if p['args'].split == 'validation' else 'evaluation',
            exploration_seed=p['args'].exploration_seed)

    def baseline(self,p,env,observation,robot):
        import torch
        from our_mani.aligned.physmani import build_policy
        from our_mani.multitask.physmani_executor_v7 import PhysManiAdapter
        torch.manual_seed(env.seed)
        config=dict(p['config'],baseline_output=str(p['out']/'disabled_wm_logger'))
        model=build_policy(ROOT,config,p['args'].task)
        adapter=PhysManiAdapter(ROOT,config,p['args'].task,model)
        adapter.reset(env,p['args'].task,int(env.reset_receipt['variation']),observation,robot)
        return adapter,model

    def dispatch(self,env,command):
        from our_mani.multitask.actuation import dispatch_counted_command
        return dispatch_counted_command(env,command)


def censored_attempts(actor,time_value,error):
    if actor is None: return []
    ledger=actor.controller.attempts
    if not ledger.finalized:
        return ledger.finish(None,time_value,'infrastructure_error:'+type(error).__name__)
    # An I/O failure after finalization still cannot release these labels as a
    # completed training/metric case. Preserve them only in a censored copy.
    rows=ledger.to_report()
    for row in rows:
        row.update(official_task_success=None,complete=False,censored=True,
                   outcome_reason='infrastructure_error_after_native_observation')
    return rows


def run_case(p,backend=None):
    verify_preflight(p)
    backend=backend or NativeBackend()
    args,config,out=p['args'],p['config'],p['out']
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'inputs.json',clean(dict(identity=p['identity'],manifest_sha256=p['manifest_sha256'],
        original_manifest_row_sha256=digest(p['original_row']),frozen_input_lock=dict(path=str(p['lock_path']),
        sha256=p['lock_sha256']),files=p['files'],entrypoint_sha256=file_sha(Path(__file__)),
        process_id=os.getpid(),argv=list(sys.argv),r09_release=p['r09_release'],r10_validation_release=p['r10_validation_release'])))
    env=actor=adapter=model=None
    started=time.perf_counter();compute_seconds=0.;steps=[];frames=[];last_tick=-1
    initial=alignment=None;policy_failure=None;simulation_time=0.
    def capture(observation,robot):
        nonlocal last_tick
        tick=int(env.ledger.ticks)
        if p['record_demo']:
            require(tick == last_tick+1,'recording must copy one existing observation for each counted tick')
            frames.append(video_frame(observation));last_tick=tick
    try:
        random.seed(int(config['seed']));np.random.seed(int(config['seed']))
        env,descriptions=backend.environment(p)
        observation,robot=env.observe()
        initial=backend.initial_conditions(env,observation,robot)
        require(env.ledger.ticks == 0 and env.time == 0,'fresh reset must finish before initial metric receipt')
        capture(observation,robot)
        if args.method == 'physmani':
            adapter,model=backend.baseline(p,env,observation,robot)
            continuation=None
        else:
            actor,continuation=backend.actor(p,env,descriptions)
        while env.ledger.ticks < 200 and env.ledger.remaining >= env.ledger.dt-1e-8 and not env._native_terminal_latched:
            if (ROOT/'runtime/framework/multitask_r10_STOP').exists():
                raise InterruptedError('controlled R10 stop request; current case is infrastructure-censored')
            before=env.ledger.ticks;decision_started=time.perf_counter()
            if actor is not None:
                command=actor.act(observation,robot,env.ledger.remaining)
                compute_seconds += time.perf_counter()-decision_started
                validate_actor_output(command,robot)
                execution=backend.dispatch(env,command)
                require(env.ledger.ticks == before+1,'ours execution must use exactly one native tick')
                steps.append(clean(dict(time=robot.timestamp,command=command,execution=execution,
                                        diagnostics=actor.last_diagnostics)))
                observation,robot=env.observe();capture(observation,robot)
            else:
                try:
                    action=adapter.predict(observation,robot)
                except ValueError as error:
                    if str(error) == 'nonfinite PhysMani action':
                        raise ExplicitPolicyFailure(str(error)) from error
                    raise
                try:
                    action=np.asarray(action,float)
                except (TypeError,ValueError) as error:
                    raise ExplicitPolicyFailure('malformed_PhysMani_action') from error
                if action.shape != (8,) or not np.isfinite(action).all() or np.round(action[7]) not in (0.,1.):
                    raise ExplicitPolicyFailure('malformed_or_nonfinite_PhysMani_action')
                execution=adapter.execute(env,action,observer=capture)
                require(before < env.ledger.ticks <= min(before+10,200),'baseline decision returned without 1..10 counted native ticks')
                steps.append(clean(dict(time=robot.timestamp,action=action,execution=execution)))
                observation,robot=env.observe() # normal current policy frame; recording used only tick observer frames
            simulation_time=float(env.time)
        end_reason=('native_success' if env._native_success_latched else
            'native_terminal_failure' if env._native_failure_latched else 'native_budget_exhausted')
    except ExplicitPolicyFailure as error:
        policy_failure=str(error);end_reason='explicit_policy_output_failure'
    except Exception as error:
        cleanup=None
        if env is not None:
            simulation_time=float(env.time)
            try: alignment=env.alignment_receipt()
            except Exception as receipt_error: alignment=dict(unavailable_due_to=repr(receipt_error))
            try: env.close()
            except Exception as close_error: cleanup=repr(close_error)
            env=None
        error_record=dict(version='r10_censored_infrastructure_case_v1',identity=p['identity'],complete=False,
            infrastructure_error=repr(error),traceback=traceback.format_exc(),cleanup_error=cleanup,
            counted_as_policy_failure=False,formal_metrics_included=False,process_id=os.getpid(),
            continuation_attempts=censored_attempts(actor,simulation_time,error),steps=steps,
            recorded_initial_conditions=initial,alignment_receipt=alignment,
            wall_seconds=time.perf_counter()-started,entrypoint_sha256=file_sha(Path(__file__)))
        atomic_json(out/'infrastructure_error.json',clean(error_record))
        raise
    try:
        require(env is not None,'native runtime unavailable at policy completion')
        simulation_time=float(env.time);ticks=int(env.ledger.ticks)
        success,fatal=bool(env._native_success_latched),bool(env._native_failure_latched)
        alignment=env.alignment_receipt()
        if adapter is not None: compute_seconds=float(adapter.compute_seconds)
        env.close();env=None
        verify_preflight(p)
        video=None
        if p['record_demo']:
            require(len(frames) == ticks+1,'video must use reset plus every actual counted tick')
            video=save_mp4(out/'case.mp4',frames)
        attempts=(actor.controller.attempts.finish(success,simulation_time,end_reason) if actor is not None else [])
        record=native_completion_record(p['identity'],process_id=os.getpid(),success=success,fatal=fatal,
            ticks=ticks,initial_conditions=initial,alignment_receipt=alignment,policy_failure=policy_failure)
        record.update(version='r10_native_fresh_case_v1',evaluation_process_pid=os.getpid(),
            raw_episode_filter=args.raw_episode_id,replica=args.replica,exploration_seed=args.exploration_seed,
            config_sha256=p['identity']['config_sha256'],source_row_sha256=p['identity']['source_row_sha256'],
            manifest_sha256=p['manifest_sha256'],frozen_input_lock=dict(path=str(p['lock_path']),sha256=p['lock_sha256']),
            entrypoint_sha256=file_sha(Path(__file__)),argv=list(sys.argv),compute_seconds=compute_seconds,
            wall_seconds=time.perf_counter()-started,simulation_seconds=simulation_time,steps=steps,
            native_end_reason=end_reason,cleanup_completed=True,video=video,
            source_raw_episode_id=args.raw_episode_id,source_motion_replayed=args.split == 'test',
            r09_completed_release=p['r09_release'],r10_validation_release=p['r10_validation_release'],
            continuation_attempts=attempts,unexecuted_candidates_labelled=False,
            first_selection_policy=actor.first_selection_policy if actor is not None else 'released_PhysMani_policy',
            downstream_selection_policy='geometry' if actor is not None else 'released_PhysMani_policy',
            actor_metrics=actor.metrics_report() if actor is not None else None,
            baseline_executor=EXECUTOR if adapter is not None else None,
            checkpoint_sha256=file_sha(absolute(config['baseline_checkpoint'])) if adapter is not None else None,
            recorded_video_in_formal_metrics=bool(p['record_demo'] and not args.demo_only),
            demo_selection='fixed first official manifest source, independent of outcome' if p['record_demo'] else None)
        if actor is not None:
            record.update(predictor=actor.predictor,continuation_selection_policy='single_value_selection_then_frozen_geometry',
                continuation_sha256=actor.controller.continuation_sha256,
                interaction_library_sha256=actor.controller.library.sha256,
                continuation_identity=continuation)
        atomic_json(out/'case.json',clean(record))
        return record
    except Exception as error:
        cleanup=None
        if env is not None:
            try: env.close()
            except Exception as close_error: cleanup=repr(close_error)
        atomic_json(out/'infrastructure_error.json',clean(dict(version='r10_censored_infrastructure_case_v1',
            identity=p['identity'],complete=False,infrastructure_error=repr(error),cleanup_error=cleanup,
            counted_as_policy_failure=False,formal_metrics_included=False,process_id=os.getpid(),
            continuation_attempts=censored_attempts(actor,simulation_time,error),
            recorded_initial_conditions=initial,alignment_receipt=alignment,steps=steps,
            wall_seconds=time.perf_counter()-started,traceback=traceback.format_exc())))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--frozen-inputs',required=True)
    parser.add_argument('--split',required=True,choices=('train','validation','test'))
    parser.add_argument('--task',required=True)
    parser.add_argument('--raw-episode-id',required=True)
    parser.add_argument('--method',required=True,choices=METHODS)
    parser.add_argument('--out',required=True)
    parser.add_argument('--replica',type=int,default=0)
    parser.add_argument('--exploration-seed',type=int)
    parser.add_argument('--record-demo',action='store_true')
    parser.add_argument('--demo-only',action='store_true')
    args=parser.parse_args()
    record=run_case(preflight(args))
    print(json.dumps(dict(complete=True,case=str(absolute(args.out)/'case.json'),
        raw_episode_id=record['raw_episode_id'],native_success=record['official_task_success'],
        ticks=record['counted_clock']['ticks'],demo_only=record['demo_only']),ensure_ascii=False))


if __name__ == '__main__': main()
