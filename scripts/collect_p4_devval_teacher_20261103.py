"""Collect only the eight preassigned *new development VAL* teacher parents.

Default CPU preflight never launches Native. Explicit execution requires the
complete TRAIN16 owner and a prior TRAIN-only student-model freeze manifest.
Teacher policy remains truth B; its public50ms GRU observer is passive. These
are held-out supervised-development records, not an official benchmark VAL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / 'runtime/framework/local_mpc_20261102'
spec = importlib.util.spec_from_file_location('p4_frozen_shared_collection', ROOT/'scripts/collect_p4_teacher_train_20261103.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha, write_new, read_lines = shared.sha, shared.write_new, shared.read_lines


def resolve(value, family=None):
    return shared.relative_path(value, family=family)


def initial_xy(config):
    return tuple(config['initial_target_position'][:2])


def validate_plan(plan_path):
    raw = Path(plan_path).read_bytes(); plan = json.loads(raw)
    if plan.get('schema') != 'P4_newdevVAL_truthB_shadow50_teacher_20261103_v1':
        raise ValueError('explicit new development VAL plan required')
    if plan.get('split') != 'val' or plan.get('official_VAL_or_TEST_read') is not False:
        raise ValueError('new development val only; no official VAL/TEST')
    jobs = plan.get('jobs', [])
    if len(jobs) != 8 or len({j['parent_id'] for j in jobs}) != 8:
        raise ValueError('eight preassigned independent newdevVAL parents required')
    shared.check_sources(plan['source_sha256'])
    train_path = resolve(plan['TRAIN_plan']['path'])
    if sha(train_path) != plan['TRAIN_plan']['sha256']:
        raise RuntimeError('preassigned TRAIN reference changed')
    train = json.loads(train_path.read_text())
    if len(train['jobs']) != 16 or any(j['split'] != 'train' for j in train['jobs']):
        raise ValueError('fixed TRAIN16 reference required')
    train_parents = {j['parent_id'] for j in train['jobs']}
    train_seeds = {j['seed'] for j in train['jobs']}
    train_xy = {initial_xy(json.loads(resolve(j['config']).read_text())) for j in train['jobs']}
    configs, coordinates = [], set()
    for job in jobs:
        if job.get('split') != 'val' or job['parent_id'] in train_parents or job['seed'] in train_seeds:
            raise ValueError('newdevVAL parent/split/seed must be separate from TRAIN')
        path = resolve(job['config'])
        if sha(path) != job['config_sha256']:
            raise RuntimeError('newdevVAL config identity changed')
        cfg = json.loads(path.read_text()); meta = cfg.get('P4_collection_metadata', {})
        if (cfg.get('dataset_use') != 'val' or cfg.get('seed') != job['seed'] or
                any(meta.get(k) != job[k] for k in ('parent_id','split','episode_id','variant_id')) or
                meta.get('parent_assigned_before_episode_generation') is not True):
            raise ValueError('exact parent/split/seed/episode/variant binding required')
        xy = initial_xy(cfg)
        if xy in train_xy or xy in coordinates:
            raise ValueError('newdevVAL initial XY must not reuse TRAIN or another VAL parent')
        coordinates.add(xy)
        required = dict(preserve_native_height=True, simulation_seconds=8., physics_dt=.05,
            control_dt=.1, decision_ticks=2, horizon_steps=8, num_candidates=8,
            response_factory='coupled_joint_20261103_v1',response_controller='coupled',action_space='joint',
            use_joint_goal_planner=True, public_motion_shadow=True,public_shadow_save_frames=True,
            gripper_completion_latch=True,formal_evaluation=False)
        if any(cfg.get(k) != v for k,v in required.items()) or cfg.get('cem_iteration_sequences') != [88,88,80]:
            raise ValueError('reviewed B joint/passivepublic50/Nativeheight contract required')
        if cfg.get('cem_sequences_total') != 256 or any(cfg.get(k) for k in (
                'public_grasp_prototype','public_grasp_active_visual_v4','public_grasp_active_visual_v41',
                'public_grasp_active_visual_v42','public_pregrasp_control','diagnostic_no_proxy_attachment')):
            raise ValueError('unchanged teacher budget; no public controller or counterfactual')
        if job['scene'] not in ('static','uniform','speed_change'):
            raise ValueError('unsupported worker scene')
        if (not .05 <= xy[0] <= .35 or not -.45 <= xy[1] <= .45 or
                cfg['uniform_velocity'][2] != 0. or cfg['changed_velocity'][2] != 0.):
            raise ValueError('declared planar Native-area scenarios required')
        if cfg.get('task') != 'pick_moving_target_on_the_table' or '/train_package_compressed/' not in cfg.get('calibration_train_episode',''):
            raise ValueError('shared original TRAIN calibration only; no official VAL/TEST source')
        resolve(job['path'], FAMILY)
        resolve(job['evaluation'], ROOT/'artifacts/framework/local_mpc_20261102')
        configs.append(cfg)
    if len({j['seed'] for j in jobs}) != 8:
        raise ValueError('newdevVAL seeds must also be unique')
    return plan, raw, configs


def verify_TRAIN_reaped(plan):
    """Inspect exit receipts only, never use TRAIN results to change VAL jobs."""
    train_plan = json.loads(resolve(plan['TRAIN_plan']['path']).read_text())
    owner = resolve(plan['TRAIN_plan']['owner'], FAMILY)
    result = json.loads((owner/'result.json').read_text())
    if result.get('complete') is not True or result.get('episodes') != 16:
        raise RuntimeError('TRAIN16 collection must be fully complete before newdevVAL')
    assigned = {(j['episode_id'],j['parent_id']) for j in train_plan['jobs']}
    recorded = {(r['episode_id'],r['parent_id']) for r in result.get('results',[])}
    if assigned != recorded or any(r.get('split') != 'train' for r in result['results']):
        raise RuntimeError('complete actual TRAIN owner assignment receipts required')
    receipts = {}
    for job in train_plan['jobs']:
        case = resolve(job['path'], FAMILY)
        actual = case.parent/(case.name+'.owner.actual_exit.json')
        worker = case/'worker_exit.json'
        if (json.loads(actual.read_text()).get('actual_process_returncode') != 0 or
                json.loads(worker.read_text()).get('exit_code') != 0):
            raise RuntimeError('every actual TRAIN child must have reaped successfully')
        receipts[str(actual.relative_to(ROOT))] = sha(actual)
        receipts[str(worker.relative_to(ROOT))] = sha(worker)
    return dict(owner_result_sha256=sha(owner/'result.json'), actual_exit_receipts_sha256=receipts,
                all_16_actual_children_reaped=True, VAL_assignment_adapted_to_TRAIN_results=False)


def verify_model_freeze(path):
    """A separate immutable identity is bound after TRAIN fit, before VAL."""
    path = Path(path).resolve(); raw = path.read_bytes(); manifest = json.loads(raw)
    if (manifest.get('schema') != 'P4_model_freeze_before_newdevVAL_v1' or
            manifest.get('training_complete') is not True or manifest.get('fit_split') != 'train' or
            manifest.get('new_devVAL_used_for_fit') is not False or manifest.get('formal_TEST_used') is not False):
        raise ValueError('completed TRAIN-only model freeze before newdevVAL required')
    files = manifest.get('files_sha256')
    if not isinstance(files,dict) or not files or manifest.get('checkpoint') not in files:
        raise ValueError('explicit checkpoint and nonempty frozen model/source SHA required')
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in files):
        raise ValueError('frozen model/source paths must be project-relative without parent traversal')
    shared.check_sources(files)
    return dict(path=str(path),sha256=sha(path),files_sha256=files,checkpoint=manifest['checkpoint'],
        model_is_frozen_but_not_used_by_truth_teacher=True), raw


def audit_completed_job(job, config, *, expected_split='val'):
    """New explicit-role audit. Never call the old hardcoded-TRAIN audit."""
    if expected_split != 'val' or job.get('split') != expected_split or config.get('dataset_use') != expected_split:
        raise ValueError('explicit newdevVAL role required before opening the case')
    case = resolve(job['path'],FAMILY)
    owner = json.loads((case.parent/(case.name+'.owner.actual_exit.json')).read_text())
    worker = json.loads((case/'worker_exit.json').read_text())
    result = json.loads((case/'result.json').read_text())
    inputs = json.loads((case/'run_inputs.json').read_text())
    if owner.get('actual_process_returncode') != 0 or worker.get('exit_code') != 0:
        raise RuntimeError('actual successful child/worker exits required')
    if (result.get('complete') is not True or result.get('source_unchanged') is not True or
            inputs.get('TEST_read') is not False or inputs.get('config') != config or inputs.get('scene') != job['scene']):
        raise RuntimeError('exact completed unchanged newdevVAL teacher config required')
    archives = shared.validate_archived_sources(case,inputs.get('source_sha256'))
    native = inputs.get('native_sources_sha256',{})
    expected = {'third_party/RLBench/rlbench/tasks/'+config['task']+'.py':config['native_task_source_sha256'],
                'third_party/RLBench/rlbench/task_ttms/'+config['task']+'.ttm':config['native_task_scene_sha256']}
    if any(native.get(k) != v for k,v in expected.items()):
        raise RuntimeError('original Native source receipts failed')
    snapshot=json.loads((case/'initial_snapshot.json').read_text())
    if (snapshot.get('native_height_preserved') is not True or
            abs(snapshot['state']['object_position'][2]-snapshot['native_default_target_pose_before_override'][2])>1e-8):
        raise RuntimeError('original Native table height required')
    states=[snapshot['state'],*read_lines(case/'native_ticks.jsonl')]
    sys.path.insert(0,str(ROOT))
    from our_mani.local_mpc_control_20261103.native_task_evaluation import write_case_evaluation
    report=write_case_evaluation(case,resolve(job['evaluation']))
    public,forecasts=shared.audit_public_capture(case,states,job['frozen_GRU_checkpoint_sha256'])
    labels=shared.contact_labels(case,states,forecasts,job['parent_id'])
    positives=[r for r in labels if r['accepted_positive']]
    return dict(episode_id=job['episode_id'],parent_id=job['parent_id'],split=expected_split,
        new_development_VAL_only=True,official_VAL_or_TEST_read=False,
        actual_owner_exit=owner,actual_worker_exit=worker,source_snapshot_evidence=archives,
        original_Native_sources_passed=True,native_height_preserved=True,task_success=report['metrics'],
        public_capture=public,contact_labels=labels,positive_contact_labels=len(positives),
        causal_macro_anchors=sum(a['macro_100ms_anchor'] for r in positives for a in r['causal_50ms_anchors']),
        completion_latched_actual_protocol=True,old_NN_finger_response_requalified=False,
        success_failure_both_retained=True,finished_utc=datetime.now(timezone.utc).isoformat())


def run(plan_path,owner_path=None,*,execute=False,model_freeze=None):
    plan,raw,configs=validate_plan(plan_path)
    if not execute:
        return dict(preflight_passed=True,Native_started=False,new_devVAL_parents=8,
            source_files_checked=len(plan['source_sha256']),TRAIN_parent_seed_and_initial_XY_disjoint=True,
            execution_checks_deferred=['TRAIN16 actual reaped owner','completed TRAIN-only model freeze'],
            execution_requires_explicit_execute=True)
    if model_freeze is None:
        raise ValueError('--model-freeze required before any newdevVAL launch')
    train_evidence=verify_TRAIN_reaped(plan)
    model,model_raw=verify_model_freeze(model_freeze)
    owner=Path(owner_path).resolve()
    if FAMILY not in owner.parents or owner.exists():
        raise ValueError('fresh runtime-family owner required')
    for job in plan['jobs']:
        if resolve(job['path']).exists() or resolve(job['evaluation']).exists():
            raise FileExistsError('fresh newdevVAL case/evaluation outputs required')
    batch=resolve(plan['batch_output_parent'],FAMILY)
    batch.mkdir(parents=True,exist_ok=False);owner.mkdir(parents=True,exist_ok=False)
    write_new(owner/'inputs.json',dict(exact_preassigned_plan=plan,plan_sha256=sha(plan_path),
        runner_sha256=sha(__file__),TRAIN_reaped=train_evidence,student_model_freeze=model,
        teacher_policy_uses_student_model=False,scope='new development val teacher labels, not official benchmark evaluation'))
    results=[]
    for job,cfg in zip(plan['jobs'],configs):
        if shared.stop_requested(owner,batch):
            write_new(owner/'stopped.json',dict(complete=False,user_stop=True,results=results));return 2
        if Path(plan_path).read_bytes()!=raw or Path(model_freeze).read_bytes()!=model_raw:
            raise RuntimeError('VAL assignment or model freeze manifest changed')
        shared.check_sources(plan['source_sha256']);shared.check_sources(model['files_sha256'])
        if sha(resolve(job['config']))!=job['config_sha256']:
            raise RuntimeError('prepared newdevVAL config changed')
        print(json.dumps(dict(event='start',episode_id=job['episode_id'],parent_id=job['parent_id'],split='val')),flush=True)
        process=subprocess.Popen([sys.executable,str(ROOT/'scripts/launch_local_mpc_20261102.py'),
            '--mode','p1','--scene',job['scene'],'--config',job['config'],'--output',job['path']],cwd=ROOT)
        while True:
            try:code=process.wait(timeout=.25);break
            except subprocess.TimeoutExpired:
                if shared.stop_requested(owner,batch) and not (batch/'STOP').exists():
                    with (batch/'STOP').open('x') as stream:stream.write('newdevVAL owner STOP propagated; reap actual child\n')
        if code!=0:
            write_new(owner/'infrastructure_or_stop.json',dict(complete=False,episode_id=job['episode_id'],
                actual_launcher_returncode=code,user_stop_requested=shared.stop_requested(owner,batch),results=results))
            return 2 if shared.stop_requested(owner,batch) else code
        shared.check_sources(plan['source_sha256']);shared.check_sources(model['files_sha256'])
        row=audit_completed_job(job,cfg,expected_split='val')
        write_new(owner/(job['episode_id']+'.json'),row);results.append(row)
        print(json.dumps(dict(event='finished',episode_id=job['episode_id'],split='val',
            positive_contact_labels=row['positive_contact_labels'],causal_macro_anchors=row['causal_macro_anchors'])),flush=True)
    if Path(plan_path).read_bytes()!=raw or Path(model_freeze).read_bytes()!=model_raw:
        raise RuntimeError('VAL plan or model freeze changed')
    shared.check_sources(plan['source_sha256']);shared.check_sources(model['files_sha256'])
    write_new(owner/'result.json',dict(complete=True,episodes=8,split='val',results=results,
        new_development_VAL_only=True,official_VAL_or_TEST_read=False,training_started=False,
        failed_task_episodes_retained=True,student_model_frozen_before_collection=True))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--owner',type=Path)
    parser.add_argument('--model-freeze',type=Path);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute and args.owner is None:parser.error('--owner required with --execute')
    result=run(args.plan,args.owner,execute=args.execute,model_freeze=args.model_freeze)
    if isinstance(result,dict):print(json.dumps(result));return 0
    return result


if __name__=='__main__':raise SystemExit(main())
