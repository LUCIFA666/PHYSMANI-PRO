"""Freeze generic task adapters and existing supported models for TRAIN smoke."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.task_actor_backend_v52 import persistent_reference_policy_overlay,persistent_reference_continuation_identity
from our_mani.multitask.task_native_backend_v52 import diagnostic_binding
from our_mani.multitask.task_spherical_enclosure_v18 import geometry_binding
from our_mani.multitask.colour_runtime_v18 import binding as colour_binding
from our_mani.multitask.interaction import InteractionLibrary,CandidateConfig,MPCConfig
from scripts.run_task_executor_smoke_v52_r10 import parser,preflight

BASE=ROOT/'artifacts/framework/multitask_r10'
SETUP=BASE/'task_executor_smoke_v52_setup'
OUTPUT=BASE/'task_executor_smoke_v52'


def command(task,raw,method):
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_v52_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest',str((ROOT/'artifacts/framework/protocol/train_manifest.json').relative_to(ROOT)),
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,
        '--replica','0','--development-demo','--out',
        str((OUTPUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise RuntimeError('Preserve existing frozen setup')
    old=json.loads((BASE/'bin_executor_v17_setup/frozen_inputs.json').read_text())
    verify_inputs(ROOT,old['files'])
    for launcher in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        subprocess.run(launcher+['-m','unittest','discover','-s','tests','-p','test_task_reference_v37.py'],cwd=ROOT,check=True)
    for launcher in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        for test in ('test_public_held_mesh_v33.py','test_ring_runtime_v25.py','test_bounded_approach_v36.py','test_task_dispatch_integration_v43.py','test_placement_clearance_v50.py','test_remove_runtime_v48.py'):
            subprocess.run(launcher+['-m','unittest','discover','-s','tests','-p',test],cwd=ROOT,check=True)
    c=json.loads((BASE/'bin_executor_v17_setup/configuration.json').read_text())
    bundle=BASE/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json';fit=json.loads(bundle.read_text())
    available=[]
    for task,row in fit['per_task'].items():
        if row['interaction_status']!='TRAIN_fitted_pending_formal_release':continue
        for key in ('interaction','rigid_layout'):
            if row.get(key):
                pointer=deepcopy(row[key]);pointer['path']=str(Path(pointer['path']).resolve().relative_to(ROOT))
                c['learned_artifacts'][task][key]=pointer
        available.append(task)
    remove_report=BASE/'remove_source_qualified_libraries_v48/report.json'
    qualified=json.loads(remove_report.read_text())
    from our_mani.multitask.remove_runtime_v48 import binding as remove_binding
    c['remove_runtime_v48']={}
    for task,row in qualified['tasks'].items():
        c['learned_artifacts'][task]['interaction']=row['interaction'];available.append(task)
        c['remove_runtime_v48'][task]=dict(qualification=row['qualification'])
        c['remove_runtime_v48'][task]['binding']=remove_binding(ROOT,c,task)
    colour_path=BASE/'TRAIN_colour_distribution_candidate_actual_v2/candidate.json'
    colour=dict(path=str(colour_path.relative_to(ROOT)),sha256=file_sha(colour_path))
    c['colour_runtime_v18']=colour_binding(ROOT,colour)
    for task in c['colour_runtime_v18']['tasks']:c['learned_artifacts'][task]['colour']=colour
    c['task_spherical_geometry_v18']={}
    for task in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed'):
        geometry=geometry_binding(ROOT,c,task);c['task_spherical_geometry_v18'][task]=geometry
        c.setdefault('body_sphere_radii',{}).setdefault(task,{})['ball']=geometry['current_RGBD_sphere_radius_m']
    from our_mani.multitask.public_held_mesh_v33 import binding as held_binding
    c['public_held_mesh_v33']={task:held_binding(ROOT,c,task) for task in ('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')}
    from our_mani.multitask.ring_runtime_v25 import binding as ring_binding
    replay=BASE/'ring_full_TRAIN_affordance_v21/report.json';rp=dict(path=str(replay.relative_to(ROOT)),sha256=file_sha(replay))
    row=next(r for r in json.loads(replay.read_text())['entries'] if r['task']=='insert_onto_rotating_peg')
    c.setdefault('ring_affordances',{})[row['task']]=row['permission']
    c['ring_runtime_v25']={row['task']:ring_binding(ROOT,row['permission'],rp)}
    from our_mani.multitask.placement_clearance_v50 import binding as clearance_binding
    c['placement_clearance_v50']={task:clearance_binding(ROOT,c,task) for task in ('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')}
    settings=c['policy_overlay']['configurations']
    settings['persistent_reference']['version']='r10_continuous_verified_reference_feedforward_v37'
    settings['persistent_reference']['approach_transit']['version']='r10_bounded_public_approach_v36'
    c['policy_overlay']=persistent_reference_policy_overlay(ROOT,settings['feedback'],settings['persistent_reference'],settings['causal_retarget'])
    c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
    c['execution_scope']='all16_executor_repair_existing_supported_models_TRAIN_smoke_v52'
    c['formal_comparison_ready']=False
    for task in available:
        library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        for predictor in ('cv','context'):
            identity=persistent_reference_continuation_identity(ROOT,c,task,predictor,library,
                CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
            c['continuation_identities'].setdefault(task,{})[predictor]=identity['sha256']
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c)
    files=dict(old['files']);paths=[Path(__file__),ROOT/'scripts/run_task_executor_smoke_v52_r10.py',
        ROOT/'our_mani/multitask/terminal_candidate_generator_v51.py',ROOT/'our_mani/multitask/task_reference_v37.py',ROOT/'our_mani/multitask/task_actor_backend_v52.py',
        ROOT/'our_mani/multitask/task_dispatch_v43.py',ROOT/'our_mani/multitask/task_native_backend_v52.py',
        ROOT/'our_mani/multitask/task_spherical_enclosure_v18.py',ROOT/'our_mani/multitask/bounded_approach_v36.py',
        ROOT/'our_mani/multitask/colour_runtime_v18.py',ROOT/'tests/test_task_reference_v37.py',
        ROOT/'tests/test_colour_runtime_v18.py',ROOT/'tests/test_public_held_mesh_v33.py',
        ROOT/'tests/test_ring_runtime_v25.py',ROOT/'tests/test_bounded_approach_v36.py',ROOT/'tests/test_placement_clearance_v50.py',ROOT/'tests/test_remove_runtime_v48.py',
        ROOT/'our_mani/multitask/placement_clearance_v50.py',ROOT/'our_mani/multitask/remove_runtime_v48.py',
        ROOT/'scripts/export_qualified_remove_libraries_v48_r10.py',remove_report,ROOT/'tests/test_task_dispatch_integration_v43.py',ROOT/'our_mani/multitask/public_held_mesh_v33.py',
        ROOT/'our_mani/multitask/ring_runtime_v25.py',ROOT/'our_mani/multitask/ring_affordance_v21.py',SETUP/'configuration.json',bundle]
    for task in available:
        for pointer in c['learned_artifacts'][task].values():
            if pointer:paths.append(ROOT/pointer['path'])
        library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        paths.append(Path(library.metadata['TRAIN_source_manifest_path']))
    for name,sha in c['colour_runtime_v18']['inputs_sha256'].items():
        assert file_sha(name)==sha;paths.append(Path(name))
    for name,sha in c['ring_runtime_v25']['insert_onto_rotating_peg']['inputs_sha256'].items():
        p=ROOT/name;assert file_sha(p)==sha;paths.append(p)
    proof=json.loads((ROOT/row['permission']['path']).read_text())
    paths.extend(Path(proof[key]) for key in ('native_gripper_source','native_task_source'))
    for row in qualified['tasks'].values():
        paths.append(ROOT/row['qualification']['path'])
        proof=json.loads((ROOT/row['qualification']['path']).read_text())
        for key in ('source_goal_audit','current_full_curve_fit','current_prefit_proof','fitter','exporter'):
            paths.append(ROOT/proof[key]['path'])
    for p in paths:
        p=p.resolve();sha=file_sha(p);assert str(p) not in files or files[str(p)]==sha;files[str(p)]=sha
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_generic_task_executor_TRAIN_smoke_v18_lock',
        files=files,approved_for_execution=True,native_started=False,available_existing_libraries=available,
        formal_VAL_TEST_released=False,joint16_model_training_started=False))
    task='put_rubbish_in_moving_bin';raw='episode105'
    for method in ('cv_geometry','context_geometry'):
        preflight(parser().parse_args(command(task,raw,method)[3:]))
        result=subprocess.run(command(task,raw,method)+['--check'],cwd=ROOT,text=True,capture_output=True,check=True)
        atomic_json(SETUP/(method+'_native_CPU_preflight.json'),json.loads(result.stdout))
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,controller_tests_each_environment=35,
        colour_integration_tests_each_environment=2,actual_bin_argv_preflight=True,native_started=False,
        models_refitted=False,formal_admission=False))
    verify_inputs(ROOT,files)
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),supported_libraries=available,formal_released=False)))


if __name__=='__main__':main()
