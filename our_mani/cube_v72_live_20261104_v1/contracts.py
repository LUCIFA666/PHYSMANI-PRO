"""Fixed paired generated-TRAIN development assignments and source contracts."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'Cube_V72_current_runtime_paired_candidate_development_20261104_v1'
TASK = 'pick_moving_target_on_the_table'
ASSIGNMENTS = (
    ('ordinary_00',TASK,2611041100), ('ordinary_01',TASK,2611041101),
    ('highspeed_00',TASK+'_high_speed',2611041110), ('highspeed_01',TASK+'_high_speed',2611041111))
REFERENCE = ROOT/'config/framework/P4_position_dwell_v72_development_canaries_v1/00_val_static.json'
OLD_FREEZE = ROOT/'artifacts/framework/local_mpc_20261102/V72_fixed6_execution_preparation_v1/execution_source_freeze.json'
OWN_FILES = tuple('our_mani/cube_v72_live_20261104_v1/'+n for n in
    ('__init__.py','contracts.py','backend.py','candidate_runtime.py','producer.py','worker.py')) + (
    'scripts/run_cube_v72_live_20261104_v1.py',
    'scripts/launch_cube_v72_live_20261104_v1.py',
    'scripts/run_cube_v72_live_20261104_v1.sh',
    'scripts/run_cube_v72_paired_development_20261104_v1.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def validate_plan(path):
    path=Path(path).resolve(strict=True); plan=read(path)
    if plan.get('schema')!=SCHEMA or plan.get('formal_VAL_TEST')is not False:
        raise ValueError('Exact generated-TRAIN/development comparison required')
    if plan.get('reference')!={'path':str(REFERENCE.relative_to(ROOT)),'sha256':sha(REFERENCE)}:
        raise ValueError('Frozen V72 control reference required')
    if [(c['id'],c['task'],c['seed'])for c in plan['cases']]!=list(ASSIGNMENTS):
        raise ValueError('Exactly the four predeclared ordinary/high-speed assignments required')
    if any(c.get('variation')!=0 or c.get('simulation_seconds')!=8. or c.get('budget_ticks')!=160 for c in plan['cases']):
        raise ValueError('Original variation0/eight seconds/160tick schedule required')
    if (plan.get('profiles_order')!=['legacy_pilot','cube_candidate_v1']
            or plan.get('all_cases_regardless_of_task_outcome')is not True
            or plan.get('retry_or_replacement')is not False):
        raise ValueError('Complete paired profiles; no outcome-based retry/selection')
    for case in plan['cases']:
        calibration=Path(case['calibration_episode']).resolve(strict=True)
        if 'train_package_compressed'not in calibration.parts or calibration.name!='data.h5':
            raise ValueError('Explicit TRAIN calibration only')
    return plan


def case_config(plan,case,profile):
    if profile not in plan['profiles_order']:raise ValueError('Explicit candidate profile required')
    config=deepcopy(read(REFERENCE)); candidate=plan['candidate_profiles'][profile]
    config.update(schema=SCHEMA,task=case['task'],seed=case['seed'],variation=0,
        simulation_seconds=8.,dataset_use='train',formal_evaluation=False,
        calibration_train_episode=case['calibration_episode'],candidate_profile=profile,
        candidate_transformer_checkpoint=candidate['checkpoint'],
        candidate_transformer_checkpoint_sha256=candidate['sha256'],
        candidate_model_freeze_manifest=None,native_generated_motion=True,
        Native_labels_used_by_policy=False,official_VAL_or_TEST_source=False)
    config['P4_collection_metadata']=dict(parent_id=case['id']+'_seed'+str(case['seed']),split='train',
        source='new_generated_development',official_VAL_or_TEST_source=False,seed_predeclared=True)
    config['native_task_source_sha256']=case['native_source_sha256']
    config['native_task_scene_sha256']=case['native_scene_sha256']
    return config


def sources_for_plan(path):
    plan=validate_plan(path)
    sources=deepcopy(read(OLD_FREEZE)['files_sha256'])
    sources[str(OLD_FREEZE.relative_to(ROOT))]=sha(OLD_FREEZE)
    sources[str(Path(path).resolve().relative_to(ROOT))]=sha(path)
    for name in OWN_FILES:
        sources[name]=sha(ROOT/name)
    for name in ('scripts/feedback_current_environment_20261104_v2.py',
        'our_mani/candidate_transformer_cube_20261104_v1/__init__.py',
        'our_mani/candidate_transformer_cube_20261104_v1/model.py',
        'our_mani/candidate_transformer_cube_20261104_v1/losses.py'):
        if (ROOT/name).is_file(): sources[name]=sha(ROOT/name)
    for profile in plan['candidate_profiles'].values():
        p=ROOT/profile['checkpoint']
        if not p.is_file() or sha(p)!=profile['sha256']: raise ValueError('Actual frozen candidate checkpoint required: '+str(p))
        sources[str(p.relative_to(ROOT))]=profile['sha256']
        for pointer in profile.get('evidence',[]):
            p=Path(pointer['path']);p=p if p.is_absolute()else ROOT/p
            if sha(p)!=pointer['sha256']:raise ValueError('Candidate fit evidence changed')
            sources[str(p.relative_to(ROOT))]=pointer['sha256']
    for case in plan['cases']:
        native=Path(plan['native_repo'])
        for key,sub in (('native_source_sha256','tasks/'+case['task']+'.py'),('native_scene_sha256','task_ttms/'+case['task']+'.ttm')):
            p=native/'third_party/RLBench/rlbench'/sub
            if sha(p)!=case[key]:raise ValueError('Original actual task source/scene changed')
    for name,value in sources.items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen source/model changed: '+name)
    return sources


def native_sources_for_plan(path):
    """Both actual task classes/scenes and shared Native success/action code."""
    from ..multitask.specs import NATIVE_COMMON_SOURCE_SHA256
    plan=validate_plan(path); native=Path(plan['native_repo'])
    result={str(native/name):h for name,h in NATIVE_COMMON_SOURCE_SHA256.items()}
    for case in plan['cases']:
        for key,sub in (('native_source_sha256','tasks/'+case['task']+'.py'),
                        ('native_scene_sha256','task_ttms/'+case['task']+'.ttm')):
            result[str(native/'third_party/RLBench/rlbench'/sub)]=case[key]
    for name,value in result.items():
        if sha(name)!=value:raise ValueError('Original Native task/common source changed: '+name)
    return result


def check_freeze(plan_path,freeze_path,digest):
    if sha(freeze_path)!=digest:raise ValueError('External execution freeze SHA required')
    freeze=read(freeze_path)
    if (freeze.get('schema')!='Cube_V72_current_runtime_execution_freeze_20261104_v1'
            or freeze.get('files_sha256')!=sources_for_plan(plan_path)
            or freeze.get('native_files_sha256')!=native_sources_for_plan(plan_path)):
        raise ValueError('Exact new complete source/model/plan freeze required')
    return freeze['files_sha256']
