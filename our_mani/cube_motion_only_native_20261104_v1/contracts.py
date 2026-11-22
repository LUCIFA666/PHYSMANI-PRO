"""Fixed four assignments; safe repository-only archive and separate Native7."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

from ..cube_v72_live_20261104_v1.contracts import ASSIGNMENTS,REFERENCE,OLD_FREEZE

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='Cube_corrected_motion_only_native_development_20261104_v1'
FREEZE_SCHEMA='Cube_motion_only_native_current_runtime_execution_freeze_20261104_v1'
PROFILES=('corrected_legacy',)
SOURCE_PLAN=ROOT/'config/framework/cube_v72_live_20261104_v1/plan.json'
OWN_FILES=tuple('our_mani/cube_motion_only_native_20261104_v1/'+n for n in ('__init__.py','contracts.py','worker.py'))+(
    'scripts/run_cube_motion_only_native_20261104_v1.py',
    'scripts/launch_cube_motion_only_native_20261104_v1.py',
    'scripts/run_cube_motion_only_native_20261104_v1.sh')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def normalize_repo_sources(mapping):
    """Canonical relative names only; an outside path can never reach archive."""
    if not isinstance(mapping,dict):raise TypeError('Explicit source:SHA mapping required')
    result={}
    for name,digest in mapping.items():
        if type(name)is not str or type(digest)is not str or len(digest)!=64:
            raise ValueError('Explicit source name and full digest required')
        path=Path(name);path=path if path.is_absolute()else ROOT/path
        path=path.resolve(strict=True)
        if not path.is_relative_to(ROOT) or path==ROOT:
            raise ValueError('External/native sources must remain separate, never repository archive keys: '+str(path))
        relative=path.relative_to(ROOT).as_posix()
        if relative in result and result[relative]!=digest:
            raise ValueError('Conflicting source aliases: '+relative)
        if sha(path)!=digest:raise ValueError('Frozen source/model changed: '+relative)
        result[relative]=digest
    return result


def repo_pointer(pointer,*,allow_pending=False):
    if (not isinstance(pointer,dict) or 'checkpoint'not in pointer or 'sha256'not in pointer):
        raise ValueError('Explicit checkpoint/SHA pointer required')
    if pointer['checkpoint']is None or pointer['sha256']is None:
        if allow_pending and pointer['checkpoint']is None and pointer['sha256']is None:return None
        raise ValueError('Actual checkpoint identity is pending')
    mapping=normalize_repo_sources({pointer['checkpoint']:pointer['sha256']})
    return ROOT/next(iter(mapping))


def validate_plan(path):
    path=Path(path).resolve(strict=True);plan=read(path)
    if (plan.get('schema')!=SCHEMA or plan.get('formal_VAL_TEST')is not False
            or type(plan.get('preparation_only'))is not bool or type(plan.get('execution_authorized'))is not bool):
        raise ValueError('Exact explicit prepared/generated-TRAIN development plan required')
    if plan.get('reference')!={'path':str(REFERENCE.relative_to(ROOT)),'sha256':sha(REFERENCE)}:
        raise ValueError('Unchanged V72 control/budget reference required')
    if plan.get('baseline_execution_plan')!={'path':str(SOURCE_PLAN.relative_to(ROOT)),'sha256':sha(SOURCE_PLAN)}:
        raise ValueError('Frozen prior four-seed assignment provenance required')
    if [(c['id'],c['task'],c['seed'])for c in plan['cases']]!=list(ASSIGNMENTS):
        raise ValueError('Exactly ordinary1100/1101 and high1110/1111 assignments required')
    if plan['cases']!=read(SOURCE_PLAN)['cases'] or plan.get('native_repo')!=read(REFERENCE)['native_repo']:
        raise ValueError('Original calibration/task scene/native repo assignments must remain exact')
    if any(c.get('variation')!=0 or c.get('simulation_seconds')!=8. or c.get('budget_ticks')!=160 for c in plan['cases']):
        raise ValueError('Original variation0/eight seconds/160 actual physics ticks required')
    if (plan.get('profiles_order')!=list(PROFILES) or set(plan.get('candidate_profiles',{}))!=set(PROFILES)
            or plan.get('all_cases_regardless_of_task_outcome')is not True
            or plan.get('retry_or_replacement')is not False):
        raise ValueError('Fixed complete profiles and denominator with no retry/replacement required')
    reserved=plan.get('reserved_profiles')
    if reserved!={'corrected_matched_reobserve':{'enabled':False,'reason':'requires separate add-only supervisor/runner protocol'}}:
        raise ValueError('Reobserve is a disabled reserved profile, not an implicit live feature')
    for case in plan['cases']:
        p=Path(case['calibration_episode']).resolve(strict=True)
        if 'train_package_compressed'not in p.parts or p.name!='data.h5':
            raise ValueError('TRAIN camera/home calibration only')
    repo_pointer(plan['motion_forecast'])
    for pointer in plan['candidate_profiles'].values():repo_pointer(pointer,allow_pending=True)
    return plan


def case_config(plan,case,profile):
    if profile not in PROFILES:raise ValueError('Explicit implemented corrected profile required')
    config=deepcopy(read(REFERENCE));candidate=plan['candidate_profiles'][profile];forecast=plan['motion_forecast']
    config.update(schema=SCHEMA,task=case['task'],seed=case['seed'],variation=0,
        simulation_seconds=8.,dataset_use='train',formal_evaluation=False,
        calibration_train_episode=case['calibration_episode'],candidate_profile=profile,
        candidate_transformer_checkpoint=candidate['checkpoint'],candidate_transformer_checkpoint_sha256=candidate['sha256'],
        corrected_forecast_checkpoint=forecast['checkpoint'],corrected_forecast_checkpoint_sha256=forecast['sha256'],
        candidate_model_freeze_manifest=None,native_generated_motion=True,
        Native_labels_used_by_policy=False,official_VAL_or_TEST_source=False,
        cube_corrected_motion_v1=True,candidate_corrected_forecast_development=True,
        cube_matched_candidate_v1=profile=='corrected_matched',reobserve_enabled=False)
    config['P4_collection_metadata']=dict(parent_id=case['id']+'_seed'+str(case['seed']),split='train',
        source='same_fixed_generated_development_assignment_new_correction_profile',official_VAL_or_TEST_source=False,seed_predeclared=True)
    config['native_task_source_sha256']=case['native_source_sha256'];config['native_task_scene_sha256']=case['native_scene_sha256']
    return config


def code_sources_for_plan(path):
    plan=validate_plan(path)
    from ..cube_corrected_live_20261104_v1.matched_sources import matched_corrected_source_map
    sources=deepcopy(read(OLD_FREEZE)['files_sha256'])
    sources.update(matched_corrected_source_map())
    for p in (OLD_FREEZE,SOURCE_PLAN,REFERENCE,Path(path).resolve(),
              ROOT/'scripts/feedback_current_environment_20261104_v2.py',*(ROOT/n for n in OWN_FILES)):
        sources[str(p)]=sha(p)
    return normalize_repo_sources(sources)


def sources_for_plan(path):
    plan=validate_plan(path);sources=code_sources_for_plan(path)
    for pointer in (plan['motion_forecast'],*plan['candidate_profiles'].values()):
        p=repo_pointer(pointer);sources[str(p.relative_to(ROOT))]=pointer['sha256']
        for evidence in pointer.get('evidence',[]):
            if not isinstance(evidence,dict) or set(evidence)!={'path','sha256'}:
                raise ValueError('Explicit immutable fit/evaluation evidence pointer required')
            sources.update(normalize_repo_sources({evidence['path']:evidence['sha256']}))
    native_sources_for_plan(path)
    return normalize_repo_sources(sources)


def native_sources_for_plan(path):
    from ..multitask.specs import NATIVE_COMMON_SOURCE_SHA256
    plan=validate_plan(path);native=Path(plan['native_repo']).resolve(strict=True)
    result={str((native/name).resolve()):h for name,h in NATIVE_COMMON_SOURCE_SHA256.items()}
    for case in plan['cases']:
        for key,sub in (('native_source_sha256','tasks/'+case['task']+'.py'),
                        ('native_scene_sha256','task_ttms/'+case['task']+'.ttm')):
            p=native/'third_party/RLBench/rlbench'/sub;result[str(p.resolve())]=case[key]
    if len(result)!=7:raise ValueError('Exactly separate seven Native task/common/scene sources required')
    for name,value in result.items():
        if sha(name)!=value:raise ValueError('Original Native task/common source changed: '+name)
    return result


def check_freeze(plan_path,freeze_path,digest):
    if type(digest)is not str or sha(freeze_path)!=digest:raise ValueError('External complete execution freeze SHA required')
    freeze=read(freeze_path)
    if (freeze.get('schema')!=FREEZE_SCHEMA or freeze.get('files_sha256')!=sources_for_plan(plan_path)
            or freeze.get('native_files_sha256')!=native_sources_for_plan(plan_path)):
        raise ValueError('Exact new repo-relative plus Native7 execution freeze required')
    return freeze['files_sha256']
