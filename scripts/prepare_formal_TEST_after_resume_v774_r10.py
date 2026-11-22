"""Prepare TEST after the actual independent resumed945-case VAL audit.

The immutable752 suite, physical routes and future753/754 TEST owner/auditor
are retained. Actual VAL provenance is explicitly773, with its frozen resume
control graph; it is never relabelled as an audit produced by754.
"""
import argparse,json,pickle,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,validate_roster
from our_mani.multitask.three_method_release_v531 import require_test_release
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v728 import admit,locked_pointer,locked_json
from scripts.prepare_formal_TEST_author_outcome_v755_r10 import require_driver_release_evidence


def point(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def require_resumed_release_evidence(suite,release,files,proof):
    """Validate actual773 provenance and the entire authorized resume graph."""
    require_test_release(release,proof['routes'])
    if (not isinstance(release.get('cases'),list) or len(release['cases'])!=945
        or release.get('policy_sha256')!=proof['policy_sha256']):
        raise ValueError('complete unchanged945-case resumed VAL audit required before TEST data access')
    auditor=point(ROOT/'scripts/audit_resumed_formal_VAL_v773_r10.py')
    original=point(ROOT/'scripts/audit_formal_suite16_results_v754_r10.py')
    if (suite.get('formal_result_auditor')!=original or release.get('base_result_auditor')!=original
        or release.get('source_sha256')!=auditor['sha256']):
        raise ValueError('actual773 VAL source and unchanged754 future TEST auditor required')
    evidence=release.get('inputs_sha256')
    if not isinstance(evidence,dict) or not evidence:raise ValueError('complete frozen resumed evidence map required')
    for name,checksum in evidence.items():
        if not Path(name).is_absolute() or not Path(name).resolve().is_relative_to(ROOT):
            raise ValueError('absolute project paths required in resumed evidence map')
        if name in files and files[name]!=checksum:raise ValueError('resumed evidence conflicts with original frozen suite')
    verify_inputs(ROOT,evidence)
    for name in ('scripts/audit_resumed_formal_VAL_v773_r10.py',
                 'scripts/run_resumed_formal_VAL_owner_v772_r10.py',
                 'scripts/continue_resumed_VAL_TEST_v775_r10.py'):
        locked_pointer(ROOT,evidence,point(ROOT/name))
    locked_pointer(ROOT,evidence,point(Path(__file__)))
    control_pointer=release.get('resume_control');lock_pointer=release.get('resume_control_lock')
    control=locked_json(ROOT,evidence,control_pointer)
    lock=locked_json(ROOT,evidence,lock_pointer)
    if (control.get('schema')!='r10_user_authorized_formal_VAL_resume_control_v772'
        or control.get('user_authorized_resume') is not True or control.get('split')!='validation'
        or control.get('expected_cases')!=945 or control.get('policy_sha256')!=proof['policy_sha256']):
        raise ValueError('explicit user-authorized unchanged full945 VAL resume control required')
    if (lock.get('schema')!='r10_user_authorized_formal_VAL_resume_inputs_v772'
        or lock.get('approved_for_execution') is not True or not isinstance(lock.get('files'),dict)
        or not lock['files']):
        raise ValueError('approved complete resumed control input lock required')
    verify_inputs(ROOT,lock['files'])
    verify_inputs(ROOT,files)
    # The old complete graph can include external author/model inputs. Keep
    # every original entry unchanged and verify its bytes; carry the frozen
    # lock itself rather than expanding that old graph into the project-only
    # independent release evidence accepted by the unchanged755 interface.
    for name,checksum in files.items():
        if lock['files'].get(name)!=checksum:raise ValueError('resume input lock must retain the exact original752 input graph')
    for name,checksum in lock['files'].items():
        if name not in files:
            if not Path(name).is_absolute() or not Path(name).resolve().is_relative_to(ROOT):
                raise ValueError('new resume-control inputs must remain inside the project')
            if evidence.get(name)!=checksum:raise ValueError('resumed release must freeze every newly declared resume input')
    locked_pointer(ROOT,lock['files'],control_pointer)
    base=ROOT/'artifacts/framework/multitask_r10/formal_suite16_v752_setup'
    expected=dict(suite=point(base/'suite.json'),suite_lock=point(base/'frozen_inputs.json'),
        source_inputs=point(base/'validation_source_inputs.json'),
        resume_owner=point(ROOT/'scripts/run_resumed_formal_VAL_owner_v772_r10.py'),resume_auditor=auditor)
    for key,pointer in expected.items():
        if control.get(key)!=pointer:raise ValueError('frozen resume control differs from the actual base or source: '+key)
        locked_pointer(ROOT,lock['files'],pointer)
    locked_pointer(ROOT,files,original)
    return dict(evidence=evidence,control=control_pointer,control_lock=lock_pointer,auditor=auditor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-release',default='artifacts/framework/multitask_r10/formal_VAL_resume_v772/VAL_release.json')
    args=parser.parse_args()
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'formal_suite16_v752_setup';setup=base/'formal_TEST_after_resume_v774_setup'
    if setup.exists():raise FileExistsError('preserve the original post-VAL TEST setup')
    suite=json.loads((old/'suite.json').read_text());files=json.loads((old/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    proof=admit(ROOT,suite,files);release_path=(ROOT/args.validation_release).resolve()
    if not release_path.is_relative_to(ROOT):raise ValueError('resumed VAL release must be frozen project evidence')
    release=json.loads(release_path.read_text())
    require_test_release(release,proof['routes'])
    resumed=require_resumed_release_evidence(suite,release,files,proof)
    for pointer in release['cases']:
        if file_sha(ROOT/pointer['path'])!=pointer['sha256']:raise ValueError('completed frozen VAL native case changed')
    evidence=require_driver_release_evidence(suite,release,files)
    files.update(evidence)
    suite['validation_release_path']=str(release_path.relative_to(ROOT))
    suite['resumed_validation_auditor']=resumed['auditor']
    suite['resumed_validation_control']=resumed['control']
    suite['resumed_validation_control_lock']=resumed['control_lock']
    # The original formal_result_auditor754 remains the actual future TEST auditor.
    # This is the first TEST source-selection read in this workflow. It happens
    # after the complete immutable VAL release, with no parameter selection.
    original=ROOT/'artifacts/framework/protocol/test_manifest_all16.json'
    source=json.loads(original.read_text());rows=[]
    from our_mani.multitask.specs import TASK_SPECS
    from our_mani.multitask.joint16_tracks_v31 import BLOCKS
    from our_mani.multitask.simulation import OFFICIAL_SOURCE_FILES,verify_official_source_files
    from our_mani.multitask.native_tasks import validate_reset_metadata
    core_path=base/'official_test_manifest.json'
    block_path=ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json'
    core=json.loads(core_path.read_text());blocks=json.loads(block_path.read_text())
    core_rows={(r['task'],r['raw_episode_id']):r for r in core['episodes'] if r['task'] in TASK_SPECS}
    block_rows={(r['task'],r['raw_episode_id']):r for r in blocks['episodes'] if r['task'] in BLOCKS}
    if len(core_rows)!=560 or len(block_rows)!=80:raise ValueError('existing verified560 core plus original80 block identities required')
    aliases=set()
    for original_row in source['episodes']:
        key=(original_row['task'],original_row['raw_episode_id'])
        mapping=core_rows if key[0] in TASK_SPECS else block_rows
        if key not in mapping:raise ValueError('no original verified reset mapping for '+str(key))
        row=dict(mapping[key]);path=Path(row['path']).resolve()
        if not path.is_dir():raise ValueError('original official reset source missing')
        # Retain the original variation-local alias and episode index. A
        # global0..39 list position is not an official variation-local index.
        variation=int(pickle.loads((path/'variation_number.pkl').read_bytes()))
        index=row.get('official_episode_index')
        if type(index) is not int or index<0 or row.get('variation')!=variation:
            raise ValueError('existing original official variation/index metadata required')
        alias=(row['task'],variation,index)
        if alias in aliases:raise ValueError('duplicate original variation-local reset alias')
        aliases.add(alias)
        if Path(original_row['path']).resolve()!=path:
            for relative in ('low_dim_obs.pkl','variation_number.pkl','target_state.pkl'):
                if file_sha(Path(original_row['path'])/relative)!=file_sha(path/relative):
                    raise ValueError('existing mapping is not the same original raw source recording')
        if row['task'] in TASK_SPECS:
            spec=TASK_SPECS[row['task']]
            configuration=next(r for r in proof['routes']['routes'] if r['task']==row['task'] and r['method']=='physmani')['configuration']
            native=json.loads((ROOT/configuration['path']).read_text())['baseline_repo']
            validate_reset_metadata(spec,row,archived_variation=variation)
            verify_official_source_files(row,spec,native)
            named=Path(row['official_alias_path'])
            if (not named.is_dir() or named.resolve()!=path or named.name!='episode'+str(index)
                or named.parent.parent.name!='variation'+str(variation)):
                raise ValueError('exact original variation-local core reset alias required')
        else:
            # The legacy two-block reset keeps its established original map;
            # record every first-frame/reset file without changing its sequence.
            row['source_required_files']={relative:dict(path=str(path/relative),sha256=file_sha(path/relative),
                bytes=(path/relative).stat().st_size) for relative in OFFICIAL_SOURCE_FILES}
        row.update(split='test',mapping_status='verified')
        rows.append(row)
    if Counter(r['task'] for r in rows)!=Counter({t:40 for t in TASKS}):raise ValueError('forty original sources per full16 task required')
    manifest=dict(split='test',mapping_status='verified',episode_count=640,episodes=rows,
        training_calibration_or_parameter_selection_used=False,original_source=point(original),
        complete_unchanged_VAL_release_observed_before_source_selection=point(release_path))
    validate_roster(manifest,'test',completed_VAL_audit=True)
    setup.mkdir();manifest_path=setup/'official_test_manifest16.json';atomic_json(manifest_path,manifest)
    suite['source_manifests']['test']=point(manifest_path);suite['original_TEST_source_not_yet_selected']=False
    suite['TEST_data_scope_extension_after_complete_VAL']=point(release_path)
    for path in [manifest_path,original,core_path,block_path,release_path,Path(__file__)]:files[str(path)]=file_sha(path)
    for pointer in release['cases']:files[str(ROOT/pointer['path'])]=pointer['sha256']
    atomic_json(setup/'suite.json',suite);files[str(setup/'suite.json')]=file_sha(setup/'suite.json')
    verify_inputs(ROOT,files);again=admit(ROOT,suite,files);require_test_release(release,again['routes'])
    if again['policy_sha256']!=proof['policy_sha256']:raise ValueError('TEST data scope must not change executable/model policy')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,actual_formal_split='test',
        policy_sha256=again['policy_sha256'],original_945_VAL_release=point(release_path),
        source_bytes_require_separate_post_admission_scope=True,model_or_policy_parameters_changed=False))
    print(json.dumps(dict(complete=True,TEST_permitted=True,official_sources=640,method_cases=1920,
        unchanged_policy_sha256=again['policy_sha256'],native_started=False)),flush=True)


if __name__=='__main__':main()
