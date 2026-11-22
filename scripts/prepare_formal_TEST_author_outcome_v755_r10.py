"""Extend only TEST data evidence after the unchanged-policy945-case VAL audit."""
import json,pickle,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,validate_roster
from our_mani.multitask.three_method_release_v531 import require_test_release
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.formal_suite_admission_v728 import admit,locked_pointer


def point(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def require_driver_release_evidence(suite,release,files):
    """Carry the independently audited VAL admission evidence into TEST scope."""
    evidence=release.get('inputs_sha256')
    if not isinstance(evidence,dict) or not evidence:
        raise ValueError('explicit frozen VAL admission evidence inputs required')
    for name,checksum in evidence.items():
        path=Path(name)
        if not path.is_absolute() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('VAL admission evidence requires absolute project paths')
        if name in files and files[name]!=checksum:
            raise ValueError('VAL admission evidence conflicts with frozen suite inputs')
    verify_inputs(ROOT,evidence)
    protocol=suite['author_outcome_protocol']
    if release.get('author_outcome_protocol')!=protocol:
        raise ValueError('complete VAL must use the same new author-outcome protocol')
    worker=point(ROOT/'scripts/run_task_executor_smoke_author_outcome_v743_r10.py')
    pointers=release.get('admission_driver_receipts')
    cases=release.get('cases')
    if not isinstance(pointers,list) or len(pointers)!=945 or not isinstance(cases,list) or len(cases)!=945:
        raise ValueError('all945 VAL native cases and admission driver receipts required')
    def key(pointer):
        if not isinstance(pointer,dict) or set(pointer)!={'path','sha256'}:
            raise ValueError('explicit exact VAL evidence pointer required')
        return pointer['path'],pointer['sha256']
    expected={key(pointer) for pointer in cases}
    if len(expected)!=945 or len({key(pointer) for pointer in pointers})!=945:
        raise ValueError('VAL cases and admission receipts must be unique')
    observed=set()
    for pointer in pointers:
        receipt=json.loads(locked_pointer(ROOT,evidence,pointer).read_text())
        if (receipt.get('schema')!='r10_explicit_author_outcome_formal_driver_v751' or receipt.get('complete') is not True
            or receipt.get('actual_split')!='validation' or receipt.get('execution_protocol_changed') is not True
            or receipt.get('admission_driver')!=suite['formal_case_driver']
            or receipt.get('admission_guard')!=suite['formal_admission_guard']
            or receipt.get('case_worker')!=worker
            or receipt.get('native_runtime')!=protocol['runtime_consumer']
            or receipt.get('original_native_runtime')!=protocol['original_runtime']
            or receipt.get('author_outcome_protocol')!=protocol):
            raise ValueError('actual VAL driver, guard, worker and author-outcome protocol receipts required')
        case_key=key(receipt.get('case'))
        if case_key not in expected or case_key in observed:
            raise ValueError('VAL admission receipts must bind the exact945 native cases once each')
        observed.add(case_key)
    if observed!=expected:raise ValueError('complete VAL native/admission evidence grid required')
    return evidence


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'formal_suite16_v752_setup';setup=base/'formal_TEST_suite16_v755_setup'
    if setup.exists():raise FileExistsError('preserve the original post-VAL TEST setup')
    suite=json.loads((old/'suite.json').read_text());files=json.loads((old/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    proof=admit(ROOT,suite,files);release_path=ROOT/suite['validation_release_path'];release=json.loads(release_path.read_text())
    require_test_release(release,proof['routes'])
    if (release.get('source_sha256')!=file_sha(ROOT/suite['formal_result_auditor']['path'])
        or len(release.get('cases',[]))!=945):raise ValueError('actual independent945-case VAL audit required before TEST data access')
    for pointer in release['cases']:
        if file_sha(ROOT/pointer['path'])!=pointer['sha256']:raise ValueError('completed frozen VAL native case changed')
    evidence=require_driver_release_evidence(suite,release,files)
    files.update(evidence)
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
