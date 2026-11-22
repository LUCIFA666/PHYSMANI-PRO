"""Freeze genuine full16 VAL after fresh training and exact192 TRAIN admission.

TEST manifests/reset files are absent here. They are admitted in a later data
scope after the complete unchanged-policy945-case VAL release.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
from our_mani.multitask.formal_suite_admission_v532 import admit
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def point(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    base=ROOT/'artifacts/framework/multitask_r10';qualified=base/'post_training_qualification_v680_setup'
    setup=base/'formal_suite16_v688_setup'
    if setup.exists():raise FileExistsError('preserve the complete frozen VAL suite')
    readiness=base/'post_training_qualification_v680/readiness.json'
    training=base/'joint16_after_executor_training_acceptance_v672/report.json'
    train=ROOT/'artifacts/framework/protocol/train_manifest.json';val=ROOT/'artifacts/framework/protocol/validation_manifest.json'
    suite=dict(schema='r10_full16_fresh_joint16_common_RML_formal_suite_v688',tasks=list(TASKS),methods=list(METHODS),
        routes=point(qualified/'routes.json'),readiness=point(readiness),
        readiness_auditor=point(ROOT/'scripts/audit_post_training_readiness_v685_r10.py'),training_acceptance=point(training),
        TRAIN_manifest=point(train),source_manifests=dict(validation=point(val)),
        validation_release_path=str((base/'formal_suite16_v688/VAL_release.json').relative_to(ROOT)),
        formal_result_auditor=point(ROOT/'scripts/audit_formal_suite16_results_v690_r10.py'),
        original_TEST_source_not_yet_selected=True,policy_changes_after_VAL_started_permitted=False)
    files=dict(json.loads((qualified/'frozen_inputs.json').read_text())['files'])
    for path in [qualified/'frozen_inputs.json',qualified/'routes.json',readiness,training,train,val,
                 ROOT/'scripts/audit_post_training_readiness_v685_r10.py',Path(__file__),
                 ROOT/'scripts/prepare_formal_split_source_lock_v686_r10.py',
                 ROOT/'scripts/run_formal_suite16_owner_v689_r10.py',ROOT/'scripts/audit_formal_suite16_results_v690_r10.py',
                 ROOT/'scripts/prepare_formal_TEST_suite16_v691_r10.py']:
        files[str(path)]=file_sha(path)
    report=json.loads(readiness.read_text())
    for row in report['cases']:
        path=ROOT/row['case']['path'];files[str(path)]=row['case']['sha256']
    accepted=json.loads(training.read_text())
    for name,sha in accepted['inputs_sha256'].items():
        if name in files and files[name]!=sha:raise ValueError('new completed training input changed')
        files[name]=sha
    verify_inputs(ROOT,files);proof=admit(ROOT,suite,files)
    setup.mkdir();atomic_json(setup/'suite.json',suite);files[str(setup/'suite.json')]=file_sha(setup/'suite.json')
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,actual_formal_split='validation',
        full16_admission_policy_sha256=proof['policy_sha256'],TEST_source_manifest_not_read=True,
        source_bytes_require_separate_post_admission_scope=True))
    atomic_json(setup/'admission.json',dict(proof,full_routes_retained_in=suite['routes'],TEST_source_not_selected=True))
    print(json.dumps(dict(complete=True,VAL_permitted=True,policy_sha256=proof['policy_sha256'],
        TEST_permitted=False,new_shared_checkpoint=proof['training_checkpoint'])),flush=True)


if __name__=='__main__':main()
