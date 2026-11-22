"""Freeze full16 VAL after all192 new author-outcome TRAIN qualifications.

The accepted joint16 weights and operation models are retained. The outcome
protocol is versioned explicitly; no former TRAIN or VAL outcome is reused.
TEST manifests/reset files are absent here. They are admitted in a later data
scope after the complete unchanged-policy945-case VAL release.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS,METHODS
from our_mani.multitask.formal_suite_admission_v728 import admit
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.rml_case_preflight_v742 import outcome_protocol_binding


def point(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def require_author_qualification(routes,readiness,protocol):
    if (routes.get('author_outcome_protocol')!=protocol
        or readiness.get('author_outcome_protocol')!=protocol):
        raise ValueError('new author-outcome qualification and routes must bind one protocol')
    worker=point(ROOT/'scripts/run_task_executor_smoke_author_outcome_v743_r10.py')
    for route in routes.get('routes',[]):
        if (route.get('worker')!=worker or route.get('consumer')!=protocol['runtime_consumer']
            or route.get('author_outcome_protocol')!=protocol):
            raise ValueError('every qualified route must use the explicit743/744 outcome protocol')


def main():
    base=ROOT/'artifacts/framework/multitask_r10';qualified=base/'author_outcome_qualification_v745_setup'
    setup=base/'formal_suite16_v752_setup'
    if setup.exists():raise FileExistsError('preserve the complete frozen VAL suite')
    readiness=base/'author_outcome_qualification_v745/readiness.json'
    training=base/'joint16_after_executor_training_acceptance_v672/report.json'
    train=ROOT/'artifacts/framework/protocol/train_manifest.json';val=ROOT/'artifacts/framework/protocol/validation_manifest.json'
    protocol=outcome_protocol_binding(ROOT)
    report=json.loads(readiness.read_text())
    require_author_qualification(json.loads((qualified/'routes.json').read_text()),report,protocol)
    suite=dict(schema='r10_full16_fresh_joint16_common_RML_formal_suite_v752',tasks=list(TASKS),methods=list(METHODS),
        routes=point(qualified/'routes.json'),readiness=point(readiness),
        readiness_auditor=point(ROOT/'scripts/audit_author_outcome_readiness_v746_r10.py'),training_acceptance=point(training),
        TRAIN_manifest=point(train),source_manifests=dict(validation=point(val)),
        validation_release_path=str((base/'formal_suite16_v752/VAL_release.json').relative_to(ROOT)),
        formal_result_auditor=point(ROOT/'scripts/audit_formal_suite16_results_v754_r10.py'),
        formal_case_driver=point(ROOT/'scripts/run_formal_case_author_outcome_v751_r10.py'),
        formal_admission_guard=point(ROOT/'our_mani/multitask/formal_suite_admission_v728.py'),
        author_outcome_protocol=protocol,execution_protocol_changed=True,
        old_protocol_TRAIN_or_VAL_cases_reused=False,existing_trained_models_unchanged=True,
        original_TEST_source_not_yet_selected=True,policy_changes_after_VAL_started_permitted=False)
    files=dict(json.loads((qualified/'frozen_inputs.json').read_text())['files'])
    for path in [qualified/'frozen_inputs.json',qualified/'routes.json',readiness,training,train,val,
                 ROOT/'scripts/audit_author_outcome_readiness_v746_r10.py',Path(__file__),
                 ROOT/'scripts/prepare_formal_split_source_lock_v756_r10.py',
                 ROOT/'our_mani/multitask/formal_suite_admission_v728.py',
                 ROOT/'our_mani/multitask/formal_suite_admission_v718.py',
                 ROOT/'our_mani/multitask/rml_case_preflight_v742.py',
                 ROOT/'our_mani/multitask/author_outcome_receipt_v749.py',
                 ROOT/'scripts/run_formal_case_author_outcome_v751_r10.py',
                 ROOT/'scripts/check_formal_admission_v757_r10.py',
                 ROOT/'scripts/continue_author_outcome_VAL_TEST_v758_r10.py',
                 ROOT/'scripts/run_formal_suite16_owner_v753_r10.py',ROOT/'scripts/audit_formal_suite16_results_v754_r10.py',
                 ROOT/'scripts/prepare_formal_TEST_author_outcome_v755_r10.py',
                 ROOT/'tests/test_author_outcome_receipt_v749.py',
                 ROOT/'tests/test_formal_author_outcome_driver_v751.py',
                 ROOT/'tests/test_formal_author_outcome_owner_v753.py',
                 ROOT/'tests/test_formal_author_outcome_results_v754.py',
                 ROOT/'tests/test_author_outcome_formal_setup_v752.py']:
        files[str(path)]=file_sha(path)
    for name in ('implementation','runtime_consumer','original_runtime'):
        pointer=protocol[name];path=ROOT/pointer['path']
        if file_sha(path)!=pointer['sha256']:raise ValueError('author-outcome protocol source changed')
        if str(path) in files and files[str(path)]!=pointer['sha256']:raise ValueError('author-outcome protocol conflicts with frozen inputs')
        files[str(path)]=pointer['sha256']
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
