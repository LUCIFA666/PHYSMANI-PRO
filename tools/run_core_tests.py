"""Run the executor/protocol/time-limit regression set without Native."""
import argparse,json,sys,time,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path)
    parser.add_argument('--with-local-artifacts',action='store_true',help='Also run the two original-machine artifact integration tests')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_executor*2026111[23]_v*.py')
    for name in ['test_execution_interface_20261111_v1','test_all16_time_limits_20261111_v1','test_task_time_limits_60s_20261111_v1']:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('tests.'+name))
    artifact_tests={
        'PoseContract.test_existing_policy_reference_reaches_same_typed_executor':'Requires original STOP / retired revision runtime records.',
        'Active60sContracts.test_both_cube_entry_tasks_prepare_full60s_without_launching_native':'Requires original model checkpoints and frozen artifact source manifests.'}
    marked=[]
    def mark_artifact_tests(group):
        for test in group:
            if isinstance(test,unittest.TestSuite):mark_artifact_tests(test);continue
            for suffix,reason in artifact_tests.items():
                if test.id().endswith('.'+suffix):
                    marked.append(dict(test=test.id(),reason=reason))
                    method=getattr(test,test._testMethodName)
                    setattr(test,test._testMethodName,unittest.skip(reason)(method))
    if not args.with_local_artifacts:mark_artifact_tests(suite)
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=1).run(suite)
    report=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),passed=result.wasSuccessful(),wall_seconds=time.monotonic()-started,
        executed_from=str(ROOT),Native_started=False,model_weights_loaded=False,
        local_artifact_tests_enabled=args.with_local_artifacts,explicitly_skipped_artifact_tests=marked)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x')as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if result.wasSuccessful()else 1

if __name__=='__main__':raise SystemExit(main())
