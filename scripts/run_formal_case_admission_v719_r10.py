"""Explicit, frozen admission driver around the unchanged v709 case worker.

Only the optional-colour metadata guard changes. The original worker, native
run function, constructors, models, controls and case records remain intact.
The outer entrypoint and guard are independently locked and receipted.
"""
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io,json,sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_rml_inputs_v709_r10 as worker
from our_mani.multitask import formal_suite_admission_v532 as original_guard
from our_mani.multitask import formal_suite_admission_v718 as guard
from our_mani.multitask.evaluation import atomic_json,file_sha


def point(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def option(argv,name):
    if argv.count(name)!=1:raise ValueError('one explicit '+name+' required')
    return argv[argv.index(name)+1]


def bindings(argv):
    split=option(argv,'--split')
    if split not in ('validation','test'):raise ValueError('admission driver is formal-only')
    suite_path=(ROOT/option(argv,'--suite')).resolve()
    files=json.loads((ROOT/option(argv,'--suite-lock')).read_text())['files']
    guard.locked_pointer(ROOT,files,point(suite_path))
    suite=json.loads(suite_path.read_text())
    expected=dict(admission_driver=point(__file__),admission_guard=point(guard.__file__),
        unchanged_case_worker=point(worker.__file__),native_runtime=point(worker.runtime.__file__))
    for field,key in [('formal_case_driver','admission_driver'),('formal_admission_guard','admission_guard')]:
        if suite.get(field)!=expected[key]:raise ValueError('explicit frozen admission binding changed: '+field)
        guard.locked_pointer(ROOT,files,suite[field])
    routes=guard.locked_json(ROOT,files,suite['routes'])
    selected=[r for r in routes['routes'] if (r['task'],r['method'])==
        (option(argv,'--task'),option(argv,'--method'))]
    if len(selected)!=1:raise ValueError('one unchanged physical case route required')
    route=selected[0]
    if route['worker']!=expected['unchanged_case_worker'] or route['consumer']!=expected['native_runtime']:
        raise ValueError('guard-only driver cannot replace the physical case worker or runtime')
    for key in ('worker','consumer'):guard.locked_pointer(ROOT,files,route[key])
    return expected,split


def main():
    argv=sys.argv[1:];expected,split=bindings(argv)
    # The capability factory imports this module's source gate at call time.
    # Scoped injection selects the versioned guard without editing frozen code.
    with patch.object(original_guard,'admit',guard.admit),patch.object(original_guard,'read_formal_source',guard.read_formal_source):
        if '--check' in argv:
            captured=io.StringIO()
            with redirect_stdout(captured):worker.main()
            proof=json.loads(captured.getvalue())
            if proof.get('native_started') is not False or proof.get('GPU_started') is not False:
                raise ValueError('actual argv CPU proof must remain CPU-only')
            proof.update({k:expected[k] for k in ('admission_driver','admission_guard','unchanged_case_worker')})
            print(json.dumps(proof));return
        worker.main()
    out=(ROOT/option(argv,'--out')).resolve();case=out/'case.json'
    record=json.loads(case.read_text())
    if record.get('complete') is not True or record.get('cleanup_completed') is not True or record.get('split')!=split:
        raise ValueError('only the original complete actual-split native case can be receipted')
    atomic_json(out/'admission_driver_receipt.json',dict(schema='r10_guard_only_formal_driver_v719',
        complete=True,actual_split=split,physical_policy_changed=False,case=point(case),**expected))


if __name__=='__main__':
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope,declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    with declared_edge_scope(),evaluator_scope(evaluator):main()
