"""Frozen formal driver for the explicitly versioned author-outcome policy."""
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io,json,sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_author_outcome_v743_r10 as worker
from our_mani.multitask import formal_suite_admission_v532 as original_guard
from our_mani.multitask import formal_suite_admission_v728 as guard
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.author_outcome_receipt_v749 import require_sampling_case

BINDING_KEYS=('admission_driver','admission_guard','case_worker','native_runtime',
              'original_native_runtime','author_outcome_protocol')
DRIVER_SCHEMA='r10_explicit_author_outcome_formal_driver_v751'


def point(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def option(argv,name):
    if argv.count(name)!=1:raise ValueError('one explicit '+name+' required')
    index=argv.index(name)
    if index+1==len(argv) or argv[index+1].startswith('--'):raise ValueError('explicit '+name+' value required')
    return argv[index+1]


def require_CPU_protocol(proof,expected):
    for key in ('native_runtime','original_native_runtime','author_outcome_protocol'):
        if proof.get(key)!=expected[key]:raise ValueError('actual CPU author-outcome binding changed: '+key)


def require_native_protocol(case,protocol):
    """Use the one independently tested native action transcript verifier."""
    return require_sampling_case(case,protocol,block=case.get('task') in BLOCKS)


def bindings(argv):
    split=option(argv,'--split')
    if split not in ('validation','test'):raise ValueError('admission driver is formal-only')
    suite_path=(ROOT/option(argv,'--suite')).resolve()
    files=json.loads((ROOT/option(argv,'--suite-lock')).read_text())['files']
    guard.locked_pointer(ROOT,files,point(suite_path))
    suite=json.loads(suite_path.read_text())
    protocol=worker.contract.outcome_protocol_binding(ROOT)
    expected=dict(admission_driver=point(__file__),admission_guard=point(guard.__file__),
        case_worker=point(worker.__file__),native_runtime=point(worker.runtime.__file__),
        original_native_runtime=protocol['original_runtime'],author_outcome_protocol=protocol)
    for field,key in [('formal_case_driver','admission_driver'),('formal_admission_guard','admission_guard')]:
        if suite.get(field)!=expected[key]:raise ValueError('explicit frozen admission binding changed: '+field)
        guard.locked_pointer(ROOT,files,suite[field])
    if suite.get('author_outcome_protocol')!=protocol:raise ValueError('explicit frozen suite outcome protocol required')
    for key in ('implementation','runtime_consumer','original_runtime'):guard.locked_pointer(ROOT,files,protocol[key])
    guard.locked_pointer(ROOT,files,point(worker.contract.__file__))
    routes=guard.locked_json(ROOT,files,suite['routes'])
    selected=[r for r in routes['routes'] if (r['task'],r['method'])==
        (option(argv,'--task'),option(argv,'--method'))]
    if len(selected)!=1:raise ValueError('one frozen author-outcome physical case route required')
    route=selected[0]
    if route['worker']!=expected['case_worker'] or route['consumer']!=expected['native_runtime']:
        raise ValueError('formal driver requires the explicitly frozen743/744 physical route')
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
            require_CPU_protocol(proof,expected)
            proof.update({k:expected[k] for k in ('admission_driver','admission_guard','case_worker')})
            proof['execution_protocol_changed']=True
            proof['admission_batch_metrics']=guard.admission_metrics()
            print(json.dumps(proof));return
        worker.main()
    out=(ROOT/option(argv,'--out')).resolve();case=out/'case.json'
    record=json.loads(case.read_text())
    if record.get('complete') is not True or record.get('cleanup_completed') is not True or record.get('split')!=split:
        raise ValueError('only a complete actual-split native case can be receipted')
    require_native_protocol(record,expected['author_outcome_protocol'])
    atomic_json(out/'admission_driver_receipt.json',dict(schema=DRIVER_SCHEMA,
        complete=True,actual_split=split,execution_protocol_changed=True,case=point(case),
        admission_batch_metrics=guard.admission_metrics(),**expected))


if __name__=='__main__':
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope,declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    with declared_edge_scope(),evaluator_scope(evaluator):main()
