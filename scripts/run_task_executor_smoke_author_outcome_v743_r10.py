"""Explicit author-outcome worker with genuine TRAIN/VAL/TEST source domains.

The physical v664 flow is versioned as v744 solely to place explicit action
boundaries around its original executor/hold calls and install opt-in core
simulation subclasses. Models, public geometry, RML and resets are inherited.
"""
from pathlib import Path
import json,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_author_suite16_v744_r10 as runtime
from our_mani.multitask import rml_case_preflight_v742 as contract
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask import rml_component_factory_v643 as base_factory
from our_mani.multitask import rml_component_extensions_v708 as extension
from our_mani.multitask import rml_policy_components_v669 as base_policy
from our_mani.multitask import rml_policy_components_v702 as latest_policy


def main():
    parser=runtime.parser();parser.add_argument('--source-inputs')
    args=parser.parse_args();p=contract.prepare(ROOT,args,Path(__file__))
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=p['identity'],
            component_domain=p['component_domain'].receipt(),model_preflight=p['model_preflight'],actual_split=args.split,
            native_runtime=dict(path=str(Path(runtime.__file__).relative_to(ROOT)),sha256=file_sha(runtime.__file__)),
            original_native_runtime=contract.outcome_protocol_binding(ROOT)['original_runtime'],
            author_outcome_protocol=contract.outcome_protocol_binding(ROOT),
            split_source_input_lock_sha256=p['source_lock_sha256'])))
        return
    # Retain the exact v709 component selection. The separately frozen v744
    # consumer installs explicit action-outcome scopes before environment init.
    with patch.object(base_factory,'create_actor',extension.create_actor),patch.object(base_policy,'construct',latest_policy.construct):
        result=runtime.run(p)
    print(json.dumps(dict(complete=True,native_success=result['official_task_success'],case=str(p['out']/'case.json'))))


if __name__=='__main__':
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope,declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    with declared_edge_scope(),evaluator_scope(evaluator):main()
