"""Freeze one fixed TRAIN source per remaining task for GPU integration only.

These fourteen released-policy executions are not a formal VAL/TEST result,
critic-training data, or a model-selection experiment. No videos are saved.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.specs import TASK_SPECS
from scripts.evaluate_multitask_case_r10 import required_local_sources,EXECUTOR
from scripts.audit_train_resets_r10 import native_inputs
from scripts.run_multitask_fresh_queue_r10 import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='artifacts/framework/multitask_r10/baseline_integration_v1')
    args=p.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        raise ValueError('use a fresh baseline integration directory')
    out.mkdir(parents=True)
    source=ROOT/'artifacts/framework/protocol/train_manifest.json'
    rows=json.loads(source.read_text())['episodes'];first={}
    for row in rows:
        if row['task'] in TASK_SPECS:first.setdefault(row['task'],row)
    if set(first)!=set(TASK_SPECS):raise ValueError('all fourteen fixed TRAIN sources required')
    manifest=out/'fixed_TRAIN_sources.json'
    atomic_json(manifest,dict(split='train',episodes=[first[t] for t in TASK_SPECS],
        development_only=True,formal_metrics_included=False,critic_training_permitted=False,
        selection='first source in canonical TRAIN order per task, independent of outcomes',
        canonical_source=dict(path=str(source.relative_to(ROOT)),sha256=file_sha(source))))
    protocol=json.loads((ROOT/'artifacts/framework/multitask_r10/common_training_pilot_config_v6.json').read_text())
    old=json.loads((ROOT/'config/framework/aligned_protocol_r09_baseline_v6.json').read_text())
    protocol.update(version='r10_fourteen_task_baseline_integration_v1',
        purpose='released PhysMani TRAIN-only fresh-process engineering smoke',
        baseline_checkpoint=old['baseline_checkpoint'],baseline_decision_ticks=10,
        baseline_executor=EXECUTOR,baseline_reference_filter='author_direct_setpoint',
        formal_comparison_ready=False,formal_metrics_included=False,critic_training_permitted=False,
        videos_per_task=0,baseline_output=str(out/'disabled_wm_logger'),
        r09_completed_release='artifacts/framework/aligned_r09/final_report.json')
    config=out/'protocol.json';atomic_json(config,protocol)
    files={str(path.resolve()):file_sha(path) for path in required_local_sources('physmani')}
    files.update(native_inputs(protocol))
    repo=Path(protocol['baseline_repo'])
    # Freeze the released policy/online-WM implementation used by this pilot.
    for directory in ('physmani_policy','physmani_wm','physmani_bench'):
        for path in (repo/directory).rglob('*.py'):
            files[str(path.resolve())]=file_sha(path)
    extra=[config,manifest,source,Path(__file__),ROOT/'scripts/run_multitask_fresh_queue_r10.py',
        ROOT/'scripts/audit_train_resets_r10.py',ROOT/protocol['r09_completed_release'],
        Path(protocol['baseline_checkpoint']),repo/'tasks/18_peract_tasks_location_bounds.json',
        repo/'instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl']
    for path in extra:files[str(path.resolve())]=file_sha(path)
    lock=out/'frozen_inputs.json';atomic_json(lock,dict(files=files,development_only=True,
        formal_metrics_included=False,TEST_read=False,source_selection_used_success=False))
    plan=out/'queue_plan.json'
    prepared=prepare(SimpleNamespace(config=str(config.relative_to(ROOT)),manifest=str(manifest.relative_to(ROOT)),
        frozen_inputs=str(lock.relative_to(ROOT)),split='train',tasks=None,expected_sources=14,
        methods=['physmani'],replicas=1,explore_first=False,record_demos=False,
        output=str((out/'executions').relative_to(ROOT)),plan=str(plan.relative_to(ROOT))))
    print(json.dumps(dict(plan=str(plan.relative_to(ROOT)),jobs=len(prepared['jobs']),
        native_started=False,GPU_started=False,videos=0,formal_metrics_included=False)),flush=True)


if __name__=='__main__':main()
