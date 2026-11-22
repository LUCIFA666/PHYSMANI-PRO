"""Fit the new R10 executed-continuation critic using CPU TRAIN and VAL only.

Example (after actual collection): --train-manifest ... --validation-manifest
... --task ... --predictor context --interaction-library ... --continuation-sha256
... --out artifacts/framework/multitask_r10/critics/TASK_context. Output paths
must be new. The script cannot launch native simulation or read TEST labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from our_mani.multitask.candidate_success import (CONTINUATION_POLICY,
    FEATURE_NAMES,FEATURE_SCHEMA_SHA256,LABEL_SCOPE,SCHEMA,CandidateSuccessModel)
from our_mani.multitask.candidate_training import (CriticFitConfig,fit_phase_head,
    load_executed_manifest,parent_group,sha)
from our_mani.multitask.interaction import InteractionLibrary


def write(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest',required=True)
    parser.add_argument('--validation-manifest',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--predictor',required=True,choices=('cv','context'))
    parser.add_argument('--interaction-library',required=True)
    parser.add_argument('--continuation-sha256',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seed',type=int,default=10)
    args=parser.parse_args()
    if len(args.continuation_sha256) != 64 or any(character not in '0123456789abcdef' for character in args.continuation_sha256):
        raise ValueError('bind an actual SHA256 of the frozen full geometry continuation')
    out=Path(args.out)
    out=out if out.is_absolute() else ROOT/out
    if out.exists():
        raise ValueError('preserve existing trained critic/report; choose a new output directory')
    library=InteractionLibrary.load(args.interaction_library)
    if library.task_spec.task_name != args.task:
        raise ValueError('interaction artifact belongs to a different task')
    common=dict(task_name=args.task,predictor=args.predictor,continuation_sha256=args.continuation_sha256,
                library_sha256=library.sha256,project_root=ROOT)
    train,train_subset,train_audit=load_executed_manifest(args.train_manifest,split='train',**common)
    val,val_subset,val_audit=load_executed_manifest(args.validation_manifest,split='validation',**common)
    if train_audit['censored_rollouts'] or val_audit['censored_rollouts']:
        raise ValueError('finish/retry all declared native TRAIN/VAL sources before fitting; infrastructure censoring is not a policy negative')
    train_groups={parent_group(row) for row in train_subset['entries']}
    val_groups={parent_group(row) for row in val_subset['entries']}
    strip_split=lambda names:{name.split(':',1)[1] for name in names}
    if strip_split(train_groups) & strip_split(val_groups):
        raise ValueError('TRAIN/VAL raw parent sources overlap')
    # The deployed critic is queried once, at the first actionable selection.
    # Later phase/retry labels remain in the immutable source ledger but cannot
    # change the classifier or CAL target distribution for that first decision.
    train=[row for row in train if row['first_actionable']]
    val=[row for row in val if row['first_actionable']]
    fit_config=CriticFitConfig(seed=args.seed)
    heads,reports={},{}
    phases=sorted({row['phase'] for row in train},key=library.phase_order.index)
    if set(phases) != {library.phase_order[0]}:
        raise ValueError('no actual TRAIN first actionable phase decisions for the episode-level critic')
    omitted=[dict(phase=phase,reason='unused downstream phase; frozen geometry continuation has no critic query')
             for phase in library.phase_order[1:]]
    for phase_index,phase in enumerate(phases):
        train_phase=[row for row in train if row['phase'] == phase]
        val_phase=[row for row in val if row['phase'] == phase]
        if len({row['group'] for row in train_phase}) < 2:
            if phase == library.phase_order[0]:
                raise ValueError('first actionable critic lacks two independent TRAIN parent sources')
            omitted.append(dict(phase=phase,reason='fewer than two TRAIN parents; unused downstream phase is geometry-only'))
            continue
        head,report=fit_phase_head(train_phase,val_phase,fit_config,seed_offset=phase_index*1009)
        key='|'.join((args.task,args.predictor,phase));heads[key]=head;reports[phase]=report
    out.mkdir(parents=True)
    train_subset_path=out/'TRAIN_fit_manifest.json';val_subset_path=out/'VAL_calibration_manifest.json'
    write(train_subset_path,train_subset);write(val_subset_path,val_subset)
    metadata=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),task=args.task,
        predictor=args.predictor,training_complete=True,classifier_fit_split='train',
        classifier_selection_split='train',normalization_fit_split='train',
        TEST_used_for_fitting=False,synthetic=False,label_scope=LABEL_SCOPE,
        continuation_sha256=args.continuation_sha256,interaction_library_sha256=library.sha256,
        interaction_library_path=str(Path(args.interaction_library).resolve()),
        continuation_selection_policy=CONTINUATION_POLICY,
        probability_target='selected candidate survives same-phase abandonment and frozen geometry downstream reaches native success',
        value_use='first actionable selection once per episode including fallback; all later selections geometry',
        classifier_and_calibration_population='episode_first_actionable_decision_only',
        feature_names=list(FEATURE_NAMES),feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        TRAIN_source_manifest_path=str(train_subset_path),TRAIN_source_manifest_sha256=sha(train_subset_path),
        TRAIN_parent_source_grouping=['split','task','parent_raw_episode_id_or_raw_episode_id'],
        TRAIN_parent_source_groups=sorted(train_groups),
        VAL_parent_source_groups=sorted(val_groups),
        TRAIN_eligible_first_decision_parent_source_groups=sorted({row['group'] for row in train}),
        VAL_eligible_first_decision_parent_source_groups=sorted({row['group'] for row in val}),
        execution_coverage_proof=dict(TRAIN=train_audit,VAL=val_audit),
        formal_model_coverage_requires_independent_preflight=True,
        VAL_calibration_manifest_path=str(val_subset_path),VAL_calibration_manifest_sha256=sha(val_subset_path),
        classifier_selection_uses_VAL=False,calibration_selector_used_VAL=True,
        fit_configuration=fit_config.__dict__,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (
            Path(__file__),ROOT/'our_mani/multitask/candidate_training.py',
            ROOT/'our_mani/multitask/candidate_success.py',ROOT/'our_mani/multitask/interaction.py')},
        unexecuted_candidates_labelled=False,omitted_downstream_heads=omitted)
    artifact_path=out/'value.json';write(artifact_path,dict(metadata=metadata,heads=heads))
    loaded=CandidateSuccessModel(artifact_path,continuation_sha256=args.continuation_sha256,library_sha256=library.sha256)
    first_phase=library.phase_order[0]
    first_report=reports[first_phase]
    report=dict(metadata=metadata,artifact=str(artifact_path),artifact_sha256=sha(artifact_path),
        TRAIN_input_audit=train_audit,VAL_input_audit=val_audit,per_phase=reports,
        first_actionable_phase=first_phase,episode_value_ranking_enabled=first_report['ranking_enabled'],
        calibration_gate_passed=first_report['calibration_gate_passed'],
        ready_for_fixed_policy_validation=True,
        formal_model_coverage_passed=False,
        task_success_rate_gain_validated=False,formal_TEST_used=False,
        interpretation='TRAIN classifier/source CV; nested VAL calibration. Disabled ranking falls back to geometry; no SR gain claimed.')
    if not loaded.heads:
        raise ValueError('fitted critic has no executable trained heads')
    write(out/'report.json',report)
    print(json.dumps(dict(artifact=str(artifact_path),heads=len(heads),
        first_phase=first_phase,ranking_enabled=first_report['ranking_enabled'],
        calibration_gate_passed=first_report['calibration_gate_passed'],
        formal_TEST_used=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
