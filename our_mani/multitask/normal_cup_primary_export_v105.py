"""Admit an explicit initial-sequence policy from previously fitted TRAIN curves.

Every original phase fit and completion label remains in the audit, including
unsupported released-body retries. This is a new policy domain declaration,
not a claim that the old all-observed-partitions fitting gate passed.
"""
from dataclasses import asdict
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

from . import interaction
from .specs import TASK_SPECS

TASK='place_cups_on_rotating_frame'
SCHEMA='r10_normal_cup_initial_sequence_admission_v105'


def admit_templates(draft,production,configuration):
    c=configuration
    if (c.minimum_completed_parent_sources!=8 or c.maximum_position_tolerance_m!=.03
            or c.maximum_orientation_tolerance_rad!=.15 or c.development_only is not False):
        raise ValueError('original8/.03/.15 support and terminal caps required')
    original=draft['phase_fits'];other=production['phase_fits']
    if set(original)!=set(other):raise ValueError('original independent producer partition coverage differs')
    admitted=[];omitted=[]
    for key,row in original.items():
        parts=key.split('|')
        if len(parts)!=7:raise ValueError('exact phase/reference/mode/time partition required')
        phase,anchor,role,kind,manipulated,mode,basis=parts
        if not anchor.startswith(TASK+':'+phase+':'):
            raise ValueError('partition belongs to another task or phase')
        check=other[key]
        for name in ('passed','completed_parent_count','completed_attempt_count'):
            if row.get(name)!=check.get(name):raise ValueError('independent original fit support differs: '+key)
        for name in ('selected_lag','position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds'):
            x,y=row.get(name),check.get(name)
            if (x is None)!=(y is None) or (x is not None and abs(x-y)>1e-12):
                raise ValueError('independent original numeric fit differs: '+key)
        if mode=='released':
            if phase not in ('approach','acquire') or role!='cup0':
                raise ValueError('only explicit released-cup recovery lies outside this policy domain')
            # Unconditional domain restriction, even if a future fit happened
            # to support that partition; no outcome-based admission rule.
            omitted.append(dict(key=key,original_completed_parent_count=row['completed_parent_count'],
                original_fit_passed=row['passed'],reason='released-body recovery outside this declared policy domain'))
            continue
        if mode not in ('external','not_applicable'):
            raise ValueError('initial-sequence reference mode is undeclared')
        if (not row.get('passed') or row['completed_parent_count']<8
                or row['position_tolerance_m']>.03 or row['orientation_tolerance_rad']>.15):
            raise ValueError('mandatory initial-sequence partition is unqualified: '+key)
        if basis!=interaction.CURRENT:raise ValueError('this scope only exports the verified original current-reference cup fits')
        curve=row['fitted_curve']
        template=interaction.PhaseTemplate(phase=phase,anchor_id=anchor,reference_role=role,
            reference_kind=kind,manipulated_role=manipulated,lag_seconds=row['selected_lag'],
            progress=curve['progress'],positions=curve['positions'],
            rotations=Rotation.from_quat(np.asarray(curve['rotation_xyzw'])).as_matrix(),opening=curve['opening'],
            duration_seconds=row['duration_seconds'],duration_budget_seconds=row['duration_budget_seconds'],
            position_tolerance_m=row['position_tolerance_m'],orientation_tolerance_rad=row['orientation_tolerance_rad'],
            reference_mode=mode,reference_time_basis=basis)
        admitted.append(template)
    wanted=[p.name for p in TASK_SPECS[TASK].phase_skills]
    if set(t.phase for t in admitted)!=set(wanted):raise ValueError('every original mandatory task phase must remain trained')
    return admitted,dict(schema=SCHEMA,task=TASK,declared_reference_domain='initial_sequence; external/phase_origin; no released-body recovery',
        admitted_partition_keys=[key for key in original if '|released|' not in key],
        unadmitted_partitions=omitted,original_all_observed_partitions_gate_passed=not draft['failed_fit_keys'],
        original_required_phase_failures=list(draft['required_phase_failures']),
        admitted_partition_support_and_error_gates_passed=True,
        original_raw_completion_labels_not_changed=True,original_source_parents_not_filtered=True,
        old_full_scope_library_release_claimed=False,new_numeric_refit_performed=False,
        native_or_formal_release_granted=False)


def payload(templates,scope,*,sources,manifest,fit_configuration,production_report,export_proof):
    # The actual numeric helpers are the unchanged interaction.py kernels;
    # the exporter/admission source is additionally explicit and source-bound.
    module=Path(interaction.__file__)
    m=dict(schema=interaction.LIBRARY_SCHEMA,task_name=TASK,training_complete=True,
        fit_split='train',selection_split='train',TEST_used_for_fitting=False,synthetic=False,
        TRAIN_source_ids=sorted(s.source_id for s in sources),
        TRAIN_parent_source_groups=sorted({s.grouping_source for s in sources}),
        TRAIN_source_provenance={s.source_id:dict(source_kind=s.source_kind,parent_source_group=s.grouping_source,
            augmentation_id=s.augmentation_id,raw_episode_id=s.raw_episode_id,parent_raw_episode_id=s.parent_raw_episode_id) for s in sources},
        TRAIN_source_manifest_path=str(manifest),TRAIN_source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        fitting_module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        task_spec_module_sha256=hashlib.sha256((module.parent/'specs.py').read_bytes()).hexdigest(),
        fit_configuration=asdict(fit_configuration),source_grouped_lag_selection=production_report['phase_fits'],
        completed_skill_fitting=production_report,initial_sequence_policy_scope=scope,
        actual_curve_producer_and_admission_export=export_proof,
        library_exported_from_previously_fitted_curves=True,original_full_scope_fitter_returned_library=False,
        BC_used_source_ids=sorted(s.source_id for s in sources),BC_excluded_source_ids=[],
        all_original_parent_sources_retained=True,completion_tolerances_are_terminal_only=True,
        reference_time_schema=interaction.REFERENCE_TIME_SCHEMA,
        reference_time_helper_sha256=hashlib.sha256((module.parent/'reference_time.py').read_bytes()).hexdigest(),
        reference_time_rule_sha256_values=[],formal_comparison_ready=False,
        development_support_override=False,target_label_column_is_not_a_body_track=True,
        expert_actions_are_training_labels_not_runtime_inputs=True,
        fitted_phase_order=[p.name for p in TASK_SPECS[TASK].phase_skills],omitted_optional_phases=[],
        TRAIN_stage_coverage={s.source_id:dict(Counter(s.stages)) for s in sources})
    return dict(metadata=m,templates=[asdict(t) for t in templates])
