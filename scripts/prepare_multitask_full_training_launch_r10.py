"""CPU-only explicit config checks and deferred full-data training argv."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.acquisition_lineage import file_sha,pointer,require
from our_mani.multitask.interaction import InteractionFitConfig,InteractionArtifactError,InteractionFittingPending,fit_interaction_library
from our_mani.multitask.public_roles import ColourFitConfig
from our_mani.multitask.artifact_fitting import fit_layout
from tests.test_multitask_completed_skill_fitting import completed_source
from tests.test_multitask_artifact_fitting import teacher_case

BASE=ROOT/'artifacts/framework/multitask_r10'
FIT_CONFIG=ROOT/'config/framework/multitask_r10_artifact_fit_v1.json'
OUT=BASE/'launch_preparation_v2'


def cpu_checks(config):
    interaction=InteractionFitConfig(**config['interaction']);colour=ColourFitConfig(**config['colour'])
    require(asdict(interaction)==config['interaction'] and asdict(colour)==config['colour'],'constructor field roundtrip differs')
    require(interaction.minimum_sources==interaction.minimum_completed_parent_sources==8
        and interaction.maximum_position_tolerance_m==.03 and interaction.maximum_orientation_tolerance_rad==.15
        and interaction.development_only is False,'existing formal8/caps must not be weakened')
    sources=[completed_source(index) for index in range(8)]
    library=fit_interaction_library(sources,interaction)
    failures={}
    try:fit_interaction_library(sources,replace(interaction,minimum_completed_parent_sources=7))
    except InteractionArtifactError as error:failures['min7_configuration_rejected']=str(error)
    else:raise AssertionError('formal min7 accepted')
    try:fit_interaction_library(sources[:7],interaction)
    except InteractionFittingPending as error:failures['seven_completed_parents_pending']=str(error)
    else:raise AssertionError('seven actual completed parent sources accepted')
    for kind in ('position','orientation'):
        shifted=[]
        for index,source in enumerate(sources):
            if index<4:shifted.append(source);continue
            poses=source.measured_tcp_xyzw.copy()
            if kind=='position':poses[:,0]+=.1
            else:poses[:,3:]=Rotation.from_euler('z',.4).as_quat()
            events=[]
            for event in source.completion_events:
                terminal=event.terminal_tcp_xyzw.copy()
                if kind=='position':terminal[0]+=.1
                else:terminal[3:]=Rotation.from_euler('z',.4).as_quat()
                events.append(replace(event,terminal_tcp_xyzw=terminal))
            shifted.append(replace(source,measured_tcp_xyzw=poses,curve_tcp_xyzw=poses,completion_events=tuple(events)))
        try:fit_interaction_library(shifted,interaction)
        except InteractionFittingPending as error:
            diagnostics=list(error.diagnostics['phase_fits'].values())
            key='position_tolerance_m' if kind=='position' else 'orientation_tolerance_rad'
            cap=.03 if kind=='position' else .15
            require(any(row.get(key,0)>cap and row.get('passed') is False for row in diagnostics),'terminal residual was not rejected above hard cap')
            failures[kind+'_cap_excess_pending_without_clipping']=dict(error=str(error),actual_residuals=[row.get(key) for row in diagnostics],cap=cap)
        else:raise AssertionError(kind+' residual cap silently relaxed')
    layout=fit_layout([teacher_case(index) for index in range(8)],'push_moving_button',config['rigid_layout'])
    layout7=fit_layout([teacher_case(index) for index in range(7)],'push_moving_button',config['rigid_layout'])
    require(len(layout['accepted'])==1 and not layout7['accepted'],'explicit layout source support differs')
    return dict(passed=True,constructor_roundtrip=True,synthetic_completed_parent_fixture_count=8,
        synthetic_library_template_count=len(library.templates),development_support_override=False,
        hard_gate_negative_checks=failures,rigid_layout_synthetic8parent_passed=True,rigid_layout_synthetic7parent_rejected=True,
        colour_actual_native_pixel_fit_performed=False,real_TRAIN_artifacts_fitted=False,GPU_used=False,native_started=False)


def main():
    require(not OUT.exists(),'fresh launch-preparation directory required')
    config=json.loads(FIT_CONFIG.read_text());require(config['fit_split']=='train' and config['VAL_used'] is False and config['TEST_used'] is False,'TRAIN-only config required')
    checks=cpu_checks(config);python=str(ROOT/'.conda-env/bin/python')
    train_release='artifacts/framework/multitask_r10/TRAIN_full_lineage_release_v1'
    train_manifest=train_release+'/observed_lineage_manifest.json'
    val_manifest='artifacts/framework/multitask_r10/VAL_hold_collection_v16/manifest.json'
    val_registration='artifacts/framework/multitask_r10/VAL_hold_collection_v16_setup/frozen/execution_registration.json'
    view='artifacts/framework/multitask_r10/context_training_view_full_v1'
    config_name=str(FIT_CONFIG.relative_to(ROOT));registrations='artifacts/framework/multitask_r10/TRAIN_composite_registration_preflight_v2/registrations.json'
    jobs=[
        dict(id='committed_counts_preflight',parallel_group=None,argv=[python,'scripts/preflight_multitask_full_acquisition_counts_r10.py',
            '--train-registrations',registrations,'--validation-registration',val_registration,'--require-complete',
            '--out','artifacts/framework/multitask_r10/full_acquisition_counts_release_preflight_v1'],
            run_only_after=['all original TRAIN and recovery native workers have finished','VAL275 queue finished writing its actual ordinary manifest',
                'use a new --out directory on each renewed check; never overwrite an earlier partial snapshot']),
        dict(id='complete_TRAIN_lineage',parallel_group=None,argv=[python,'scripts/audit_acquisition_lineage_r10.py','--registrations',registrations,
            '--require-complete','--out',train_release],run_only_after=['committed_counts_preflight ready=true for exact TRAIN2800 andVAL275 with all raw parents committed and stable source/inventory',
                'v11 original queue idle with all parent receipts committed','v15 two completed preserved','v16 known54 either committed or represented by explicit additional recovery plans','all actual historical plans retained in registrations']),
        dict(id='complete_actual_VAL_lineage',parallel_group=None,argv=[python,'scripts/audit_validation_acquisition_lineage_r10.py',
            '--manifest',val_manifest,'--out','artifacts/framework/multitask_r10/VAL_full_lineage_release_v1'],
            run_only_after=['committed_counts_preflight ready=true','actual VAL_hold_collection_v16/manifest.json complete275, includes shared_profile_execution_registration',
                'all raw parent receipts committed, source/archive audits and inventory stable']),
        dict(id='merge_complete_context_view',parallel_group=None,argv=[python,'scripts/merge_multitask_context_training_view_r10.py',
            '--train-manifest',train_manifest,'--validation-manifest',val_manifest,
            '--train-sources','artifacts/framework/protocol/train_manifest.json','--validation-sources','artifacts/framework/protocol/validation_manifest.json','--out',view],
            run_only_after=['complete_TRAIN_lineage','complete_actual_VAL_lineage','final consumer source freeze']),
        dict(id='context_GPU',parallel_group='full_fits',environment=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),
            argv=[python,'scripts/train_multitask_context_r10.py','--manifest',view+'/manifest.json',
                '--out','artifacts/framework/multitask_r10/context_se3_full_v1','--device','cuda','--epochs','30','--batch-size','256',
                '--windows-per-parent','64','--steps','22','--expected-train','1400','--expected-validation','275','--seed','1018','--learning-rate','0.0003'],
            run_only_after=['merge_complete_context_view reports training_view_ready=true','1400 distinct TRAIN and275 VAL hold parents each have usable causal windows','each of14 tasks per split has a full1.1s window','R09 protocol400/400 and orchestrator complete','fresh model directory']),
        dict(id='interaction_colour_layout_CPU',parallel_group='full_fits',environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),
            argv=[python,'scripts/fit_multitask_interaction_artifacts_r10.py','--collector-manifest',train_manifest,
                '--demonstrations','artifacts/framework/multitask_r10/demonstrations/manifest.json',
                '--original-prior','artifacts/framework/multitask_r10/original_TRAIN_action_phase_priors.json',
                '--fit-config',config_name,'--out','artifacts/framework/multitask_r10/TRAIN_artifacts_full_v1'],
            run_only_after=['complete_TRAIN_lineage','all original1400 action-phase prior hashes verified','new explicit TRAIN-only fit config fixed and hashed','fresh fit directory'],
            pending_is_a_valid_result='Insufficient completed phase/partition support or terminal caps/colour/layout rejection remain visible; never downgrade a threshold or fabricate missing phases.')]
    progress_path=BASE/'TRAIN_supervision_progress_20261018T0540Z.json';progress=json.loads(progress_path.read_text())
    low=[]
    for task,row in progress['per_task'].items():
        for phase,values in row.get('phases',{}).items():
            if not values['optional_after_native_completion'] and values['completed_parent_count']<8:
                low.append(dict(task=task,phase=phase,observed_teacher_acquisitions=row.get('teacher',0),completed_parents=values['completed_parent_count']))
    report=dict(schema='r10_full_data_training_launch_preparation_v1',created_utc=datetime.now(timezone.utc).isoformat(),status='prepared_waiting_for_complete_data_and_strict_source_audits',
        working_directory=str(ROOT),actual_training_launched=False,native_or_GPU_training_started=False,current_main_acquisition_complete_claimed=False,
        fit_config=pointer(ROOT,FIT_CONFIG),earliest_preregistration_claimed=False,CPU_configuration_checks=checks,
        original_fixed_contract=dict(TRAIN_parents=1400,TRAIN_teacher_cases=1400,TRAIN_hold_cases=1400,VAL_hold_parents=275,K_per_behavior=1,
            minimum_completed_parents_per_skill_partition=8,max_terminal_position_m=.03,max_terminal_angle_rad=.15),
        actual_validation_manifest_binding=dict(path=val_manifest,registration=val_registration,
            status='actual VAL queue output path explicitly bound; completeness is not claimed at preparation',
            standalone_strict_audit_report_directory='artifacts/framework/multitask_r10/VAL_full_lineage_release_v1',
            audit_report_directory_is_not_a_replacement_acquisition_manifest=True,required_complete_distinct_VAL_parents=275),
        known_registrations=pointer(ROOT,ROOT/registrations),registrations_must_include_any_future_real_recovery_plans=True,
        command_jobs=jobs,parallelism='Only after readiness: Context GPU and interaction/colour/layout CPU can run together. CPU fitter uses TRAIN only; grouped VAL is solely Context selection.',
        directories_are_proposed_fresh_targets=True,existing_outputs_must_never_be_overwritten=True,
        existing_CLI_help_schema_checks_passed=['preflight_multitask_full_acquisition_counts_r10.py','audit_acquisition_lineage_r10.py',
            'audit_validation_acquisition_lineage_r10.py','merge_multitask_context_training_view_r10.py','train_multitask_context_r10.py','fit_multitask_interaction_artifacts_r10.py'],
        latest_reviewed_supervision_snapshot=pointer(ROOT,progress_path),latest_reviewed_snapshot_created_utc=progress['created_utc'],
        latest_reviewed_snapshot_canonical_count=progress['snapshot_canonical_count'],snapshot_is_observed_so_far_not_final=True,
        required_phases_observed_below8=low,
        concrete_supervision_risk='High-speed basketball deposit has6 completed parents from96 teacher acquisitions in the reviewed snapshot. Four original teacher infra cases were pending at that snapshot; reaching8 needs at least2 additional completed deposit parents. This is pending observation, not a final fit failure or permission to resample.',
        additional_fit_risks=['8 aggregate phase parents are necessary but insufficient: each exact anchor/reference/manipulated-role/mode partition needs8 with source-grouped lag/terminal-cap crossfit.',
            'Colour two-fold source CV requires at least8 fit parents per class in each fold, at least16 total per class; actual support/separability may remain pending.',
            'Rigid fitting checks every declared rollout; fewer than5 valid native pose samples or inconsistent relations reject the relation without dropping a parent.',
            'A full acquisition does not guarantee1400/275 usable hold parents or full-horizon supervision; context-view gates remain mandatory.',
            'Raw teacher rows and original1400 HDF5 phase priors never substitute for completed native skill attempts. Native-terminal failures remain censored.'],
        frozen_core_sources_modified=False)
    OUT.mkdir();(OUT/'CPU_config_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'launch_plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['Prepared metadata/CPU validation only. No actual full-data fitting started.',
        'All audit/model output directories below are fresh destinations. VAL acquisition input is the actual VAL_hold_collection_v16/manifest.json.',
        'Config numerical choices are fixed current TRAIN engineering settings, not claimed earliest-preregistered.',
        'Keep existing min8/.03m/.15rad hard gates; missing support remains pending.',
        'Run countpreflight then complete TRAIN/VAL audits and merge in order, only after all2800 TRAIN+275VAL raw parent commits and stable source audits.',
        'After those gates pass, Context GPU and interaction CPU jobs may run in parallel. Renew countpreflight using a fresh --out each time.',
        'Working directory: '+str(ROOT),'cd '+shlex.quote(str(ROOT)),'']
    for index,job in enumerate(jobs,1):
        lines.append(str(index)+'. '+job['id'])
        env=' '.join(name+'='+shlex.quote(value) for name,value in job.get('environment',{}).items())
        lines.append((env+' ' if env else '')+shlex.join(job['argv']));lines.append('Requires: '+'; '.join(job['run_only_after']));lines.append('')
    lines.extend(['Reviewed observed-so-far supervision snapshot:'+progress['created_utc'],report['concrete_supervision_risk'],
        'Final source freeze and actual model/artifact acceptance remain required. No VAL/TEST fit-parameter tuning.'])
    (OUT/'commands.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(prepared=True,actual_training_launched=False,config=pointer(ROOT,FIT_CONFIG),launch_plan=pointer(ROOT,OUT/'launch_plan.json'))))


if __name__=='__main__':main()
