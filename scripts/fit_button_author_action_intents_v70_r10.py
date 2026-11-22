"""Fit original author command goals, separate from actual press completion.

Every original TRAIN parent and every executed press command is retained,
including unsuccessful/censored cases. This artifact grants no skill-completion
capability and is not an InteractionLibrary. Public/native completion labels
are not used to fit or select the action model.
"""
from pathlib import Path
from dataclasses import asdict
import json,ast,sys
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal,translation_command_goal_reference as translation
from our_mani.multitask.evaluation import atomic_json,file_sha
BASE=ROOT/'artifacts/framework/multitask_r10';OUT=BASE/'button_author_action_intents_v70'
TASKS=('push_moving_button','push_moving_button_high_speed')


def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))


def main():
    if OUT.exists():raise RuntimeError('preserve existing candidate fit')
    rows=json.loads((BASE/'button_press_native_success200_endpoint_regions_diagnostic_v2/all200_source_endpoint_audit.json').read_text())
    assert len(rows)==200 and len({r['parent'] for r in rows})==200
    ruleptr=pointer(BASE/'command_goal_reference_rule_v6_counted_final_setup_v1/command_goal_rule.json')
    rule=goal.load_rule(ROOT,ruleptr,purpose='production_fitting')
    inputs=dict(rule.verified_files_sha256);inputs[str(Path(__file__).resolve())]=file_sha(__file__)
    # Only the exact static boundary function is imported; no native handle or
    # dynamic state is created. This is an offline TRAIN source replay.
    native=ROOT.parent/'phsimani/PhysMani/third_party/RLBench/rlbench/tasks'
    helper=native/'reach_single_moving_target_on_the_table_high_speed.py'
    tree=ast.parse(helper.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='handle_boundary')
    namespace=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),'<TRAIN-static-button-boundary>','exec'),namespace)
    inputs[str(helper)]=file_sha(helper)
    for task in TASKS:
        archived=BASE/'command_goal_reference_rule_v5_translation_revision3_setup/readonly_source'/(task+'.py')
        assert archived.is_file() and file_sha(archived)==file_sha(native/(task+'.py'))
        pointer(archived)
    samples=defaultdict(list);parents=defaultdict(set);proofs=[]
    for old in rows:
        binding=old['raw_bindings']
        for pt in binding.values():assert file_sha(ROOT/pt['path'])==pt['sha256'];inputs[str((ROOT/pt['path']).resolve())]=pt['sha256']
        path=ROOT/binding['canonical_receipt']['path'];record=json.loads(path.read_text());cp=ROOT/binding['commands']['path'];commands=json.loads(cp.read_text())
        with np.load(ROOT/binding['cache']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=commands,
            command_receipt_verified=True,command_receipt_sha256=binding['commands']['sha256'])
        source=adapter.interaction_source(case)[0];verified=goal.verify_original_source(case,source,rule);inputs.update(verified.derivation_proof['verified_files_sha256'])
        states,kernel,replay=translation.replay_states(case,rule.translation_permission)
        task=source.task_name;assert task in TASKS;parents[task].add(source.grouping_source)
        high=task.endswith('high_speed');delay=.25 if high else .5;_,area=translation._kernel(rule.translation_permission,task)
        ids=arrays['body_ids'].tolist();bi=ids.index('button');times=arrays['times'];kept=[]
        for command in source.command_ledger:
            phase=command['raw_phase'];assert phase in ('approach','press')
            if command['executed_tick_count']<1:continue
            found=np.flatnonzero(np.isclose(times,command['generated_at'],rtol=0,atol=1e-8));assert len(found)==1;i=int(found[0])
            if phase=='press':
                tt,state=states[i];reference=np.asarray(kernel['compute_target_position'](t=tt+delay,t0=state['t0'],x0=state['x'],v0=state['v'],a0=state['a'],dt=float(np.float32(.05))))
                if high:reference=namespace['handle_boundary'](reference.copy(),area)
            else:reference=arrays['target_positions'][i,bi]
            raw=np.asarray(command['raw_waypoint_xyzw']);position=raw[:3]-reference
            assert np.linalg.norm(position[:2])<=1e-6,'source command is not its declared button intention'
            sample=dict(task=task,parent=source.grouping_source,source_id=source.source_id,command_id=command['command_id'],
                phase=phase,generated_at=float(command['generated_at']),position=position.tolist(),rotation_xyzw=raw[3:].tolist(),
                desired_opening=float(command['desired_opening']),executed_ticks=command['executed_tick_count'],
                command_completed_original=bool('command_completed_at' in command),raw_native_outcome_used_for_fit=False,
                source_forecast_seconds=delay if phase=='press' else 0.)
            samples[(task,phase)].append(sample);kept.append(command['command_id'])
        proofs.append(dict(task=task,parent=source.grouping_source,original_bindings=binding,kept_executed_commands=kept,
            all_original_commands_and_labels_preserved=True,source_replay_maximum_error_m=replay['maximum_body_replay_error_m']))
    result={}
    for task in TASKS:
        assert len(parents[task])==100;phases={}
        for phase in ('approach','press'):
            ss=samples[(task,phase)];pp=sorted({r['parent'] for r in ss});assert len(pp)>=8
            # One total weight per original parent, without outcome filtering.
            parent_p=np.asarray([np.mean([r['position'] for r in ss if r['parent']==p],axis=0) for p in pp])
            parent_r=np.asarray([Rotation.from_quat([r['rotation_xyzw'] for r in ss if r['parent']==p]).mean().as_matrix() for p in pp])
            folds={p:i%2 for i,p in enumerate(pp)};errors=[]
            for fold in (0,1):
                train=[i for i,p in enumerate(pp) if folds[p]!=fold];fitp=parent_p[train].mean(0);fitr=Rotation.from_matrix(parent_r[train]).mean().as_matrix()
                for i,p in enumerate(pp):
                    if folds[p]!=fold:continue
                    errors.append(dict(parent=p,position_m=float(np.linalg.norm(parent_p[i]-fitp)),orientation_rad=float(Rotation.from_matrix(parent_r[i]@fitr.T).magnitude())))
            pt=max(.005,float(np.quantile([r['position_m'] for r in errors],.95)));rt=max(.03,float(np.quantile([r['orientation_rad'] for r in errors],.95)))
            phases[phase]=dict(supervision='original_author_command_goal_not_actual_completion_TCP',parents=len(pp),commands=len(ss),
                minimum_independent_command_parent_support=8,source_to_fold=folds,grouped_TRAIN_CV_errors=errors,
                position_tolerance_m=pt,orientation_tolerance_rad=rt,original_point03_point15_error_caps_passed=pt<=.03 and rt<=.15,
                position=parent_p.mean(0).tolist(),rotation_xyzw=Rotation.from_matrix(parent_r).mean().as_quat().tolist(),
                desired_opening=float(np.mean([np.mean([r['desired_opening'] for r in ss if r['parent']==p]) for p in pp])),
                command_completion_claimed=False,press_or_task_completion_capability_granted=False)
        source=BASE/'command_goal_reference_rule_v5_translation_revision3_setup/readonly_source'/(task+'.py')
        assert source.is_file() and file_sha(source)==file_sha(native/(task+'.py'))
        inputs[str(source)]=file_sha(source);tree=ast.parse(source.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
        radius=[n.value.value for n in ast.walk(expert) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='th_wp0' for t in n.targets) and isinstance(n.value,ast.Constant)]
        assert len(radius)==1 and radius[0]==.3
        model=dict(schema='r10_TRAIN_original_author_button_command_intent_model_v70',task=task,fit_split='train',TEST_used=False,
            original_parent_count=100,all_failed_censored_original_parents_preserved=True,phases=phases,
            source_stage_guard=dict(approach_to_press_distance_m=radius[0],source=pointer(source),guard_is_not_verified_skill_completion=True),
            source_press_forecast_seconds=.25 if task.endswith('high_speed') else .5,
            native_success_labels_used_for_fit=False,public_press_deformation_verifier_granted=False,
            runtime_enabled=False,formal_release=False,original_completed_skill_library_impersonated=False)
        atomic_json(OUT/task/'model.json',model);result[task]=dict(model=pointer(OUT/task/'model.json'),phase_summary={p:{k:v for k,v in row.items() if k in ('parents','commands','position_tolerance_m','orientation_tolerance_rad','original_point03_point15_error_caps_passed')} for p,row in phases.items()})
    atomic_json(OUT/'raw_command_proofs.json',proofs);atomic_json(OUT/'command_intent_samples.json',{task:{phase:samples[(task,phase)] for phase in ('approach','press')} for task in TASKS})
    for path,sha in inputs.items():assert file_sha(path)==sha
    atomic_json(OUT/'report.json',dict(schema='r10_button_original200_command_intent_fit_v70',tasks=result,all200_original_parents_retained=True,
        original_inputs_sha256=inputs,raw_proofs=pointer(OUT/'raw_command_proofs.json'),command_samples=pointer(OUT/'command_intent_samples.json'),
        all_action_error_caps_passed=all(row['original_point03_point15_error_caps_passed'] for t in result.values() for row in t['phase_summary'].values()),
        actual_completed_skill_labels_not_replaced=True,public_native_completion_capability_granted=False,runtime_enabled=False,formal_release=False))
    print(json.dumps(dict(report=str((OUT/'report.json').relative_to(ROOT)),tasks=result)))

if __name__=='__main__':main()
