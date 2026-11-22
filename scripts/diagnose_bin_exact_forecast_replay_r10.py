"""Replay original bin task arithmetic on all100 TRAIN raw tracks; CPU only."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import List,Tuple

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def replay(manifest_path,output):
    output=a.local(ROOT,output);a.require(not output.exists(),'fresh replay output required');output.mkdir()
    files={};manifest=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,manifest_path)),files))
    a.require(manifest['task']=='put_rubbish_in_moving_bin' and len(manifest['case_proofs'])==100,
        'strict actual all100 production raw audit required')
    native=Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/tasks')
    names=['put_rubbish_in_moving_bin.py','reach_single_moving_target_on_the_table_high_speed.py']
    sources={}
    for name in names:
        p=native/name;q=output/'archived_readonly_source'/name;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);sources[str(p)]=dict(sha256=a.file_sha(p),archive=a.pointer(ROOT,q))
    tree=ast.parse((native/names[1]).read_text());helper_nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)
        and node.name in ('compute_target_position','cross_boundary')]
    a.require(len(helper_nodes)==2,'exact original arithmetic helpers required')
    task_tree=ast.parse((native/names[0]).read_text());cls=next(node for node in task_tree.body if isinstance(node,ast.ClassDef))
    step=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='step')
    ns={'np':np,'deepcopy':deepcopy,'List':List,'Tuple':Tuple,'Shape':object}
    exec(compile(ast.Module(body=helper_nodes+[step],type_ignores=[]),'<original-bin-arithmetic-AST>','exec'),ns)
    class Bin:
        def __init__(self,value):self.value=np.array(value,dtype=np.float32)
        def get_position(self):return self.value.astype(float).tolist()
        def set_position(self,value):self.value=np.array(value,dtype=np.float32)
    cases=[];open_commands=[];closed_commands=[];all_errors=[]
    for row in manifest['case_proofs']:
        record=a.read(a.checked_pointer(ROOT,row['canonical_receipt'],files))
        with np.load(a.checked_pointer(ROOT,row['cache'],files),allow_pickle=False) as stored:
            arrays={k:stored[k].copy() for k in stored.files}
        commands=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
        j=list(arrays['body_ids']).index('bin');positions=arrays['target_positions'][:,j]
        initial=record['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule']
        # The original C API declares simFloat simGetSimulationTimeStep();
        # retain that binary32 value for task.t, independently of the policy
        # ledger's nominal 50ms integer-index clock.
        dt=float(np.float32(.05));ledger_dt=.05
        # Retain the source's Python arithmetic, including the upper-x float
        # 0.4-0.2/2 rather than a rounded decimal replacement at equality.
        area=[.15,-.5,.85,.4,.5,.85];size=[.2,.2]
        area=[area[0]+size[0]/2,area[1]+size[1]/2,area[2],
            area[3]-size[0]/2,area[4]-size[1]/2,area[5]]
        fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),
            target_state_list=[deepcopy(initial)],bin=Bin(positions[0]),area=area,
            _bool_expert=False,t=0.,step_id=0)
        states=[];predicted=[];flips=[]
        for i in range(len(positions)):
            states.append((fake.t,deepcopy(fake.target_state_list[-1])))
            predicted.append(fake.bin.get_position())
            if i+1<len(positions):
                count=len(fake.target_state_list);ns['step'](fake)
                if len(fake.target_state_list)>count:flips.append(i)
        errors=np.linalg.norm(np.asarray(predicted)-positions,axis=1);all_errors.extend(errors.tolist())
        a.require(np.isfinite(errors).all() and errors.max()<1e-6,
            'original source arithmetic does not reproduce actual complete bin track: '+record['case_id'])
        per_commands=[]
        for command in commands['commands']:
            if command['raw_phase']!='deposit':continue
            i=int(round(command['generated_at']/ledger_dt));t,state=states[i]
            a.require(abs(arrays['times'][i]-command['generated_at'])<1e-8,'unique raw issue frame required')
            predicted_forecast=ns['compute_target_position'](t=t+.5,x0=state['x'],v0=state['v'],a0=state['a'],t0=state['t0'],dt=dt)
            raw=np.asarray(command['raw_waypoint_xyzw'][:3]);opening=command['desired_opening']
            if opening>=.5:
                error=float(np.linalg.norm(raw[:2]-predicted_forecast[:2]))
                a.require(error<1e-6,'original source opening forecast differs from raw committed goal')
                target=open_commands;reference=np.asarray(predicted_forecast)
            else:
                error=float(np.linalg.norm(raw[:2]-positions[i,:2]));target=closed_commands;reference=positions[i]
            item=dict(command_id=command['command_id'],issued_at=command['generated_at'],desired_opening=opening,
                intent_xy=reference[:2].tolist(),raw_waypoint_xy=raw[:2].tolist(),raw_goal_xy_difference_m=error,
                source_forecast_is_actual_future_observation=False)
            per_commands.append(item);target.append(dict(case_id=record['case_id'],**item))
        cases.append(dict(case_id=record['case_id'],parent_source_group=record['entry']['parent_source_group'],
            raw_binding=row['canonical_receipt'],cache=row['cache'],commands=row['command_receipt'],
            frames=len(positions),maximum_actual_track_replay_error_m=float(errors.max()),
            boundary_state_update_frames=flips,all_deposit_commands_preserved=per_commands))
    for name,item in sources.items():a.require(a.file_sha(name)==item['sha256'],'readonly source changed during replay')
    report=dict(schema='r10_original_bin_exact_arithmetic_replay_all100_TRAIN_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        original100_sources=len(cases),all_failed_censored_and_deposit_commands_preserved=True,
        prior_strict_production_manifest=a.pointer(ROOT,a.local(ROOT,manifest_path)),
        AST_functions_executed=['compute_target_position','cross_boundary','PutRubbishInMovingBin.step'],
        AST_no_native_import_or_step=True,source=sources,
        nominal_ledger_dt=.05,original_simFloat_task_dt=float(np.float32(.05)),
        historical_source_binding_claimed=False,source_closure_still_requires_separate_review=True,
        actual_body_frames=len(all_errors),maximum_actual_track_replay_error_m=max(all_errors),
        measured_track_replay_tolerance_m=1e-6,opening_forecast_commands=len(open_commands),
        maximum_opening_forecast_raw_goal_xy_error_m=max([x['raw_goal_xy_difference_m'] for x in open_commands],default=0.),
        closing_current_reference_commands=len(closed_commands),
        closing_raw_goal_has_original_waypoint_child_offset=True,cases=cases,
        production_rule_or_library_changed=False,runtime_forecast_accuracy_or_closed_loop_claimed=False,
        native_GPU_or_collection_started=False,verified_files_sha256=files)
    write(output/'report.json',report)
    return dict(report=a.pointer(ROOT,output/'report.json'),actual_frames=len(all_errors),
        maximum_track_error=max(all_errors),opening_commands=len(open_commands))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(replay(args.manifest,args.out),ensure_ascii=False))
