"""Check all old remove teacher goals against new TRAIN hierarchy evidence.

Reconstruct source-declared goals, not measured successful endpoints. This
tests historical raw commands directly instead of assuming old TTM identity.
No label, event, outcome or model is changed by this audit.
"""
from pathlib import Path
import json
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs


def compose(a,b):return Pose(a.position+a.rotation@b.position,a.rotation@b.rotation)


def main():
    base=ROOT/'artifacts/framework/multitask_r10';out=base/'remove_original_goal_hierarchy_audit_v45'
    if out.exists():raise FileExistsError('preserve previous TRAIN audit')
    old_path=base/'remove_reference_independent_actual200_CPU_candidate_v2/report.json';old=json.loads(old_path.read_text())
    # The old numeric study predates current executable revisions. Audit its
    # immutable raw evidence, and disclose old code differences explicitly;
    # this script neither reuses its model nor claims a current model replay.
    historical_code={name:sha for name,sha in old['verified_files_sha256'].items()
        if Path(name).suffix in ('.py','.sh') and '/artifacts/' not in name}
    inputs={name:sha for name,sha in old['verified_files_sha256'].items() if name not in historical_code}
    verify_inputs(ROOT,inputs)
    prefit_path=ROOT/old['prefit_proof']['path']
    if file_sha(prefit_path)!=old['prefit_proof']['sha256']:raise ValueError('old raw proof changed')
    prefit=json.loads(prefit_path.read_text());inputs[str(prefit_path)]=file_sha(prefit_path)
    rows=[];inputs[str(old_path)]=file_sha(old_path)
    changed_code={name:dict(historical_sha256=sha,current_sha256=file_sha(name)) for name,sha in historical_code.items() if file_sha(name)!=sha}
    tasks=('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed');hierarchies={}
    for task in tasks:
        paths=sorted((base/'remove_hierarchy_TRAIN_v35'/task).glob('*/hierarchy.json'))
        if len(paths)!=4:raise ValueError('all fixed first4 hierarchy probes required')
        hs=[json.loads(p.read_text()) for p in paths]
        if not all(h['complete'] and h['clock_unchanged'] and not h['policy_used'] and h['counted_actions']==0 for h in hs):raise ValueError('TRAIN hierarchy probe did not complete')
        for p in paths:inputs[str(p)]=file_sha(p)
        for h in hs:
            for wp in ('waypoint0','waypoint1','waypoint2','waypoint3'):
                names=[r['name'] for r in h['waypoints'][wp]['parent_chain']]
                if 'place_cups_holder_base' not in names:raise ValueError('source rotating helper cannot move a waypoint outside the frame hierarchy')
        # Source wp0 inherits wp1, source wp3 inherits wp2; all transforms are
        # fixed TRAIN-local labels, kept wholly outside online policy inputs.
        reference=hs[0]['waypoints'];locals={wp:Pose.from_xyzw(reference[wp]['cup_local_xyzw']['mug0'] if wp in ('waypoint0','waypoint1')
            else reference[wp]['spoke_local_xyzw']['place_cups_holder_spoke0']) for wp in ('waypoint0','waypoint1','waypoint2','waypoint3')}
        hierarchies[task]=locals
    for proof in prefit['case_proofs']:
        raw=proof['original_raw_bindings'];record=json.loads((ROOT/raw['canonical_receipt']['path']).read_text())
        commands=json.loads((ROOT/raw['commands']['path']).read_text())['commands'];ledger={c['command_id']:c for c in commands}
        with np.load(ROOT/raw['cache']['path'],allow_pickle=False) as z:
            ids=list(z['body_ids']);times=z['times'];task=record['task'];initial=record['native_reset_receipt']['native_initial_snapshot']
            yaw=float(np.asarray(initial['active_motion_schedule']['yaw_speed']).reshape(-1)[0])
            for c in commands:
                stage=c['raw_stage']
                if stage not in ('wp0','wp1','wp2','wp3'):continue
                i=int(np.argmin(abs(times-c['generated_at'])))
                if abs(times[i]-c['generated_at'])>1e-8:raise ValueError('original issue frame is missing')
                cup=c['manipulated_body'];ci=ids.index(cup)
                spoke=min((r for r in ids if r.startswith('spoke')),key=lambda r:np.linalg.norm(z['target_positions'][i,ids.index(r)]-z['target_positions'][i,ci]))
                role=cup if stage in ('wp0','wp1') else spoke;j=ids.index(role)
                body=Pose(z['target_positions'][i,j],z['target_rotations'][i,j]);expected=compose(body,hierarchies[task]['waypoint'+stage[-1]])
                previous=ledger.get(c['command_id']-1)
                delay=0. if stage in ('wp0','wp2') else (1. if stage=='wp1' and (previous['raw_stage'] if previous else initial['task_internal_fields']['stage'])=='wp0' else .5)
                fi=ids.index('frame');center=z['target_positions'][i,fi];rot=Rotation.from_rotvec(z['target_rotations'][i,fi,:,2]*np.deg2rad(yaw)*delay).as_matrix()
                expected=Pose(center+rot@(expected.position-center),rot@expected.rotation)
                observed=Pose.from_xyzw(c['raw_waypoint_xyzw']);pe=float(np.linalg.norm(expected.position-observed.position));re=float(Rotation.from_matrix(expected.rotation@observed.rotation.T).magnitude())
                rows.append(dict(task=task,parent=record['entry']['parent_raw_episode_id'],command_id=c['command_id'],stage=stage,delay=delay,
                    position_error_m=pe,orientation_error_rad=re,source_reference=role))
    out.mkdir();atomic_json(out/'commands.json',rows)
    summary={}
    for task in tasks:
        summary[task]={stage:dict(commands=len(v),max_position_error_m=max(r['position_error_m'] for r in v),
            max_orientation_error_rad=max(r['orientation_error_rad'] for r in v)) for stage in ('wp0','wp1','wp2','wp3')
            if (v:=[r for r in rows if r['task']==task and r['stage']==stage])}
    report=dict(complete=True,source_hierarchy_report_rows=8,original_parents=200,per_task=summary,
        commands=len(rows),inputs_sha256=inputs,source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
        raw_goals_reconstructed_not_measured_success_endpoints=True,original_TTM_identity_assumed=False,
        source_qualification_passed=all(r['position_error_m']<=1e-4 and r['orientation_error_rad']<=1e-4 for r in rows),
        numeric_audit_bound_m_rad=1e-4,historical_executable_differences=changed_code,
        old_numeric_model_not_readmitted=True,model_fit=False,runtime_admission=False,TEST_read=False)
    atomic_json(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'}),flush=True)


if __name__=='__main__':main()
