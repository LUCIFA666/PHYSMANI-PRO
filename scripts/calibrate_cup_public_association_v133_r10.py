"""Fit the phase-free public association radius from every TRAIN acquisition."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.mode_observer import PublicModeObserver
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import checked_sha,verify_inputs

B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'cup_public_association_radius_TRAIN_v133'


def main():
    if OUT.exists():raise FileExistsError('preserve previous calibration')
    OUT.mkdir();files={};entries=[];samples=[];rows=[]
    for name in ('normal_cup_primary_policy_v108/TRAIN_source_manifest.json',
            'high_cup_all300_issue_operation_v125/TRAIN_source_manifest.json'):
        path=B/name;manifest=json.loads(path.read_text());verify_inputs(ROOT,manifest['verified_files_sha256'])
        files.update(manifest['verified_files_sha256']);files[str(path)]=file_sha(path);entries.extend(manifest['entries'])
    assert len(entries)==400
    for entry in entries:
        path=ROOT/entry['cache'];checked_sha(path,entry['cache_sha256']);files[str(path)]=entry['cache_sha256']
        with np.load(path,allow_pickle=False) as z:
            contacts=np.flatnonzero(z['robot_contact']);row=dict(source_id=entry['source_id'],parent=entry['parent_source_group'],
                task=entry['task'],contact_seen=bool(len(contacts)),raw_cache=dict(path=entry['cache'],sha256=entry['cache_sha256']))
            if len(contacts):
                i=int(contacts[0]);assert i>0
                ids=tuple(map(str,z['body_ids']));cups=tuple(n for n in ids if n.startswith('cup'))
                intended=str(z['teacher_manipulated_body'][i]);assert intended in cups
                previous=[SimpleNamespace(object_id=n,visible=bool(z['valid'][i-1,j]),
                    pose=SimpleNamespace(position=z['positions'][i-1,j].copy())) for j,n in enumerate(ids)]
                robot=SimpleNamespace(contact=True,pose=SimpleNamespace(position=z['measured_tcp_xyzw'][i,:3].copy()))
                target=next(s for s in previous if s.object_id==intended)
                assert target.visible
                others=[np.linalg.norm(s.pose.position-robot.pose.position) for s in previous if s.object_id in cups and s.object_id!=intended and s.visible]
                assert others
                distance=float(np.linalg.norm(target.pose.position-robot.pose.position));minimum_other=float(min(others))
                row.update(first_contact_time=float(z['times'][i]),first_contact_phase=str(z['teacher_phase'][i]),intended_body=intended,
                    distance_to_intended=distance,minimum_other_visible_distance=minimum_other,
                    previous_public_target_visible=True,previous_frame_age_seconds=float(z['times'][i]-z['times'][i-1]))
                assert abs(row['previous_frame_age_seconds']-.05)<1e-7
                samples.append((row,ids,cups,previous,robot));rows.append(row)
            else:rows.append(row)
    radius=max(r['distance_to_intended'] for r,_,_,_,_ in samples)+.02
    minimum_other=min(r['minimum_other_visible_distance'] for r,_,_,_,_ in samples)
    assert radius<minimum_other and radius<.15
    old_ambiguous=0
    for row,ids,cups,previous,robot in samples:
        old=PublicModeObserver(ids,cups,.15);_,invalid=old.observe(previous,robot);old_ambiguous+=bool(invalid)
        new=PublicModeObserver(ids,cups,radius);modes,invalid=new.observe(previous,robot)
        assert not invalid and new.held==row['intended_body'] and modes[new.held]=='attached'
        row.update(original_radius_ambiguous=bool(old.held is None),new_radius_unique_intended=True)
    parents={t:len({r['parent'] for r in rows if r['task']==t and r['contact_seen']}) for t in set(r['task'] for r in rows)}
    assert all(n==100 for n in parents.values())
    for path in (Path(__file__),ROOT/'our_mani/multitask/mode_observer.py'):files[str(path.resolve())]=file_sha(path)
    report=dict(schema='r10_phase_free_cup_public_contact_radius_TRAIN_v133',
        total_original_parents=200,total_acquisitions=400,first_public_contact_samples=len(samples),original_parents_with_public_contact=parents,
        original_unvalidated_radius_m=.15,association_radius_m=radius,
        radius_derivation='maximum actual TRAIN first-contact public intended-center distance plus unchanged physical close2cm margin',
        maximum_intended_distance_m=radius-.02,minimum_other_visible_distance_m=minimum_other,
        original_observer_replayed_on_all_first_contact_samples=True,new_radius_unique_intended_samples=len(samples),
        old_radius_ambiguous_samples=old_ambiguous,all400_sources_and_failures_retained=True,
        all100_original_sources_each_task_retained=True,calibration_conditioned_on_task_success=False,
        source_selected_identity_is_TRAIN_intent_not_native_held_ID_label=True,
        native_held_identity_not_read=True,runtime_uses_only_previous_visible_public_poses_and_robot_contact=True,
        existing_phase_free_observer_algorithm_unchanged=True,physical_grasp_gate_not_changed=True,
        v128_overstrict_readonly_summary_superseded=True,
        v128_diagnosis='required all3 cups visible and acquire phase; actual observer uses each visible cup and no phase',
        rows=rows,verified_files_sha256=files,VAL_used=False,TEST_used=False,formal_release=False)
    verify_inputs(ROOT,files);atomic_json(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','verified_files_sha256')}),flush=True)


if __name__=='__main__':main()
