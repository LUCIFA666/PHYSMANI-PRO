"""Fit the actual author wp0 command offset from every original TRAIN parent."""
from pathlib import Path
from collections import defaultdict
import ast
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.specs import TASK_SPECS
BASE=ROOT/'artifacts/framework/multitask_r10'


def main():
    out=BASE/'bin_author_approach_v427'
    if out.exists():raise FileExistsError('preserve previously fitted model')
    manifest_path=BASE/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/TRAIN_fit_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['fit_split']!='train' or manifest['VAL_used'] or manifest['TEST_used']:raise ValueError('TRAIN only')
    lock=json.loads((BASE/'registered_tasks_v408_setup/frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,lock)
    files={str(manifest_path):file_sha(manifest_path),str(Path(__file__)):file_sha(__file__)}
    for task in ('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed'):
        config=json.loads((BASE/'registered_tasks_v408_setup'/
            ('put_rubbish_in_moving_bin_high_speed_configuration.json')).read_text())
        native=Path(config['baseline_repo'])/'third_party/RLBench/rlbench/tasks'/(task+'.py')
        if file_sha(native)!=TASK_SPECS[task].native_source_sha256:raise ValueError('native author source changed')
        tree=ast.parse(native.read_text());expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
        values=[n.value.value for n in ast.walk(expert) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='th_wp0' for t in n.targets) and isinstance(n.value,ast.Constant)]
        if values!=[.1]:raise ValueError('original author approach threshold changed')
        files[str(native)]=file_sha(native)
        rows=[r for r in manifest['entries'] if r['task']==task and r['source_kind']=='TRAIN_native_generated_augmentation']
        parents=defaultdict(list);all_parents={r['parent_source_group'] for r in rows}
        if len(all_parents)!=100:raise ValueError('all100 original TRAIN parents required')
        for row in rows:
            path=ROOT/row['teacher_command_receipt'];sha=file_sha(path)
            if lock.get(str(path))!=sha:raise ValueError('original locked command ledger changed')
            files[str(path)]=sha;ledger=json.loads(path.read_text())
            if ledger['version']!='r10_TRAIN_committed_waypoint_commands_v1':raise ValueError('exact author command ledger required')
            for command in ledger['commands']:
                if command['raw_stage']!='wp0':continue
                if (command['raw_phase']!='approach' or command['reference_body']!='rubbish'
                    or command['reference_kind']!='translation_only' or command['issued_reference_motion_mode']!='external'):
                    raise ValueError('source author approach frame differs')
                point=np.asarray(command['raw_waypoint_xyzw'],float)[:3]
                body=np.asarray(command['issued_reference_native_pose_xyzw'],float)[:3]
                parents[row['parent_source_group']].append(point-body)
        if set(parents)!=all_parents:raise ValueError('do not omit missing or failed TRAIN sources')
        parent_values=np.asarray([np.mean(parents[k],axis=0) for k in sorted(parents)])
        mean=parent_values.mean(axis=0);loo=(parent_values.sum(axis=0)-parent_values)/(len(parent_values)-1)
        errors=np.linalg.norm(parent_values-loo,axis=1)
        if np.max(errors)>.03+1e-8:raise ValueError('original3cm grouped target fit cap failed')
        model=dict(schema='r10_TRAIN_author_bin_approach_command_offset_v1',task=task,
            reference_role='rubbish',reference_kind='translation_only',target_offset_world=mean.tolist(),
            author_approach_distance_m=.1,author_native_source=dict(path=str(native),sha256=file_sha(native)),
            original_TRAIN_parents=sorted(all_parents),parent_count=100,all_parents_used_equally=True,
            raw_author_command_supervision=True,completed_TCP_not_used_as_waypoint=True,
            native_success_or_completion_used_for_source_selection=False,
            leave_one_parent_out_maximum_error_m=float(np.max(errors)),original_fit_cap_m=.03,
            TRAIN_manifest=dict(path=str(manifest_path.relative_to(ROOT)),sha256=file_sha(manifest_path)),
            privileged_reference_labels_are_offline_TRAIN_only=True,runtime_inputs='current public RGBD body pose only',
            original_physical_grasp_collision_motion_clock_and_native_success_unchanged=True,
            source_sha256=file_sha(__file__),VAL_used=False,TEST_used=False,formal_VAL_TEST=False)
        atomic_json(out/task/'model.json',model)
        print(json.dumps(dict(task=task,parents=100,offset=mean.tolist(),crossfit_max_error_m=float(np.max(errors)))),flush=True)
    atomic_json(out/'inputs.json',dict(files=files,TRAIN_only=True,VAL_used=False,TEST_used=False))


if __name__=='__main__':main()
