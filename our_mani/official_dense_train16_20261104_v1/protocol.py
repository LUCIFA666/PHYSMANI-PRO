"""Fixed16 generated TRAIN attempt metadata, no HDF payloads or Native imports."""
from pathlib import Path
import ast
import json
import hashlib
from ..multitask_execution_pipeline_20261104_v1.registry import TASKS,NATIVE_REPO
from . import VERSION

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'artifacts/framework/local_mpc_20261102/current_network16_data_manifest_v1/manifest.json'
MANIFEST_SHA='c03a4fb31140a5aab7a49ad5d771955a56e5adfb6facfe31aa6b269769b9559b'
NOTES=ROOT/'artifacts/framework/local_mpc_20261102/official_dense_TRAIN16_trueclock_feasibility_v1'
NOTES_SHA={'assignment.json':'4183306571bcad69b409a695bde2c33f426b05793f2273e2d12e27614c326bbb',
    'report.json':'72f1ad3f9a166e1b5e7def4590f112c5575e9511409a2590a9379101fb4d75c2',
    'planning_gap_addendum.json':'25364d37b1824ea5c0b1a5ef743dea7b23f93f5d44f33cdabcc95c42c4cfbdf5'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pointer(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def source_map():
    paths=[Path(__file__).with_name(p)for p in ('__init__.py','protocol.py','recording.py','native_hooks.py','collector.py')]
    paths+=[ROOT/'scripts/prepare_official_dense_TRAIN16_20261104_v1.py',
        ROOT/'our_mani/multitask_execution_pipeline_20261104_v1/registry.py',
        ROOT/'our_mani/generic_reference_mpc_20261104_v1/public_observation.py',
        ROOT/'our_mani/generic_reference_mpc_20261104_v1/train_labels.py',
        ROOT/'our_mani/multitask/specs.py',ROOT/'our_mani/multitask/native_tasks.py']
    paths+=[NATIVE_REPO/p for p in ('packages/physmani_bench/physmani_bench/collect.py',
        'packages/physmani_bench/physmani_bench/env.py','packages/physmani_bench/physmani_bench/rlbench_env.py',
        'third_party/RLBench/rlbench/task_environment.py','third_party/RLBench/rlbench/backend/scene.py',
        'third_party/RLBench/rlbench/backend/task.py','third_party/PyRep/pyrep/pyrep.py',
        'third_party/PyRep/pyrep/robots/robot_component.py')]
    paths+=[NATIVE_REPO/'third_party/RLBench/rlbench/tasks'/(task+'.py')for task in TASKS]
    return {str(p.resolve()):sha(p)for p in paths}


def prepare():
    if any(sha(NOTES/name)!=checksum for name,checksum in NOTES_SHA.items()):
        raise ValueError('Frozen source metadata feasibility/assignment changed')
    if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('Original frozen source-parent metadata manifest changed')
    manifest=json.loads(MANIFEST.read_text());rows=[]
    for ordinal,task in enumerate(TASKS):
        eligible=[r for r in manifest['episodes']if r['task']==task and r['split']=='train'
                  and r['role']=='fit'and r['offline_eligible']]
        # Existing frozen metadata assignment fixes numeric-first source parents.
        selected=next(r for r in json.loads((NOTES/'assignment.json').read_text())['parents']if r['task']==task)
        matches=[r for r in eligible if r['source_parent_group']==selected['source_parent_group']]
        if len(matches)!=1:raise ValueError('One eligible TRAIN source parent required')
        path=NATIVE_REPO/'third_party/RLBench/rlbench/tasks'/(task+'.py')
        tree=ast.parse(path.read_text());classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef)
            and any(isinstance(b,ast.Name)and b.id=='Task'for b in n.bases)]
        if len(classes)!=1:raise ValueError('One actual task class required')
        seed=261104800+ordinal
        rows.append(dict(id='official_dense_TRAIN_'+task+'_seed'+str(seed),task=task,seed=seed,variation=0,
            native_class=classes[0],native_module='rlbench.tasks.'+task,native_source=pointer(path),
            source_parent_group=matches[0]['source_parent_group'],source_metadata_only=True,
            original_HDF_scene_replay=False,source_HDF_payload_not_read=True,
            all_private_label_roles=selected['all_private_label_roles'],
            maximum_command_calls=40,maximum_physics_steps=400))
    sources=source_map()
    return dict(schema='official_dense_TRAIN16_one_attempt_preparation_20261104_v1',version=VERSION,
        preparation_only=True,execution_authorized=False,Native_started=False,GPU=False,TEST_reads=0,HDF_payload_reads=0,
        source_parent_manifest=pointer(MANIFEST),prior_feasibility=pointer(NOTES/'report.json'),
        planning_gap=pointer(NOTES/'planning_gap_addendum.json'),source_metadata_assignment=pointer(NOTES/'assignment.json'),
        source_sha256=sources,attempts=rows,attempt_count=16,attempts_per_task=1,
        reset_protocol='new_generated_TRAIN_seed_after_Root_explicit_launch; never archived_HDF_scene_replay',
        teacher='original privileged official ExpertActioner command intent, TRAIN teacher only, never model input',
        public_inputs='four actual current RGBD/K/T/clips plus q7/TCP7; gripper scalar zero/unobserved',
        private_labels='all body-role identities/poses; actual robot/fingers/qtargets; expert command/stage; Native grasped handles labels only',
        physics_time='actual simulator before/after; task.t separately observed, gripper task-clock pauses retained',
        future_resource_owner_locks=[str(ROOT/'runtime/framework'/name)for name in
            ('fixed_TRAIN104_geometry_native.lock','TRAIN_postcomplete_diagnostics_v1106.lock')],
        source_only_entry_owns_no_resource_locks=True,
        missing_commands=dict(vref=None,aref=None,h=None,mask=False),contact_force=None,
        first_terminal_failure_retained=True,failed_or_partial_attempts_retained=True,retry_or_success_only_filter=False,
        official_lowlevel_planning_gap_supported=False,
        blockers=['get_action_traj direct simExtStep/temporary kinematic teleports require separate planning-event segmentation; current observer faults rather than silently train on them',
                  'Native launch/reset/camera setup/Root resource owner deliberately absent; attach-only prototype, not execution admission'],
        end_to_end_current_network_ready=False,physical_motion_training_ready=False)
