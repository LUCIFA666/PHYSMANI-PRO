"""TRAIN-development author targets through the common template interface.

The distinct schema does not pass legacy completed-skill factory admission.

This model supervises what the author commanded. Its press state remains open
until the native benchmark terminates; neither command arrival nor its fitted
position tolerance grants a physical press-completion capability.
"""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .interaction import PhaseTemplate,COMMAND_GOAL,InteractionLibrary
from .evaluation import file_sha,digest
from .specs import get_task_spec

SCHEMA='r10_TRAIN_development_button_author_command_targets_v71'
TASKS=('push_moving_button','push_moving_button_high_speed')


class ButtonIntentLibrary(InteractionLibrary):
    def __init__(self,root,path,*,purpose):
        if purpose!='train_development':raise ValueError('v71 author command candidate cannot enter formal VAL/TEST')
        root=Path(root).resolve();path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('source model must remain in project')
        model=json.loads(path.read_text());task=model['task']
        if (model['schema']!='r10_TRAIN_original_author_button_command_intent_model_v70' or task not in TASKS
            or model['original_parent_count']!=100 or not model['all_failed_censored_original_parents_preserved']
            or model['native_success_labels_used_for_fit'] or model['public_press_deformation_verifier_granted']
            or model['original_completed_skill_library_impersonated']):raise ValueError('exact independently sourced action model required')
        report_path=path.parent.parent/'report.json';report=json.loads(report_path.read_text())
        if report['tasks'][task]['model']!={'path':str(path.relative_to(root)),'sha256':file_sha(path)}:raise ValueError('command model/report source binding differs')
        self.model=model;self.task_spec=get_task_spec(task);self.phase_order=('approach','press');self.templates=[]
        rule=digest(dict(model=report['tasks'][task]['model'],source=dict(path=str(Path(__file__).relative_to(root)),sha256=file_sha(__file__))))
        for phase in self.phase_order:
            row=model['phases'][phase]
            if (row['parents']<8 or row['minimum_independent_command_parent_support']!=8
                or not row['original_point03_point15_error_caps_passed']
                or row['position_tolerance_m']>.03 or row['orientation_tolerance_rad']>.15
                or row['command_completion_claimed'] or row['press_or_task_completion_capability_granted']):raise ValueError('independent author command support/error gates failed')
            p=np.asarray(row['position']);r=Rotation.from_quat(row['rotation_xyzw']).as_matrix();xyzw=np.r_[p,row['rotation_xyzw']]
            self.templates.append(PhaseTemplate(phase,f'{task}:{phase}:author_command_intent_v71','button','translation_only','button',0.,
                np.array([0.,1.]),np.tile(p,(2,1)),np.tile(r,(2,1,1)),np.full(2,row['desired_opening']),.5,.5,
                row['position_tolerance_m'],row['orientation_tolerance_rad'],'external',COMMAND_GOAL,rule,xyzw))
        self.templates=tuple(self.templates)
        proof=json.loads((path.parent.parent/'raw_command_proofs.json').read_text());parents=sorted(r['parent'] for r in proof if r['task']==task)
        if len(parents)!=100 or len(set(parents))!=100:raise ValueError('all original parent identities required')
        self.metadata=dict(schema=SCHEMA,task_name=task,synthetic=False,training_complete=True,fit_split='train',selection_split='train',
            TRAIN_source_ids=parents,TRAIN_parent_source_groups=parents,TRAIN_source_manifest_path=str(report_path),TRAIN_source_manifest_sha256=file_sha(report_path),
            source_model=report['tasks'][task]['model'],supervision='original_author_command_goals',
            independent_command_parent_support={p:model['phases'][p]['parents'] for p in self.phase_order},
            completed_skill_support_claimed=False,public_press_completion_granted=False,formal_release=False,development_only=True)
        self.sha256=digest(dict(metadata=self.metadata,rule=rule));self.payload=dict(metadata=deepcopy(self.metadata))

    def for_phase(self,phase):return tuple(t for t in self.templates if t.phase==phase)
