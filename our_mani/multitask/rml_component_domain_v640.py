"""Source-bound TRAIN qualification and genuine evaluation component purposes.

A frozen completed full16 admission is the only way to obtain an evaluation
capability. No VAL/TEST caller is relabelled as TRAIN to enter old loaders.
"""
from dataclasses import dataclass
from pathlib import Path
import json
from .evaluation import file_sha
from .evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from .frozen_subset_batch_v132 import checked_subset

_TOKEN=object()


@dataclass(frozen=True)
class ComponentDomain:
    purpose:str
    task:str
    method:str
    split:str
    raw:str
    policy_sha256:str|None
    _token:object

    def require(self,task,method=None):
        if self._token is not _TOKEN or self.task!=task or self.method not in METHODS or method is not None and self.method!=method:
            raise ValueError('source-bound case component capability required')
        if self.purpose not in ('train_qualification','evaluation'):
            raise ValueError('explicit qualification or evaluation component purpose required')
        if self.purpose=='evaluation' and (self.split not in ('validation','test') or not self.policy_sha256):
            raise ValueError('actual formal admission and split required')
        if self.purpose=='train_qualification' and self.split!='train':raise ValueError('TRAIN qualification only')
        return self.purpose

    def receipt(self):
        self.require(self.task)
        return dict(schema='r10_source_bound_RML_component_purpose_v640',purpose=self.purpose,task=self.task,
            method=self.method,split=self.split,raw_episode_id=self.raw,policy_sha256=self.policy_sha256,
            VAL_TEST_relabelled_as_TRAIN=False)


def for_train(root,files,task,raw,method):
    if task not in TASKS or method not in METHODS:raise ValueError('full16 and exactly three methods')
    path=Path(root)/'artifacts/framework/protocol/train_manifest.json'
    checked_subset(root,files,{str(path):file_sha(path)})
    rows=validate_roster(json.loads(path.read_text()),'train')
    fixed=[r['raw_episode_id'] for r in rows if r['task']==task][:4]
    if raw not in fixed:raise ValueError('original fixed four TRAIN sources only')
    return ComponentDomain('train_qualification',task,method,'train',raw,None,_TOKEN)


def for_evaluation(root,suite,files,split,task,raw,method,*,validation_release=None):
    if task not in TASKS or method not in METHODS:raise ValueError('full16 and exactly three methods')
    from .formal_suite_admission_v532 import read_formal_source
    manifest,rows,proof=read_formal_source(root,suite,files,split,validation_release=validation_release)
    selected=[r for r in rows if r['task']==task and r['raw_episode_id']==raw]
    if len(selected)!=1:raise ValueError('one exact admitted formal source required')
    return ComponentDomain('evaluation',task,method,split,raw,proof['policy_sha256'],_TOKEN),selected[0],proof
