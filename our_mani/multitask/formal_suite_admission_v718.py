"""Versioned full16 admission with explicit optional colour-model semantics.

All v532 admission and source-read guards are retained. A colour entry that is
absent or null means that the task has no colour model. Geometry, interaction
and rigid layout remain required, and every present model is still checked
against the accepted TRAIN model and its frozen file hash. No evidence or
frozen configuration is rewritten to repair a representation mismatch.
"""
from pathlib import Path
from .evaluation import file_sha
from .evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from .three_method_release_v531 import policy_fingerprint,require_test_release
from .common_rml_protocol_v530 import fingerprint,require_attestation
from .formal_suite_admission_v532 import locked_pointer,locked_json


def _normalise_optional_colour(artifacts):
 if not isinstance(artifacts,dict):raise ValueError('operation model mapping required')
 if any(artifacts.get(key) is None for key in ('geometry','interaction','rigid_layout')):
  raise ValueError('geometry, interaction and rigid_layout frozen models remain required')
 normalised=dict(artifacts)
 if normalised.get('colour') is None:normalised.pop('colour',None)
 return normalised


def admit(root,suite,files):
 root=Path(root).resolve()
 if suite.get('methods')!=list(METHODS) or suite.get('tasks')!=list(TASKS):raise ValueError('full16 and exactly three authorized methods')
 routes=locked_json(root,files,suite['routes']);policy=policy_fingerprint(routes)
 readiness=locked_json(root,files,suite['readiness'])
 if (readiness.get('schema')!='r10_exact16_fixed_four_common_author_RML_readiness_v532'
  or readiness.get('policy_sha256')!=policy or readiness.get('all16_fixed_four_runtime_ready') is not True
  or readiness.get('completed')!=192 or readiness.get('method_case_count')!=192
  or readiness.get('source_count')!=64 or readiness.get('task_ready_count')!=16
  or readiness.get('motion_is_diagnostic_only') is not True
  or readiness.get('common_RML_profile_sha256')!=fingerprint(routes['common_author_RML_profile'])
  or readiness.get('TEST_read') is not False):
  raise ValueError('complete unchanged full16x4x3 TRAIN runtime/common-RML audit required')
 auditor=locked_pointer(root,files,suite['readiness_auditor'])
 if readiness['source_sha256']!=file_sha(auditor):raise ValueError('readiness audit executable changed')
 training=locked_json(root,files,suite['training_acceptance']);trained=training.get('training',{})
 if (training.get('schema')!='r10_joint16_completed_training_independent_acceptance_v1'
  or training.get('training_complete') is not True or trained.get('TRAIN_original_parents')!=1600
  or trained.get('completed_epochs')!=30 or trained.get('exact_roster_passed') is not True
  or trained.get('tasks')!=list(TASKS) or trained.get('buzz_parents_with_conditional_rotation')!=200
  or training.get('TEST_read') is not False or training.get('Value_NGFF_FASTER_opened_or_started') is not False
  or training['checkpoint']['sha256']!=routes['shared_context_checkpoint_sha256']):
  raise ValueError('independent completed shared1600-parent training acceptance required')
 locked_pointer(root,files,training['checkpoint'])
 # Source identity is checked against TRAIN only. Formal manifests are not read.
 roster=locked_json(root,files,suite['TRAIN_manifest']);sources=validate_roster(roster,'train')
 fixed={(t,r['raw_episode_id']) for t in TASKS for r in [x for x in sources if x['task']==t][:4]}
 expected={(t,r,m) for t,r in fixed for m in METHODS}
 rows=readiness.get('cases',[]);keys=[(r['task'],r['raw'],r['method']) for r in rows]
 if len(keys)!=192 or len(set(keys))!=192 or set(keys)!=expected:raise ValueError('exact original fixed four source/method grid required')
 groups=readiness.get('groups',[])
 if len(groups)!=48 or {(g['task'],g['method']) for g in groups}!={(t,m) for t in TASKS for m in METHODS}:
  raise ValueError('all48 executor method routes need fixed-four evidence')
 for row in rows:
  if row.get('status')!='completed' or type(row.get('native_success')) is not bool or type(row.get('motion_pass')) is not bool:
   raise ValueError('all fixed-source runs need clean explicit native and measured outcomes')
  require_attestation(row.get('common_author_RML_parameters',{}),fingerprint(routes['common_author_RML_profile']))
  locked_pointer(root,files,row['case'])
 for r in routes['routes']:
  for key in ('configuration','frozen_inputs','worker'):
   locked_pointer(root,files,r[key])
  locked_pointer(root,files,{k:r['consumer'][k] for k in ('path','sha256')})
  if r['method']!='physmani':
   config=locked_json(root,files,r['configuration'])
   if config.get('context_checkpoint')!=training['checkpoint']:raise ValueError('all task consumers must share accepted joint16 weights')
   trained_artifacts=next(x for x in training['operation_artifacts'] if x['task']==r['task'])
   if not trained_artifacts['block_uses_original_geometry_kernel']:
    current={k:v for k,v in config['learned_artifacts'][r['task']].items() if k in ('geometry','interaction','rigid_layout','colour')}
    current=_normalise_optional_colour(current)
    accepted=_normalise_optional_colour(trained_artifacts['existing_TRAIN_operation_artifacts'])
    if current!=accepted:raise ValueError('operation model changed after independent training acceptance')
    for pointer in current.values():locked_pointer(root,files,pointer)
 return dict(schema='r10_full16_training_fixed_four_common_author_RML_admission_v532',
  complete=True,policy_sha256=policy,routes=routes,training_checkpoint=training['checkpoint'],
  TRAIN_original_parents=1600,fixed_TRAIN_source_cases=64,fixed_TRAIN_method_cases=192,
  native_success_is_not_an_admission_threshold=True,baseline_algorithm_unchanged=True,
  common_RML_profile_sha256=fingerprint(routes['common_author_RML_profile']),motion_is_diagnostic_only=True,
  formal_VAL_permitted=True,formal_TEST_permitted=False,TEST_read=False)


def read_formal_source(root,suite,files,split,*,validation_release=None):
 if split not in ('validation','test'):raise ValueError('formal VAL/TEST only')
 proof=admit(root,suite,files)
 if split=='test':require_test_release(validation_release or {},proof['routes'])
 # This is the first and only formal-source read in this admission interface.
 manifest=locked_json(root,files,suite['source_manifests'][split])
 rows=validate_roster(manifest,split,completed_VAL_audit=split=='test')
 return manifest,rows,proof
