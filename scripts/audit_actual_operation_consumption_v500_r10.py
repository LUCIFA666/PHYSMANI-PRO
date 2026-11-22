"""CPU-only actual library loaders and fixed first TRAIN native consumption."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.executor_suite16_v467 import route,CHECKPOINT_SHA
from our_mani.multitask.evaluation_roster16_v283 import TASKS,validate_roster
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.evaluation import atomic_json,file_sha,digest
BASE=ROOT/'artifacts/framework/multitask_r10'

def main():
 out=BASE/'actual_operation_consumption_v500'
 if out.exists():raise FileExistsError('preserve audit')
 roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text());sources=validate_roster(roster,'train')
 rows=[];issues=[]
 for task in TASKS:
  routing=route(task,'context_geometry');config_path=BASE/routing.configuration(task);config=json.loads(config_path.read_text())
  if config['context_checkpoint']['sha256']!=CHECKPOINT_SHA:raise ValueError('all tasks share actual157 dynamics')
  row=dict(task=task,configuration=dict(path=str(config_path.relative_to(ROOT)),sha256=file_sha(config_path)),
   context_checkpoint=config['context_checkpoint'],formal_execution_admission_granted=False)
  if task in BLOCKS:
   row.update(block_original_geometry_kernel=True,interaction_model=None,canonical_library_consumption_not_applicable=True)
  else:
   pointer=config['learned_artifacts'][task]['interaction'];path=ROOT/pointer['path']
   if file_sha(path)!=pointer['sha256']:raise ValueError('declared trained model changed')
   loader=PrimarySequenceLibrary if task=='place_cups_on_rotating_frame' else InteractionLibrary
   model=loader.load(path);metadata=model.metadata
   if metadata.get('training_complete') is not True or metadata.get('TEST_used_for_fitting') is not False or metadata.get('fit_split')!='train':
    raise ValueError('completed TRAIN-only library required')
   first=next(r for r in sources if r['task']==task);proofs=[]
   acceptance=json.loads((BASE/routing.setup/'CPU_acceptance.json').read_text())
   for candidate in acceptance['actual_argv_preflights']:
    if candidate['task']!=task or candidate['raw']!=first['raw_episode_id']:continue
    argv=candidate['argv'];p=ROOT/argv[argv.index('--out')+1]/'case.json'
    if not p.exists():issues.append(dict(task=task,method=candidate['method'],reason='fixed first native case missing'));continue
    case=json.loads(p.read_text())
    if case['split']!='train' or case['identity']['source_row_sha256']!=digest(first):raise ValueError('fixed original TRAIN source only')
    a=case['actor_metrics'];observed=a['interaction_library_sha256']
    if observed!=model.sha256:raise ValueError('actual native actor library differs from declared CPU loader: '+task)
    proofs.append(dict(method=candidate['method'],actual_consumed_library_canonical_sha256=observed,
     case=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))))
   if {r['method'] for r in proofs}!={'cv_geometry','context_geometry'}:issues.append(dict(task=task,reason='two fixed first-source consumption witnesses required'))
   row.update(block_original_geometry_kernel=False,interaction_model=pointer,
    loader='normal_cup_primary_library_v109.PrimarySequenceLibrary' if loader is PrimarySequenceLibrary else 'interaction.InteractionLibrary',
    loaded_canonical_sha256=model.sha256,loaded_phase_order=list(model.phase_order),native_first_source_consumption=proofs,
    original_physical_and_fitting_caps_unchanged=True,
    declared_model_path_equals_consumed_model_path=True,normal_cup_custom_loader_is_not_a_pointer_replacement=loader is PrimarySequenceLibrary)
  rows.append(row)
 report=dict(schema='r10_actual16_operation_library_consumption_TRAIN_audit_v500',created_utc=datetime.now(timezone.utc).isoformat(),
  complete=not issues,tasks=list(TASKS),task_count=len(rows),operations=rows,issues=issues,
  actual_context_checkpoint_sha256=CHECKPOINT_SHA,TEST_read=False,Value_NGFF_FASTER_loaded=False,
  formal_native_execution_admission_granted=False,native_outcomes_not_used_to_select_sources=True,
  previous438_model_pointer_mismatch_not_established=True,source_sha256=file_sha(__file__))
 atomic_json(out/'report.json',report)
 print(json.dumps(dict(complete=not issues,tasks=len(rows),issues=issues)))
if __name__=='__main__':main()
