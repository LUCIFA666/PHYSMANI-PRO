"""Strict complete100 periodic-teacher audit, with all outcomes preserved."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import periodic_acquisition
from our_mani.multitask.evaluation import atomic_json,file_sha


def main():
    base=ROOT/'artifacts/framework/multitask_r10'
    collection=base/'TRAIN_high_cup_periodic10_all100_candidate_v2';manifest=collection/'manifest.json'
    out=base/'high_cup_periodic_complete_audit_v114'
    if out.exists():raise FileExistsError('preserve completed or failed audit')
    terminal=ROOT/'runtime/framework/native_resource_owner_v19/jobs/highcup_remaining83/terminal.json'
    status=json.loads(terminal.read_text());assert status['complete'] and status['returncode']==0
    assert not Path('/proc/1093112').exists()
    out.mkdir()
    try:
        proof=periodic_acquisition.audit_manifest(ROOT,manifest,require_complete=True)
        assert proof['audit_passed'] and proof['complete']
        full=out/'strict_raw100_audit.json';atomic_json(full,proof)
        m=json.loads(manifest.read_text());counts=defaultdict(set);commands=Counter();successes=0
        for row in m['case_receipts']:
            path=ROOT/row['path'];assert file_sha(path)==row['sha256'];record=json.loads(path.read_text())
            entry=record['entry'];command=ROOT/entry['teacher_command_receipt'];raw=json.loads(command.read_text())
            successes+=bool(record.get('official_task_success'))
            for event in raw['phase_completion_events']:
                key=(event['phase'],event.get('reference_mode','unknown'))
                counts[key].add(entry['parent_source_group']);commands[key]+=1
        atomic_json(out/'report.json',dict(schema='r10_complete_highcup_periodic100_final_audit_v114',
            created_utc=datetime.now(timezone.utc).isoformat(),complete=True,original_parent_sources=100,
            strict_audit=dict(path=str(full.relative_to(ROOT)),sha256=file_sha(full)),
            manifest=dict(path=str(manifest.relative_to(ROOT)),sha256=file_sha(manifest)),
            source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)),
            phase_raw_completion_support={':'.join(k):dict(original_parents=len(v),completed_events=commands[k]) for k,v in counts.items()},
            raw_case_native_successes=successes,raw_native_success_field_name='official_task_success',
            policy_is_periodic10_TRAIN_augmentation_not_exact_original_author=True,
            all_original_sources_and_censored_attempts_retained=True,context_primary_hold_view_unmodified=True,
            interaction_fitter_not_yet_run=True,native_started=False,GPU_started=False,formal_release=False))
        print(json.dumps(dict(complete=True,original_parents=100,raw_phase_support={':'.join(k):len(v) for k,v in counts.items()},
            raw_native_successes=successes,formal_release=False)),flush=True)
    except BaseException as e:
        atomic_json(out/'audit_failed.json',dict(type=type(e).__name__,message=str(e),
            original_collection_not_modified=True,new_native_started=False));raise


if __name__=='__main__':main()
