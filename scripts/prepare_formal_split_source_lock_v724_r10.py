"""Freeze source bytes only after full16 admission for the actual formal split.

The policy/executable route fingerprint stays unchanged across VAL and TEST.
TEST reset data and its source manifest are selected only after complete VAL.
"""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs,checked_sha
from our_mani.multitask.formal_suite_admission_v718 import read_formal_source,locked_pointer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('suite','suite-lock','split','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--validation-release');args=p.parse_args()
    if args.split not in ('validation','test'):raise ValueError('formal VAL/TEST source locks only')
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('fresh project source lock required')
    global_lock=ROOT/args.suite_lock;files=json.loads(global_lock.read_text())['files'];verify_inputs(ROOT,files)
    suite_path=(ROOT/args.suite).resolve()
    locked_pointer(ROOT,files,dict(path=str(suite_path.relative_to(ROOT)),sha256=file_sha(suite_path)))
    suite=json.loads(suite_path.read_text());release=None
    if args.split=='test':
        if not args.validation_release or (ROOT/args.validation_release).resolve()!=(ROOT/suite['validation_release_path']).resolve():
            raise ValueError('exact completed VAL release required before TEST source selection')
        release=json.loads((ROOT/args.validation_release).read_text())
    manifest,rows,proof=read_formal_source(ROOT,suite,files,args.split,validation_release=release)
    inputs={};source_keys=[]
    for row in rows:
        path=Path(row['path']).resolve();source_keys.append((row['task'],row['raw_episode_id']))
        if args.split=='validation':
            if not path.is_file() or path.name!='data.h5' or 'val_package_compressed' not in path.parts:
                raise ValueError('actual original validation HDF5 required')
            selected=[path]
        else:
            if not path.is_dir() or row.get('mapping_status')!='verified':raise ValueError('verified official TEST reset alias required')
            required=row.get('source_required_files',{})
            if not required:raise ValueError('complete original first-frame/reset hash evidence required')
            selected=[Path(v['path']).resolve() for v in required.values()]
            for relative,item in required.items():
                if file_sha(path/relative)!=item['sha256']:raise ValueError('official reset/first-frame bytes changed')
            for field in ('task_source_path','task_scene_path'):
                if row.get(field):selected.append(Path(row[field]).resolve())
        for source in selected:
            checksum=file_sha(source);checked_sha(source,checksum);inputs[str(source)]=checksum
    if len(rows)!=(315 if args.split=='validation' else 640) or len(set(source_keys))!=len(rows):
        raise ValueError('complete exact full16 split source grid required')
    atomic_json(out,dict(schema='r10_full16_post_admission_split_source_lock_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        split=args.split,source_count=len(rows),policy_sha256=proof['policy_sha256'],
        source_manifest_sha256=suite['source_manifests'][args.split]['sha256'],files=inputs,
        full16_admission_observed_before_source_selection=True,complete_VAL_release_observed=args.split=='test',
        source_content_used_for_policy_fitting=False,TEST_used_for_fitting=False,
        source_lock_preparer_sha256=file_sha(__file__),native_started=False,GPU_started=False))
    print(json.dumps(dict(complete=True,split=args.split,sources=len(rows),files=len(inputs),out=str(out.relative_to(ROOT)),
        native_started=False,GPU_started=False)),flush=True)


if __name__=='__main__':main()
