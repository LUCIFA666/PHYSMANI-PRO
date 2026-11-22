"""Read-only R10 formal model coverage gate; no native/fit/TEST work."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.release_preflight import audit_formal_model_coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--context-training-view',required=True)
    parser.add_argument('--train-source-manifest',default='artifacts/framework/protocol/train_manifest.json')
    parser.add_argument('--validation-source-manifest',default='artifacts/framework/protocol/validation_manifest.json')
    parser.add_argument('--frozen-inputs')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();config_path=(ROOT/args.config).resolve();view_path=(ROOT/args.context_training_view).resolve()
    out=(ROOT/args.out).resolve()
    if not all(path.is_relative_to(ROOT) for path in (out,config_path,view_path)):
        raise ValueError('coverage report/config/view must stay in desktop project')
    config_sha=file_sha(config_path);config=json.loads(config_path.read_text())
    files=json.loads((ROOT/args.frozen_inputs).read_text())['files'] if args.frozen_inputs else None
    report=audit_formal_model_coverage(ROOT,config,
        context_training_view=dict(path=str(view_path),sha256=file_sha(view_path)),
        train_source_manifest=args.train_source_manifest,validation_source_manifest=args.validation_source_manifest,
        frozen_files=files)
    report['config_file']=dict(path=str(config_path.relative_to(ROOT)),sha256=config_sha)
    report['input_files_sha256'][str(config_path.relative_to(ROOT))]=config_sha
    if file_sha(config_path)!=config_sha:
        report['errors'].append(dict(component='stable_config',error='config changed during model coverage audit'))
        report['formal_model_coverage_passed']=False
    if files is not None and not any((ROOT/name).resolve()==config_path and checksum==config_sha for name,checksum in files.items()):
        report['errors'].append(dict(component='config_lock',error='config file missing from formal input lock'))
        report['formal_model_coverage_passed']=False
    atomic_json(out,report)
    print(json.dumps(dict(path=str(out.relative_to(ROOT)),formal_model_coverage_passed=report['formal_model_coverage_passed'],
        errors=len(report['errors']),actual_TEST_read=False,model_fitting_performed=False)))
    if not report['formal_model_coverage_passed']:raise SystemExit(2)


if __name__=='__main__':main()
