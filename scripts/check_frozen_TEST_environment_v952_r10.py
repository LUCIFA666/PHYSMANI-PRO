"""Read-only CPU recheck of the original frozen OS and admission topology.

This observer starts no model, simulator, admission or TEST job. It verifies the
actual fresh v820 namespace, then runs the unchanged v900 topology constructor
on the original suite lock. It does not read declared model/case/raw contents.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/hybrid_environment_manifest_v820.json'
MANIFEST_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
SUITE_LOCK = ROOT / 'artifacts/framework/multitask_r10/formal_TEST_after_environment_resume_v826_setup/frozen_inputs.json'
SUITE_LOCK_SHA = '522f8937cb72d34e9558bec6efba36bdd542cf09fa7f6fcd93a710677e3a2f23'
PINS = {
    'scripts/run_frozen_system_environment_v820_r10.py': '3c0db5da80f7a9d66400bde1b949d2e4a3333ec666f9fa9e118eba0652e804c3',
    'our_mani/multitask/frozen_lookup_batch_v900.py': '35776d7e6fd2389e8ad363b0de8a7dd86e59b59e9b34eafc9399e2c8664cbf89',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    output = Path(args.report).resolve()
    if not output.is_relative_to(ROOT / 'runtime/framework') or output.exists():
        raise ValueError('new project runtime report required')
    report = dict(schema='r10_fresh_frozen_TEST_environment_CPU_observation_v952',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)),
                  complete=False, Native_started=False, GPU_started=False,
                  admission_or_TEST_started=False, declared_model_case_raw_bytes_read=False,
                  scientific_inputs_or_old_locks_modified=False)
    try:
        for relative, expected in PINS.items():
            if sha(ROOT / relative) != expected:
                raise ValueError('original diagnostic dependency changed: ' + relative)
        if sha(SUITE_LOCK) != SUITE_LOCK_SHA:
            raise ValueError('original TEST suite lock changed')
        spec = importlib.util.spec_from_file_location('frozen_environment_v820_observer952', ROOT / 'scripts/run_frozen_system_environment_v820_r10.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['actual_environment'] = module.capture_namespace_evidence(
            MANIFEST, MANIFEST_SHA,
            ROOT / 'artifacts/framework/multitask_r10/frozen_environment_restore_v820_setup/cpu_namespace_proof_v820.json',
            '1358fb0611b9677b4cc1487aa2593a89d81c5651252489ac09e340e4cc4e3bfa')
        from our_mani.multitask.frozen_lookup_batch_v900 import _CompleteEdgeIndex
        files = json.loads(SUITE_LOCK.read_text())['files']
        index = _CompleteEdgeIndex(ROOT, files)
        if not index.valid():
            raise ValueError('original complete edge index changed after construction')
        report.update(complete=True, declared_files=len(files),
                      unique_canonical_paths=len(index.by_path),
                      watched_directories=len(index.watches),
                      original_v900_topology_constructor_passed=True,
                      original_full_topology_final_check_passed=True,
                      topology_is_point_observation_not_continuous=True,
                      suite_lock=dict(path=str(SUITE_LOCK.relative_to(ROOT)), sha256=SUITE_LOCK_SHA))
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print(json.dumps(dict(report=str(output), sha256=sha(output),
                              complete=report['complete'], error=report.get('error'))))


if __name__ == '__main__':
    main()
