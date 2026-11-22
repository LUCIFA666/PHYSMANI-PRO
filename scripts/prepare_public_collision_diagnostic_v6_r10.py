"""Freeze metric-only collision witnesses around the unchanged v5 controller."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.public_collision_diagnostic_v6 import binding


def main():
    base=ROOT/'artifacts/framework/multitask_r10/geometry_native_actuator_diagnostic_v5_2_setup'
    out=ROOT/'artifacts/framework/multitask_r10/geometry_public_collision_diagnostic_v6_setup'
    if out.exists():raise RuntimeError('Preserve the prepared namespace')
    old=json.loads((base/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    config=json.loads((base/'configuration.json').read_text())
    b=binding(ROOT,config['baseline_repo']);config['public_collision_query_diagnostics']=b
    out.mkdir();atomic_json(out/'configuration.json',config)
    files=dict(old['files'])
    for p in [out/'configuration.json',Path(__file__),ROOT/b['source']['path'],Path(b['arm_API']['path']),
            ROOT/'scripts/run_geometry_public_collision_diagnostic_v6_r10.py',
            ROOT/'scripts/run_public_collision_diagnostic_pair_v6_r10.py',ROOT/'tests/test_public_collision_diagnostic_v6.py']:
        files[str(p.resolve())]=file_sha(p)
    atomic_json(out/'frozen_inputs.json',dict(old,created_utc=datetime.now(timezone.utc).isoformat(),files=files,
        schema='r10_metric_public_collision_witness_actual_lock_v6',native_started=False,
        scope='Fixed normal basketball TRAIN104 CV/Context. Unchanged v5 policy/model/physics/deadlines; metric-only collision witnesses, no additional physics/camera or policy truth input; no fitting/VAL/TEST/video/retry.'))
    archive=out/'source_snapshot';archive.mkdir();rows=[]
    for i,(name,sha)in enumerate(sorted(files.items())):
        p=Path(name)
        if p.suffix not in ('.py','.sh'):continue
        q=archive/f'{i:04d}_{p.name}';shutil.copyfile(p,q);assert file_sha(q)==sha
        rows.append(dict(original_path=name,sha256=sha,archived_path=str(q.relative_to(ROOT))))
    atomic_json(out/'source_snapshot.json',dict(files=rows,native_started=False))
    verify_inputs(ROOT,files)
    atomic_json(out/'setup_receipt.json',dict(previous_setup=str(base.relative_to(ROOT)),
        original_inputs_preserved=True,frozen_input_count=len(files),policy_identity_unchanged=True,
        configuration_sha256=file_sha(out/'configuration.json'),frozen_inputs_sha256=file_sha(out/'frozen_inputs.json')))
    print(json.dumps(dict(setup=str(out),frozen_input_count=len(files))))


if __name__=='__main__':main()
