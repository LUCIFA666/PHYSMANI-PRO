"""Prepare one fixed TRAIN diagnostic; preserve the previous source lock."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha, verify_inputs
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
from our_mani.multitask.continuous_reference_actor_backend_v2 import (
    persistent_reference_policy_overlay, persistent_reference_continuation_identity,
)


def main():
    base = ROOT / 'artifacts/framework/multitask_r10/geometry_persistent_reference_smoke_v1_setup'
    out = ROOT / 'artifacts/framework/multitask_r10/geometry_continuous_reference_smoke_v2_setup'
    if out.exists():
        raise RuntimeError('Preserve the existing diagnostic setup; choose a new version.')
    original_lock = json.loads((base / 'frozen_inputs.json').read_text())
    verify_inputs(ROOT, original_lock['files'])
    config = json.loads((base / 'configuration.json').read_text())
    feedback = deepcopy(config['policy_overlay']['configurations']['feedback'])
    if feedback['acceleration_fraction'] != .5 or feedback['phase_tracking_slack_seconds'] != 2.:
        raise RuntimeError('This diagnostic preserves the previous feedback settings.')
    config['policy_overlay'] = persistent_reference_policy_overlay(ROOT, feedback, {}, deepcopy(config['policy_overlay']['configurations']['causal_retarget']))
    task = 'moving_basketball_in_hoop'
    pointer = config['learned_artifacts'][task]['interaction']
    library = InteractionLibrary.load(ROOT / pointer['path'])
    candidate = CandidateConfig(**config['candidate_config'])
    mpc = MPCConfig(**config['mpc_config'])
    identities = {
        predictor: persistent_reference_continuation_identity(ROOT, config, task, predictor, library, candidate, mpc)
        for predictor in ('cv', 'context')
    }
    config['continuation_identities'] = {task: {
        predictor: identity['sha256'] for predictor, identity in identities.items()
    }}
    out.mkdir()
    config_path = out / 'configuration.json'
    atomic_json(config_path, config)
    files = deepcopy(original_lock['files'])
    for path in (
        config_path, Path(__file__),
        ROOT / 'scripts/run_geometry_continuous_reference_smoke_v2_r10.py',
        ROOT / 'scripts/run_continuous_reference_pair_v2_r10.py',
        *(ROOT / source['path'] for source in config['policy_overlay']['sources'].values()),
    ):
        files[str(path.resolve())] = file_sha(path)
    lock = dict(original_lock, schema='r10_geometry_continuous_reference_direct_nonfitting_TRAIN_actual_lock_v2',
        created_utc=datetime.now(timezone.utc).isoformat(), files=files,
        scope='Fixed TRAIN episode104, CV/Context geometry; continuity-verified retarget and reference braking with actual-issued confirmation; direct setpoint unchanged from v1; no g model; no Value, fitting, VAL or TEST.',
        native_started=False)
    atomic_json(out / 'frozen_inputs.json', lock)
    # Archive the actual executable bytes, preserving the complete old lock.
    verify_inputs(ROOT, files)
    source_rows=[]
    snapshot=out/'source_snapshot'
    snapshot.mkdir()
    for index,(name,checksum) in enumerate(sorted(files.items())):
        path=Path(name)
        if path.suffix not in ('.py','.sh'): continue
        target=snapshot/f'{index:04d}_{path.name}'
        shutil.copyfile(path,target)
        if file_sha(target)!=checksum: raise RuntimeError('source snapshot copy changed')
        source_rows.append(dict(original_path=name,sha256=checksum,
            archived_path=str(target.relative_to(ROOT))))
    verify_inputs(ROOT, files)
    atomic_json(out/'source_snapshot.json',dict(
        schema='r10_fixed_TRAIN_smoke_actual_executable_archive_v1',
        files=source_rows,all_frozen_inputs_verified_before_and_after_copy=True,
        original_files_preserved=len(original_lock['files']),
        current_frozen_input_count=len(files),native_started=False))
    atomic_json(out / 'setup_receipt.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        previous_setup=str(base.relative_to(ROOT)),
        previous_config_sha256=file_sha(base / 'configuration.json'),
        previous_lock_sha256=file_sha(base / 'frozen_inputs.json'),
        configuration_sha256=file_sha(config_path),
        frozen_lock_sha256=file_sha(out / 'frozen_inputs.json'),
        overlay=config['policy_overlay'], identities=identities,
        fixed_original_TRAIN_parent='episode104', task=task,
        methods=['cv_geometry', 'context_geometry'],
        original_physics_and_model_sources_verified_unchanged=True,
        original_frozen_input_count=len(original_lock['files']),
        original_frozen_files_preserved_exact=True,
        source_snapshot_sha256=file_sha(out/'source_snapshot.json'),
        original_task_seconds=10., original_dt=.05,
        formal_metric=False, fitting_source=False, native_started=False,
    ))
    print(json.dumps({'setup': str(out), 'configuration_sha256': file_sha(config_path),
        'lock_sha256': file_sha(out / 'frozen_inputs.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
