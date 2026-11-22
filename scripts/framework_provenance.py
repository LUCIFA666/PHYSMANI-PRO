"""Record code, frozen weights, and read-only external asset provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args])


def main():
    external = Path('/home/ubuntu/桌面/phsimani/PhysMani')
    faster = ROOT/'third_party/FASTER'
    files = list((ROOT/'our_mani').glob('*.py'))+list((ROOT/'config/framework').glob('*.json'))
    files += list((ROOT/'scripts').glob('*.py'))
    files += [faster/'src/openpi/policies/our_mani_policy.py']
    baseline_files = ['packages/physmani_bench/physmani_bench/rlbench_env.py',
                      'packages/physmani_wm/physmani_wm/freegave.py',
                      'packages/physmani_bench/physmani_bench/contracts.py']
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'ngff_source_commit': git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
              'ngff_frozen_checkpoint': 'artifacts/framework/checkpoints/ngff_best.pt',
              'ngff_frozen_checkpoint_sha256': digest(ROOT/'artifacts/framework/checkpoints/ngff_best.pt'),
              'ngff_best_update': 2751, 'ngff_final_resume_update': 3083, 'ngff_training_resumed': False,
              'faster_source_commit': git(faster, 'rev-parse', 'HEAD').decode().strip(),
              'faster_extension': 'src/openpi/policies/our_mani_policy.py',
              'faster_upstream_tracked_diff_sha256': hashlib.sha256(git(faster, 'diff', 'HEAD')).hexdigest(),
              'baseline_source_commit': git(external, 'rev-parse', 'HEAD').decode().strip(),
              'baseline_worktree_diff_sha256': hashlib.sha256(git(external, 'diff', 'HEAD')).hexdigest(),
              'baseline_local_file_sha256': {p: digest(external/p) for p in baseline_files},
              'baseline_worktree_modified_by_this_task': False,
              'source_file_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
              'framework_smoke_report': 'artifacts/framework/smoke/report.json',
              'primary_protocol': 'config/framework/comparison.json',
              'physical_robot_model': 'not supplied; simulation Panda default'}
    out = ROOT/'artifacts/framework/provenance.json'
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ['ngff_frozen_checkpoint_sha256', 'faster_source_commit', 'baseline_source_commit']}, indent=2))


if __name__ == '__main__':
    main()
