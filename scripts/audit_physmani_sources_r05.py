"""Read-only baseline provenance and equivalence of the existing memory patch."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import einops
from physmani_wm.freegave import FreeGave

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    config=json.loads((ROOT/'config/framework/aligned_protocol_r05.json').read_text())
    repo=Path(config['baseline_repo'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    filename='packages/physmani_wm/physmani_wm/freegave.py'
    original=subprocess.check_output(['git','show',f'{commit}:{filename}'],cwd=repo,text=True)
    node=next(n for n in ast.walk(ast.parse(original)) if isinstance(n,ast.FunctionDef) and n.name=='segment_gaussian')
    namespace=dict(torch=torch,einops=einops,Dict=dict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'committed_segment_gaussian','exec'),namespace)
    torch.manual_seed(148)
    points=torch.randn(513,3)
    cloud=torch.randn(2,3,20,30)
    masks=torch.randint(0,80,(2,3,20,30)).float()
    # Include duplicate nearest points to check the original first-index tie rule.
    cloud[0,:,0,0]=points[0];cloud[0,:,0,20]=points[0]
    data=dict(pcds=cloud,masks=masks)
    original_fake=types.SimpleNamespace(_cano_gaussians=types.SimpleNamespace(get_xyz=points))
    patched_fake=types.SimpleNamespace(_cano_gaussians=types.SimpleNamespace(get_xyz=points))
    namespace['segment_gaussian'](original_fake,data)
    FreeGave.segment_gaussian(patched_fake,data)
    if not torch.equal(original_fake._cano_semantic_mask,patched_fake._cano_semantic_mask):
        raise RuntimeError('existing baseline memory patch changes semantic assignments')
    files=[p for folder in ('packages/physmani_policy/physmani_policy','packages/physmani_wm/physmani_wm','packages/physmani_bench/physmani_bench') for p in (repo/folder).rglob('*.py')]
    files += [repo/'third_party/RLBench/rlbench/backend/scene.py',repo/'third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py',repo/'third_party/PyRep/pyrep/robots/arms/arm.py']
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),repo=str(repo),commit=commit,
        files={str(p):sha(p) for p in sorted(set(files))},baseline_checkpoint_sha256=sha(Path(config['baseline_checkpoint'])),
        checkout_modified_by_our_r05=False,existing_FreeGave_patch='independent-Gaussian nearest-neighbor chunking to limit memory; same torch.norm/argmin and tie rule',
        patch_equivalence_test_passed=True,
        working_tree_status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        current_instance_masks='released evaluator renders object-ID masks; world model uses them for robot-part identification; same sensor available to both methods')
    path=ROOT/'artifacts/framework/aligned_r05/baseline_source_audit.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','working_tree_status')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
