"""Queue only the four repaired basketball cases; keep useful cup runs unchanged."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base=ROOT/'artifacts/framework/multitask_r10';parent=base/'physmani_skills_v550_setup'
    setup=base/'physmani_basketball_v553_setup'
    if setup.exists():raise FileExistsError('preserve original selective submission')
    accepted=json.loads((parent/'CPU_acceptance.json').read_text())
    if accepted.get('complete') is not True:raise ValueError('all actual CPU checks must finish first')
    rows=[r for r in accepted['actual_argv_preflights'] if r['task'].startswith('moving_basketball_in_hoop')]
    if len(rows)!=4 or len({(r['task'],r['method']) for r in rows})!=4:raise ValueError('exact two-task two-method repair subset')
    files=json.loads((parent/'frozen_inputs.json').read_text())['files'];files[str(Path(__file__))]=file_sha(__file__)
    verify_inputs(ROOT,files);setup.mkdir();atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=rows,
        source_CPU_proofs=[dict(path=str((parent/(r['job_id']+'_CPU_preflight.json')).relative_to(ROOT)),
                              sha256=file_sha(parent/(r['job_id']+'_CPU_preflight.json'))) for r in rows],
        actual_argv_unchanged=True,existing_cup_cases_not_repeated=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,actual_CPU_proofs_reused=4,native_cases_to_submit=4,cup_native_repeats=0)))


if __name__=='__main__':main()
